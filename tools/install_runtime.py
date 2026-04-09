from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
VENDOR_DIR = BACKEND_DIR / "vendor"
WHEELHOUSE_DIR = PROJECT_ROOT / "wheelhouse"
RELEASE_MANIFEST_PATH = PROJECT_ROOT / "release_manifest.json"
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"
LOCAL_TEMP_DIR = PROJECT_ROOT / ".tmp"
LOCAL_CONDA_ENV_DIR = PROJECT_ROOT / ".conda-env"
RUNTIME_PACKAGE_MODE_PATH = PROJECT_ROOT / ".runtime_package_mode"
DEFAULT_WINDOWS_TORCH_VERSION = "2.7.0"
DEFAULT_WINDOWS_TORCHVISION_VERSION = "0.22.0"
DEFAULT_WINDOWS_TORCHAUDIO_VERSION = "2.7.0"
DEFAULT_WINDOWS_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
_RUNTIME_PACKAGE_MODE: str | None = None

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.append(str(VENDOR_DIR))

REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "torchaudio",
    "einops",
    "transformers",
    "accelerate",
    "fastapi",
    "uvicorn",
    "pydantic",
    "numpy",
    "soundfile",
    "huggingface_hub",
    "multipart",
    "librosa",
)

def log(message: str) -> None:
    print(message, flush=True)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def current_prefix_path() -> Path:
    return Path(sys.prefix).resolve(strict=False)


def is_managed_conda_env() -> bool:
    managed = LOCAL_CONDA_ENV_DIR.resolve(strict=False)
    prefix = current_prefix_path()
    return prefix == managed or managed in prefix.parents


def is_isolated_python_environment() -> bool:
    if os.getenv("CONDA_PREFIX"):
        return True
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return True
    return is_managed_conda_env()


def preferred_runtime_package_mode() -> str:
    if _truthy(os.getenv("TADA_FORCE_PIP_TARGET")):
        return "target"
    if is_isolated_python_environment():
        return "env"
    return "target"


def set_runtime_package_mode(mode: str) -> None:
    global _RUNTIME_PACKAGE_MODE
    normalized = "target" if str(mode).strip().lower() == "target" else "env"
    _RUNTIME_PACKAGE_MODE = normalized
    if normalized == "target":
        LOCAL_SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
        site_packages_path = str(LOCAL_SITE_PACKAGES)
        if site_packages_path not in sys.path:
            sys.path.insert(1, site_packages_path)


def runtime_package_mode() -> str:
    global _RUNTIME_PACKAGE_MODE
    if _RUNTIME_PACKAGE_MODE is None:
        set_runtime_package_mode(preferred_runtime_package_mode())
    return _RUNTIME_PACKAGE_MODE or "env"


def persist_runtime_package_mode() -> None:
    RUNTIME_PACKAGE_MODE_PATH.write_text(f"{runtime_package_mode()}\n", encoding="ascii")


def active_pip_target_dir() -> Path | None:
    if runtime_package_mode() == "target":
        return LOCAL_SITE_PACKAGES
    return None


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    log(f"+ {' '.join(command)}")
    merged_env = os.environ.copy()
    merged_env["TMP"] = str(LOCAL_TEMP_DIR)
    merged_env["TEMP"] = str(LOCAL_TEMP_DIR)
    merged_env["TMPDIR"] = str(LOCAL_TEMP_DIR)
    if env:
        merged_env.update(env)
    subprocess.run(command, cwd=str(cwd or PROJECT_ROOT), env=merged_env, check=True)


def run_pip(arguments: list[str], *, cwd: Path | None = None) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys, tempfile; "
            f"tempfile.tempdir = r'{LOCAL_TEMP_DIR}'; "
            "from pip._internal.cli.main import main; "
            "raise SystemExit(main(sys.argv[1:]))"
        ),
        *arguments,
    ]
    run(command, cwd=cwd)


def run_pip_install(arguments: list[str], *, offline: bool, index_url: str = "", target_dir: Path | None = None) -> None:
    command = ["install", "--upgrade"]
    if offline:
        command.extend(["--no-index", "--find-links", str(WHEELHOUSE_DIR)])
    elif index_url:
        command.extend(["--index-url", index_url])
    if target_dir is not None and "--target" not in arguments:
        target_dir.mkdir(parents=True, exist_ok=True)
        command.extend(["--target", str(target_dir)])
    command.extend(arguments)
    run_pip(command)


def missing_modules() -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def load_release_manifest() -> dict[str, object]:
    if not RELEASE_MANIFEST_PATH.exists():
        return {}
    return json.loads(RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))


def non_torch_requirement_lines() -> list[str]:
    requirements_path = BACKEND_DIR / "requirements.txt"
    lines: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = line.split(";", 1)[0].strip().lower()
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            normalized = normalized.split(separator, 1)[0].strip()
        if normalized in {"torch", "torchvision", "torchaudio"}:
            continue
        lines.append(line)
    return lines


def materialize_runtime_requirements_file() -> Path:
    requirement_lines = non_torch_requirement_lines()
    runtime_requirements_path = LOCAL_TEMP_DIR / "backend-runtime-requirements.txt"
    runtime_requirements_path.write_text("\n".join(requirement_lines) + "\n", encoding="utf-8")
    return runtime_requirements_path


def installed_package_version(distribution_name: str) -> str | None:
    try:
        from importlib import metadata as importlib_metadata

        return importlib_metadata.version(distribution_name)
    except Exception:
        return None


def installed_torch_packages() -> dict[str, str | None]:
    return {
        name: installed_package_version(name)
        for name in ("torch", "torchvision", "torchaudio")
    }


def manifest_torch_package_specs() -> list[str]:
    payload = load_release_manifest()
    packages = payload.get("torch_packages")
    if not isinstance(packages, dict):
        return []
    specs: list[str] = []
    for name in ("torch", "torchvision", "torchaudio"):
        version = str(packages.get(name) or "").strip()
        if version:
            specs.append(f"{name}=={version}")
    return specs


def manifest_torch_matches_installed() -> bool:
    specs = manifest_torch_package_specs()
    if not specs:
        return True
    installed = installed_torch_packages()
    for spec in specs:
        name, version = spec.split("==", 1)
        if installed.get(name) != version:
            return False
    return True


def configured_torch_package_specs() -> list[str]:
    manual_specs = str(os.getenv("TADA_TORCH_PACKAGE_SPECS") or "").strip()
    if manual_specs:
        return [item.strip() for item in manual_specs.split() if item.strip()]

    manifest_specs = manifest_torch_package_specs()
    if manifest_specs:
        return manifest_specs

    torch_version = str(os.getenv("TADA_TORCH_VERSION") or "").strip()
    torchvision_version = str(os.getenv("TADA_TORCHVISION_VERSION") or "").strip()
    torchaudio_version = str(os.getenv("TADA_TORCHAUDIO_VERSION") or "").strip()
    if torch_version or torchvision_version or torchaudio_version:
        return [
            f"torch=={torch_version or DEFAULT_WINDOWS_TORCH_VERSION}",
            f"torchvision=={torchvision_version or DEFAULT_WINDOWS_TORCHVISION_VERSION}",
            f"torchaudio=={torchaudio_version or DEFAULT_WINDOWS_TORCHAUDIO_VERSION}",
        ]

    if platform.system() == "Windows":
        return [
            f"torch=={DEFAULT_WINDOWS_TORCH_VERSION}",
            f"torchvision=={DEFAULT_WINDOWS_TORCHVISION_VERSION}",
            f"torchaudio=={DEFAULT_WINDOWS_TORCHAUDIO_VERSION}",
        ]

    versions = resolved_torch_package_versions()
    if versions:
        return [f"{name}=={versions[name]}" for name in ("torch", "torchvision", "torchaudio") if name in versions]
    return ["torch", "torchvision", "torchaudio"]


def pinned_torch_specs_match_installed(specs: list[str]) -> bool:
    installed = installed_torch_packages()
    for spec in specs:
        if "==" not in spec:
            continue
        name, version = spec.split("==", 1)
        if installed.get(name) != version:
            return False
    return True


def resolve_torch_index_url() -> str:
    override = os.getenv("TADA_TORCH_INDEX_URL", "").strip()
    if override:
        return override
    if _truthy(os.getenv("TADA_PREFER_CPU")):
        return ""
    if platform.system() != "Windows":
        return ""
    return DEFAULT_WINDOWS_TORCH_INDEX_URL


def resolved_torch_package_versions() -> dict[str, str]:
    try:
        import torch

        version = str(torch.__version__)
        public_version, _, local_suffix = version.partition("+")
        major, minor, patch = public_version.split(".")
        torchvision_version = f"0.{int(minor) + 15}.{patch}"
        include_local_suffix = bool(local_suffix and local_suffix.lower() != "cpu" and resolve_torch_index_url())
        suffix = f"+{local_suffix}" if include_local_suffix else ""
        return {
            "torch": public_version + suffix,
            "torchvision": torchvision_version + suffix,
            "torchaudio": public_version + suffix,
        }
    except Exception:
        resolved: dict[str, str] = {}
        for name in ("torch", "torchvision", "torchaudio"):
            version = installed_package_version(name)
            if version:
                resolved[name] = version
        return resolved


def torch_package_specs() -> list[str]:
    return configured_torch_package_specs()


def pip_install(arguments: list[str], *, offline: bool, index_url: str = "") -> None:
    explicit_target = "--target" in arguments
    target_dir = active_pip_target_dir()
    if explicit_target or target_dir is not None:
        run_pip_install(arguments, offline=offline, index_url=index_url, target_dir=target_dir)
        return

    try:
        run_pip_install(arguments, offline=offline, index_url=index_url, target_dir=None)
    except subprocess.CalledProcessError:
        log(
            "Direct pip installation into the managed environment failed. "
            "Retrying with a project-local package target (.python_packages)."
        )
        set_runtime_package_mode("target")
        run_pip_install(arguments, offline=offline, index_url=index_url, target_dir=LOCAL_SITE_PACKAGES)


def ensure_pip_available() -> None:
    if importlib.util.find_spec("pip") is not None:
        return
    try:
        run([sys.executable, "-m", "ensurepip", "--upgrade"])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "The selected Python interpreter does not provide a usable pip installation. "
            "Use Python 3.10+ with venv/pip support or set TADA_PYTHON to another interpreter."
        ) from exc


def upgrade_pip(*, offline: bool) -> None:
    if offline:
        log("Wheelhouse detected. Skipping pip self-upgrade for offline install mode.")
        return
    try:
        run_pip(["install", "--upgrade", "pip"])
    except subprocess.CalledProcessError:
        log("Warning: pip self-upgrade failed. Continuing with the existing pip version.")


def detect_npm() -> str | None:
    return shutil.which("npm.cmd") or shutil.which("npm")


def should_force_frontend_build() -> bool:
    return os.getenv("TADA_FORCE_FRONTEND_BUILD", "0").lower() in {"1", "true", "yes"}


def validate_release_manifest() -> None:
    payload = load_release_manifest()
    if not payload:
        return
    expected_platform = str(payload.get("platform") or "").strip()
    expected_arch = str(payload.get("architecture") or "").strip().lower()
    expected_python = str(payload.get("python_version") or "").strip()

    actual_platform = platform.system()
    actual_arch = platform.machine().lower()
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"

    if expected_platform and expected_platform != actual_platform:
        raise RuntimeError(
            f"This release bundle targets {expected_platform}, but the current host is {actual_platform}."
        )
    if expected_arch and expected_arch not in actual_arch:
        raise RuntimeError(
            f"This release bundle targets architecture '{expected_arch}', but the current host is '{actual_arch}'."
        )
    if expected_python and expected_python != actual_python:
        raise RuntimeError(
            f"This release bundle expects Python {expected_python}, but the current interpreter is {actual_python}."
        )


def require_supported_python() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"Python 3.10 or newer is required for this project. Current interpreter: {sys.version.split()[0]}."
        )


def install_base_runtime(*, offline: bool, force: bool = False) -> None:
    missing = missing_modules()
    strict_bundle_runtime = offline and os.getenv("TADA_STRICT_BUNDLE_RUNTIME", "0").lower() in {"1", "true", "yes"}
    manifest_requires_torch_sync = strict_bundle_runtime and not manifest_torch_matches_installed()
    if offline and not strict_bundle_runtime and not manifest_torch_matches_installed():
        log(
            "Release bundle torch package versions differ from the current interpreter. "
            "Continuing because the runtime will be validated by the smokecheck. "
            "Set TADA_STRICT_BUNDLE_RUNTIME=1 to force the bundle's exact wheel set."
        )
    torch_specs = torch_package_specs()
    torch_versions_require_sync = pinned_torch_specs_match_installed(torch_specs) is False
    if not missing and not force and not manifest_requires_torch_sync:
        if not torch_versions_require_sync:
            log(f"Base runtime already available: {json.dumps({'missing': []})}")
            return
        log(
            "Base runtime modules are importable, but the managed torch stack does not match the requested versions. "
            "Reinstalling torch packages into the active environment."
        )

    log(f"Missing base runtime modules: {json.dumps({'missing': missing})}")
    should_install_torch_stack = force or manifest_requires_torch_sync or any(
        name in missing for name in ("torch", "torchvision", "torchaudio")
    ) or torch_versions_require_sync
    if should_install_torch_stack:
        pip_install(torch_specs, offline=offline, index_url=resolve_torch_index_url())

    pip_install(["-r", str(materialize_runtime_requirements_file())], offline=offline)


def install_vendor_packages(*, offline: bool) -> None:
    vendor_entry = VENDOR_DIR / "tada" / "modules" / "tada.py"
    if vendor_entry.exists() and os.getenv("TADA_FORCE_VENDOR_REINSTALL", "0") not in {"1", "true", "yes"}:
        log("Repo vendor is already present. Keeping local patches and skipping vendor reinstall.")
        return

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    pip_install(
        ["--target", str(VENDOR_DIR), "--no-deps", "hume-tada", "descript-audio-codec"],
        offline=offline,
    )
    pip_install(
        ["--target", str(VENDOR_DIR), "argbind", "descript-audiotools"],
        offline=offline,
    )


def install_optional_windows_triton(*, offline: bool) -> None:
    if platform.system() != "Windows":
        return
    if os.getenv("TADA_SKIP_TRITON_WINDOWS", "0").lower() in {"1", "true", "yes"}:
        log("Skipping optional triton-windows install because TADA_SKIP_TRITON_WINDOWS=1.")
        return
    if importlib.util.find_spec("triton") is not None:
        log("Optional triton runtime is already importable.")
        return
    try:
        pip_install(["triton-windows<3.7"], offline=offline)
    except subprocess.CalledProcessError:
        log("Warning: optional triton-windows installation failed. Continuing without torch.compile support.")


def ensure_frontend_build() -> None:
    if FRONTEND_DIST_DIR.joinpath("index.html").exists() and not should_force_frontend_build():
        log("Frontend dist already present. Skipping npm install/build.")
        return

    npm = detect_npm()
    if not npm:
        raise RuntimeError(
            "Frontend build assets are missing and npm is not available. "
            "Either install Node.js/npm or provide a bundle that already contains frontend/dist."
        )

    install_command = [npm, "ci"] if (FRONTEND_DIR / "package-lock.json").exists() else [npm, "install"]
    build_command = [npm, "run", "build"]
    run(install_command, cwd=FRONTEND_DIR)
    run(build_command, cwd=FRONTEND_DIR)


def smokecheck_runtime_import() -> bool:
    env = os.environ.copy()
    env["TMP"] = str(LOCAL_TEMP_DIR)
    env["TEMP"] = str(LOCAL_TEMP_DIR)
    env["TMPDIR"] = str(LOCAL_TEMP_DIR)
    pythonpath_items: list[str] = []
    if runtime_package_mode() == "target" and LOCAL_SITE_PACKAGES.exists():
        pythonpath_items.append(str(LOCAL_SITE_PACKAGES))
    pythonpath_items.append(str(PROJECT_ROOT))
    if env.get("PYTHONPATH"):
        pythonpath_items.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_items)
    env.setdefault("TADA_DISABLE_WARMUP", "1")
    command = [
        sys.executable,
        "-c",
        (
            "import json; "
            "import backend.server_app as app_module; "
            "status = app_module.runtime_service.status(); "
            "print(json.dumps(status, ensure_ascii=True)); "
            "raise SystemExit(0 if status.get('ok') else 3)"
        ),
    ]
    log(f"+ {' '.join(command)}")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env)
    return result.returncode == 0


def main() -> int:
    os.chdir(PROJECT_ROOT)
    LOCAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    offline = WHEELHOUSE_DIR.exists() and any(WHEELHOUSE_DIR.iterdir())
    set_runtime_package_mode(preferred_runtime_package_mode())
    install_mode = "active environment" if runtime_package_mode() == "env" else ".python_packages fallback"

    log(f"[1/6] Validating release manifest (if present) with Python {sys.version.split()[0]}...")
    require_supported_python()
    validate_release_manifest()
    ensure_pip_available()
    log(f"Install mode: {install_mode}")

    log("[2/6] Upgrading pip...")
    upgrade_pip(offline=offline)

    log("[3/6] Installing base runtime...")
    install_base_runtime(offline=offline)

    log("[4/6] Ensuring local vendor packages...")
    install_vendor_packages(offline=offline)

    log("[5/6] Building or validating frontend assets...")
    ensure_frontend_build()

    log("[6/6] Running runtime smokecheck...")
    install_optional_windows_triton(offline=offline)
    if not smokecheck_runtime_import():
        log("Runtime smokecheck failed. Reinstalling the pinned core runtime once and retrying...")
        install_base_runtime(offline=offline, force=True)
        if not smokecheck_runtime_import():
            raise RuntimeError("Runtime smokecheck failed even after reinstalling the pinned core runtime.")
    persist_runtime_package_mode()
    log(f"Runtime package mode recorded in {RUNTIME_PACKAGE_MODE_PATH.name}: {runtime_package_mode()}")
    log("Install/runtime validation finished successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        log(f"[ERROR] {exc}")
        raise SystemExit(1) from exc
