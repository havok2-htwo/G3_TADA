from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from importlib import metadata as importlib_metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
RELEASES_DIR = PROJECT_ROOT / "releases"
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".python_packages"
RUNTIME_PACKAGE_MODE_PATH = PROJECT_ROOT / ".runtime_package_mode"
KNOWN_MODELS = (
    "HumeAI/tada-3b-ml",
    "HumeAI/tada-1b",
    "HumeAI/tada-codec",
    "meta-llama/Llama-3.2-1B",
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if RUNTIME_PACKAGE_MODE_PATH.exists():
    try:
        mode = RUNTIME_PACKAGE_MODE_PATH.read_text(encoding="ascii").strip().lower()
    except OSError:
        mode = ""
    if mode == "target" and LOCAL_SITE_PACKAGES.exists() and str(LOCAL_SITE_PACKAGES) not in sys.path:
        sys.path.insert(1, str(LOCAL_SITE_PACKAGES))

from backend.hf_cache import resolve_cached_snapshot


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    log(f"+ {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd or PROJECT_ROOT), check=True)


def installed_package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def resolve_torch_index_url() -> str:
    override = os.getenv("TADA_TORCH_INDEX_URL", "").strip()
    if override:
        return override
    if platform.system() != "Windows":
        return ""
    try:
        import torch

        cuda_version = str(torch.version.cuda or "").strip()
        if not cuda_version:
            return ""
        return f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
    except Exception:
        return ""


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
    versions = resolved_torch_package_versions()
    return [f"{name}=={versions[name]}" for name in ("torch", "torchvision", "torchaudio") if name in versions]


def read_non_torch_requirements() -> list[str]:
    requirements_path = BACKEND_DIR / "requirements.txt"
    packages: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = line.split(";", 1)[0].strip().lower()
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            normalized = normalized.split(separator, 1)[0].strip()
        if normalized in {"torch", "torchvision", "torchaudio"}:
            continue
        packages.append(line)
    return packages


def try_download(command: list[str]) -> bool:
    try:
        run(command)
        return True
    except subprocess.CalledProcessError:
        return False


def installed_torch_packages() -> dict[str, str]:
    versions = resolved_torch_package_versions()
    return {name: versions.get(name, "missing") for name in ("torch", "torchvision", "torchaudio")}


def load_settings() -> dict[str, str]:
    settings_path = BACKEND_DIR / "data" / "server_settings.json"
    if not settings_path.exists():
        return {}
    return json.loads(settings_path.read_text(encoding="utf-8"))


def model_storage_root() -> Path:
    settings = load_settings()
    configured = str(settings.get("model_storage_path") or "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve(strict=False)
        return candidate
    return (PROJECT_ROOT / ".hf_cache" / "hub").resolve(strict=False)


def copy_tree(source: Path, destination: Path, *, ignore=None) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=ignore)


def build_manifest(included_models: list[dict[str, str]]) -> dict[str, object]:
    torch_version = "unknown"
    cuda_version = "unknown"
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = str(torch.version.cuda)
    except Exception:
        pass

    return {
        "bundle_format": "tada-windows-fat-bundle",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "torch_packages": installed_torch_packages(),
        "included_models": included_models,
    }


def create_wheelhouse(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    torch_specs = torch_package_specs()
    torch_index_url = resolve_torch_index_url()
    downloaded_torch = False
    if torch_index_url:
        downloaded_torch = try_download(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "-d",
                str(target_dir),
                *torch_specs,
                "--index-url",
                torch_index_url,
            ]
        )
        if not downloaded_torch:
            log(
                "Warning: torch wheel download from the configured torch index failed. "
                "Falling back to the default package index."
            )
    if not downloaded_torch:
        run([sys.executable, "-m", "pip", "download", "-d", str(target_dir), *torch_specs])

    non_torch_requirements = read_non_torch_requirements()
    if non_torch_requirements:
        run([sys.executable, "-m", "pip", "download", "-d", str(target_dir), *non_torch_requirements])

    if platform.system() == "Windows":
        if not try_download(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "-d",
                str(target_dir),
                "triton-windows<3.7",
            ]
        ):
            log("Warning: optional triton-windows wheel could not be downloaded. Continuing without it.")


def copy_model_caches(bundle_root: Path) -> list[dict[str, str]]:
    included: list[dict[str, str]] = []
    cache_root = model_storage_root()
    target_cache_root = bundle_root / ".hf_cache" / "hub"
    target_cache_root.mkdir(parents=True, exist_ok=True)

    for model_id in KNOWN_MODELS:
        snapshot = resolve_cached_snapshot(model_id, PROJECT_ROOT, extra_roots=[cache_root])
        if snapshot is None:
            included.append({"model_id": model_id, "status": "missing"})
            continue
        repo_root = snapshot.parent.parent
        destination = target_cache_root / repo_root.name
        copy_tree(repo_root, destination)
        included.append(
            {
                "model_id": model_id,
                "status": "ready",
                "relative_path": str(Path(".hf_cache") / "hub" / repo_root.name),
            }
        )
    return included


def populate_bundle(bundle_root: Path) -> dict[str, object]:
    log("Copying backend runtime...")
    copy_tree(
        BACKEND_DIR,
        bundle_root / "backend",
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "generated",
            "cache",
            "server_secrets.json",
            "server_settings.json.tmp",
        ),
    )

    settings_path = BACKEND_DIR / "data" / "server_settings.json"
    if settings_path.exists():
        destination_settings = bundle_root / "backend" / "data" / "server_settings.json"
        destination_settings.parent.mkdir(parents=True, exist_ok=True)
        settings_payload = json.loads(settings_path.read_text(encoding="utf-8"))
        settings_payload["model_storage_path"] = str(Path(".hf_cache") / "hub")
        destination_settings.write_text(
            json.dumps(settings_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    log("Copying built frontend...")
    if not FRONTEND_DIST_DIR.joinpath("index.html").exists():
        raise RuntimeError("frontend/dist is missing. Build the frontend before packaging a release bundle.")
    copy_tree(FRONTEND_DIST_DIR, bundle_root / "frontend" / "dist")
    copy_tree(PROJECT_ROOT / "tools", bundle_root / "tools", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    copy_tree(PROJECT_ROOT / "torchvision", bundle_root / "torchvision", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    for file_name in ("README.md", "API_DOCUMENTATION.md", "install.bat", "start.bat", "install.sh", "start.sh", "package_release.bat"):
        source = PROJECT_ROOT / file_name
        if source.exists():
            shutil.copy2(source, bundle_root / file_name)

    log("Downloading wheelhouse...")
    create_wheelhouse(bundle_root / "wheelhouse")

    log("Copying local model snapshots...")
    included_models = copy_model_caches(bundle_root)
    manifest = build_manifest(included_models)
    (bundle_root / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest


def zip_bundle(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in source_dir.rglob("*"):
            archive.write(path, path.relative_to(source_dir))


def main() -> int:
    if platform.system() != "Windows":
        raise RuntimeError("The fat release bundle is currently only supported from Windows hosts.")

    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bundle_name = f"G3_TADA3B-windows-fat-bundle-{stamp}"
    staging_root = RELEASES_DIR / "_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    bundle_root = staging_root / bundle_name
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    manifest = populate_bundle(bundle_root)
    zip_path = RELEASES_DIR / f"{bundle_name}.zip"
    log(f"Creating ZIP at {zip_path} ...")
    zip_bundle(bundle_root, zip_path)
    log(f"Release bundle created: {zip_path}")
    log(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        log(f"[ERROR] {exc}")
        raise SystemExit(1) from exc
