from __future__ import annotations

import os
from pathlib import Path


def _absolute_from_project(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def candidate_hf_cache_roots(project_root: Path, extra_roots: list[Path] | None = None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        normalized = os.path.normcase(str(path.resolve(strict=False)))
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append(path)

    hf_hub_cache = os.getenv("HF_HUB_CACHE")
    if hf_hub_cache:
        add(_absolute_from_project(hf_hub_cache, project_root))

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        add(_absolute_from_project(hf_home, project_root) / "hub")

    add((project_root / ".hf_cache" / "hub").resolve(strict=False))
    add((project_root / "backend" / "cache" / "huggingface" / "hub").resolve(strict=False))
    for extra_root in extra_roots or []:
        add(extra_root.resolve(strict=False))

    return candidates


def resolve_cached_snapshot(repo_id: str, project_root: Path, extra_roots: list[Path] | None = None) -> Path | None:
    if "/" not in repo_id:
        return None

    repo_dir_name = f"models--{repo_id.replace('/', '--')}"
    for cache_root in candidate_hf_cache_roots(project_root, extra_roots=extra_roots):
        repo_root = cache_root / repo_dir_name
        if not repo_root.exists():
            continue

        ref_main = repo_root / "refs" / "main"
        if ref_main.exists():
            revision = ref_main.read_text(encoding="utf-8").strip()
            if revision:
                snapshot = repo_root / "snapshots" / revision
                if snapshot.exists():
                    return snapshot

        snapshots_dir = repo_root / "snapshots"
        if snapshots_dir.exists():
            snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
            if snapshots:
                return max(snapshots, key=lambda path: path.stat().st_mtime)

    return None


def resolve_pretrained_source(
    name_or_path: str,
    project_root: Path,
    extra_roots: list[Path] | None = None,
) -> tuple[str, bool]:
    direct_path = _absolute_from_project(name_or_path, project_root)
    if direct_path.exists():
        return str(direct_path), True

    cached_snapshot = resolve_cached_snapshot(name_or_path, project_root, extra_roots=extra_roots)
    if cached_snapshot is not None:
        return str(cached_snapshot), True

    return name_or_path, False
