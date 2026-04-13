from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.config_store import ConfigStore
from backend.runtime_service import BENCHMARK_FIXTURES_DIR, TadaRuntimeService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark voice fixtures from the raw voices corpus.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--source-dir", default="voices", help="Directory with raw voice reference files.")
    parser.add_argument(
        "--output-dir",
        default=str(BENCHMARK_FIXTURES_DIR),
        help="Directory where prepared fixtures and the manifest will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_store = ConfigStore(project_root)
    runtime_service = TadaRuntimeService(project_root, config_store)
    result = runtime_service.build_benchmark_corpus(
        source_dir=(project_root / args.source_dir),
        output_dir=(project_root / args.output_dir),
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
