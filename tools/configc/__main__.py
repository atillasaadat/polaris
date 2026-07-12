"""CLI for the config compiler (design doc §19.3).

    PYTHONPATH=tools uv run python -m configc \
        --config config/spacecraft/leo_smallsat.yaml \
        --hardware config/hardware \
        --out build/config

Resolves hardware model-IDs, validates the config, and writes the F´-param /
sim / analysis stub artifacts to ``--out``. Exits non-zero with a readable
message on any validation or resolution failure (REQ-CFG-001).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .compiler import ConfigError, compile_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="configc", description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="spacecraft+scenario YAML config"
    )
    parser.add_argument(
        "--hardware",
        required=True,
        type=Path,
        help="hardware-model-library directory (*.yaml)",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="output directory for emitted artifacts"
    )
    args = parser.parse_args(argv)

    try:
        resolved = compile_config(args.config, args.hardware, args.out)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"configc: {exc}", file=sys.stderr)
        return 1

    print(
        f"configc: ok — config_hash={resolved['provenance']['config_hash'][:12]} "
        f"-> {args.out}/{{fprime_params,sim_setup,analysis_inputs}}.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
