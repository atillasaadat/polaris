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

from .compiler import PRMDB_FILENAME, ConfigError, compile_config


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
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=None,
        help=(
            "FPP topology dictionary JSON (build-artifacts/Linux/"
            "flight_PolarisFsw/dict/PolarisFswTopologyDictionary.json). Supplying "
            "it emits the binary PrmDb.dat parameter file, whose IDs come from "
            "the dictionary; omit it to emit only the JSON artifacts"
        ),
    )
    args = parser.parse_args(argv)

    try:
        resolved = compile_config(
            args.config, args.hardware, args.out, dictionary_path=args.dictionary
        )
    except (ConfigError, FileNotFoundError) as exc:
        print(f"configc: {exc}", file=sys.stderr)
        return 1

    artifacts = "{fprime_params,sim_setup,analysis_inputs}.json"
    if args.dictionary is not None:
        artifacts += f" + {PRMDB_FILENAME}"
    print(
        f"configc: ok — config_hash={resolved['provenance']['config_hash'][:12]} "
        f"-> {args.out}/{artifacts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
