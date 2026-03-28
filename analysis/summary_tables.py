#!/usr/bin/env python3
#
# summary_tables.py  Andrew Belles  Mar 27th, 2026
#
# Leakage-adjusted summary tables built from canonical nowcast outputs.
#

import argparse
from pathlib import Path

from analysis.loaders import load_analysis_bundle
from analysis.shared import build_leakage_adjusted_summary_table, write_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write leakage-adjusted summary tables from nowcast runtime outputs.")
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/config.hypothesis.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_analysis_bundle(args.config)
    table = build_leakage_adjusted_summary_table(bundle)
    output = args.output if args.output is not None else bundle.config.paths.leakage_summary_parquet
    write_frame(table, output)


if __name__ == "__main__":
    main()
