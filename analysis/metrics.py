#!/usr/bin/env python3
#
# metrics.py  Andrew Belles  Mar 28th, 2026
#
# Compact model-level censal metrics table for README/reporting.
#

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.loaders import AnalysisBundle, load_analysis_bundle
from analysis.shared import write_frame


LOGGER = logging.getLogger("analysis.metrics")


def setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[keep]
    yv = yv[keep]
    if xv.size <= 1:
        return float("nan")
    if float(np.std(xv, ddof=0)) <= 0.0 or float(np.std(yv, ddof=0)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _format_metric(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns.tolist()]
    header = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_format_metric(row[c]) for c in cols) + " |")
    return "\n".join([header, rule] + rows)


def build_censal_metrics_table(bundle: AnalysisBundle) -> pd.DataFrame:
    summary = bundle.censal_summary.copy()
    fold = bundle.censal_fold_metrics.copy()
    abs_df = bundle.censal_abs_errors.copy()

    fold["model"] = fold["model"].astype(str).str.strip().str.lower()
    abs_df["model"] = abs_df["model"].astype(str).str.strip().str.lower()
    summary["model"] = summary["model"].astype(str).str.strip().str.lower()

    fold_scale = fold.loc[:, ["fold", "model", "topology_leakage_proxy", "relative_error_improvement_pct"]].copy()
    fold_scale["topology_leakage_proxy"] = pd.to_numeric(fold_scale["topology_leakage_proxy"], errors="coerce").fillna(0.0)
    fold_scale["relative_error_improvement_pct"] = pd.to_numeric(fold_scale["relative_error_improvement_pct"], errors="coerce").fillna(0.0)
    fold_scale["fold_adjustment_scale"] = np.where(
        np.asarray(fold_scale["relative_error_improvement_pct"], dtype=np.float64) > 0.0,
        1.0 - np.asarray(fold_scale["topology_leakage_proxy"], dtype=np.float64),
        1.0,
    )
    fold_scale["fold_adjustment_scale"] = np.clip(np.asarray(fold_scale["fold_adjustment_scale"], dtype=np.float64), 0.0, None)

    long_df = abs_df.loc[
        :,
        [
            "fips",
            "state",
            "fold",
            "model",
            "y_log",
            "y_level",
            "pep_log",
            "pep_level",
            "pred_log",
            "pred_level",
            "true_resid_log",
            "pred_correction_log",
            "ape_pop_pct",
        ],
    ].copy()
    long_df = long_df.merge(
        fold_scale.loc[:, ["fold", "model", "fold_adjustment_scale"]],
        on=["fold", "model"],
        how="left",
        validate="many_to_one",
    )
    long_df["fold_adjustment_scale"] = pd.to_numeric(long_df["fold_adjustment_scale"], errors="coerce").fillna(1.0)
    long_df["adjusted_pred_correction_log"] = (
        np.asarray(long_df["pred_correction_log"], dtype=np.float64) * np.asarray(long_df["fold_adjustment_scale"], dtype=np.float64)
    )
    long_df["adjusted_pred_log"] = np.asarray(long_df["pep_log"], dtype=np.float64) + np.asarray(long_df["adjusted_pred_correction_log"], dtype=np.float64)

    rows: list[dict[str, object]] = []
    for model, part in long_df.groupby("model", sort=True):
        true_resid = np.asarray(part["true_resid_log"], dtype=np.float64)
        pred_corr = np.asarray(part["pred_correction_log"], dtype=np.float64)
        adjusted_pred_corr = np.asarray(part["adjusted_pred_correction_log"], dtype=np.float64)
        y_log = np.asarray(part["y_log"], dtype=np.float64)
        pred_log = np.asarray(part["pred_log"], dtype=np.float64)
        adjusted_pred_log = np.asarray(part["adjusted_pred_log"], dtype=np.float64)
        resid_err = np.asarray(y_log - pred_log, dtype=np.float64)
        adjusted_resid_err = np.asarray(y_log - adjusted_pred_log, dtype=np.float64)
        rows.append(
            {
                "model": str(model),
                "mape_pop_pct_mean": float(summary.loc[summary["model"] == model, "mape_pop_pct_mean"].iloc[0]),
                "adjusted_mape_pop_pct_mean": float(summary.loc[summary["model"] == model, "adjusted_mape_pop_pct_mean"].iloc[0]),
                "residual_error_corr_pearson": _safe_pearson(pred_log, resid_err),
                "signal_corr_pearson": _safe_pearson(y_log, pred_log),
                "adjusted_residual_error_corr_pearson": _safe_pearson(adjusted_pred_log, adjusted_resid_err),
                "adjusted_signal_corr_pearson": _safe_pearson(y_log, adjusted_pred_log),
                "n_eval": int(part.shape[0]),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values(["adjusted_mape_pop_pct_mean", "mape_pop_pct_mean", "model"], ascending=[True, True, True]).reset_index(drop=True)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write compact censal metrics tables from canonical nowcast outputs.")
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/config.metrics.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    bundle = load_analysis_bundle(args.config)
    table = build_censal_metrics_table(bundle)
    output = args.output if args.output is not None else (bundle.config.paths.output_root / "censal_metrics.parquet")
    write_frame(table, output)
    keep = [
        "model",
        "mape_pop_pct_mean",
        "adjusted_mape_pop_pct_mean",
        "residual_error_corr_pearson",
        "signal_corr_pearson",
        "adjusted_residual_error_corr_pearson",
        "adjusted_signal_corr_pearson",
    ]
    present = [c for c in keep if c in table.columns]
    if present:
        LOGGER.info("censal metrics\n%s", _markdown_table(table.loc[:, present].copy()))
    LOGGER.info("wrote metrics table to %s", output)


if __name__ == "__main__":
    main()
