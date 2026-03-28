#!/usr/bin/env python3
#
# hypothesis.py  Andrew Belles  Mar 27th, 2026
#
# Config-driven hypothesis testing for meaningful improvement and nowcast safety.
#

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.loaders import AnalysisBundle, load_analysis_bundle
from analysis.shared import (
    build_county_pair_frame,
    build_nowcast_safety_rows,
    build_state_pair_frame,
    build_year_safety_frame,
    one_sided_bootstrap_test,
    one_sided_majority_test,
    resolve_treatment_model,
    write_frame,
)


LOGGER = logging.getLogger("analysis.hypothesis")


def setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout)


def _result_row(
    *,
    hypothesis_id: str,
    family: str,
    subset: str,
    metric: str,
    threshold: float,
    test_name: str,
    stats: dict[str, float | int | bool],
    note: str,
) -> dict[str, object]:
    return {
        "hypothesis_id": hypothesis_id,
        "family": family,
        "subset": subset,
        "metric": metric,
        "threshold": float(threshold),
        "test_name": str(test_name),
        "estimate": float(stats["estimate"]),
        "ci_low": float(stats["ci_low"]) if np.isfinite(float(stats["ci_low"])) else np.nan,
        "ci_high": float(stats["ci_high"]) if np.isfinite(float(stats["ci_high"])) else np.nan,
        "p_value": float(stats["p_value"]),
        "n_obs": int(stats["n_obs"]),
        "n_groups": int(stats["n_groups"]),
        "passed": bool(stats["passed"]),
        "note": str(note),
    }


def run_hypothesis_tests(bundle: AnalysisBundle) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = bundle.config
    resolved_treatment = resolve_treatment_model(bundle)
    if resolved_treatment != str(cfg.comparison.treatment_model).strip().lower():
        LOGGER.warning(
            "configured treatment_model=%s not present in strict outputs; using best available model=%s",
            str(cfg.comparison.treatment_model),
            resolved_treatment,
        )
    county_pairs = build_county_pair_frame(bundle)
    state_pairs = build_state_pair_frame(
        county_pairs,
        equal_tolerance_pct=float(cfg.hypothesis.state_equal_tolerance_pct),
    )
    year_safety = build_year_safety_frame(bundle)
    safety_rows = build_nowcast_safety_rows(bundle, cfg.safety)
    results: list[dict[str, object]] = []

    for i, stratum in enumerate(cfg.selection.improvement_strata):
        part = county_pairs.loc[county_pairs["analysis_stratum"].astype(str) == str(stratum)].copy()
        if part.empty:
            LOGGER.warning("skip stratum hypothesis for %s because no counties matched", stratum)
            continue
        stats = one_sided_bootstrap_test(
            part,
            value_col="ape_improvement_pct",
            threshold=float(cfg.hypothesis.non_negligible_ape_pct),
            draws=int(cfg.hypothesis.bootstrap_draws),
            seed=int(cfg.hypothesis.random_seed) + i,
            alpha=float(cfg.hypothesis.alpha),
            group_col="state",
            strata_col="division",
        )
        results.append(
            _result_row(
                hypothesis_id="strict_stratum_improvement",
                family="meaningful_improvement",
                subset=str(stratum),
                metric="mean_county_ape_improvement_pct",
                threshold=float(cfg.hypothesis.non_negligible_ape_pct),
                test_name=str(cfg.hypothesis.paired_test),
                stats=stats,
                note="state-cluster bootstrap stratified by census division over strict 2020 OOF county deltas",
            )
        )

    cutoff = float(np.quantile(np.asarray(county_pairs["ape_improvement_pct"], dtype=np.float64), float(cfg.selection.worst_regression_quantile)))
    worst = county_pairs.loc[np.asarray(county_pairs["ape_improvement_pct"], dtype=np.float64) <= cutoff].copy()
    majority_stats = one_sided_majority_test(
        worst["small_pop_lt_25k"],
        threshold=float(cfg.hypothesis.majority_threshold),
        alpha=float(cfg.hypothesis.alpha),
    )
    results.append(
        _result_row(
            hypothesis_id="worst_regressions_concentrated_lt25k",
            family="meaningful_improvement",
            subset=f"worst_{int(round(float(cfg.selection.worst_regression_quantile) * 100.0))}pct",
            metric="share_small_pop_lt_25k",
            threshold=float(cfg.hypothesis.majority_threshold),
            test_name="exact_binomial",
            stats=majority_stats,
            note="tests whether the worst county-level disimprovements are majority concentrated below 25k population",
        )
    )

    state_success = np.asarray(state_pairs["state_win"], dtype=bool)
    state_stats = one_sided_majority_test(
        state_success,
        threshold=float(cfg.hypothesis.majority_threshold),
        alpha=float(cfg.hypothesis.alpha),
    )
    results.append(
        _result_row(
            hypothesis_id="majority_states_improve",
            family="meaningful_improvement",
            subset="states",
            metric="share_states_with_win_outcome",
            threshold=float(cfg.hypothesis.majority_threshold),
            test_name="exact_binomial",
            stats=state_stats,
            note=(
                "tests whether a majority of state-level strict OOF estimates are wins over PEP "
                f"using equal tolerance={cfg.hypothesis.state_equal_tolerance_pct:.6g} MAPE points"
            ),
        )
    )

    fold_metrics = bundle.censal_fold_metrics.copy()
    fold_metrics["model"] = fold_metrics["model"].astype(str).str.strip().str.lower()
    treatment_fold = fold_metrics.loc[fold_metrics["model"] == resolved_treatment].copy()
    leakage_stats = one_sided_majority_test(
        np.asarray(treatment_fold["adjusted_relative_improvement_pct"], dtype=np.float64) > float(cfg.hypothesis.adjusted_relative_pct_threshold),
        threshold=float(cfg.hypothesis.majority_threshold),
        alpha=float(cfg.hypothesis.alpha),
    )
    results.append(
        _result_row(
            hypothesis_id="leakage_adjusted_improvement_positive",
            family="meaningful_improvement",
            subset="strict_folds",
            metric="share_folds_with_positive_adjusted_relative_improvement",
            threshold=float(cfg.hypothesis.majority_threshold),
            test_name="exact_binomial",
            stats=leakage_stats,
            note=(
                "tests whether a majority of held-out strict folds retain positive adjusted relative improvement "
                f"after subtracting leakage-attributable gain; mean_adjusted_relative_improvement_pct="
                f"{float(np.asarray(treatment_fold['adjusted_relative_improvement_pct'], dtype=np.float64).mean()):.6f}"
            ),
        )
    )

    postcensal = safety_rows.loc[np.asarray(safety_rows["year"], dtype=np.int64) > int(cfg.selection.anchor_year)].copy()
    if postcensal.empty:
        LOGGER.warning("no postcensal county-year rows found after anchor year=%d", int(cfg.selection.anchor_year))
    else:
        correction_stats = one_sided_majority_test(
            postcensal["within_abs_correction_bound"],
            threshold=float(cfg.safety.bounded_share_threshold),
            alpha=float(cfg.hypothesis.alpha),
        )
        results.append(
            _result_row(
                hypothesis_id="postcensal_abs_correction_bounded",
                family="safety",
                subset="post_anchor_county_years",
                metric="share_within_abs_correction_bound",
                threshold=float(cfg.safety.bounded_share_threshold),
                test_name="exact_binomial",
                stats=correction_stats,
                note=f"bound is |corrected_log - pep_log| <= {cfg.safety.max_abs_correction_log_warn:.3f}",
            )
        )
        growth_stats = one_sided_majority_test(
            postcensal["within_growth_ratio_bound"],
            threshold=float(cfg.safety.bounded_share_threshold),
            alpha=float(cfg.hypothesis.alpha),
        )
        results.append(
            _result_row(
                hypothesis_id="postcensal_growth_ratio_bounded",
                family="safety",
                subset="post_anchor_county_years",
                metric="share_within_growth_ratio_bound",
                threshold=float(cfg.safety.bounded_share_threshold),
                test_name="exact_binomial",
                stats=growth_stats,
                note=f"bound is max(corrected/pep, pep/corrected) <= {cfg.safety.max_growth_ratio_warn:.3f}",
            )
        )

    result_df = pd.DataFrame(results).sort_values(["family", "hypothesis_id", "subset"]).reset_index(drop=True)
    return result_df, county_pairs, state_pairs, year_safety


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run meaningful-improvement and safety hypothesis tests on nowcast parquet outputs.")
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/config.hypothesis.yaml"))
    parser.add_argument("--skip", action=argparse.BooleanOptionalAction, default=False, help="skip if hypothesis result parquet already exists")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def log_hypothesis_summary(*, results: pd.DataFrame, state_pairs: pd.DataFrame, summary: dict[str, object]) -> None:
    LOGGER.info(
        "hypothesis summary baseline=%s treatment=%s passed=%d/%d state_wins=%d state_losses=%d state_equal=%d",
        str(summary["baseline_model"]),
        str(summary["treatment_model"]),
        int(summary["n_passed"]),
        int(summary["n_hypotheses"]),
        int(summary["state_win_count"]),
        int(summary["state_loss_count"]),
        int(summary["state_equal_count"]),
    )
    state_cols = [
        "state",
        "state_abbr",
        "ape_improvement_pct_mean",
        "adjusted_relative_improvement_pct_mean",
        "state_outcome",
    ]
    keep_state = [c for c in state_cols if c in state_pairs.columns]
    if keep_state:
        state_table = state_pairs.loc[:, keep_state].copy().sort_values(
            ["ape_improvement_pct_mean", "state"],
            ascending=[False, True],
        )
        LOGGER.info(
            "state outcomes\n%s",
            state_table.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"),
        )
    result_cols = [
        "family",
        "hypothesis_id",
        "subset",
        "estimate",
        "threshold",
        "p_value",
        "passed",
        "n_obs",
        "n_groups",
    ]
    keep_result = [c for c in result_cols if c in results.columns]
    if keep_result:
        result_table = results.loc[:, keep_result].copy()
        LOGGER.info(
            "hypothesis tests\n%s",
            result_table.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"),
        )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    bundle = load_analysis_bundle(args.config)
    cfg = bundle.config
    output_path = cfg.paths.hypothesis_results_parquet
    if bool(args.skip) and output_path.exists():
        LOGGER.info("skip requested and existing hypothesis results found at %s", output_path)
        return
    results, county_pairs, state_pairs, year_safety = run_hypothesis_tests(bundle)
    write_frame(county_pairs, cfg.paths.county_pairs_parquet)
    write_frame(state_pairs, cfg.paths.state_pairs_parquet)
    write_frame(year_safety, cfg.paths.year_safety_parquet)
    write_frame(results, cfg.paths.hypothesis_results_parquet)
    summary = {
        "baseline_model": str(cfg.comparison.baseline_model),
        "treatment_model": str(county_pairs["treatment_model"].iloc[0]),
        "state_equal_tolerance_pct": float(cfg.hypothesis.state_equal_tolerance_pct),
        "state_win_count": int(np.count_nonzero(np.asarray(state_pairs["state_win"], dtype=bool))),
        "state_loss_count": int(np.count_nonzero(np.asarray(state_pairs["state_loss"], dtype=bool))),
        "state_equal_count": int(np.count_nonzero(np.asarray(state_pairs["state_equal"], dtype=bool))),
        "n_hypotheses": int(results.shape[0]),
        "n_passed": int(np.count_nonzero(np.asarray(results["passed"], dtype=bool))),
        "passed_hypotheses": (
            results.loc[np.asarray(results["passed"], dtype=bool), ["hypothesis_id", "subset"]]
            .apply(lambda row: f"{row['hypothesis_id']}:{row['subset']}", axis=1)
            .astype(str)
            .tolist()
        ),
        "output_root": str(cfg.paths.output_root),
    }
    cfg.paths.hypothesis_summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.paths.hypothesis_summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log_hypothesis_summary(results=results, state_pairs=state_pairs, summary=summary)
    LOGGER.info("wrote hypothesis artifacts to %s", cfg.paths.output_root)


if __name__ == "__main__":
    main()
