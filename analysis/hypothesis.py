#!/usr/bin/env python3
#
# hypothesis.py  Andrew Belles  Mar 27th, 2026
#
# Config-driven hypothesis testing for meaningful improvement and nowcast safety.
#

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.loaders import AnalysisBundle, load_analysis_bundle
from analysis.shared import (
    attach_spatial_blocks,
    build_county_pair_frame,
    build_nowcast_safety_rows,
    build_state_pair_frame,
    build_state_worst_regression_frame,
    build_year_safety_frame,
    one_sided_exact_sign_test,
    one_sided_majority_test,
    one_sided_sign_flip_permutation_test,
    one_sided_spatial_block_hac_ratio_test,
    resolve_treatment_model,
    select_hard_case_counties,
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
    county_pairs = attach_spatial_blocks(
        county_pairs,
        county_shapefile=bundle.nowcast_config.paths.county_shapefile,
        block_side_km=float(cfg.hypothesis.spatial_hac_block_km),
    )
    state_pairs = build_state_pair_frame(
        county_pairs,
        equal_tolerance_pct=float(cfg.hypothesis.state_equal_tolerance_pct),
        adjusted_relative_tolerance_pct=float(cfg.hypothesis.adjusted_delta_tolerance_pct),
    )
    worst_state = build_state_worst_regression_frame(
        county_pairs,
        worst_regression_quantile=float(cfg.selection.worst_regression_quantile),
    )
    year_safety = build_year_safety_frame(bundle)
    safety_rows = build_nowcast_safety_rows(bundle, cfg.safety)
    results: list[dict[str, object]] = []

    county_hac_stats = one_sided_spatial_block_hac_ratio_test(
        county_pairs,
        numerator_col="adjusted_ape_improvement_pct",
        denominator_col="baseline_ape_pop_pct",
        threshold=0.0,
        alpha=float(cfg.hypothesis.alpha),
        bandwidth_km=float(cfg.hypothesis.spatial_hac_bandwidth_km),
        fallback_max_blocks=int(cfg.hypothesis.spatial_block_permutation_fallback_max_blocks),
        exact_max_blocks=int(cfg.hypothesis.spatial_block_permutation_exact_max_blocks),
        draws=int(cfg.hypothesis.permutation_draws),
        seed=int(cfg.hypothesis.random_seed),
    )
    results.append(
        _result_row(
            hypothesis_id="spatial_block_hac_adjusted_relative_mape_positive",
            family="meaningful_improvement",
            subset="county_blocks",
            metric="pooled_adjusted_relative_mape_improvement_pct",
            threshold=0.0,
            test_name=str(county_hac_stats.get("selected_test_name", "spatial_block_hac")),
            stats=county_hac_stats,
            note=(
                "county-level spatial block test over pooled adjusted relative MAPE improvement; "
                "uses division-restricted Bartlett HAC when block count is high and a mass-weighted block sign-flip permutation fallback when block count is low "
                f"(block={float(cfg.hypothesis.spatial_hac_block_km):.1f}km, bandwidth={float(cfg.hypothesis.spatial_hac_bandwidth_km):.1f}km, "
                f"fallback_max_blocks={int(cfg.hypothesis.spatial_block_permutation_fallback_max_blocks)})"
            ),
        )
    )

    hard_cases = select_hard_case_counties(
        county_pairs,
        hard_case_quantile=float(cfg.selection.hard_case_quantile),
    )
    if hard_cases.empty:
        LOGGER.warning("skip hard-case hypothesis because no county rows matched quantile=%.3f", float(cfg.selection.hard_case_quantile))
    else:
        hard_case_stats = one_sided_spatial_block_hac_ratio_test(
            hard_cases,
            numerator_col="adjusted_ape_improvement_pct",
            denominator_col="baseline_ape_pop_pct",
            threshold=0.0,
            alpha=float(cfg.hypothesis.alpha),
            bandwidth_km=float(cfg.hypothesis.spatial_hac_bandwidth_km),
            fallback_max_blocks=int(cfg.hypothesis.spatial_block_permutation_fallback_max_blocks),
            exact_max_blocks=int(cfg.hypothesis.spatial_block_permutation_exact_max_blocks),
            draws=int(cfg.hypothesis.permutation_draws),
            seed=int(cfg.hypothesis.random_seed) + 11,
        )
        results.append(
            _result_row(
                hypothesis_id="spatial_block_hac_hard_case_adjusted_relative_mape_positive",
                family="meaningful_improvement",
                subset=f"top_{int(round((1.0 - float(cfg.selection.hard_case_quantile)) * 100.0))}pct_baseline_ape",
                metric="pooled_adjusted_relative_mape_improvement_pct",
                threshold=0.0,
                test_name=str(hard_case_stats.get("selected_test_name", "spatial_block_hac")),
                stats=hard_case_stats,
                note=(
                    "county-level spatial block test over pooled adjusted relative MAPE improvement on hard cases "
                    "with mass-weighted block sign-flip fallback for low block counts; "
                    f"hard cases are defined by baseline county APE >= q{float(cfg.selection.hard_case_quantile):.2f}, "
                    f"block={float(cfg.hypothesis.spatial_hac_block_km):.1f}km, bandwidth={float(cfg.hypothesis.spatial_hac_bandwidth_km):.1f}km"
                ),
            )
        )

    state_adjusted = np.asarray(state_pairs["adjusted_state_mape_error_delta_pct"], dtype=np.float64)
    state_aggregate_adjusted = np.asarray(state_pairs["adjusted_state_aggregate_error_delta_pct"], dtype=np.float64)
    state_population_weights = np.asarray(state_pairs["population_total"], dtype=np.float64)
    aggregate_perm_stats = one_sided_sign_flip_permutation_test(
        state_aggregate_adjusted,
        threshold=0.0,
        draws=int(cfg.hypothesis.permutation_draws),
        seed=int(cfg.hypothesis.random_seed) + 101,
        alpha=float(cfg.hypothesis.alpha),
        weights=state_population_weights,
        n_groups=int(state_pairs["state"].nunique()),
    )
    results.append(
        _result_row(
            hypothesis_id="population_weighted_state_aggregate_error_delta_positive",
            family="meaningful_improvement",
            subset="states_population_weighted",
            metric="population_weighted_adjusted_state_aggregate_error_delta_pct",
            threshold=0.0,
            test_name=str(cfg.hypothesis.paired_test),
            stats=aggregate_perm_stats,
            note="state sign-flip permutation test over adjusted aggregate error deltas weighted by true state population",
        )
    )

    state_sign_stats = one_sided_exact_sign_test(
        state_adjusted,
        effect_threshold=0.0,
        success_threshold=float(cfg.hypothesis.majority_threshold),
        alpha=float(cfg.hypothesis.alpha),
        n_groups=int(state_pairs["state"].nunique()),
    )
    results.append(
        _result_row(
            hypothesis_id="majority_states_improve_adjusted",
            family="meaningful_improvement",
            subset="states",
            metric="share_states_with_positive_adjusted_mape_error_delta",
            threshold=float(cfg.hypothesis.majority_threshold),
            test_name="exact_sign",
            stats=state_sign_stats,
            note=(
                "exact sign test over adjusted state MAPE deltas "
                "with effect threshold=0"
            ),
        )
    )

    for stratum in cfg.selection.improvement_strata:
        part = county_pairs.loc[county_pairs["analysis_stratum"].astype(str) == str(stratum)].copy()
        if part.empty:
            LOGGER.warning("skip stratum hypothesis for %s because no county rows matched", stratum)
            continue
        stats = one_sided_spatial_block_hac_ratio_test(
            part,
            numerator_col="adjusted_ape_improvement_pct",
            denominator_col="baseline_ape_pop_pct",
            threshold=0.0,
            alpha=float(cfg.hypothesis.alpha),
            bandwidth_km=float(cfg.hypothesis.spatial_hac_bandwidth_km),
            fallback_max_blocks=int(cfg.hypothesis.spatial_block_permutation_fallback_max_blocks),
            exact_max_blocks=int(cfg.hypothesis.spatial_block_permutation_exact_max_blocks),
            draws=int(cfg.hypothesis.permutation_draws),
            seed=int(cfg.hypothesis.random_seed),
        )
        results.append(
            _result_row(
                hypothesis_id="spatial_block_hac_stratum_adjusted_relative_mape_positive",
                family="meaningful_improvement",
                subset=str(stratum),
                metric="pooled_adjusted_relative_mape_improvement_pct",
                threshold=0.0,
                test_name=str(stats.get("selected_test_name", "spatial_block_hac")),
                stats=stats,
                note=(
                    "county-level spatial block test within population stratum over pooled adjusted relative MAPE improvement; "
                    "uses division-restricted Bartlett HAC when block count is high and a mass-weighted block sign-flip permutation fallback when block count is low "
                    f"(block={float(cfg.hypothesis.spatial_hac_block_km):.1f}km, bandwidth={float(cfg.hypothesis.spatial_hac_bandwidth_km):.1f}km)"
                ),
            )
        )

    if worst_state.empty:
        LOGGER.warning("skip worst-regression enrichment hypothesis because no state rows were available")
    else:
        enrichment_stats = one_sided_exact_sign_test(
            np.asarray(worst_state["small_pop_enrichment"], dtype=np.float64),
            effect_threshold=0.0,
            success_threshold=float(cfg.hypothesis.majority_threshold),
            alpha=float(cfg.hypothesis.alpha),
            n_groups=int(worst_state["state"].nunique()),
        )
        results.append(
            _result_row(
                hypothesis_id="worst_regressions_concentrated_lt25k",
                family="meaningful_improvement",
                subset=f"worst_{int(round(float(cfg.selection.worst_regression_quantile) * 100.0))}pct_states",
                metric="share_states_with_positive_small_pop_worst_regression_enrichment",
                threshold=float(cfg.hypothesis.majority_threshold),
                test_name="exact_sign",
                stats=enrichment_stats,
                note=(
                    "exact sign test over state-level enrichment of <25k counties among within-state worst regressions "
                    f"using quantile={cfg.selection.worst_regression_quantile:.3f}"
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
        "hypothesis summary baseline=%s treatment=%s passed=%d/%d adj_state_wins=%d adj_state_losses=%d adj_state_equal=%d",
        str(summary["baseline_model"]),
        str(summary["treatment_model"]),
        int(summary["n_passed"]),
        int(summary["n_hypotheses"]),
        int(summary["state_adjusted_win_count"]),
        int(summary["state_adjusted_loss_count"]),
        int(summary["state_adjusted_equal_count"]),
    )
    state_cols = [
        "state",
        "state_abbr",
        "adjusted_state_mape_error_delta_pct",
        "state_adjusted_outcome",
        "state_mape_error_delta_pct",
        "state_outcome",
        "adjusted_state_aggregate_error_delta_pct",
    ]
    keep_state = [c for c in state_cols if c in state_pairs.columns]
    if keep_state:
        state_table = state_pairs.loc[:, keep_state].copy().sort_values(
            ["adjusted_state_mape_error_delta_pct", "state"],
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
        "test_name",
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
        "state_adjusted_threshold_pct": float(cfg.hypothesis.adjusted_delta_tolerance_pct),
        "state_adjusted_win_count": int(np.count_nonzero(np.asarray(state_pairs["state_adjusted_win"], dtype=bool))),
        "state_adjusted_loss_count": int(np.count_nonzero(np.asarray(state_pairs["state_adjusted_loss"], dtype=bool))),
        "state_adjusted_equal_count": int(np.count_nonzero(np.asarray(state_pairs["state_adjusted_equal"], dtype=bool))),
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
