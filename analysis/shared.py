#!/usr/bin/env python3
#
# shared.py  Andrew Belles  Mar 27th, 2026
#
# Shared transforms and statistics for hypothesis testing, tables, and future figures.
#

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import binomtest, t as student_t

from analysis.loaders import AnalysisBundle, SafetyConfig
from nowcast.common import state_division, state_region


COARSE_STRATA_ORDER = ("<5k", "5k-50k", "50k-250k", "250k-1M", ">1M")


def write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def assign_analysis_population_strata(population: pd.Series | np.ndarray) -> pd.Categorical:
    arr = np.asarray(population, dtype=np.float64).reshape(-1)
    labels = np.full(arr.shape[0], COARSE_STRATA_ORDER[-1], dtype=object)
    labels[arr < 5000.0] = COARSE_STRATA_ORDER[0]
    labels[(arr >= 5000.0) & (arr < 50000.0)] = COARSE_STRATA_ORDER[1]
    labels[(arr >= 50000.0) & (arr < 250000.0)] = COARSE_STRATA_ORDER[2]
    labels[(arr >= 250000.0) & (arr < 1000000.0)] = COARSE_STRATA_ORDER[3]
    return pd.Categorical(labels, categories=list(COARSE_STRATA_ORDER), ordered=True)


def add_state_geography(frame: pd.DataFrame, *, state_col: str = "state") -> pd.DataFrame:
    out = frame.copy()
    state = out[state_col].astype(str).str.strip().str.zfill(2)
    out[state_col] = state
    out["region"] = state.map(state_region)
    out["division"] = state.map(state_division)
    return out


def classify_outcome(delta: np.ndarray, *, tolerance: float) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float64).reshape(-1)
    out = np.full(arr.shape[0], "equal", dtype=object)
    out[arr > float(tolerance)] = "win"
    out[arr < -float(tolerance)] = "loss"
    return out


def resolve_treatment_model(bundle: AnalysisBundle) -> str:
    baseline = str(bundle.config.comparison.baseline_model).strip().lower()
    requested = str(bundle.config.comparison.treatment_model).strip().lower()
    abs_models = {
        str(m).strip().lower()
        for m in bundle.censal_abs_errors["model"].astype(str).tolist()
        if str(m).strip()
    }
    if requested and requested in abs_models:
        return requested
    summary = bundle.censal_summary.copy()
    summary["model"] = summary["model"].astype(str).str.strip().str.lower()
    contenders = summary.loc[summary["model"] != baseline].copy()
    if contenders.empty:
        raise ValueError(f"no non-baseline models available beyond baseline={baseline!r}")
    sort_col = "adjusted_mape_pop_pct_mean" if "adjusted_mape_pop_pct_mean" in contenders.columns else "mape_pop_pct_mean"
    contenders = contenders.sort_values(sort_col, ascending=True).reset_index(drop=True)
    return str(contenders["model"].iloc[0])


def build_county_pair_frame(bundle: AnalysisBundle) -> pd.DataFrame:
    baseline = str(bundle.config.comparison.baseline_model)
    requested_treatment = str(bundle.config.comparison.treatment_model)
    treatment = resolve_treatment_model(bundle)
    abs_df = bundle.censal_abs_errors.copy()
    abs_df["fips"] = abs_df["fips"].astype(str).str.strip().str.zfill(5)
    abs_df["state"] = abs_df["state"].astype(str).str.strip().str.zfill(2)
    keep_cols = [
        "fips",
        "state",
        "fold",
        "heldout_states",
        "y_level",
        "pep_level",
        "model",
        "pred_level",
        "pred_log",
        "pred_correction_log",
        "abs_err_log",
        "ape_pop_pct",
    ]
    base_df = abs_df.loc[abs_df["model"].astype(str).str.lower() == baseline, keep_cols].copy()
    base_df = base_df.rename(
        columns={
            "pred_level": "baseline_pred_level",
            "pred_log": "baseline_pred_log",
            "pred_correction_log": "baseline_pred_correction_log",
            "abs_err_log": "baseline_abs_err_log",
            "ape_pop_pct": "baseline_ape_pop_pct",
        }
    ).drop(columns=["model"])
    treat_df = abs_df.loc[abs_df["model"].astype(str).str.lower() == treatment, keep_cols].copy()
    treat_df = treat_df.rename(
        columns={
            "pred_level": "treatment_pred_level",
            "pred_log": "treatment_pred_log",
            "pred_correction_log": "treatment_pred_correction_log",
            "abs_err_log": "treatment_abs_err_log",
            "ape_pop_pct": "treatment_ape_pop_pct",
        }
    ).drop(columns=["model"])
    if base_df.empty:
        raise ValueError(f"baseline model={baseline!r} missing from censal abs-errors")
    if treat_df.empty:
        raise ValueError(f"treatment model={treatment!r} missing from censal abs-errors")
    pairs = base_df.merge(
        treat_df,
        on=["fips", "state", "fold", "heldout_states"],
        how="inner",
        suffixes=("_base", "_treat"),
        validate="one_to_one",
    )
    if pairs.empty:
        raise ValueError(f"no overlapping baseline={baseline!r} and treatment={treatment!r} county rows")
    pairs["y_level"] = pd.to_numeric(pairs["y_level_base"], errors="coerce")
    pairs["pep_level"] = pd.to_numeric(pairs["pep_level_base"], errors="coerce")
    pairs = pairs.drop(columns=["y_level_base", "pep_level_base", "y_level_treat", "pep_level_treat"])
    fold_metrics = bundle.censal_fold_metrics.copy()
    fold_metrics["model"] = fold_metrics["model"].astype(str).str.lower()
    treat_fold = fold_metrics.loc[fold_metrics["model"] == treatment, ["fold", "topology_leakage_proxy"]].drop_duplicates(subset=["fold"])
    pairs = pairs.merge(treat_fold, on="fold", how="left")
    pairs["topology_leakage_proxy"] = pd.to_numeric(pairs["topology_leakage_proxy"], errors="coerce").fillna(0.0)
    baseline_ape = np.asarray(pairs["baseline_ape_pop_pct"], dtype=np.float64)
    treatment_ape = np.asarray(pairs["treatment_ape_pop_pct"], dtype=np.float64)
    delta_ape = baseline_ape - treatment_ape
    relative = (baseline_ape - treatment_ape) / np.clip(baseline_ape, 1e-9, None) * 100.0
    pairs["ape_improvement_pct"] = baseline_ape - treatment_ape
    fold_delta = pairs.groupby("fold", sort=False)["ape_improvement_pct"].mean().rename("fold_ape_improvement_pct_mean").reset_index()
    pairs = pairs.merge(fold_delta, on="fold", how="left", validate="many_to_one")
    fold_delta_mean = np.asarray(pairs["fold_ape_improvement_pct_mean"], dtype=np.float64)
    leakage_proxy = np.asarray(pairs["topology_leakage_proxy"], dtype=np.float64)
    fold_adjustment_scale = np.where(fold_delta_mean > 0.0, 1.0 - leakage_proxy, 1.0)
    fold_adjustment_scale = np.clip(fold_adjustment_scale, 0.0, None)
    adjusted_delta_ape = delta_ape * fold_adjustment_scale
    adjusted_relative = adjusted_delta_ape / np.clip(baseline_ape, 1e-9, None) * 100.0
    attributable = relative - adjusted_relative
    pairs["fold_adjustment_scale"] = fold_adjustment_scale
    pairs["relative_error_improvement_pct"] = relative
    pairs["attributable_relative_improvement_pct"] = attributable
    pairs["adjusted_relative_improvement_pct"] = adjusted_relative
    pairs["relative_error_improvement_pct_capped"] = np.clip(np.asarray(pairs["relative_error_improvement_pct"], dtype=np.float64), -200.0, 200.0)
    pairs["adjusted_relative_improvement_pct_capped"] = np.clip(np.asarray(pairs["adjusted_relative_improvement_pct"], dtype=np.float64), -200.0, 200.0)
    pairs["adjusted_ape_improvement_pct"] = adjusted_delta_ape
    pairs["adjusted_treatment_ape_pop_pct"] = baseline_ape - adjusted_delta_ape
    pairs["log_error_improvement"] = np.asarray(pairs["baseline_abs_err_log"], dtype=np.float64) - np.asarray(pairs["treatment_abs_err_log"], dtype=np.float64)
    pairs["improved"] = np.asarray(pairs["ape_improvement_pct"], dtype=np.float64) > 0.0
    pairs["worse"] = np.asarray(pairs["ape_improvement_pct"], dtype=np.float64) < 0.0
    pairs["small_pop_lt_25k"] = np.asarray(pairs["y_level"], dtype=np.float64) < 25000.0
    pairs["analysis_stratum"] = assign_analysis_population_strata(pairs["y_level"])
    pairs = add_state_geography(pairs, state_col="state")
    county_lookup = bundle.county_lookup.copy()
    county_lookup["fips"] = county_lookup["fips"].astype(str).str.strip().str.zfill(5)
    county_lookup["state"] = county_lookup["state"].astype(str).str.strip().str.zfill(2)
    pairs = pairs.merge(
        county_lookup.loc[:, ["fips", "state", "state_abbr", "county", "region", "division", "aland_sqkm"]],
        on=["fips", "state", "region", "division"],
        how="left",
    )
    pairs["baseline_model"] = baseline
    pairs["treatment_model"] = treatment
    pairs["requested_treatment_model"] = requested_treatment
    return pairs.sort_values(["state", "fips"]).reset_index(drop=True)


def attach_spatial_blocks(
    county_pairs: pd.DataFrame,
    *,
    county_shapefile: Path,
    block_side_km: float,
) -> pd.DataFrame:
    side_km = float(block_side_km)
    if not np.isfinite(side_km) or side_km <= 0.0:
        raise ValueError(f"block_side_km must be positive and finite, got {block_side_km!r}")
    gdf = gpd.read_file(county_shapefile)
    if "GEOID" not in gdf.columns:
        raise ValueError(f"{county_shapefile}: missing GEOID")
    if "geometry" not in gdf.columns:
        raise ValueError(f"{county_shapefile}: missing geometry")
    gdf = gdf.loc[:, ["GEOID", "geometry"]].copy()
    gdf["fips"] = gdf["GEOID"].astype(str).str.strip().str.zfill(5)
    gdf = gdf.loc[gdf["fips"].isin(county_pairs["fips"].astype(str).str.zfill(5))].copy()
    if gdf.empty:
        raise ValueError(f"{county_shapefile}: no county geometries matched county pairs")
    if gdf.crs is None:
        raise ValueError(f"{county_shapefile}: missing CRS for county geometry")
    gdf = gdf.to_crs("EPSG:5070")
    centroids = gdf.geometry.centroid
    geo = pd.DataFrame(
        {
            "fips": gdf["fips"].astype(str).str.zfill(5),
            "centroid_x_km": np.asarray(centroids.x, dtype=np.float64) / 1000.0,
            "centroid_y_km": np.asarray(centroids.y, dtype=np.float64) / 1000.0,
        }
    )
    out = county_pairs.merge(geo, on="fips", how="left", validate="many_to_one")
    if np.any(~np.isfinite(np.asarray(out["centroid_x_km"], dtype=np.float64))) or np.any(~np.isfinite(np.asarray(out["centroid_y_km"], dtype=np.float64))):
        missing = out.loc[
            ~np.isfinite(np.asarray(out["centroid_x_km"], dtype=np.float64)) | ~np.isfinite(np.asarray(out["centroid_y_km"], dtype=np.float64)),
            "fips",
        ].astype(str).tolist()
        raise ValueError(f"missing county centroids for {len(missing)} counties, e.g. {missing[:5]}")
    out["spatial_block_col"] = np.floor(np.asarray(out["centroid_x_km"], dtype=np.float64) / side_km).astype(np.int64)
    out["spatial_block_row"] = np.floor(np.asarray(out["centroid_y_km"], dtype=np.float64) / side_km).astype(np.int64)
    out["spatial_block_id"] = (
        out["division"].astype(str)
        + ":"
        + out["spatial_block_col"].astype(str)
        + ":"
        + out["spatial_block_row"].astype(str)
    )
    return out


def build_state_pair_frame(
    county_pairs: pd.DataFrame,
    *,
    equal_tolerance_pct: float,
    adjusted_relative_tolerance_pct: float = 0.0,
) -> pd.DataFrame:
    grouped = (
        county_pairs.groupby(["state", "region", "division"], as_index=False)
        .agg(
            state_abbr=("state_abbr", "first"),
            n_counties=("fips", "nunique"),
            baseline_state_mape_pop_pct=("baseline_ape_pop_pct", "mean"),
            treatment_state_mape_pop_pct=("treatment_ape_pop_pct", "mean"),
            adjusted_treatment_state_mape_pop_pct=("adjusted_treatment_ape_pop_pct", "mean"),
            truth_total=("y_level", "sum"),
            baseline_pred_total=("baseline_pred_level", "sum"),
            treatment_pred_total=("treatment_pred_level", "sum"),
            topology_leakage_proxy_mean=("topology_leakage_proxy", "mean"),
            improved_county_share=("improved", "mean"),
        )
        .sort_values(["state", "region", "division"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    truth_total = np.asarray(grouped["truth_total"], dtype=np.float64)
    baseline_pred_total = np.asarray(grouped["baseline_pred_total"], dtype=np.float64)
    treatment_pred_total = np.asarray(grouped["treatment_pred_total"], dtype=np.float64)
    grouped["state_mape_error_delta_pct"] = (
        np.asarray(grouped["baseline_state_mape_pop_pct"], dtype=np.float64)
        - np.asarray(grouped["treatment_state_mape_pop_pct"], dtype=np.float64)
    )
    grouped["adjusted_state_mape_error_delta_pct"] = (
        np.asarray(grouped["baseline_state_mape_pop_pct"], dtype=np.float64)
        - np.asarray(grouped["adjusted_treatment_state_mape_pop_pct"], dtype=np.float64)
    )
    grouped["state_mape_relative_improvement_pct"] = (
        np.asarray(grouped["state_mape_error_delta_pct"], dtype=np.float64)
        / np.clip(np.asarray(grouped["baseline_state_mape_pop_pct"], dtype=np.float64), 1e-9, None)
        * 100.0
    )
    grouped["adjusted_state_mape_relative_improvement_pct"] = (
        np.asarray(grouped["adjusted_state_mape_error_delta_pct"], dtype=np.float64)
        / np.clip(np.asarray(grouped["baseline_state_mape_pop_pct"], dtype=np.float64), 1e-9, None)
        * 100.0
    )
    grouped["population_total"] = truth_total
    grouped["baseline_state_ape_pop_pct"] = np.abs(baseline_pred_total - truth_total) / np.clip(truth_total, 1e-9, None) * 100.0
    grouped["treatment_state_ape_pop_pct"] = np.abs(treatment_pred_total - truth_total) / np.clip(truth_total, 1e-9, None) * 100.0
    grouped["state_aggregate_error_delta_pct"] = (
        np.asarray(grouped["baseline_state_ape_pop_pct"], dtype=np.float64)
        - np.asarray(grouped["treatment_state_ape_pop_pct"], dtype=np.float64)
    )
    grouped["state_aggregate_relative_improvement_pct"] = (
        np.asarray(grouped["state_aggregate_error_delta_pct"], dtype=np.float64)
        / np.clip(np.asarray(grouped["baseline_state_ape_pop_pct"], dtype=np.float64), 1e-9, None)
        * 100.0
    )
    grouped["state_attributable_relative_improvement_pct"] = (
        np.maximum(np.asarray(grouped["state_aggregate_relative_improvement_pct"], dtype=np.float64), 0.0)
        * np.asarray(grouped["topology_leakage_proxy_mean"], dtype=np.float64)
    )
    grouped["adjusted_state_aggregate_relative_improvement_pct"] = (
        np.asarray(grouped["state_aggregate_relative_improvement_pct"], dtype=np.float64)
        - np.asarray(grouped["state_attributable_relative_improvement_pct"], dtype=np.float64)
    )
    grouped["adjusted_treatment_state_ape_pop_pct"] = (
        np.asarray(grouped["baseline_state_ape_pop_pct"], dtype=np.float64)
        * (1.0 - np.asarray(grouped["adjusted_state_aggregate_relative_improvement_pct"], dtype=np.float64) / 100.0)
    )
    grouped["adjusted_state_aggregate_error_delta_pct"] = (
        np.asarray(grouped["baseline_state_ape_pop_pct"], dtype=np.float64)
        - np.asarray(grouped["adjusted_treatment_state_ape_pop_pct"], dtype=np.float64)
    )
    grouped["ape_improvement_pct_mean"] = np.asarray(grouped["state_mape_error_delta_pct"], dtype=np.float64)
    grouped["relative_error_improvement_pct_mean"] = np.asarray(grouped["state_mape_relative_improvement_pct"], dtype=np.float64)
    grouped["adjusted_relative_improvement_pct_mean"] = np.asarray(grouped["adjusted_state_mape_relative_improvement_pct"], dtype=np.float64)
    grouped["state_equal_tolerance_pct"] = float(equal_tolerance_pct)
    grouped["state_outcome"] = classify_outcome(
        np.asarray(grouped["state_mape_error_delta_pct"], dtype=np.float64),
        tolerance=float(equal_tolerance_pct),
    )
    grouped["state_win"] = grouped["state_outcome"].astype(str) == "win"
    grouped["state_loss"] = grouped["state_outcome"].astype(str) == "loss"
    grouped["state_equal"] = grouped["state_outcome"].astype(str) == "equal"
    grouped["state_improved"] = np.asarray(grouped["state_win"], dtype=bool)
    grouped["state_adjusted_threshold_pct"] = float(adjusted_relative_tolerance_pct)
    grouped["state_adjusted_outcome"] = classify_outcome(
        np.asarray(grouped["adjusted_state_mape_error_delta_pct"], dtype=np.float64),
        tolerance=float(adjusted_relative_tolerance_pct),
    )
    grouped["state_adjusted_win"] = grouped["state_adjusted_outcome"].astype(str) == "win"
    grouped["state_adjusted_loss"] = grouped["state_adjusted_outcome"].astype(str) == "loss"
    grouped["state_adjusted_equal"] = grouped["state_adjusted_outcome"].astype(str) == "equal"
    return grouped.sort_values(["adjusted_state_mape_error_delta_pct", "state"], ascending=[False, True]).reset_index(drop=True)


def build_state_stratum_pair_frame(county_pairs: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        county_pairs.groupby(["state", "state_abbr", "region", "division", "analysis_stratum"], as_index=False, observed=True)
        .agg(
            n_counties=("fips", "nunique"),
            baseline_state_stratum_mape_pop_pct=("baseline_ape_pop_pct", "mean"),
            treatment_state_stratum_mape_pop_pct=("treatment_ape_pop_pct", "mean"),
            adjusted_treatment_state_stratum_mape_pop_pct=("adjusted_treatment_ape_pop_pct", "mean"),
            truth_total=("y_level", "sum"),
            baseline_pred_total=("baseline_pred_level", "sum"),
            treatment_pred_total=("treatment_pred_level", "sum"),
            topology_leakage_proxy_mean=("topology_leakage_proxy", "mean"),
            improved_county_share=("improved", "mean"),
        )
        .sort_values(["analysis_stratum", "state"], ascending=[True, True])
        .reset_index(drop=True)
    )
    truth_total = np.asarray(grouped["truth_total"], dtype=np.float64)
    baseline_pred_total = np.asarray(grouped["baseline_pred_total"], dtype=np.float64)
    treatment_pred_total = np.asarray(grouped["treatment_pred_total"], dtype=np.float64)
    grouped["state_stratum_mape_error_delta_pct"] = (
        np.asarray(grouped["baseline_state_stratum_mape_pop_pct"], dtype=np.float64)
        - np.asarray(grouped["treatment_state_stratum_mape_pop_pct"], dtype=np.float64)
    )
    grouped["adjusted_state_stratum_mape_error_delta_pct"] = (
        np.asarray(grouped["baseline_state_stratum_mape_pop_pct"], dtype=np.float64)
        - np.asarray(grouped["adjusted_treatment_state_stratum_mape_pop_pct"], dtype=np.float64)
    )
    grouped["state_stratum_mape_relative_improvement_pct"] = (
        np.asarray(grouped["state_stratum_mape_error_delta_pct"], dtype=np.float64)
        / np.clip(np.asarray(grouped["baseline_state_stratum_mape_pop_pct"], dtype=np.float64), 1e-9, None)
        * 100.0
    )
    grouped["adjusted_state_stratum_mape_relative_improvement_pct"] = (
        np.asarray(grouped["adjusted_state_stratum_mape_error_delta_pct"], dtype=np.float64)
        / np.clip(np.asarray(grouped["baseline_state_stratum_mape_pop_pct"], dtype=np.float64), 1e-9, None)
        * 100.0
    )
    grouped["population_total"] = truth_total
    grouped["baseline_state_stratum_ape_pop_pct"] = np.abs(baseline_pred_total - truth_total) / np.clip(truth_total, 1e-9, None) * 100.0
    grouped["treatment_state_stratum_ape_pop_pct"] = np.abs(treatment_pred_total - truth_total) / np.clip(truth_total, 1e-9, None) * 100.0
    grouped["state_stratum_error_delta_pct"] = (
        np.asarray(grouped["baseline_state_stratum_ape_pop_pct"], dtype=np.float64)
        - np.asarray(grouped["treatment_state_stratum_ape_pop_pct"], dtype=np.float64)
    )
    grouped["state_stratum_relative_improvement_pct"] = (
        np.asarray(grouped["state_stratum_error_delta_pct"], dtype=np.float64)
        / np.clip(np.asarray(grouped["baseline_state_stratum_ape_pop_pct"], dtype=np.float64), 1e-9, None)
        * 100.0
    )
    grouped["state_stratum_attributable_relative_improvement_pct"] = (
        np.maximum(np.asarray(grouped["state_stratum_relative_improvement_pct"], dtype=np.float64), 0.0)
        * np.asarray(grouped["topology_leakage_proxy_mean"], dtype=np.float64)
    )
    grouped["adjusted_state_stratum_relative_improvement_pct"] = (
        np.asarray(grouped["state_stratum_relative_improvement_pct"], dtype=np.float64)
        - np.asarray(grouped["state_stratum_attributable_relative_improvement_pct"], dtype=np.float64)
    )
    grouped["adjusted_relative_improvement_pct_mean"] = np.asarray(grouped["adjusted_state_stratum_relative_improvement_pct"], dtype=np.float64)
    grouped["adjusted_state_stratum_ape_pop_pct"] = (
        np.asarray(grouped["baseline_state_stratum_ape_pop_pct"], dtype=np.float64)
        * (1.0 - np.asarray(grouped["adjusted_state_stratum_relative_improvement_pct"], dtype=np.float64) / 100.0)
    )
    grouped["adjusted_state_stratum_error_delta_pct"] = (
        np.asarray(grouped["baseline_state_stratum_ape_pop_pct"], dtype=np.float64)
        - np.asarray(grouped["adjusted_state_stratum_ape_pop_pct"], dtype=np.float64)
    )
    return grouped


def build_state_worst_regression_frame(county_pairs: pd.DataFrame, *, worst_regression_quantile: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    q = float(min(max(worst_regression_quantile, 0.0), 1.0))
    for (state, region, division), part in county_pairs.groupby(["state", "region", "division"], sort=True):
        vals = np.asarray(part["ape_improvement_pct"], dtype=np.float64)
        if vals.size <= 0:
            continue
        cutoff = float(np.quantile(vals, q))
        worst_mask = vals <= cutoff
        if not np.any(worst_mask):
            continue
        base_small_share = float(np.mean(np.asarray(part["small_pop_lt_25k"], dtype=bool)))
        worst_small_share = float(np.mean(np.asarray(part.loc[worst_mask, "small_pop_lt_25k"], dtype=bool)))
        rows.append(
            {
                "state": str(state),
                "state_abbr": str(part["state_abbr"].iloc[0]) if "state_abbr" in part.columns and part.shape[0] > 0 else str(state),
                "region": str(region),
                "division": str(division),
                "n_counties": int(part.shape[0]),
                "n_worst": int(np.count_nonzero(worst_mask)),
                "worst_regression_cutoff": cutoff,
                "small_pop_base_share": base_small_share,
                "small_pop_worst_share": worst_small_share,
                "small_pop_enrichment": worst_small_share - base_small_share,
            }
        )
    return pd.DataFrame(rows).sort_values(["small_pop_enrichment", "state"], ascending=[False, True]).reset_index(drop=True)


def select_hard_case_counties(
    county_pairs: pd.DataFrame,
    *,
    hard_case_quantile: float,
    value_col: str = "baseline_ape_pop_pct",
) -> pd.DataFrame:
    q = float(min(max(hard_case_quantile, 0.0), 1.0))
    vals = np.asarray(county_pairs[value_col], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size <= 0:
        return county_pairs.iloc[0:0].copy()
    cutoff = float(np.quantile(finite, q))
    mask = np.asarray(county_pairs[value_col], dtype=np.float64) >= cutoff
    out = county_pairs.loc[mask].copy()
    out["hard_case_cutoff"] = cutoff
    out["hard_case_quantile"] = q
    return out.sort_values([value_col, "fips"], ascending=[False, True]).reset_index(drop=True)


def build_year_safety_frame(bundle: AnalysisBundle) -> pd.DataFrame:
    cfg = bundle.config
    traj = bundle.county_trajectory.copy()
    required_traj = {"fips", "state", "year", "corrected_level", "pep_level", "corrected_log", "pep_log"}
    if traj.empty or not required_traj.issubset(set(traj.columns)):
        return pd.DataFrame(
            columns=[
                "year",
                "n_counties",
                "is_nowcast_year",
                "mean_abs_correction_log",
                "max_abs_correction_log",
                "within_abs_correction_bound_share",
                "mean_growth_ratio",
                "max_growth_ratio",
                "within_growth_ratio_bound_share",
            ]
        )
    traj["fips"] = traj["fips"].astype(str).str.strip().str.zfill(5)
    traj["state"] = traj["state"].astype(str).str.strip().str.zfill(2)
    traj["year"] = pd.to_numeric(traj["year"], errors="raise").astype(np.int64)
    traj = traj.loc[(traj["year"] >= int(cfg.selection.start_year)) & (traj["year"] <= int(cfg.selection.end_year))].copy()
    traj = add_state_geography(traj, state_col="state")
    corrected = np.asarray(traj["corrected_level"], dtype=np.float64)
    pep = np.asarray(traj["pep_level"], dtype=np.float64)
    correction_log = np.asarray(traj["corrected_log"], dtype=np.float64) - np.asarray(traj["pep_log"], dtype=np.float64)
    ratio = corrected / np.clip(pep, 1e-9, None)
    traj["abs_correction_log"] = np.abs(correction_log)
    traj["growth_ratio"] = ratio
    traj["within_abs_correction_bound"] = np.asarray(traj["abs_correction_log"], dtype=np.float64) <= float(cfg.safety.max_abs_correction_log_warn)
    traj["within_growth_ratio_bound"] = np.maximum(np.asarray(ratio, dtype=np.float64), 1.0 / np.clip(np.asarray(ratio, dtype=np.float64), 1e-9, None)) <= float(cfg.safety.max_growth_ratio_warn)
    traj["is_nowcast_year"] = np.asarray(traj["year"], dtype=np.int64) > int(cfg.selection.anchor_year)
    year_rows = []
    for year, part in traj.groupby("year", sort=True):
        year_rows.append(
            {
                "year": int(year),
                "n_counties": int(part.shape[0]),
                "is_nowcast_year": bool(np.asarray(part["is_nowcast_year"], dtype=bool).any()),
                "mean_abs_correction_log": float(np.nanmean(np.asarray(part["abs_correction_log"], dtype=np.float64))),
                "max_abs_correction_log": float(np.nanmax(np.asarray(part["abs_correction_log"], dtype=np.float64))),
                "within_abs_correction_bound_share": float(np.mean(np.asarray(part["within_abs_correction_bound"], dtype=bool))),
                "mean_growth_ratio": float(np.nanmean(np.asarray(part["growth_ratio"], dtype=np.float64))),
                "max_growth_ratio": float(np.nanmax(np.asarray(part["growth_ratio"], dtype=np.float64))),
                "within_growth_ratio_bound_share": float(np.mean(np.asarray(part["within_growth_ratio_bound"], dtype=bool))),
            }
        )
    year_df = pd.DataFrame(year_rows).sort_values("year").reset_index(drop=True)
    metrics = bundle.year_metrics.copy()
    if metrics.empty or "year" not in metrics.columns:
        return year_df
    metrics["year"] = pd.to_numeric(metrics["year"], errors="raise").astype(np.int64)
    keep_cols = [
        "year",
        "fit_mode",
        "has_truth",
        "graph_tag",
        "graph_kind",
        "graph_train_loss",
        "basis_align_mean_abs_corr",
        "community_ari",
        "grassmann_sqdist",
        "topology_common_counties",
    ]
    keep = [c for c in keep_cols if c in metrics.columns]
    if "year" in keep:
        year_df = year_df.merge(metrics.loc[:, keep].drop_duplicates(subset=["year"]), on="year", how="left")
    return year_df


def bootstrap_grouped_mean(
    frame: pd.DataFrame,
    *,
    value_col: str,
    group_col: str,
    strata_col: str,
    draws: int,
    seed: int,
) -> np.ndarray:
    part = frame.loc[:, [value_col, group_col, strata_col]].dropna().copy()
    if part.empty:
        raise ValueError(f"cannot bootstrap empty frame for value_col={value_col!r}")
    grouped: dict[str, list[np.ndarray]] = {}
    for (stratum, _group), sub in part.groupby([strata_col, group_col], sort=False):
        grouped.setdefault(str(stratum), []).append(np.asarray(sub[value_col], dtype=np.float64))
    rng = np.random.default_rng(int(seed))
    dist = np.empty(int(draws), dtype=np.float64)
    strata_items = list(grouped.items())
    for i in range(int(draws)):
        samples: list[np.ndarray] = []
        for _stratum, clusters in strata_items:
            idx = rng.integers(0, len(clusters), size=len(clusters))
            samples.extend(clusters[int(j)] for j in idx.tolist())
        dist[i] = float(np.mean(np.concatenate(samples, axis=0)))
    return dist


def one_sided_sign_flip_permutation_test(
    values: pd.Series | np.ndarray,
    *,
    threshold: float,
    draws: int,
    seed: int,
    alpha: float,
    weights: pd.Series | np.ndarray | None = None,
    n_groups: int | None = None,
) -> dict[str, float | int | bool]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights is None:
        w_full = np.ones(arr.shape[0], dtype=np.float64)
    else:
        w_full = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w_full.shape[0] != arr.shape[0]:
            raise ValueError("weights must match values length for sign-flip permutation test")
    keep = np.isfinite(arr) & np.isfinite(w_full)
    arr = arr[keep]
    w = w_full[keep]
    if arr.size <= 0:
        raise ValueError("sign-flip permutation test requires at least one finite observation")
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise ValueError("sign-flip permutation test requires positive finite weights")
    w = w / w_sum
    centered = arr - float(threshold)
    observed = float(np.dot(w, centered))
    estimate = float(np.dot(w, arr))
    rng = np.random.default_rng(int(seed))
    draws_eff = int(max(1, draws))
    null_stats = np.empty(draws_eff, dtype=np.float64)
    for i in range(draws_eff):
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=centered.shape[0], replace=True)
        null_stats[i] = float(np.dot(w, centered * signs))
    p_value = float((1 + np.count_nonzero(null_stats >= observed)) / (draws_eff + 1))
    return {
        "estimate": estimate,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_value": p_value,
        "n_obs": int(arr.shape[0]),
        "n_groups": int(n_groups if n_groups is not None else arr.shape[0]),
        "passed": bool(estimate > float(threshold) and p_value < float(alpha)),
    }


def _one_sided_block_sign_flip_test(
    block_scores: np.ndarray,
    *,
    threshold: float,
    alpha: float,
    exact_max_blocks: int,
    draws: int,
    seed: int,
) -> dict[str, float | int | bool]:
    scores = np.asarray(block_scores, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if scores.size <= 0:
        raise ValueError("block sign-flip test requires at least one finite block score")
    observed = float(np.sum(scores))
    centered = scores.copy()
    n_blocks = int(centered.shape[0])
    exact_limit = int(max(1, exact_max_blocks))
    if n_blocks <= exact_limit:
        sign_grid = ((np.arange(1 << n_blocks, dtype=np.uint64)[:, None] >> np.arange(n_blocks, dtype=np.uint64)) & 1).astype(np.int8)
        signs = np.where(sign_grid > 0, 1.0, -1.0).astype(np.float64)
        null_stats = signs @ centered
        p_value = float((1 + np.count_nonzero(null_stats >= observed)) / (null_stats.shape[0] + 1))
        test_name = "mass_weighted_block_permutation_exact"
    else:
        rng = np.random.default_rng(int(seed))
        draws_eff = int(max(1, draws))
        null_stats = np.empty(draws_eff, dtype=np.float64)
        for i in range(draws_eff):
            signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=n_blocks, replace=True)
            null_stats[i] = float(np.dot(signs, centered))
        p_value = float((1 + np.count_nonzero(null_stats >= observed)) / (draws_eff + 1))
        test_name = "mass_weighted_block_permutation_mc"
    return {
        "estimate": observed,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_value": p_value,
        "n_obs": n_blocks,
        "n_groups": n_blocks,
        "passed": bool(observed > float(threshold) and p_value < float(alpha)),
        "selected_test_name": test_name,
    }


def one_sided_spatial_block_hac_ratio_test(
    frame: pd.DataFrame,
    *,
    numerator_col: str,
    denominator_col: str,
    threshold: float,
    alpha: float,
    bandwidth_km: float,
    block_col: str = "spatial_block_id",
    x_col: str = "centroid_x_km",
    y_col: str = "centroid_y_km",
    division_col: str = "division",
    fallback_max_blocks: int | None = None,
    exact_max_blocks: int = 20,
    draws: int = 20000,
    seed: int = 0,
) -> dict[str, float | int | bool]:
    cols = [numerator_col, denominator_col, block_col, x_col, y_col, division_col]
    part = frame.loc[:, cols].dropna().copy()
    if part.empty:
        raise ValueError(f"cannot test empty frame for numerator_col={numerator_col!r} denominator_col={denominator_col!r}")
    num = np.asarray(part[numerator_col], dtype=np.float64)
    den = np.asarray(part[denominator_col], dtype=np.float64)
    keep = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    part = part.loc[keep].copy()
    num = num[keep]
    den = den[keep]
    if num.size <= 1:
        raise ValueError("spatial block HAC ratio test requires at least two finite county observations")
    den_mean = float(np.mean(den))
    if not np.isfinite(den_mean) or den_mean <= 0.0:
        raise ValueError("spatial block HAC ratio test requires positive baseline denominator mean")
    num_mean = float(np.mean(num))
    ratio_frac = float(num_mean / den_mean)
    estimate = float(ratio_frac * 100.0)
    metric_mass = den / float(np.sum(den))
    county_effect_pct = num / np.clip(den, 1e-9, None) * 100.0
    part["_metric_mass_score"] = metric_mass * county_effect_pct
    influence = (num - ratio_frac * den) / den_mean * 100.0
    part["_if"] = np.asarray(influence, dtype=np.float64) / float(num.size)
    block = (
        part.groupby(block_col, as_index=False)
        .agg(
            block_score=("_if", "sum"),
            block_metric_mass_score=("_metric_mass_score", "sum"),
            block_x_km=(x_col, "mean"),
            block_y_km=(y_col, "mean"),
            division=(division_col, "first"),
            n_counties=(numerator_col, "size"),
        )
        .sort_values(block_col)
        .reset_index(drop=True)
    )
    n_blocks = int(block.shape[0])
    if n_blocks <= 1:
        raise ValueError("spatial block HAC ratio test requires at least two populated spatial blocks")
    fallback_limit = None if fallback_max_blocks is None else int(max(0, fallback_max_blocks))
    if fallback_limit is not None and n_blocks <= fallback_limit:
        out = _one_sided_block_sign_flip_test(
            np.asarray(block["block_metric_mass_score"], dtype=np.float64),
            threshold=float(threshold),
            alpha=float(alpha),
            exact_max_blocks=int(exact_max_blocks),
            draws=int(draws),
            seed=int(seed),
        )
        out["estimate"] = estimate
        out["n_obs"] = int(num.size)
        out["n_groups"] = n_blocks
        return out
    scores = np.asarray(block["block_score"], dtype=np.float64)
    x = np.asarray(block["block_x_km"], dtype=np.float64)
    y = np.asarray(block["block_y_km"], dtype=np.float64)
    div = block["division"].astype(str).to_numpy(dtype=object)
    bw = float(bandwidth_km)
    if not np.isfinite(bw) or bw <= 0.0:
        raise ValueError(f"bandwidth_km must be positive and finite, got {bandwidth_km!r}")
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    same_division = div[:, None] == div[None, :]
    kernel = np.clip(1.0 - dist / bw, a_min=0.0, a_max=None)
    kernel = np.where(same_division, kernel, 0.0)
    np.fill_diagonal(kernel, 1.0)
    variance = float(scores @ kernel @ scores)
    variance *= float(n_blocks / max(n_blocks - 1, 1))
    variance = max(variance, 0.0)
    se = float(np.sqrt(variance))
    df = int(max(n_blocks - 1, 1))
    if se <= 0.0:
        p_value = 0.0 if estimate > float(threshold) else 1.0
        ci_low = estimate
        ci_high = estimate
    else:
        t_stat = float((estimate - float(threshold)) / se)
        p_value = float(1.0 - student_t.cdf(t_stat, df=df))
        crit = float(student_t.ppf(1.0 - float(alpha) / 2.0, df=df))
        ci_low = float(estimate - crit * se)
        ci_high = float(estimate + crit * se)
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n_obs": int(num.size),
        "n_groups": n_blocks,
        "passed": bool(estimate > float(threshold) and p_value < float(alpha)),
        "selected_test_name": "spatial_block_hac",
    }


def one_sided_exact_sign_test(
    values: pd.Series | np.ndarray,
    *,
    effect_threshold: float,
    success_threshold: float,
    alpha: float,
    tie_tolerance: float = 1e-12,
    n_groups: int | None = None,
) -> dict[str, float | int | bool]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        raise ValueError("exact sign test requires at least one finite observation")
    pos = arr > float(effect_threshold) + float(tie_tolerance)
    neg = arr < float(effect_threshold) - float(tie_tolerance)
    keep = np.asarray(pos | neg, dtype=bool)
    if not np.any(keep):
        raise ValueError("exact sign test has no non-tied observations after thresholding")
    k = int(np.count_nonzero(pos[keep]))
    n = int(np.count_nonzero(keep))
    share = float(k / max(n, 1))
    res = binomtest(k, n=n, p=0.5, alternative="greater")
    ci = res.proportion_ci(confidence_level=max(0.0, min(1.0, 1.0 - float(alpha))), method="exact")
    return {
        "estimate": share,
        "ci_low": float(ci.low),
        "ci_high": float(ci.high),
        "p_value": float(res.pvalue),
        "n_obs": n,
        "n_groups": int(n_groups if n_groups is not None else n),
        "passed": bool(share > float(success_threshold) and float(res.pvalue) < float(alpha)),
    }


def one_sided_bootstrap_test(
    frame: pd.DataFrame,
    *,
    value_col: str,
    threshold: float,
    draws: int,
    seed: int,
    alpha: float,
    group_col: str = "state",
    strata_col: str = "division",
) -> dict[str, float | int | bool]:
    part = frame.loc[:, [value_col, group_col, strata_col]].dropna().copy()
    if part.empty:
        raise ValueError(f"cannot test empty frame for value_col={value_col!r}")
    dist = bootstrap_grouped_mean(
        part,
        value_col=value_col,
        group_col=group_col,
        strata_col=strata_col,
        draws=int(draws),
        seed=int(seed),
    )
    estimate = float(np.mean(np.asarray(part[value_col], dtype=np.float64)))
    p_value = float((1 + np.count_nonzero(dist <= float(threshold))) / (int(draws) + 1))
    lo = float(np.quantile(dist, float(alpha) / 2.0))
    hi = float(np.quantile(dist, 1.0 - float(alpha) / 2.0))
    return {
        "estimate": estimate,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_value,
        "n_obs": int(part.shape[0]),
        "n_groups": int(part[group_col].nunique()),
        "passed": bool(estimate > float(threshold) and p_value < float(alpha)),
    }


def one_sided_majority_test(successes: pd.Series | np.ndarray, *, threshold: float, alpha: float) -> dict[str, float | int | bool]:
    arr = np.asarray(successes, dtype=bool).reshape(-1)
    n = int(arr.shape[0])
    if n <= 0:
        raise ValueError("majority test requires at least one observation")
    k = int(np.count_nonzero(arr))
    share = float(k / max(n, 1))
    res = binomtest(k, n=n, p=float(threshold), alternative="greater")
    return {
        "estimate": share,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_value": float(res.pvalue),
        "n_obs": n,
        "n_groups": n,
        "passed": bool(share > float(threshold) and float(res.pvalue) < float(alpha)),
    }


def build_leakage_adjusted_summary_table(bundle: AnalysisBundle) -> pd.DataFrame:
    strict = bundle.censal_summary.copy()
    strict = strict.loc[
        :,
        [
            "model",
            "mape_pop_pct_mean",
            "adjusted_mape_pop_pct_mean",
            "relative_error_improvement_pct_mean",
            "attributable_relative_improvement_pct_mean",
            "adjusted_relative_improvement_pct_mean",
            "topology_leakage_proxy_mean",
        ],
    ].copy()
    strict = strict.rename(
        columns={
            "mape_pop_pct_mean": "strict_mape_pop_pct_mean",
            "adjusted_mape_pop_pct_mean": "strict_adjusted_mape_pop_pct_mean",
            "relative_error_improvement_pct_mean": "strict_relative_error_improvement_pct_mean",
            "attributable_relative_improvement_pct_mean": "strict_attributable_relative_improvement_pct_mean",
            "adjusted_relative_improvement_pct_mean": "strict_adjusted_relative_improvement_pct_mean",
            "topology_leakage_proxy_mean": "strict_topology_leakage_proxy_mean",
        }
    )
    year_metrics = bundle.year_metrics.copy()
    if year_metrics.empty or "year" not in year_metrics.columns or "has_truth" not in year_metrics.columns:
        strict["postcensal_anchor_year"] = pd.NA
        strict["postcensal_anchor_delta_mape_pct"] = pd.NA
        return strict.sort_values("strict_adjusted_mape_pop_pct_mean").reset_index(drop=True)
    anchor_year = year_metrics.loc[year_metrics["has_truth"].astype(bool)].sort_values("year").head(1)
    if anchor_year.empty:
        strict["postcensal_anchor_year"] = pd.NA
        strict["postcensal_anchor_delta_mape_pct"] = pd.NA
    else:
        strict["postcensal_anchor_year"] = int(anchor_year["year"].iloc[0])
        strict["postcensal_anchor_delta_mape_pct"] = float(anchor_year["delta_mape_pct"].iloc[0])
    return strict.sort_values("strict_adjusted_mape_pop_pct_mean").reset_index(drop=True)


def build_nowcast_safety_rows(bundle: AnalysisBundle, safety: SafetyConfig) -> pd.DataFrame:
    traj = bundle.county_trajectory.copy()
    required_traj = {"state", "year", "corrected_level", "pep_level", "corrected_log", "pep_log"}
    if traj.empty or not required_traj.issubset(set(traj.columns)):
        return pd.DataFrame(
            columns=[
                "year",
                "state",
                "within_abs_correction_bound",
                "within_growth_ratio_bound",
                "abs_correction_log",
                "growth_ratio",
                "region",
                "division",
            ]
        )
    traj["year"] = pd.to_numeric(traj["year"], errors="raise").astype(np.int64)
    traj = traj.loc[traj["year"] > int(bundle.config.selection.anchor_year)].copy()
    corrected = np.asarray(traj["corrected_level"], dtype=np.float64)
    pep = np.asarray(traj["pep_level"], dtype=np.float64)
    ratio = corrected / np.clip(pep, 1e-9, None)
    abs_correction_log = np.abs(np.asarray(traj["corrected_log"], dtype=np.float64) - np.asarray(traj["pep_log"], dtype=np.float64))
    traj["within_abs_correction_bound"] = abs_correction_log <= float(safety.max_abs_correction_log_warn)
    traj["within_growth_ratio_bound"] = np.maximum(ratio, 1.0 / np.clip(ratio, 1e-9, None)) <= float(safety.max_growth_ratio_warn)
    traj["abs_correction_log"] = abs_correction_log
    traj["growth_ratio"] = ratio
    traj["region"] = traj["state"].astype(str).str.zfill(2).map(state_region)
    traj["division"] = traj["state"].astype(str).str.zfill(2).map(state_division)
    return traj
