#!/usr/bin/env python3
#
# loaders.py  Andrew Belles  Mar 27th, 2026
#
# Parent-level analysis config and readers over parquet-native nowcast outputs.
#

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from nowcast import NowcastConfig, load_config as load_nowcast_config
from nowcast.common import load_county_display_lookup


LOGGER = logging.getLogger("analysis.loaders")


@dataclass(slots=True)
class AnalysisPaths:
    nowcast_config: Path
    nowcast_root: Path
    graph_runs_parquet: Path
    graph_basis_parquet: Path
    graph_edges_parquet: Path
    output_root: Path
    county_pairs_parquet: Path
    state_pairs_parquet: Path
    year_safety_parquet: Path
    hypothesis_results_parquet: Path
    hypothesis_summary_json: Path
    leakage_summary_parquet: Path


@dataclass(slots=True)
class AnalysisInputs:
    censal_summary: Path
    censal_fold_metrics: Path
    censal_abs_errors: Path
    county_trajectory: Path
    year_metrics: Path
    county_summary: Path
    summary_json: Path


@dataclass(slots=True)
class ComparisonConfig:
    baseline_model: str
    treatment_model: str


@dataclass(slots=True)
class SelectionConfig:
    anchor_year: int
    start_year: int
    end_year: int
    hard_case_quantile: float
    worst_regression_quantile: float
    improvement_strata: list[str]


@dataclass(slots=True)
class HypothesisTestConfig:
    paired_test: str
    permutation_draws: int
    alpha: float
    random_seed: int
    adjusted_relative_pct_threshold: float
    majority_threshold: float
    state_equal_tolerance_pct: float


@dataclass(slots=True)
class SafetyConfig:
    max_growth_ratio_warn: float
    max_abs_correction_log_warn: float
    interval_coverage_target: float
    bounded_share_threshold: float


@dataclass(slots=True)
class VisualizationConfig:
    county_examples_per_group: int
    export_dpi: int


@dataclass(slots=True)
class HypothesisAnalysisConfig:
    paths: AnalysisPaths
    inputs: AnalysisInputs
    comparison: ComparisonConfig
    selection: SelectionConfig
    hypothesis: HypothesisTestConfig
    safety: SafetyConfig
    visualization: VisualizationConfig


@dataclass(slots=True)
class AnalysisBundle:
    config: HypothesisAnalysisConfig
    nowcast_config: NowcastConfig
    county_lookup: pd.DataFrame
    censal_summary: pd.DataFrame
    censal_fold_metrics: pd.DataFrame
    censal_abs_errors: pd.DataFrame
    county_trajectory: pd.DataFrame
    year_metrics: pd.DataFrame
    county_summary: pd.DataFrame
    summary_json: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must load to a mapping")
    return raw


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must load to a mapping")
    return raw


def _read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        LOGGER.warning("optional analysis input missing: %s", path)
        return {}
    return _read_json(path)


def _read_parquet_optional(path: Path, *, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("optional analysis input missing: %s", path)
        return pd.DataFrame(columns=list(columns))
    return pd.read_parquet(path)


def _resolve_input(base: Path, value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _repo_root(config_path: Path) -> Path:
    return config_path.resolve().parent.parent.parent


def load_analysis_config(config_path: str | Path = "configs/analysis/config.hypothesis.yaml") -> HypothesisAnalysisConfig:
    cfg_path = Path(str(config_path)).expanduser()
    raw = _read_yaml(cfg_path)
    repo_root = _repo_root(cfg_path)
    paths_cfg = dict(raw.get("paths", {}))
    inputs_cfg = dict(raw.get("inputs", {}))
    comparison_cfg = dict(raw.get("comparison", {}))
    selection_cfg = dict(raw.get("selection", {}))
    hypothesis_cfg = dict(raw.get("hypothesis", {}))
    safety_cfg = dict(raw.get("safety", {}))
    visualization_cfg = dict(raw.get("visualization", {}))

    nowcast_cfg_path = _resolve_input(repo_root, paths_cfg.get("nowcast_config", "configs/nowcast/config.nowcast.yaml"))
    nowcast_cfg = load_nowcast_config(nowcast_cfg_path)
    output_root = _resolve_input(repo_root, paths_cfg.get("output_root", "analysis/artifacts/hypothesis"))
    nowcast_root = _resolve_input(repo_root, paths_cfg.get("nowcast_root", nowcast_cfg.paths.outputs.root))
    default_treatment = str(comparison_cfg.get("treatment_model", nowcast_cfg.downstream.selected)).strip().lower()
    default_anchor_year = int(selection_cfg.get("anchor_year", nowcast_cfg.anchor_year))
    default_end_year = int(selection_cfg.get("end_year", nowcast_cfg.years.end))

    paths = AnalysisPaths(
        nowcast_config=nowcast_cfg_path,
        nowcast_root=nowcast_root,
        graph_runs_parquet=_resolve_input(repo_root, paths_cfg.get("graph_runs_parquet", nowcast_cfg.paths.topology_runs_parquet)),
        graph_basis_parquet=_resolve_input(repo_root, paths_cfg.get("graph_basis_parquet", nowcast_cfg.paths.topology_basis_parquet)),
        graph_edges_parquet=_resolve_input(repo_root, paths_cfg.get("graph_edges_parquet", nowcast_cfg.paths.topology_edges_parquet)),
        output_root=output_root,
        county_pairs_parquet=_resolve_input(repo_root, paths_cfg.get("county_pairs_parquet", output_root / "county_pairs.parquet")),
        state_pairs_parquet=_resolve_input(repo_root, paths_cfg.get("state_pairs_parquet", output_root / "state_pairs.parquet")),
        year_safety_parquet=_resolve_input(repo_root, paths_cfg.get("year_safety_parquet", output_root / "year_safety.parquet")),
        hypothesis_results_parquet=_resolve_input(repo_root, paths_cfg.get("hypothesis_results_parquet", output_root / "hypothesis_results.parquet")),
        hypothesis_summary_json=_resolve_input(repo_root, paths_cfg.get("hypothesis_summary_json", output_root / "hypothesis_summary.json")),
        leakage_summary_parquet=_resolve_input(repo_root, paths_cfg.get("leakage_summary_parquet", output_root / "topology_leakage_summary.parquet")),
    )
    inputs = AnalysisInputs(
        censal_summary=_resolve_input(repo_root, inputs_cfg.get("censal_summary", nowcast_cfg.paths.outputs.censal_dir / "summary.parquet")),
        censal_fold_metrics=_resolve_input(repo_root, inputs_cfg.get("censal_fold_metrics", nowcast_cfg.paths.outputs.censal_dir / "fold_metrics.parquet")),
        censal_abs_errors=_resolve_input(repo_root, inputs_cfg.get("censal_abs_errors", nowcast_cfg.paths.outputs.censal_dir / "abs_errors.parquet")),
        county_trajectory=_resolve_input(repo_root, inputs_cfg.get("county_trajectory", nowcast_cfg.paths.outputs.postcensal_dir / "county_trajectory.parquet")),
        year_metrics=_resolve_input(repo_root, inputs_cfg.get("year_metrics", nowcast_cfg.paths.outputs.postcensal_dir / "year_metrics.parquet")),
        county_summary=_resolve_input(repo_root, inputs_cfg.get("county_summary", nowcast_cfg.paths.outputs.postcensal_dir / "county_summary.parquet")),
        summary_json=_resolve_input(repo_root, inputs_cfg.get("summary_json", nowcast_cfg.paths.outputs.postcensal_dir / "summary.json")),
    )
    comparison = ComparisonConfig(
        baseline_model=str(comparison_cfg.get("baseline_model", "pep")).strip().lower(),
        treatment_model=default_treatment,
    )
    selection = SelectionConfig(
        anchor_year=default_anchor_year,
        start_year=int(selection_cfg.get("start_year", default_anchor_year)),
        end_year=default_end_year,
        hard_case_quantile=float(selection_cfg.get("hard_case_quantile", nowcast_cfg.evaluation.selection_hard_case_quantile)),
        worst_regression_quantile=float(selection_cfg.get("worst_regression_quantile", 0.10)),
        improvement_strata=[str(x).strip() for x in list(selection_cfg.get("improvement_strata", ["<5k", "5k-50k", "250k-1M", ">1M"]))],
    )
    hypothesis = HypothesisTestConfig(
        paired_test=str(hypothesis_cfg.get("paired_test", "sign_flip_permutation")).strip().lower(),
        permutation_draws=int(hypothesis_cfg.get("permutation_draws", hypothesis_cfg.get("bootstrap_draws", 20000))),
        alpha=float(hypothesis_cfg.get("alpha", 0.05)),
        random_seed=int(hypothesis_cfg.get("random_seed", 0)),
        adjusted_relative_pct_threshold=float(hypothesis_cfg.get("adjusted_relative_pct_threshold", 0.0)),
        majority_threshold=float(hypothesis_cfg.get("majority_threshold", 0.50)),
        state_equal_tolerance_pct=float(hypothesis_cfg.get("state_equal_tolerance_pct", 1e-4)),
    )
    safety = SafetyConfig(
        max_growth_ratio_warn=float(safety_cfg.get("max_growth_ratio_warn", 1.10)),
        max_abs_correction_log_warn=float(safety_cfg.get("max_abs_correction_log_warn", 0.25)),
        interval_coverage_target=float(safety_cfg.get("interval_coverage_target", 0.68)),
        bounded_share_threshold=float(safety_cfg.get("bounded_share_threshold", 0.95)),
    )
    visualization = VisualizationConfig(
        county_examples_per_group=int(visualization_cfg.get("county_examples_per_group", 3)),
        export_dpi=int(visualization_cfg.get("export_dpi", 180)),
    )
    return HypothesisAnalysisConfig(
        paths=paths,
        inputs=inputs,
        comparison=comparison,
        selection=selection,
        hypothesis=hypothesis,
        safety=safety,
        visualization=visualization,
    )


def load_analysis_bundle(config_path: str | Path = "configs/analysis/config.hypothesis.yaml") -> AnalysisBundle:
    cfg = load_analysis_config(config_path)
    nowcast_cfg = load_nowcast_config(cfg.paths.nowcast_config)
    county_lookup = load_county_display_lookup(nowcast_cfg.paths.county_shapefile)
    return AnalysisBundle(
        config=cfg,
        nowcast_config=nowcast_cfg,
        county_lookup=county_lookup,
        censal_summary=pd.read_parquet(cfg.inputs.censal_summary),
        censal_fold_metrics=pd.read_parquet(cfg.inputs.censal_fold_metrics),
        censal_abs_errors=pd.read_parquet(cfg.inputs.censal_abs_errors),
        county_trajectory=_read_parquet_optional(
            cfg.inputs.county_trajectory,
            columns=["fips", "state", "year", "corrected_level", "pep_level", "corrected_log", "pep_log"],
        ),
        year_metrics=_read_parquet_optional(
            cfg.inputs.year_metrics,
            columns=[
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
                "delta_mape_pct",
            ],
        ),
        county_summary=_read_parquet_optional(
            cfg.inputs.county_summary,
            columns=["year", "fips", "state"],
        ),
        summary_json=_read_json_optional(cfg.inputs.summary_json),
    )
