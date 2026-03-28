#!/usr/bin/env python3
#
# config.py  Andrew Belles  Mar 27th, 2026
#
# YAML contract loader for parquet-native censal and postcensal nowcast runs.
#

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


RESERVED_MODALITY_KEYS = {"years", "modalities", "paths", "evaluation", "graph", "downstream", "analysis"}


@dataclass(slots=True)
class YearRange:
    start: int
    end: int

    @property
    def values(self) -> list[int]:
        if int(self.end) < int(self.start):
            raise ValueError(f"invalid year range start={self.start} end={self.end}")
        return list(range(int(self.start), int(self.end) + 1))


@dataclass(slots=True)
class OutputPaths:
    root: Path
    censal_dir: Path
    postcensal_dir: Path
    analysis_dir: Path


@dataclass(slots=True)
class PathsConfig:
    county_shapefile: Path
    pep_parquet: Path
    topology_runs_parquet: Path
    topology_basis_parquet: Path
    topology_edges_parquet: Path
    outputs: OutputPaths


@dataclass(slots=True)
class ModalityConfig:
    enabled: bool
    name: str
    kind: str
    input_parquet: Path
    family_tag_base: str


@dataclass(slots=True)
class EvaluationConfig:
    strict_year: int
    n_splits: int
    seed: int
    tile_pool_mode: str
    model_pca_reduce: bool
    model_pca_dim: int
    model_pca_mode: str
    fold_strategy: str
    fold_region_level: str
    postcensal_direct_modalities: list[str]
    postcensal_use_mem: bool
    postcensal_full_prediction: bool
    selection_hard_case_quantile: float


@dataclass(slots=True)
class GraphConfig:
    enabled: bool
    graph_tag_base: str
    graph_kind: str
    mem_top_k: int


@dataclass(slots=True)
class DownstreamModelConfig:
    model: str
    kr_kernel: str
    kr_gamma: float
    kr_alpha: float
    enet_alpha: float
    enet_l1_ratio: float
    enet_max_iter: int
    enet_tol: float
    huber_alpha: float
    huber_epsilon: float
    huber_max_iter: int
    huber_tol: float
    huber_kernelize: bool
    rolling_online_update: bool
    rolling_alpha_mult: float
    rolling_weight_drift_frac: float


@dataclass(slots=True)
class DownstreamConfig:
    selected: str
    strict_direct_modalities: list[str]
    strict_feature_specs: list[str]
    models: dict[str, DownstreamModelConfig]

    def model_cfg(self, key: str | None = None) -> DownstreamModelConfig:
        model_key = str(key if key is not None else self.selected).strip()
        if model_key not in self.models:
            raise KeyError(f"unknown downstream model={model_key!r}; known={sorted(self.models)}")
        return self.models[model_key]


@dataclass(slots=True)
class AnalysisConfig:
    leakage_proxy_mode: str
    leakage_summary_parquet: Path


@dataclass(slots=True)
class NowcastConfig:
    years: YearRange
    modalities: list[str]
    paths: PathsConfig
    evaluation: EvaluationConfig
    graph: GraphConfig
    downstream: DownstreamConfig
    analysis: AnalysisConfig
    blocks: dict[str, ModalityConfig]

    @property
    def anchor_year(self) -> int:
        return int(self.years.start)

    def block_cfg(self, modality: str) -> ModalityConfig:
        key = str(modality).strip().lower()
        if key not in self.blocks:
            raise KeyError(f"unknown nowcast modality={modality!r}; known={sorted(self.blocks)}")
        return self.blocks[key]


def _as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


def _require(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise KeyError(f"missing required config key: {key}")
    return section[key]


def _parse_output_paths(section: dict[str, Any]) -> OutputPaths:
    root = _as_path(section.get("root", "nowcast/data"))
    censal_dir = _as_path(section.get("censal_dir", root / "censal"))
    postcensal_dir = _as_path(section.get("postcensal_dir", root / "postcensal"))
    analysis_dir = _as_path(section.get("analysis_dir", root / "analysis"))
    return OutputPaths(root=root, censal_dir=censal_dir, postcensal_dir=postcensal_dir, analysis_dir=analysis_dir)


def _parse_paths(section: dict[str, Any]) -> PathsConfig:
    return PathsConfig(
        county_shapefile=_as_path(_require(section, "county_shapefile")),
        pep_parquet=_as_path(_require(section, "pep_parquet")),
        topology_runs_parquet=_as_path(_require(section, "topology_runs_parquet")),
        topology_basis_parquet=_as_path(_require(section, "topology_basis_parquet")),
        topology_edges_parquet=_as_path(_require(section, "topology_edges_parquet")),
        outputs=_parse_output_paths(dict(section.get("outputs", {}))),
    )


def _parse_modality(section: dict[str, Any], name: str) -> ModalityConfig:
    kind = str(section.get("kind", "bag" if str(name).strip().lower() != "admin" else "dense")).strip().lower()
    if kind not in {"dense", "bag"}:
        raise ValueError(f"unsupported modality kind={kind!r} for {name}")
    return ModalityConfig(
        enabled=bool(section.get("enabled", True)),
        name=str(name).strip().lower(),
        kind=kind,
        input_parquet=_as_path(_require(section, "input_parquet")),
        family_tag_base=str(_require(section, "family_tag_base")),
    )


def _parse_evaluation(section: dict[str, Any], *, anchor_year: int) -> EvaluationConfig:
    return EvaluationConfig(
        strict_year=int(section.get("strict_year", anchor_year)),
        n_splits=int(section.get("n_splits", 5)),
        seed=int(section.get("seed", 0)),
        tile_pool_mode=str(section.get("tile_pool_mode", "mean_max")).strip().lower(),
        model_pca_reduce=bool(section.get("model_pca_reduce", False)),
        model_pca_dim=int(section.get("model_pca_dim", 64)),
        model_pca_mode=str(section.get("model_pca_mode", "global")).strip().lower(),
        fold_strategy=str(section.get("fold_strategy", "region_balanced")).strip().lower(),
        fold_region_level=str(section.get("fold_region_level", "division")).strip().lower(),
        postcensal_direct_modalities=[str(x).strip().lower() for x in list(section.get("postcensal_direct_modalities", ["admin"]))],
        postcensal_use_mem=bool(section.get("postcensal_use_mem", True)),
        postcensal_full_prediction=bool(section.get("postcensal_full_prediction", False)),
        selection_hard_case_quantile=float(section.get("selection_hard_case_quantile", 0.90)),
    )


def _parse_graph(section: dict[str, Any]) -> GraphConfig:
    return GraphConfig(
        enabled=bool(section.get("enabled", True)),
        graph_tag_base=str(section.get("graph_tag_base", "gsl_topology")).strip(),
        graph_kind=str(section.get("graph_kind", "learned")).strip().lower(),
        mem_top_k=int(section.get("mem_top_k", 11)),
    )


def _parse_downstream_model(section: dict[str, Any], *, model_key: str) -> DownstreamModelConfig:
    return DownstreamModelConfig(
        model=str(section.get("model", model_key)).strip().lower(),
        kr_kernel=str(section.get("kr_kernel", "laplacian")).strip().lower(),
        kr_gamma=float(section.get("kr_gamma", 0.0)),
        kr_alpha=float(section.get("kr_alpha", 0.8)),
        enet_alpha=float(section.get("enet_alpha", 1e-3)),
        enet_l1_ratio=float(section.get("enet_l1_ratio", 0.5)),
        enet_max_iter=int(section.get("enet_max_iter", 5000)),
        enet_tol=float(section.get("enet_tol", 1e-4)),
        huber_alpha=float(section.get("huber_alpha", 1e-4)),
        huber_epsilon=float(section.get("huber_epsilon", 1.35)),
        huber_max_iter=int(section.get("huber_max_iter", 5000)),
        huber_tol=float(section.get("huber_tol", 1e-5)),
        huber_kernelize=bool(section.get("huber_kernelize", False)),
        rolling_online_update=bool(section.get("rolling_online_update", False)),
        rolling_alpha_mult=float(section.get("rolling_alpha_mult", 50.0)),
        rolling_weight_drift_frac=float(section.get("rolling_weight_drift_frac", 0.05)),
    )


def _parse_downstream(section: dict[str, Any]) -> DownstreamConfig:
    selected = str(section.get("selected", "huber")).strip().lower()
    strict_direct_modalities = [str(x).strip().lower() for x in list(section.get("strict_direct_modalities", ["admin"]))]
    strict_feature_specs = [str(x).strip().lower() for x in list(section.get("strict_feature_specs", ["pep", "mem", "embeddings", "embeddings_only", "embeddings_mem", "embeddings_mem_only"]))]
    models: dict[str, DownstreamModelConfig] = {}
    for key, value in section.items():
        if key in {"selected", "strict_direct_modalities", "strict_feature_specs"}:
            continue
        if not isinstance(value, dict):
            continue
        models[str(key).strip().lower()] = _parse_downstream_model(dict(value), model_key=str(key))
    if selected not in models:
        raise KeyError(f"downstream.selected={selected!r} missing model section")
    return DownstreamConfig(
        selected=selected,
        strict_direct_modalities=strict_direct_modalities,
        strict_feature_specs=strict_feature_specs,
        models=models,
    )


def _parse_analysis(section: dict[str, Any], *, outputs: OutputPaths) -> AnalysisConfig:
    return AnalysisConfig(
        leakage_proxy_mode=str(section.get("leakage_proxy_mode", "bidirectional")).strip().lower(),
        leakage_summary_parquet=_as_path(section.get("leakage_summary_parquet", outputs.analysis_dir / "topology_leakage_summary.parquet")),
    )


def load_config(path: str | Path) -> NowcastConfig:
    config_path = _as_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {config_path}")

    years_raw = dict(_require(raw, "years"))
    years = YearRange(start=int(_require(years_raw, "start")), end=int(_require(years_raw, "end")))
    modalities = [str(x).strip().lower() for x in list(_require(raw, "modalities"))]
    if not modalities:
        raise ValueError("modalities must not be empty")

    paths = _parse_paths(dict(_require(raw, "paths")))
    evaluation = _parse_evaluation(dict(_require(raw, "evaluation")), anchor_year=int(years.start))
    graph = _parse_graph(dict(_require(raw, "graph")))
    downstream = _parse_downstream(dict(_require(raw, "downstream")))
    analysis = _parse_analysis(dict(raw.get("analysis", {})), outputs=paths.outputs)

    blocks: dict[str, ModalityConfig] = {}
    for key in modalities:
        if key in RESERVED_MODALITY_KEYS:
            raise ValueError(f"invalid modality name={key!r}")
        if key not in raw:
            raise KeyError(f"config missing modality section: {key}")
        blocks[key] = _parse_modality(dict(raw[key]), key)

    return NowcastConfig(
        years=years,
        modalities=modalities,
        paths=paths,
        evaluation=evaluation,
        graph=graph,
        downstream=downstream,
        analysis=analysis,
        blocks=blocks,
    )
