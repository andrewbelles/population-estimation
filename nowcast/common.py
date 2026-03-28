#!/usr/bin/env python3
#
# common.py  Andrew Belles  Mar 27th, 2026
#
# Shared data loading, fold construction, metrics, and modeling helpers for nowcast.
#

import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm, spearmanr
from scipy.linalg import subspace_angles
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ingestion.common import STATE_ABBR_BY_FIPS
from nowcast.config import DownstreamModelConfig, ModalityConfig


LOGGER = logging.getLogger("nowcast.common")

POP_STRATA_BOUNDS = (5000.0, 25000.0, 50000.0, 100000.0, 250000.0, 1000000.0)
POP_STRATA_LABELS = ("<5k", "5k-25k", "25k-50k", "50k-100k", "100k-250k", "250k-1M", "1M+")

STATE_REGION_BY_FIPS = {
    "09": "northeast", "23": "northeast", "25": "northeast", "33": "northeast", "44": "northeast", "50": "northeast",
    "34": "northeast", "36": "northeast", "42": "northeast",
    "17": "midwest", "18": "midwest", "26": "midwest", "39": "midwest", "55": "midwest",
    "19": "midwest", "20": "midwest", "27": "midwest", "29": "midwest", "31": "midwest", "38": "midwest", "46": "midwest",
    "10": "south", "11": "south", "12": "south", "13": "south", "24": "south", "37": "south", "45": "south", "51": "south", "54": "south",
    "01": "south", "21": "south", "28": "south", "47": "south",
    "05": "south", "22": "south", "40": "south", "48": "south",
    "04": "west", "08": "west", "16": "west", "30": "west", "32": "west", "35": "west", "49": "west", "56": "west",
    "02": "west", "06": "west", "15": "west", "41": "west", "53": "west",
    "72": "territory",
}

STATE_DIVISION_BY_FIPS = {
    "09": "new_england", "23": "new_england", "25": "new_england", "33": "new_england", "44": "new_england", "50": "new_england",
    "34": "middle_atlantic", "36": "middle_atlantic", "42": "middle_atlantic",
    "17": "east_north_central", "18": "east_north_central", "26": "east_north_central", "39": "east_north_central", "55": "east_north_central",
    "19": "west_north_central", "20": "west_north_central", "27": "west_north_central", "29": "west_north_central", "31": "west_north_central", "38": "west_north_central", "46": "west_north_central",
    "10": "south_atlantic", "11": "south_atlantic", "12": "south_atlantic", "13": "south_atlantic", "24": "south_atlantic", "37": "south_atlantic", "45": "south_atlantic", "51": "south_atlantic", "54": "south_atlantic",
    "01": "east_south_central", "21": "east_south_central", "28": "east_south_central", "47": "east_south_central",
    "05": "west_south_central", "22": "west_south_central", "40": "west_south_central", "48": "west_south_central",
    "04": "mountain", "08": "mountain", "16": "mountain", "30": "mountain", "32": "mountain", "35": "mountain", "49": "mountain", "56": "mountain",
    "02": "pacific", "06": "pacific", "15": "pacific", "41": "pacific", "53": "pacific",
    "72": "territory",
}

DIVISION_ORDER = [
    "new_england",
    "middle_atlantic",
    "east_north_central",
    "west_north_central",
    "south_atlantic",
    "east_south_central",
    "west_south_central",
    "mountain",
    "pacific",
    "territory",
]


@dataclass(slots=True)
class FoldSplit:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    heldout_states: list[str]
    heldout_regions: list[str]
    heldout_divisions: list[str]


@dataclass(slots=True)
class BlockRows:
    fips: np.ndarray
    x: np.ndarray


@dataclass(slots=True)
class TopologyRows:
    fips: np.ndarray
    x: np.ndarray
    graph_tag: str
    graph_kind: str
    graph_loss: float
    graph_counties: int


@dataclass(slots=True)
class YearSlice:
    family_year: int
    source_year: int
    sample_ids: np.ndarray
    states: np.ndarray
    has_truth: bool
    y_log: np.ndarray
    y_level: np.ndarray
    pep_log: np.ndarray
    pep_level: np.ndarray
    raw_pep_log: np.ndarray
    raw_pep_level: np.ndarray
    p_t_minus_1: np.ndarray
    births: np.ndarray
    deaths: np.ndarray
    migration: np.ndarray
    residual: np.ndarray
    direct_blocks: dict[str, np.ndarray]
    mem_x: np.ndarray
    x: np.ndarray
    graph_tag: str
    graph_kind: str
    graph_loss: float
    graph_counties: int
    basis_align_mean_abs_corr: float


@dataclass(slots=True)
class LinearHuberState:
    mean_: np.ndarray
    scale_: np.ndarray
    coef_: np.ndarray
    intercept_: float
    sigma_: float


def canon_fips_vec(arr: Any) -> np.ndarray:
    vals = np.asarray(arr).reshape(-1)
    out: list[str] = []
    for v in vals:
        if isinstance(v, bytes):
            s = v.decode("utf-8", errors="ignore")
        else:
            s = str(v)
        s = s.strip().replace("'", "").replace('"', "")
        if s.isdigit():
            s = s.zfill(5)
        out.append(s)
    return np.asarray(out, dtype="U5")


def family_tag(family_tag_base: str, family_end_year: int) -> str:
    return f"{str(family_tag_base)}_y{int(family_end_year)}_nowcast"


def graph_tag(graph_tag_base: str, family_end_year: int) -> str:
    return f"{str(graph_tag_base)}_y{int(family_end_year)}_nowcast"


def fips_state_groups(fips: np.ndarray) -> np.ndarray:
    f = np.asarray(fips).astype("U5")
    out = np.empty(f.shape[0], dtype="U2")
    for i, s in enumerate(f.tolist()):
        out[i] = str(s).zfill(5)[:2]
    return out


def state_region(state_fips: str) -> str:
    return str(STATE_REGION_BY_FIPS.get(str(state_fips).zfill(2), "unknown"))


def state_division(state_fips: str) -> str:
    return str(STATE_DIVISION_BY_FIPS.get(str(state_fips).zfill(2), "unknown"))


def load_county_display_lookup(path: Path) -> pd.DataFrame:
    gdf = gpd.read_file(path)
    if "GEOID" not in gdf.columns:
        raise ValueError(f"{path}: missing GEOID")
    if "STATEFP" not in gdf.columns:
        raise ValueError(f"{path}: missing STATEFP")
    county_col = "NAMELSAD" if "NAMELSAD" in gdf.columns else "NAME" if "NAME" in gdf.columns else None
    if county_col is None:
        raise ValueError(f"{path}: missing NAMELSAD/NAME")
    keep_cols = ["GEOID", "STATEFP", county_col]
    if "ALAND" in gdf.columns:
        keep_cols.append("ALAND")
    out = gdf.loc[:, keep_cols].copy()
    out["fips"] = out["GEOID"].astype(str).str.strip().str.zfill(5)
    out["state"] = out["STATEFP"].astype(str).str.strip().str.zfill(2)
    out["state_abbr"] = out["state"].map(STATE_ABBR_BY_FIPS).fillna(out["state"])
    out["county"] = out[county_col].astype(str).str.strip()
    out["region"] = out["state"].map(state_region)
    out["division"] = out["state"].map(state_division)
    if "ALAND" in out.columns:
        out["aland_sqkm"] = pd.to_numeric(out["ALAND"], errors="coerce") / 1e6
    else:
        out["aland_sqkm"] = np.nan
    return out.loc[:, ["fips", "state", "state_abbr", "county", "region", "division", "aland_sqkm"]].drop_duplicates(subset=["fips"])


def assign_population_strata(pop: np.ndarray) -> pd.Categorical:
    arr = np.asarray(pop, dtype=np.float64).reshape(-1)
    bins = np.asarray(POP_STRATA_BOUNDS, dtype=np.float64)
    labels = np.full(arr.shape[0], POP_STRATA_LABELS[-1], dtype=object)
    labels[arr < bins[0]] = POP_STRATA_LABELS[0]
    labels[(arr >= bins[0]) & (arr < bins[1])] = POP_STRATA_LABELS[1]
    labels[(arr >= bins[1]) & (arr < bins[2])] = POP_STRATA_LABELS[2]
    labels[(arr >= bins[2]) & (arr < bins[3])] = POP_STRATA_LABELS[3]
    labels[(arr >= bins[3]) & (arr < bins[4])] = POP_STRATA_LABELS[4]
    labels[(arr >= bins[4]) & (arr < bins[5])] = POP_STRATA_LABELS[5]
    return pd.Categorical(labels, categories=list(POP_STRATA_LABELS), ordered=True)


def build_state_group_splits(
    sample_ids: np.ndarray,
    *,
    n_splits: int,
    strategy: str,
    region_level: str,
) -> list[FoldSplit]:
    groups = fips_state_groups(np.asarray(sample_ids))
    unique_states, counts = np.unique(groups, return_counts=True)
    if unique_states.size < 2:
        raise ValueError("state-group CV requires at least two unique states")
    n_splits_eff = int(max(2, min(int(n_splits), int(unique_states.size))))
    strategy_name = str(strategy).strip().lower()
    region_level_name = str(region_level).strip().lower()
    if strategy_name == "group_kfold":
        splitter = GroupKFold(n_splits=n_splits_eff)
        out: list[FoldSplit] = []
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(np.arange(sample_ids.shape[0]), groups=groups), start=1):
            heldout_states = sorted(np.unique(groups[test_idx]).tolist())
            out.append(
                FoldSplit(
                    fold_id=int(fold_id),
                    train_idx=np.asarray(train_idx, dtype=np.int64),
                    test_idx=np.asarray(test_idx, dtype=np.int64),
                    heldout_states=heldout_states,
                    heldout_regions=sorted({state_region(s) for s in heldout_states}),
                    heldout_divisions=sorted({state_division(s) for s in heldout_states}),
                )
            )
        return out

    if strategy_name != "region_balanced":
        raise ValueError(f"unsupported fold strategy={strategy!r}")
    if region_level_name not in {"region", "division"}:
        raise ValueError(f"unsupported fold region level={region_level!r}")

    states_df = pd.DataFrame(
        {
            "state": unique_states.astype("U2"),
            "n": counts.astype(np.int64),
        }
    )
    states_df["region"] = states_df["state"].map(state_region)
    states_df["division"] = states_df["state"].map(state_division)
    fold_states: list[list[str]] = [[] for _ in range(n_splits_eff)]
    fold_counts = np.zeros((n_splits_eff,), dtype=np.int64)
    fold_region_counts = [Counter() for _ in range(n_splits_eff)]
    fold_division_counts = [Counter() for _ in range(n_splits_eff)]

    for division in DIVISION_ORDER:
        part = states_df.loc[states_df["division"] == division].sort_values(["n", "state"], ascending=[False, True]).reset_index(drop=True)
        for row in part.itertuples(index=False):
            primary_key = str(row.region) if region_level_name == "region" else str(row.division)
            best_fold = min(
                range(n_splits_eff),
                key=lambda j: (
                    fold_region_counts[j][primary_key] if region_level_name == "region" else fold_division_counts[j][primary_key],
                    fold_region_counts[j][str(row.region)],
                    fold_division_counts[j][str(row.division)],
                    int(fold_counts[j]),
                    len(fold_states[j]),
                    j,
                ),
            )
            fold_states[best_fold].append(str(row.state))
            fold_counts[best_fold] += int(row.n)
            fold_region_counts[best_fold][str(row.region)] += 1
            fold_division_counts[best_fold][str(row.division)] += 1

    out: list[FoldSplit] = []
    for fold_id, heldout_states in enumerate(fold_states, start=1):
        heldout_set = set(heldout_states)
        if not heldout_set:
            continue
        test_idx = np.flatnonzero(np.asarray([s in heldout_set for s in groups.tolist()], dtype=bool))
        train_idx = np.flatnonzero(np.asarray([s not in heldout_set for s in groups.tolist()], dtype=bool))
        out.append(
            FoldSplit(
                fold_id=int(fold_id),
                train_idx=np.asarray(train_idx, dtype=np.int64),
                test_idx=np.asarray(test_idx, dtype=np.int64),
                heldout_states=sorted(heldout_set),
                heldout_regions=sorted({state_region(s) for s in heldout_set}),
                heldout_divisions=sorted({state_division(s) for s in heldout_set}),
            )
        )
    if len(out) < 2:
        raise RuntimeError("region-balanced state folding produced fewer than two non-empty folds")
    return out


def finite_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    keep = np.isfinite(arr)
    if not np.any(keep):
        return float("nan")
    return float(np.mean(arr[keep]))


def finite_max(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    keep = np.isfinite(arr)
    if not np.any(keep):
        return float("nan")
    return float(np.max(arr[keep]))


def weighted_std(x: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return 0.0
    if sample_weight is None:
        return float(np.std(arr, ddof=0))
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    mu = float(np.average(arr, weights=w))
    var = float(np.average((arr - mu) ** 2, weights=w))
    return float(np.sqrt(max(var, 0.0)))


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    muv = np.asarray(mu, dtype=np.float64).reshape(-1)
    sig = np.clip(np.asarray(sigma, dtype=np.float64).reshape(-1), 1e-6, None)
    z = (yv - muv) / sig
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    per = sig * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - (1.0 / np.sqrt(np.pi)))
    return float(np.mean(np.clip(per, 0.0, None)))


def mape_pop_pct(y_log: np.ndarray, p_log: np.ndarray) -> float:
    t = np.exp(np.asarray(y_log, dtype=np.float64))
    p = np.exp(np.asarray(p_log, dtype=np.float64))
    return float(np.mean(np.abs(p - t) / np.clip(np.abs(t), 1e-9, None)) * 100.0)


def read_parquet_frame(path: Path, *, columns: list[str] | None = None, filters: list[tuple[str, str, object]] | None = None) -> pd.DataFrame:
    table = pq.read_table(str(path), columns=columns, filters=filters)
    return table.to_pandas()


def load_pep_year(path: Path, *, year: int) -> pd.DataFrame:
    frame = read_parquet_frame(
        Path(path),
        columns=[
            "fips",
            "county_name",
            "state_abbr",
            "year",
            "pep_population",
            "pep_population_prev",
            "pep_births",
            "pep_deaths",
            "pep_domestic_migration",
            "pep_international_migration",
            "pep_residual",
            "pep_net_migration",
            "label",
            "label_level",
            "label_prev",
            "label_delta",
            "target_correction_log",
            "target_correction_level",
        ],
        filters=[("year", "=", int(year))],
    )
    if frame.empty:
        raise ValueError(f"{path}: no rows for year={int(year)}")
    frame["fips"] = frame["fips"].astype(str).str.strip().str.zfill(5)
    frame["state"] = frame["fips"].str[:2]
    frame["state_abbr"] = frame["state"].map(STATE_ABBR_BY_FIPS).fillna(frame["state_abbr"])
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(np.int64)
    frame["pep_population"] = pd.to_numeric(frame["pep_population"], errors="coerce").astype(np.float64)
    frame["pep_log"] = np.log(np.clip(np.asarray(frame["pep_population"], dtype=np.float64), 1e-9, None))
    frame["y_log"] = pd.to_numeric(frame["label"], errors="coerce").astype(np.float64)
    frame["y_level"] = pd.to_numeric(frame["label_level"], errors="coerce").astype(np.float64)
    frame["y_prev"] = pd.to_numeric(frame["label_prev"], errors="coerce").astype(np.float64)
    frame["y_delta"] = pd.to_numeric(frame["label_delta"], errors="coerce").astype(np.float64)
    frame["p_t_minus_1"] = pd.to_numeric(frame["pep_population_prev"], errors="coerce").astype(np.float64)
    frame["births"] = pd.to_numeric(frame["pep_births"], errors="coerce").astype(np.float64)
    frame["deaths"] = pd.to_numeric(frame["pep_deaths"], errors="coerce").astype(np.float64)
    frame["migration"] = pd.to_numeric(frame["pep_net_migration"], errors="coerce").astype(np.float64)
    frame["migration_domestic"] = pd.to_numeric(frame["pep_domestic_migration"], errors="coerce").astype(np.float64)
    frame["migration_international"] = pd.to_numeric(frame["pep_international_migration"], errors="coerce").astype(np.float64)
    frame["residual"] = pd.to_numeric(frame["pep_residual"], errors="coerce").astype(np.float64)
    frame["has_truth"] = np.isfinite(np.asarray(frame["y_log"], dtype=np.float64)) & np.isfinite(np.asarray(frame["y_level"], dtype=np.float64))
    return frame.sort_values("fips").reset_index(drop=True)


def parquet_rows_for_family(input_parquet: Path, *, family_tag_name: str, source_year: int) -> pd.DataFrame:
    frame = read_parquet_frame(
        Path(input_parquet),
        columns=["fips", "item_index", "item_count", "embedding"],
        filters=[("family_tag", "=", str(family_tag_name)), ("source_year", "=", int(source_year))],
    )
    if frame.empty:
        raise ValueError(f"{input_parquet}: no rows for family={family_tag_name} source_year={int(source_year)}")
    frame["fips"] = frame["fips"].astype(str).str.strip().str.zfill(5)
    frame["item_index"] = pd.to_numeric(frame["item_index"], errors="raise").astype(np.int64)
    frame["item_count"] = pd.to_numeric(frame["item_count"], errors="raise").astype(np.int64)
    frame["embedding"] = frame["embedding"].apply(lambda x: np.asarray(list(x), dtype=np.float64))
    return frame.sort_values(["fips", "item_index"]).reset_index(drop=True)


def load_dense_block(modality_cfg: ModalityConfig, *, family_end_year: int, source_year: int) -> BlockRows:
    fam_tag = family_tag(str(modality_cfg.family_tag_base), int(family_end_year))
    frame = parquet_rows_for_family(modality_cfg.input_parquet, family_tag_name=fam_tag, source_year=int(source_year))
    rows: list[np.ndarray] = []
    fips: list[str] = []
    for fid, part in frame.groupby("fips", sort=False):
        if int(part.shape[0]) != 1:
            raise ValueError(f"{modality_cfg.input_parquet}: dense block expected one row per fips; got {part.shape[0]} for {fid}")
        fips.append(str(fid))
        rows.append(np.asarray(part["embedding"].iloc[0], dtype=np.float64))
    return BlockRows(fips=np.asarray(fips, dtype="U5"), x=np.vstack(rows).astype(np.float64, copy=False))


def tile_bag_mean_max(emb: np.ndarray) -> np.ndarray:
    arr = np.asarray(emb, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        raise ValueError("bag embeddings must be 2D and non-empty")
    mu = np.mean(arr, axis=0)
    mx = np.max(arr, axis=0)
    return np.concatenate([mu, mx], axis=0).astype(np.float64, copy=False)


def load_bag_block(modality_cfg: ModalityConfig, *, family_end_year: int, source_year: int, pool_mode: str) -> BlockRows:
    fam_tag = family_tag(str(modality_cfg.family_tag_base), int(family_end_year))
    frame = parquet_rows_for_family(modality_cfg.input_parquet, family_tag_name=fam_tag, source_year=int(source_year))
    mode = str(pool_mode).strip().lower()
    if mode != "mean_max":
        raise ValueError(f"unsupported bag pool mode={pool_mode!r}; parquet nowcast expects mean_max for legacy parity")
    rows: list[np.ndarray] = []
    fips: list[str] = []
    for fid, part in frame.groupby("fips", sort=False):
        part = part.sort_values("item_index").reset_index(drop=True)
        emb = np.vstack([np.asarray(v, dtype=np.float64) for v in part["embedding"].tolist()]).astype(np.float64, copy=False)
        fips.append(str(fid))
        rows.append(tile_bag_mean_max(emb))
    return BlockRows(fips=np.asarray(fips, dtype="U5"), x=np.vstack(rows).astype(np.float64, copy=False))


def load_modality_block(modality_cfg: ModalityConfig, *, family_end_year: int, source_year: int, pool_mode: str) -> BlockRows:
    if str(modality_cfg.kind) == "dense":
        return load_dense_block(modality_cfg, family_end_year=int(family_end_year), source_year=int(source_year))
    return load_bag_block(modality_cfg, family_end_year=int(family_end_year), source_year=int(source_year), pool_mode=str(pool_mode))


def load_topology_rows(
    *,
    basis_parquet: Path,
    runs_parquet: Path,
    graph_tag_base: str,
    graph_kind: str,
    family_end_year: int,
    source_year: int,
    top_k: int,
) -> TopologyRows:
    tag = graph_tag(str(graph_tag_base), int(family_end_year))
    run_frame = read_parquet_frame(
        Path(runs_parquet),
        columns=["graph_tag", "graph_kind", "graph_loss", "n_counties"],
        filters=[("graph_tag", "=", str(tag)), ("source_year", "=", int(source_year)), ("graph_kind", "=", str(graph_kind))],
    )
    if run_frame.empty:
        raise ValueError(f"{runs_parquet}: missing run metadata for graph_tag={tag} source_year={int(source_year)} graph_kind={graph_kind}")
    basis_frame = read_parquet_frame(
        Path(basis_parquet),
        columns=["fips", "basis_index", "basis_value"],
        filters=[("graph_tag", "=", str(tag)), ("source_year", "=", int(source_year)), ("graph_kind", "=", str(graph_kind))],
    )
    if basis_frame.empty:
        raise ValueError(f"{basis_parquet}: missing basis rows for graph_tag={tag} source_year={int(source_year)} graph_kind={graph_kind}")
    basis_frame["fips"] = basis_frame["fips"].astype(str).str.strip().str.zfill(5)
    basis_frame["basis_index"] = pd.to_numeric(basis_frame["basis_index"], errors="raise").astype(np.int64)
    basis_frame["basis_value"] = pd.to_numeric(basis_frame["basis_value"], errors="coerce").astype(np.float64)
    groups = list(basis_frame.groupby("fips", sort=False))
    fips: list[str] = []
    rows: list[np.ndarray] = []
    for fid, part in groups:
        part = part.sort_values("basis_index").reset_index(drop=True)
        vals = np.asarray(part["basis_value"], dtype=np.float64)
        k_eff = int(max(1, min(int(top_k), int(vals.shape[0]))))
        fips.append(str(fid))
        rows.append(vals[:k_eff])
    x = np.vstack(rows).astype(np.float64, copy=False)
    return TopologyRows(
        fips=np.asarray(fips, dtype="U5"),
        x=x,
        graph_tag=str(tag),
        graph_kind=str(graph_kind),
        graph_loss=float(pd.to_numeric(run_frame["graph_loss"], errors="coerce").iloc[0]),
        graph_counties=int(pd.to_numeric(run_frame["n_counties"], errors="coerce").iloc[0]),
    )


def load_topology_edges(
    *,
    edges_parquet: Path,
    graph_tag_name: str,
    graph_kind: str,
    source_year: int,
) -> pd.DataFrame:
    frame = read_parquet_frame(
        Path(edges_parquet),
        columns=["src_fips", "dst_fips", "edge_weight"],
        filters=[("graph_tag", "=", str(graph_tag_name)), ("source_year", "=", int(source_year)), ("graph_kind", "=", str(graph_kind))],
    )
    if frame.empty:
        return pd.DataFrame(columns=["src_fips", "dst_fips", "edge_weight"])
    frame["src_fips"] = frame["src_fips"].astype(str).str.strip().str.zfill(5)
    frame["dst_fips"] = frame["dst_fips"].astype(str).str.strip().str.zfill(5)
    frame["edge_weight"] = pd.to_numeric(frame["edge_weight"], errors="coerce").astype(np.float64)
    return frame.loc[np.isfinite(np.asarray(frame["edge_weight"], dtype=np.float64))].reset_index(drop=True)


def align_rows(
    *,
    truth_pep: pd.DataFrame,
    direct_blocks: dict[str, BlockRows],
    mem_block: TopologyRows | None,
) -> dict[str, np.ndarray]:
    sample_ids = canon_fips_vec(truth_pep["fips"].to_numpy())
    out: dict[str, np.ndarray] = {
        "sample_ids": sample_ids,
        "y_log": np.asarray(truth_pep["y_log"], dtype=np.float64),
        "y_level": np.asarray(truth_pep["y_level"], dtype=np.float64),
        "pep_log": np.asarray(truth_pep["pep_log"], dtype=np.float64),
        "pep_level": np.asarray(truth_pep["pep_population"], dtype=np.float64),
    }
    for name, block in direct_blocks.items():
        idx = {str(f): i for i, f in enumerate(np.asarray(block.fips, dtype="U5").tolist())}
        ref = np.asarray(block.x, dtype=np.float64)
        out_block = np.full((sample_ids.shape[0], ref.shape[1]), np.nan, dtype=np.float64)
        mask = np.zeros(sample_ids.shape[0], dtype=bool)
        for i, fid in enumerate(sample_ids.tolist()):
            j = idx.get(str(fid))
            if j is None:
                continue
            row = np.asarray(ref[j], dtype=np.float64)
            out_block[i] = row
            mask[i] = bool(np.all(np.isfinite(row)))
        out[str(name)] = out_block
        out[f"{name}_mask"] = mask
    if mem_block is not None:
        idx = {str(f): i for i, f in enumerate(np.asarray(mem_block.fips, dtype="U5").tolist())}
        ref = np.asarray(mem_block.x, dtype=np.float64)
        out_block = np.full((sample_ids.shape[0], ref.shape[1]), np.nan, dtype=np.float64)
        mask = np.zeros(sample_ids.shape[0], dtype=bool)
        for i, fid in enumerate(sample_ids.tolist()):
            j = idx.get(str(fid))
            if j is None:
                continue
            row = np.asarray(ref[j], dtype=np.float64)
            out_block[i] = row
            mask[i] = bool(np.all(np.isfinite(row)))
        out["mem"] = out_block
        out["mem_mask"] = mask
    return out


def apply_block_pca(
    *,
    blocks_tr: dict[str, np.ndarray],
    blocks_te: dict[str, np.ndarray],
    reduce: bool,
    dim: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if len(blocks_tr) <= 0:
        n_tr = next(iter(blocks_tr.values())).shape[0] if blocks_tr else 0
        n_te = next(iter(blocks_te.values())).shape[0] if blocks_te else 0
        return np.zeros((n_tr, 0), dtype=np.float64), np.zeros((n_te, 0), dtype=np.float64), {}
    order = list(blocks_tr.keys())
    dims_out: dict[str, int] = {}
    if not reduce:
        xtr = np.concatenate([np.asarray(blocks_tr[k], dtype=np.float64) for k in order], axis=1)
        xte = np.concatenate([np.asarray(blocks_te[k], dtype=np.float64) for k in order], axis=1)
        for k in order:
            dims_out[k] = int(blocks_tr[k].shape[1])
        return xtr, xte, dims_out
    if str(mode).strip().lower() == "per_block":
        parts_tr: list[np.ndarray] = []
        parts_te: list[np.ndarray] = []
        for k in order:
            btr = np.asarray(blocks_tr[k], dtype=np.float64)
            bte = np.asarray(blocks_te[k], dtype=np.float64)
            sc = StandardScaler()
            btr_s = sc.fit_transform(btr)
            bte_s = sc.transform(bte)
            k_eff = int(max(1, min(int(dim), int(btr_s.shape[1]), int(btr_s.shape[0]))))
            if k_eff >= int(btr_s.shape[1]):
                ztr = btr_s
                zte = bte_s
            else:
                pca = PCA(n_components=k_eff, svd_solver="full", random_state=0)
                ztr = np.asarray(pca.fit_transform(btr_s), dtype=np.float64)
                zte = np.asarray(pca.transform(bte_s), dtype=np.float64)
            parts_tr.append(ztr)
            parts_te.append(zte)
            dims_out[k] = int(ztr.shape[1])
        return np.concatenate(parts_tr, axis=1), np.concatenate(parts_te, axis=1), dims_out
    xtr = np.concatenate([np.asarray(blocks_tr[k], dtype=np.float64) for k in order], axis=1)
    xte = np.concatenate([np.asarray(blocks_te[k], dtype=np.float64) for k in order], axis=1)
    sc = StandardScaler()
    xtr_s = sc.fit_transform(xtr)
    xte_s = sc.transform(xte)
    k_eff = int(max(1, min(int(dim), int(xtr_s.shape[1]), int(xtr_s.shape[0]))))
    if k_eff >= int(xtr_s.shape[1]):
        ztr = xtr_s
        zte = xte_s
    else:
        pca = PCA(n_components=k_eff, svd_solver="full", random_state=0)
        ztr = np.asarray(pca.fit_transform(xtr_s), dtype=np.float64)
        zte = np.asarray(pca.transform(xte_s), dtype=np.float64)
    dims_out["global"] = int(ztr.shape[1])
    return ztr, zte, dims_out


def resolve_kernel_gamma(
    Xtr_s: np.ndarray,
    *,
    kernel: str,
    gamma: float,
    seed: int = 0,
    sample_limit: int = 1024,
    neighbor_k: int = 8,
) -> float:
    kernel_name = str(kernel).strip().lower()
    if kernel_name == "cosine":
        return 0.0
    gamma_eff = float(gamma)
    if np.isfinite(gamma_eff) and gamma_eff > 0.0:
        return gamma_eff
    X = np.asarray(Xtr_s, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] <= 1 or X.shape[1] <= 0:
        return 1.0
    if X.shape[0] > int(sample_limit):
        rng = np.random.default_rng(int(seed))
        take = rng.choice(int(X.shape[0]), size=int(sample_limit), replace=False)
        X = np.asarray(X[take], dtype=np.float64)
    metric = "manhattan" if kernel_name == "laplacian" else "euclidean"
    k_eff = int(max(2, min(int(neighbor_k), max(2, int(X.shape[0]) - 1))))
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nbrs.fit(X)
    d_knn, _ = nbrs.kneighbors(X, return_distance=True)
    med = float(np.median(np.asarray(d_knn[:, -1], dtype=np.float64)))
    if (not np.isfinite(med)) or med <= 1e-12:
        med = float(np.sqrt(max(1, int(X.shape[1])))) if metric == "euclidean" else float(max(1, int(X.shape[1])))
    if kernel_name == "laplacian":
        return float(1.0 / max(med, 1e-6))
    return float(1.0 / max(med * med, 1e-6))


def fit_kernel_ridge(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    model_cfg: DownstreamModelConfig,
    seed: int,
) -> tuple[np.ndarray, float]:
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(np.asarray(Xtr, dtype=np.float64))
    Xte_s = sc.transform(np.asarray(Xte, dtype=np.float64))
    gamma_eff = resolve_kernel_gamma(Xtr_s, kernel=str(model_cfg.kr_kernel), gamma=float(model_cfg.kr_gamma), seed=int(seed))
    kwargs: dict[str, object] = {"alpha": float(max(1e-12, model_cfg.kr_alpha)), "kernel": str(model_cfg.kr_kernel).strip().lower()}
    if str(model_cfg.kr_kernel).strip().lower() in {"rbf", "laplacian"}:
        kwargs["gamma"] = float(gamma_eff)
    mdl = KernelRidge(**kwargs)
    mdl.fit(Xtr_s, np.asarray(ytr, dtype=np.float64))
    pred_te = np.asarray(mdl.predict(Xte_s), dtype=np.float64).reshape(-1)
    pred_tr = np.asarray(mdl.predict(Xtr_s), dtype=np.float64).reshape(-1)
    sigma = max(weighted_std(np.asarray(ytr, dtype=np.float64) - pred_tr), 1e-6)
    return pred_te, sigma


def kernel_feature_map(
    Xtr_s: np.ndarray,
    Xte_s: np.ndarray,
    *,
    kernel: str,
    gamma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    kernel_name = str(kernel).strip().lower()
    gamma_eff = resolve_kernel_gamma(Xtr_s, kernel=kernel_name, gamma=float(gamma), seed=int(seed))
    Xtr_f = np.asarray(Xtr_s, dtype=np.float64)
    Xte_f = np.asarray(Xte_s, dtype=np.float64)
    if kernel_name == "cosine":
        Ktr = cosine_similarity(Xtr_f, Xtr_f)
        Kte = cosine_similarity(Xte_f, Xtr_f)
    elif kernel_name == "laplacian":
        Dtr = manhattan_distances(Xtr_f, Xtr_f)
        Dte = manhattan_distances(Xte_f, Xtr_f)
        Ktr = np.exp(-float(gamma_eff) * Dtr)
        Kte = np.exp(-float(gamma_eff) * Dte)
    elif kernel_name == "rbf":
        Dtr = euclidean_distances(Xtr_f, Xtr_f, squared=True)
        Dte = euclidean_distances(Xte_f, Xtr_f, squared=True)
        Ktr = np.exp(-float(gamma_eff) * Dtr)
        Kte = np.exp(-float(gamma_eff) * Dte)
    else:
        raise ValueError(f"unsupported kernel={kernel!r}")
    sc = StandardScaler()
    return np.asarray(sc.fit_transform(Ktr), dtype=np.float64), np.asarray(sc.transform(Kte), dtype=np.float64)


def fit_elastic_net(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    model_cfg: DownstreamModelConfig,
) -> tuple[np.ndarray, float]:
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(np.asarray(Xtr, dtype=np.float64))
    Xte_s = sc.transform(np.asarray(Xte, dtype=np.float64))
    ytr_f = np.asarray(ytr, dtype=np.float64).reshape(-1)
    if float(model_cfg.enet_l1_ratio) <= 1e-12:
        mdl = Ridge(alpha=float(max(1e-12, model_cfg.enet_alpha)), fit_intercept=True)
        mdl.fit(Xtr_s, ytr_f)
        pred_te = np.asarray(mdl.predict(Xte_s), dtype=np.float64).reshape(-1)
        pred_tr = np.asarray(mdl.predict(Xtr_s), dtype=np.float64).reshape(-1)
    else:
        mdl = ElasticNet(
            alpha=float(max(1e-12, model_cfg.enet_alpha)),
            l1_ratio=float(min(max(model_cfg.enet_l1_ratio, 0.0), 1.0)),
            fit_intercept=True,
            max_iter=int(model_cfg.enet_max_iter),
            tol=float(max(model_cfg.enet_tol, 1e-8)),
            random_state=0,
            selection="cyclic",
        )
        mdl.fit(Xtr_s, ytr_f)
        pred_te = np.asarray(mdl.predict(Xte_s), dtype=np.float64).reshape(-1)
        pred_tr = np.asarray(mdl.predict(Xtr_s), dtype=np.float64).reshape(-1)
    sigma = max(weighted_std(ytr_f - pred_tr), 1e-6)
    return pred_te, sigma


def fit_huber(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    model_cfg: DownstreamModelConfig,
    seed: int,
) -> tuple[np.ndarray, float]:
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(np.asarray(Xtr, dtype=np.float64))
    Xte_s = sc.transform(np.asarray(Xte, dtype=np.float64))
    if bool(model_cfg.huber_kernelize):
        Xtr_s, Xte_s = kernel_feature_map(
            Xtr_s,
            Xte_s,
            kernel=str(model_cfg.kr_kernel),
            gamma=float(model_cfg.kr_gamma),
            seed=int(seed),
        )
    ytr_f = np.asarray(ytr, dtype=np.float64).reshape(-1)
    mdl = HuberRegressor(
        alpha=float(max(model_cfg.huber_alpha, 0.0)),
        epsilon=float(max(model_cfg.huber_epsilon, 1.0 + 1e-6)),
        max_iter=int(model_cfg.huber_max_iter),
        tol=float(max(model_cfg.huber_tol, 1e-8)),
        fit_intercept=True,
    )
    mdl.fit(Xtr_s, ytr_f)
    pred_te = np.asarray(mdl.predict(Xte_s), dtype=np.float64).reshape(-1)
    pred_tr = np.asarray(mdl.predict(Xtr_s), dtype=np.float64).reshape(-1)
    sigma = max(float(getattr(mdl, "scale_", weighted_std(ytr_f - pred_tr))), 1e-6)
    return pred_te, sigma


def fit_predict(
    *,
    model_cfg: DownstreamModelConfig,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if Xtr.ndim != 2 or Xte.ndim != 2:
        raise ValueError("feature blocks must be 2D")
    if Xtr.shape[1] <= 0:
        raise ValueError("cannot fit model with zero features")
    model_name = str(model_cfg.model).strip().lower()
    if model_name == "kernel_ridge":
        pred, sigma = fit_kernel_ridge(Xtr, ytr, Xte, model_cfg=model_cfg, seed=int(seed))
    elif model_name == "elastic_net":
        pred, sigma = fit_elastic_net(Xtr, ytr, Xte, model_cfg=model_cfg)
    elif model_name == "huber":
        pred, sigma = fit_huber(Xtr, ytr, Xte, model_cfg=model_cfg, seed=int(seed))
    else:
        raise ValueError(f"unsupported model={model_name!r}")
    return pred, np.full(Xte.shape[0], sigma, dtype=np.float64), int(Xtr.shape[1])


def fit_predict_residual(
    *,
    model_cfg: DownstreamModelConfig,
    Xtr: np.ndarray,
    resid_tr: np.ndarray,
    Xte: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    mu, sigma, _ = fit_predict(model_cfg=model_cfg, Xtr=np.asarray(Xtr, dtype=np.float64), ytr=np.asarray(resid_tr, dtype=np.float64), Xte=np.asarray(Xte, dtype=np.float64), seed=int(seed))
    return np.asarray(mu, dtype=np.float64), np.asarray(sigma, dtype=np.float64)


def fit_linear_huber_state(
    *,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    model_cfg: DownstreamModelConfig,
) -> LinearHuberState:
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(np.asarray(Xtr, dtype=np.float64))
    ytr_f = np.asarray(ytr, dtype=np.float64).reshape(-1)
    mdl = HuberRegressor(
        alpha=float(max(model_cfg.huber_alpha, 0.0)),
        epsilon=float(max(model_cfg.huber_epsilon, 1.0 + 1e-6)),
        max_iter=int(model_cfg.huber_max_iter),
        tol=float(max(model_cfg.huber_tol, 1e-8)),
        fit_intercept=True,
    )
    mdl.fit(Xtr_s, ytr_f)
    pred_tr = np.asarray(mdl.predict(Xtr_s), dtype=np.float64).reshape(-1)
    sigma = max(float(getattr(mdl, "scale_", weighted_std(ytr_f - pred_tr))), 1e-6)
    return LinearHuberState(
        mean_=np.asarray(sc.mean_, dtype=np.float64),
        scale_=np.asarray(sc.scale_, dtype=np.float64),
        coef_=np.asarray(mdl.coef_, dtype=np.float64),
        intercept_=float(mdl.intercept_),
        sigma_=float(sigma),
    )


def predict_linear_huber_state(*, state: LinearHuberState, Xte: np.ndarray) -> np.ndarray:
    X = np.asarray(Xte, dtype=np.float64)
    Xs = (X - state.mean_.reshape(1, -1)) / np.clip(state.scale_.reshape(1, -1), 1e-9, None)
    return np.asarray(Xs @ state.coef_.reshape(-1) + float(state.intercept_), dtype=np.float64)


def fit_capped_huber_delta(
    *,
    base_state: LinearHuberState,
    Xcur: np.ndarray,
    delta_target: np.ndarray,
    model_cfg: DownstreamModelConfig,
) -> tuple[np.ndarray, float]:
    X = np.asarray(Xcur, dtype=np.float64)
    y = np.asarray(delta_target, dtype=np.float64).reshape(-1)
    Xs = (X - base_state.mean_.reshape(1, -1)) / np.clip(base_state.scale_.reshape(1, -1), 1e-9, None)
    mdl = HuberRegressor(
        alpha=float(max(float(model_cfg.huber_alpha) * float(max(model_cfg.rolling_alpha_mult, 1e-9)), 0.0)),
        epsilon=float(max(float(model_cfg.huber_epsilon), 1.0 + 1e-6)),
        max_iter=int(model_cfg.huber_max_iter),
        tol=float(max(float(model_cfg.huber_tol), 1e-8)),
        fit_intercept=True,
    )
    mdl.fit(Xs, y)
    delta_coef = np.asarray(mdl.coef_, dtype=np.float64).reshape(-1)
    delta_intercept = float(mdl.intercept_)
    base_vec = np.concatenate([np.asarray(base_state.coef_, dtype=np.float64).reshape(-1), np.asarray([float(base_state.intercept_)], dtype=np.float64)])
    delta_vec = np.concatenate([delta_coef, np.asarray([delta_intercept], dtype=np.float64)])
    max_frac = float(max(model_cfg.rolling_weight_drift_frac, 0.0))
    base_norm = float(np.linalg.norm(base_vec))
    delta_norm = float(np.linalg.norm(delta_vec))
    max_norm = float(max_frac * base_norm)
    if np.isfinite(max_norm) and max_norm >= 0.0 and delta_norm > max(max_norm, 1e-12):
        scale = float(max_norm / delta_norm) if max_norm > 0.0 else 0.0
        delta_vec = delta_vec * scale
        delta_coef = np.asarray(delta_vec[:-1], dtype=np.float64)
        delta_intercept = float(delta_vec[-1])
    pred_tr = np.asarray(Xs @ delta_coef + delta_intercept, dtype=np.float64)
    sigma = max(float(getattr(mdl, "scale_", weighted_std(y - pred_tr))), 1e-6)
    pred_all = np.asarray(Xs @ delta_coef + delta_intercept, dtype=np.float64)
    return pred_all, float(sigma)


def rolling_regularized_model_cfg(model_cfg: DownstreamModelConfig) -> DownstreamModelConfig:
    alpha_mult = float(max(model_cfg.rolling_alpha_mult, 1e-9))
    return DownstreamModelConfig(
        model=model_cfg.model,
        kr_kernel=model_cfg.kr_kernel,
        kr_gamma=model_cfg.kr_gamma,
        kr_alpha=float(max(1e-12, model_cfg.kr_alpha * alpha_mult)),
        enet_alpha=float(max(1e-12, model_cfg.enet_alpha * alpha_mult)),
        enet_l1_ratio=model_cfg.enet_l1_ratio,
        enet_max_iter=model_cfg.enet_max_iter,
        enet_tol=model_cfg.enet_tol,
        huber_alpha=float(max(0.0, model_cfg.huber_alpha * alpha_mult)),
        huber_epsilon=model_cfg.huber_epsilon,
        huber_max_iter=model_cfg.huber_max_iter,
        huber_tol=model_cfg.huber_tol,
        huber_kernelize=model_cfg.huber_kernelize,
        rolling_online_update=model_cfg.rolling_online_update,
        rolling_alpha_mult=model_cfg.rolling_alpha_mult,
        rolling_weight_drift_frac=model_cfg.rolling_weight_drift_frac,
    )


def restrict_basis_common(
    *,
    ref_fips: np.ndarray,
    ref_basis: np.ndarray,
    cur_fips: np.ndarray,
    cur_basis: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_fips_u8 = np.asarray(ref_fips, dtype="U5")
    cur_fips_u8 = np.asarray(cur_fips, dtype="U5")
    ref_idx = {str(f): i for i, f in enumerate(ref_fips_u8.tolist())}
    cur_idx = {str(f): i for i, f in enumerate(cur_fips_u8.tolist())}
    common = [f for f in ref_fips_u8.tolist() if f in cur_idx]
    if len(common) <= 1:
        return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64), np.asarray([], dtype="U5")
    ref_rows = np.asarray([ref_idx[f] for f in common], dtype=np.int64)
    cur_rows = np.asarray([cur_idx[f] for f in common], dtype=np.int64)
    k_eff = int(max(1, min(int(k), int(ref_basis.shape[1]), int(cur_basis.shape[1]))))
    return np.asarray(ref_basis[ref_rows, :k_eff], dtype=np.float64), np.asarray(cur_basis[cur_rows, :k_eff], dtype=np.float64), np.asarray(common, dtype="U5")


def row_normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    nrm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(nrm, 1e-9, None)


def compute_grassmann_sqdist(
    *,
    ref_fips: np.ndarray,
    ref_basis: np.ndarray,
    cur_fips: np.ndarray,
    cur_basis: np.ndarray,
    k: int,
) -> tuple[float, int]:
    u_ref, u_cur, common = restrict_basis_common(ref_fips=ref_fips, ref_basis=ref_basis, cur_fips=cur_fips, cur_basis=cur_basis, k=k)
    if common.shape[0] <= 1 or u_ref.size == 0 or u_cur.size == 0:
        return float("nan"), int(common.shape[0])
    q_ref, _ = np.linalg.qr(u_ref, mode="reduced")
    q_cur, _ = np.linalg.qr(u_cur, mode="reduced")
    ang = subspace_angles(q_ref, q_cur)
    return float(np.sum(np.square(np.asarray(ang, dtype=np.float64)))), int(common.shape[0])


def compute_community_ari(
    *,
    ref_fips: np.ndarray,
    ref_basis: np.ndarray,
    cur_fips: np.ndarray,
    cur_basis: np.ndarray,
    k: int,
    n_clusters: int,
) -> tuple[float, int]:
    x_ref, x_cur, common = restrict_basis_common(ref_fips=ref_fips, ref_basis=ref_basis, cur_fips=cur_fips, cur_basis=cur_basis, k=k)
    n_common = int(common.shape[0])
    if n_common <= 3:
        return float("nan"), n_common
    x_ref = row_normalize(x_ref)
    x_cur = row_normalize(x_cur)
    k_eff = int(max(2, min(int(n_clusters), n_common - 1)))
    ref_labels = KMeans(n_clusters=k_eff, n_init=20, random_state=0).fit_predict(x_ref)
    cur_labels = KMeans(n_clusters=k_eff, n_init=20, random_state=0).fit_predict(x_cur)
    return float(adjusted_rand_score(ref_labels, cur_labels)), n_common


def align_basis_to_reference(
    *,
    ref_fips: np.ndarray,
    ref_basis: np.ndarray,
    cur_fips: np.ndarray,
    cur_basis: np.ndarray,
) -> tuple[np.ndarray, float]:
    ref_fips_u8 = np.asarray(ref_fips, dtype="U5")
    cur_fips_u8 = np.asarray(cur_fips, dtype="U5")
    if ref_basis.shape[1] != cur_basis.shape[1]:
        raise ValueError("reference and current MEM basis must share the same retained top-k")
    ref_idx = {str(f): i for i, f in enumerate(ref_fips_u8.tolist())}
    cur_idx = {str(f): i for i, f in enumerate(cur_fips_u8.tolist())}
    common = [f for f in ref_fips_u8.tolist() if f in cur_idx]
    if len(common) < max(25, int(ref_basis.shape[1])):
        return np.asarray(cur_basis, dtype=np.float64), float("nan")
    ref_rows = np.asarray([ref_idx[f] for f in common], dtype=np.int64)
    cur_rows = np.asarray([cur_idx[f] for f in common], dtype=np.int64)
    ref_sub = np.asarray(ref_basis[ref_rows], dtype=np.float64)
    cur_sub = np.asarray(cur_basis[cur_rows], dtype=np.float64)
    ref_sub = ref_sub - ref_sub.mean(axis=0, keepdims=True)
    cur_sub = cur_sub - cur_sub.mean(axis=0, keepdims=True)
    ref_sub = ref_sub / np.clip(np.linalg.norm(ref_sub, axis=0, keepdims=True), 1e-9, None)
    cur_sub = cur_sub / np.clip(np.linalg.norm(cur_sub, axis=0, keepdims=True), 1e-9, None)
    corr = cur_sub.T @ ref_sub / float(cur_sub.shape[0])
    cost = -np.abs(corr)
    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.arange(cur_basis.shape[1], dtype=np.int64)
    signs = np.ones(cur_basis.shape[1], dtype=np.float64)
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        order[c] = int(r)
        signs[c] = 1.0 if float(corr[r, c]) >= 0.0 else -1.0
    aligned = np.asarray(cur_basis[:, order], dtype=np.float64) * signs.reshape(1, -1)
    match_vals = [float(abs(corr[r, c])) for r, c in zip(row_ind.tolist(), col_ind.tolist())]
    return aligned, float(np.mean(np.asarray(match_vals, dtype=np.float64)))


def compute_topology_leakage_proxy(
    *,
    edges: pd.DataFrame,
    sample_ids: np.ndarray,
    test_idx: np.ndarray,
    mode: str,
) -> float:
    if edges.empty or int(test_idx.shape[0]) <= 0:
        return 0.0
    ids = canon_fips_vec(sample_ids)
    test_fips = set(ids[np.asarray(test_idx, dtype=np.int64)].tolist())
    train_fips = set(ids.tolist()) - test_fips
    if not test_fips or not train_fips:
        return 0.0
    work = edges.copy()
    mode_name = str(mode).strip().lower()
    outbound = work.loc[work["src_fips"].isin(test_fips)].copy()
    if outbound.empty:
        outbound_proxy = 0.0
    else:
        outbound["cross"] = outbound["dst_fips"].isin(train_fips).astype(np.float64)
        outbound_sum = outbound.groupby("src_fips", sort=False)["edge_weight"].sum().replace(0.0, np.nan)
        outbound_cross = outbound.loc[outbound["cross"] > 0.0].groupby("src_fips", sort=False)["edge_weight"].sum()
        outbound_ratio = (outbound_cross / outbound_sum).replace([np.inf, -np.inf], np.nan)
        outbound_proxy = float(np.nanmean(np.asarray(outbound_ratio, dtype=np.float64))) if not outbound_ratio.empty else 0.0
    if mode_name == "outbound":
        return float(np.clip(outbound_proxy, 0.0, 1.0))
    inbound = work.loc[work["dst_fips"].isin(test_fips)].copy()
    if inbound.empty:
        inbound_proxy = 0.0
    else:
        inbound["cross"] = inbound["src_fips"].isin(train_fips).astype(np.float64)
        inbound_sum = inbound.groupby("dst_fips", sort=False)["edge_weight"].sum().replace(0.0, np.nan)
        inbound_cross = inbound.loc[inbound["cross"] > 0.0].groupby("dst_fips", sort=False)["edge_weight"].sum()
        inbound_ratio = (inbound_cross / inbound_sum).replace([np.inf, -np.inf], np.nan)
        inbound_proxy = float(np.nanmean(np.asarray(inbound_ratio, dtype=np.float64))) if not inbound_ratio.empty else 0.0
    if mode_name == "inbound":
        return float(np.clip(inbound_proxy, 0.0, 1.0))
    return float(np.clip(0.5 * (outbound_proxy + inbound_proxy), 0.0, 1.0))
