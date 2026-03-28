#!/usr/bin/env python3
#
# topology.py  Andrew Belles  Mar 27th, 2026
#
# Graph topology learning on top of manifold parquet embeddings.
#

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from graph.config import GraphConfig, ModalityConfig, TopologyConfig, load_config
from manifold.data import load_county_coords


LOGGER = logging.getLogger("graph.topology")


@dataclass(slots=True)
class DenseBlockRows:
    fips: np.ndarray
    x: np.ndarray


@dataclass(slots=True)
class BagBlockRows:
    fips: np.ndarray
    x: np.ndarray
    mask: np.ndarray


@dataclass(slots=True)
class FeaturePack:
    sample_ids: np.ndarray
    coords: np.ndarray
    blocks: dict[str, dict[str, object]]
    block_order: list[str]
    block_specs: dict[str, dict[str, object]]
    block_dims: dict[str, int]


@dataclass(slots=True)
class TrainedGraphArtifact:
    z: np.ndarray
    fips: np.ndarray
    coords: np.ndarray
    w_learn: np.ndarray
    evals_learn: np.ndarray
    evecs_learn: np.ndarray
    w_knn: np.ndarray
    evals_knn: np.ndarray
    evecs_knn: np.ndarray
    block_dims: dict[str, int]
    pool_stats: dict[str, float]
    graph_loss: float


def setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout)


def family_label(anchor_year: int, family_end_year: int) -> str:
    if int(anchor_year) == int(family_end_year):
        return str(int(anchor_year))
    return f"{int(anchor_year)}->{int(family_end_year)}"


def family_tag(family_tag_base: str, family_end_year: int) -> str:
    return f"{str(family_tag_base)}_y{int(family_end_year)}_nowcast"


def source_split(*, family_end_year: int, source_year: int) -> str:
    return "eval" if int(source_year) == int(family_end_year) else "pqval"


def source_suffix(*, family_end_year: int, source_year: int) -> str:
    split = source_split(family_end_year=int(family_end_year), source_year=int(source_year))
    if split == "eval":
        return str(int(source_year))
    return f"pqval_{int(source_year)}"


def graph_tag(graph_tag_base: str, family_end_year: int) -> str:
    return f"{str(graph_tag_base)}_y{int(family_end_year)}_nowcast"


def canon_fips_vec(arr: Any) -> np.ndarray:
    vals = np.asarray(arr).reshape(-1)
    out: list[str] = []
    for v in vals:
        s = str(v).strip().replace("'", "").replace('"', "")
        if s.isdigit():
            s = s.zfill(5)
        out.append(s)
    return np.asarray(out, dtype="U5")


class AppendableParquetWriter:
    def __init__(self, *, output_path: Path, schema: pa.Schema, append: bool = False):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.temp_path: Path | None = None
        if self.output_path.exists() and (not bool(append)):
            self.output_path.unlink()
        if self.output_path.exists() and bool(append):
            existing = pq.ParquetFile(str(self.output_path))
            if not existing.schema_arrow.equals(self.schema, check_metadata=False):
                raise ValueError(f"{self.output_path}: existing parquet schema does not match expected schema")
            self.temp_path = self.output_path.with_suffix(f"{self.output_path.suffix}.tmp")
            if self.temp_path.exists():
                self.temp_path.unlink()
            self.writer = pq.ParquetWriter(str(self.temp_path), self.schema, compression="zstd")
            for row_group_idx in range(int(existing.num_row_groups)):
                self.writer.write_table(existing.read_row_group(row_group_idx))
        else:
            self.writer = pq.ParquetWriter(str(self.output_path), self.schema, compression="zstd")

    def write_table(self, table: pa.Table) -> None:
        self.writer.write_table(table)

    def close(self) -> None:
        self.writer.close()
        if self.temp_path is not None:
            self.temp_path.replace(self.output_path)
            self.temp_path = None


class TopologyParquetWriter:
    def __init__(self, *, config: TopologyConfig, append: bool = False):
        self.runs_schema = pa.schema(
            [
                ("graph_tag_base", pa.string()),
                ("graph_tag", pa.string()),
                ("graph_kind", pa.string()),
                ("family_start_year", pa.int16()),
                ("family_end_year", pa.int16()),
                ("family_label", pa.string()),
                ("source_year", pa.int16()),
                ("source_split", pa.string()),
                ("source_suffix", pa.string()),
                ("modality_set", pa.string()),
                ("n_counties", pa.int32()),
                ("basis_dim", pa.int32()),
                ("graph_loss", pa.float32()),
                ("pool_mode", pa.string()),
                ("graph_objective", pa.string()),
                ("support_k", pa.int32()),
                ("final_row_topk", pa.int32()),
                ("knn_k", pa.int32()),
                ("tau_graph", pa.float32()),
                ("beta_geo", pa.float32()),
                ("block_dims_json", pa.string()),
                ("pool_stats_json", pa.string()),
            ]
        )
        self.basis_schema = pa.schema(
            [
                ("graph_tag_base", pa.string()),
                ("graph_tag", pa.string()),
                ("graph_kind", pa.string()),
                ("family_start_year", pa.int16()),
                ("family_end_year", pa.int16()),
                ("family_label", pa.string()),
                ("source_year", pa.int16()),
                ("source_split", pa.string()),
                ("source_suffix", pa.string()),
                ("fips", pa.string()),
                ("basis_index", pa.int32()),
                ("basis_value", pa.float32()),
                ("eigenvalue", pa.float32()),
            ]
        )
        self.edges_schema = pa.schema(
            [
                ("graph_tag_base", pa.string()),
                ("graph_tag", pa.string()),
                ("graph_kind", pa.string()),
                ("family_start_year", pa.int16()),
                ("family_end_year", pa.int16()),
                ("family_label", pa.string()),
                ("source_year", pa.int16()),
                ("source_split", pa.string()),
                ("source_suffix", pa.string()),
                ("src_fips", pa.string()),
                ("dst_fips", pa.string()),
                ("edge_weight", pa.float32()),
            ]
        )
        self.runs_writer = AppendableParquetWriter(output_path=config.paths.runs_parquet, schema=self.runs_schema, append=append)
        self.basis_writer = AppendableParquetWriter(output_path=config.paths.basis_parquet, schema=self.basis_schema, append=append)
        self.edges_writer = AppendableParquetWriter(output_path=config.paths.edges_parquet, schema=self.edges_schema, append=append)

    def write_run(
        self,
        *,
        graph_tag_base: str,
        graph_tag_name: str,
        graph_kind: str,
        family_start_year: int,
        family_end_year: int,
        family_label_name: str,
        source_year: int,
        source_split_name: str,
        source_suffix_name: str,
        modality_set: str,
        n_counties: int,
        basis_dim: int,
        graph_loss: float,
        graph_cfg: GraphConfig,
        block_dims: dict[str, int],
        pool_stats: dict[str, float],
    ) -> None:
        table = pa.Table.from_arrays(
            [
                pa.array([str(graph_tag_base)], type=pa.string()),
                pa.array([str(graph_tag_name)], type=pa.string()),
                pa.array([str(graph_kind)], type=pa.string()),
                pa.array(np.asarray([int(family_start_year)], dtype=np.int16)),
                pa.array(np.asarray([int(family_end_year)], dtype=np.int16)),
                pa.array([str(family_label_name)], type=pa.string()),
                pa.array(np.asarray([int(source_year)], dtype=np.int16)),
                pa.array([str(source_split_name)], type=pa.string()),
                pa.array([str(source_suffix_name)], type=pa.string()),
                pa.array([str(modality_set)], type=pa.string()),
                pa.array(np.asarray([int(n_counties)], dtype=np.int32)),
                pa.array(np.asarray([int(basis_dim)], dtype=np.int32)),
                pa.array(np.asarray([float(graph_loss)], dtype=np.float32)),
                pa.array([str(graph_cfg.pool_mode)], type=pa.string()),
                pa.array([str(graph_cfg.graph_objective)], type=pa.string()),
                pa.array(np.asarray([int(graph_cfg.support_k)], dtype=np.int32)),
                pa.array(np.asarray([int(graph_cfg.final_row_topk)], dtype=np.int32)),
                pa.array(np.asarray([int(graph_cfg.knn_k)], dtype=np.int32)),
                pa.array(np.asarray([float(graph_cfg.tau_graph)], dtype=np.float32)),
                pa.array(np.asarray([float(graph_cfg.beta_geo)], dtype=np.float32)),
                pa.array([json.dumps(block_dims, sort_keys=True)], type=pa.string()),
                pa.array([json.dumps(pool_stats, sort_keys=True)], type=pa.string()),
            ],
            schema=self.runs_schema,
        )
        self.runs_writer.write_table(table)

    def write_basis(
        self,
        *,
        graph_tag_base: str,
        graph_tag_name: str,
        graph_kind: str,
        family_start_year: int,
        family_end_year: int,
        family_label_name: str,
        source_year: int,
        source_split_name: str,
        source_suffix_name: str,
        fips: np.ndarray,
        evals: np.ndarray,
        evecs: np.ndarray,
    ) -> None:
        vecs = np.asarray(evecs, dtype=np.float32)
        vals = np.asarray(evals, dtype=np.float32).reshape(-1)
        ids = canon_fips_vec(fips)
        if int(vecs.shape[0]) != int(ids.shape[0]):
            raise ValueError("basis fips/evec rows mismatch")
        if int(vecs.shape[1]) != int(vals.shape[0]):
            raise ValueError("basis eigenvalue/vector mismatch")
        n_rows = int(vecs.shape[0] * vecs.shape[1])
        table = pa.Table.from_arrays(
            [
                pa.array([str(graph_tag_base)] * n_rows, type=pa.string()),
                pa.array([str(graph_tag_name)] * n_rows, type=pa.string()),
                pa.array([str(graph_kind)] * n_rows, type=pa.string()),
                pa.array(np.full(n_rows, int(family_start_year), dtype=np.int16)),
                pa.array(np.full(n_rows, int(family_end_year), dtype=np.int16)),
                pa.array([str(family_label_name)] * n_rows, type=pa.string()),
                pa.array(np.full(n_rows, int(source_year), dtype=np.int16)),
                pa.array([str(source_split_name)] * n_rows, type=pa.string()),
                pa.array([str(source_suffix_name)] * n_rows, type=pa.string()),
                pa.array(np.repeat(ids, int(vecs.shape[1])).tolist(), type=pa.string()),
                pa.array(np.tile(np.arange(int(vecs.shape[1]), dtype=np.int32), int(vecs.shape[0]))),
                pa.array(vecs.reshape(-1)),
                pa.array(np.tile(vals, int(vecs.shape[0]))),
            ],
            schema=self.basis_schema,
        )
        self.basis_writer.write_table(table)

    def write_edges(
        self,
        *,
        graph_tag_base: str,
        graph_tag_name: str,
        graph_kind: str,
        family_start_year: int,
        family_end_year: int,
        family_label_name: str,
        source_year: int,
        source_split_name: str,
        source_suffix_name: str,
        fips: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        ids = canon_fips_vec(fips)
        w = np.asarray(weights, dtype=np.float32)
        if int(w.shape[0]) != int(w.shape[1]) or int(w.shape[0]) != int(ids.shape[0]):
            raise ValueError("edge matrix/fips shape mismatch")
        src_idx, dst_idx = np.nonzero(np.asarray(w > 0.0, dtype=bool))
        if int(src_idx.size) <= 0:
            return
        n_rows = int(src_idx.size)
        table = pa.Table.from_arrays(
            [
                pa.array([str(graph_tag_base)] * n_rows, type=pa.string()),
                pa.array([str(graph_tag_name)] * n_rows, type=pa.string()),
                pa.array([str(graph_kind)] * n_rows, type=pa.string()),
                pa.array(np.full(n_rows, int(family_start_year), dtype=np.int16)),
                pa.array(np.full(n_rows, int(family_end_year), dtype=np.int16)),
                pa.array([str(family_label_name)] * n_rows, type=pa.string()),
                pa.array(np.full(n_rows, int(source_year), dtype=np.int16)),
                pa.array([str(source_split_name)] * n_rows, type=pa.string()),
                pa.array([str(source_suffix_name)] * n_rows, type=pa.string()),
                pa.array(ids[src_idx].tolist(), type=pa.string()),
                pa.array(ids[dst_idx].tolist(), type=pa.string()),
                pa.array(w[src_idx, dst_idx]),
            ],
            schema=self.edges_schema,
        )
        self.edges_writer.write_table(table)

    def close(self) -> None:
        self.runs_writer.close()
        self.basis_writer.close()
        self.edges_writer.close()


def read_existing_run_keys(path: Path) -> set[tuple[str, int, str]]:
    p = Path(path)
    if not p.exists():
        return set()
    table = pq.read_table(str(p), columns=["graph_tag", "source_year", "graph_kind"])
    frame = table.to_pandas()
    if frame.empty:
        return set()
    return {
        (str(row["graph_tag"]), int(row["source_year"]), str(row["graph_kind"]))
        for _, row in frame.iterrows()
    }


def apply_block_pca(x: np.ndarray, target_dim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("block PCA expects 2D array")
    if int(target_dim) <= 0:
        return arr
    sc = StandardScaler()
    xs = sc.fit_transform(arr)
    k = int(max(1, min(int(target_dim), int(xs.shape[0]), int(xs.shape[1]))))
    if k >= int(xs.shape[1]):
        return np.asarray(xs, dtype=np.float64)
    pca = PCA(n_components=int(k), svd_solver="full", random_state=0)
    return np.asarray(pca.fit_transform(xs), dtype=np.float64)


def standardize_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("standardize_array expects 2D array")
    sc = StandardScaler()
    return np.asarray(sc.fit_transform(arr), dtype=np.float64)


def detect_coord_mode(coords: np.ndarray) -> str:
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError("coords must have shape [n,2]")
    max0 = float(np.nanmax(np.abs(c[:, 0])))
    max1 = float(np.nanmax(np.abs(c[:, 1])))
    if max0 <= 90.0 and max1 <= 180.0:
        return "latlon_01"
    if max1 <= 90.0 and max0 <= 180.0:
        return "latlon_10"
    return "xy"


def to_lat_lon(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mode = detect_coord_mode(coords)
    if mode == "latlon_01":
        lat = coords[:, 0]
        lon = coords[:, 1]
    elif mode == "latlon_10":
        lon = coords[:, 0]
        lat = coords[:, 1]
    else:
        raise ValueError("coords are not recognizable as lat/lon")
    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def haversine_km(coords: np.ndarray) -> np.ndarray:
    lat_deg, lon_deg = to_lat_lon(coords)
    lat = np.deg2rad(lat_deg).reshape(-1, 1)
    lon = np.deg2rad(lon_deg).reshape(-1, 1)
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat) * np.cos(lat.T) * (np.sin(dlon * 0.5) ** 2)
    np.clip(a, 0.0, 1.0, out=a)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 1e-12, None)))
    return 6371.0 * c


def euclidean_dist(coords: np.ndarray) -> np.ndarray:
    c = np.asarray(coords, dtype=np.float64)
    d = c[:, None, :] - c[None, :, :]
    return np.sqrt(np.sum(d * d, axis=-1))


def pairwise_distance(coords: np.ndarray) -> np.ndarray:
    mode = detect_coord_mode(coords)
    if mode.startswith("latlon"):
        return haversine_km(coords)
    return euclidean_dist(coords)


def knn_weight_matrix(dist: np.ndarray, k: int, bandwidth_k: int | None = None, eps: float = 1e-9) -> np.ndarray:
    n = int(dist.shape[0])
    if dist.shape[1] != n:
        raise ValueError("dist must be square")
    work = np.asarray(dist, dtype=np.float64).copy()
    np.fill_diagonal(work, np.inf)
    k_eff = int(max(1, min(int(k), n - 1)))
    idx = np.argpartition(work, kth=k_eff - 1, axis=1)[:, :k_eff]
    dsel = np.take_along_axis(work, idx, axis=1)
    if bandwidth_k is None:
        finite = dsel[np.isfinite(dsel)]
        bw = float(np.median(finite)) if finite.size else 1.0
    else:
        kb = min(max(int(bandwidth_k), 1), k_eff)
        kth = np.partition(dsel, kth=kb - 1, axis=1)[:, kb - 1]
        finite = kth[np.isfinite(kth)]
        bw = float(np.median(finite)) if finite.size else 1.0
    if (not np.isfinite(bw)) or bw <= eps:
        bw = 1.0
    w = np.zeros((n, n), dtype=np.float64)
    wsel = np.exp(-np.square(dsel / bw))
    wsel[~np.isfinite(wsel)] = 0.0
    np.put_along_axis(w, idx, wsel, axis=1)
    rs = np.clip(w.sum(axis=1, keepdims=True), eps, None)
    return w / rs


def build_support_mask(dist: np.ndarray, support_k: int) -> np.ndarray:
    n = int(dist.shape[0])
    work = np.asarray(dist, dtype=np.float64).copy()
    np.fill_diagonal(work, np.inf)
    k_eff = int(max(1, min(int(support_k), n - 1)))
    idx = np.argpartition(work, kth=k_eff - 1, axis=1)[:, :k_eff]
    mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n, dtype=np.int64)[:, None]
    mask[rows, idx] = True
    mask = np.logical_or(mask, mask.T)
    np.fill_diagonal(mask, False)
    return mask


def _row_topk_sparsify_symmetric(w: np.ndarray, k_row: int) -> csr_matrix:
    n = int(w.shape[0])
    if k_row >= n - 1:
        ws = 0.5 * (w + w.T)
        return csr_matrix(ws)
    k_eff = int(max(1, min(int(k_row), n - 1)))
    work = np.asarray(w, dtype=np.float64).copy()
    np.fill_diagonal(work, -np.inf)
    idx = np.argpartition(work, kth=n - k_eff, axis=1)[:, -k_eff:]
    rows = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    cols = idx.reshape(-1)
    vals = work[rows, cols]
    vals[~np.isfinite(vals)] = 0.0
    s = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    s = 0.5 * (s + s.T)
    s.setdiag(0.0)
    s.eliminate_zeros()
    return s


def build_moran_basis_fast(w: np.ndarray, top_k: int, eps: float = 1e-9, row_topk: int = 96) -> tuple[np.ndarray, np.ndarray]:
    n = int(w.shape[0])
    if w.shape[1] != n:
        raise ValueError("w must be square")
    s = _row_topk_sparsify_symmetric(w, k_row=int(row_topk))
    one_over_n = 1.0 / float(n)

    def center(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v, dtype=np.float64)
        return vv - np.sum(vv) * one_over_n

    def matvec(v: np.ndarray) -> np.ndarray:
        cv = center(v)
        y = s @ cv
        return center(np.asarray(y, dtype=np.float64))

    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    k_req = int(min(max(int(top_k) + 8, 2 * int(top_k)), n - 2))
    evals, evecs = eigsh(op, k=k_req, which="LA", tol=1e-3, maxiter=max(300, 4 * n))
    order = np.argsort(evals)[::-1]
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)
    mask = evals > float(eps)
    if not np.any(mask):
        raise RuntimeError("no positive eigenvalues for Moran basis")
    pevals = evals[mask]
    pevecs = evecs[:, mask]
    k = int(min(max(1, int(top_k)), pevecs.shape[1]))
    return pevals[:k], pevecs[:, :k]


def sample_random_walk_positives(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = int(w.shape[0])
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        p = np.asarray(w[i], dtype=np.float64)
        s = float(np.sum(p))
        if s <= 0.0 or (not np.isfinite(s)):
            j = int(rng.integers(0, max(1, n - 1)))
            if j >= i and n > 1:
                j += 1
            out[i] = min(j, n - 1)
            continue
        out[i] = int(rng.choice(n, p=p / s))
    return out


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = int(z1.shape[0])
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / float(max(temperature, 1e-6))
    diag = torch.eye(2 * n, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -1e9)
    targets = torch.cat(
        [
            torch.arange(n, 2 * n, device=sim.device, dtype=torch.long),
            torch.arange(0, n, device=sim.device, dtype=torch.long),
        ],
        dim=0,
    )
    return F.cross_entropy(sim, targets)


def spatial_nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float, *, support_mask: torch.Tensor, dist_norm: torch.Tensor) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = int(z1.shape[0])
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / float(max(temperature, 1e-6))
    diag = torch.eye(2 * n, device=sim.device, dtype=torch.bool)
    neg_w = 1.0 + support_mask.to(dtype=sim.dtype) / (1.0 + torch.clamp(dist_norm.to(dtype=sim.dtype), min=0.0))
    neg_w = neg_w.repeat(2, 2)
    neg_w = torch.where(diag, torch.ones_like(neg_w), neg_w)
    sim = sim + torch.log(torch.clamp(neg_w, min=1e-6))
    sim = sim.masked_fill(diag, -1e9)
    targets = torch.cat(
        [
            torch.arange(n, 2 * n, device=sim.device, dtype=torch.long),
            torch.arange(0, n, device=sim.device, dtype=torch.long),
        ],
        dim=0,
    )
    return F.cross_entropy(sim, targets)


def _offdiag_flat(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("offdiag expects square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(y1: torch.Tensor, y2: torch.Tensor, *, offdiag_lambda: float, eps: float = 1e-9) -> torch.Tensor:
    if y1.ndim != 2 or y2.ndim != 2:
        raise ValueError("barlow_twins_loss expects 2D inputs")
    if y1.shape != y2.shape:
        raise ValueError("barlow_twins_loss expects matching shapes")
    z1 = (y1 - y1.mean(dim=0, keepdim=True)) / torch.clamp(y1.std(dim=0, keepdim=True, unbiased=False), min=eps)
    z2 = (y2 - y2.mean(dim=0, keepdim=True)) / torch.clamp(y2.std(dim=0, keepdim=True, unbiased=False), min=eps)
    n = max(int(y1.shape[0]), 1)
    c = torch.matmul(z1.T, z2) / float(n)
    on_diag = torch.sum((torch.diagonal(c) - 1.0) ** 2)
    off_diag = torch.sum(_offdiag_flat(c) ** 2)
    return on_diag + float(offdiag_lambda) * off_diag


class SignedGeMPool(nn.Module):
    def __init__(self, in_dim: int, p_init: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        p0 = float(max(p_init, 1.05))
        raw0 = math.log(math.exp(p0 - 1.0) - 1.0)
        self.in_dim = int(in_dim)
        self.raw_p = nn.Parameter(torch.tensor(float(raw0), dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = torch.clamp(m.sum(dim=1), min=1.0)
        mean = torch.sum(m * x, dim=1) / denom
        sign = torch.sign(mean)
        p = 1.0 + F.softplus(self.raw_p)
        x_mag = torch.clamp(torch.abs(x), min=self.eps)
        x_pow = torch.pow(x_mag, p)
        agg = torch.sum(m * x_pow, dim=1) / denom
        return sign * torch.pow(torch.clamp(agg, min=self.eps), 1.0 / p)


class MeanMaxPool(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = torch.clamp(m.sum(dim=1), min=1.0)
        mean = torch.sum(m * x, dim=1) / denom
        x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        mx = torch.amax(x_masked, dim=1)
        mx = torch.where(torch.isfinite(mx), mx, mean)
        return torch.cat([mean, mx], dim=1)


class MaskedAttentionPool(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 256, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.v = nn.Linear(int(in_dim), int(attn_dim))
        self.u = nn.Linear(int(in_dim), int(attn_dim))
        self.w = nn.Linear(int(attn_dim), 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(float(attn_dropout)) if float(attn_dropout) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.w(self.tanh(self.v(x)) * self.sigmoid(self.u(x))).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        attn = torch.softmax(logits, dim=1)
        attn = self.dropout(attn)
        attn = torch.where(mask, attn, torch.zeros_like(attn))
        norm = torch.clamp(attn.sum(dim=1, keepdim=True), min=1e-9)
        attn = attn / norm
        return torch.sum(attn.unsqueeze(-1) * x, dim=1)


class NetVLADPool(nn.Module):
    def __init__(self, in_dim: int, clusters: int = 8) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.clusters = int(max(2, clusters))
        self.assignment = nn.Linear(int(in_dim), int(self.clusters), bias=True)
        self.centroids = nn.Parameter(torch.randn(int(self.clusters), int(in_dim), dtype=torch.float32) * 0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.assignment(x)
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e9)
        assign = torch.softmax(logits, dim=2)
        assign = assign * mask.to(dtype=x.dtype).unsqueeze(-1)
        residual = x.unsqueeze(2) - self.centroids.view(1, 1, int(self.clusters), int(self.in_dim))
        vlad = torch.sum(assign.unsqueeze(-1) * residual, dim=1)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.reshape(x.shape[0], int(self.clusters) * int(self.in_dim))
        return F.normalize(vlad, p=2, dim=1)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        block_specs: dict[str, dict[str, object]],
        block_order: list[str],
        hidden_dim: int,
        joint_dim: int,
        dropout_p: float,
        pool_mode: str,
        bottleneck_dims: dict[str, int],
        attention_hidden_dim: int,
        attention_dropout: float,
        gem_p_init: dict[str, float],
        netvlad_clusters: int,
        remote_gating: bool,
        bag_keep_rates: dict[str, float],
        projector_hidden_dim: int,
        projector_dim: int,
    ) -> None:
        super().__init__()
        self.dropout_p = float(dropout_p)
        self.pool_mode = str(pool_mode)
        self.block_order = list(block_order)
        self.block_specs = {str(k): dict(v) for k, v in block_specs.items()}
        self.block_poolers = nn.ModuleDict()
        self.block_projectors = nn.ModuleDict()
        self.remote_gating = bool(remote_gating)
        self.bag_keep_rates = {str(k): float(min(max(v, 0.0), 1.0)) for k, v in dict(bag_keep_rates).items()}
        self.remote_block_names: list[str] = []
        total_in = 0
        for name in self.block_order:
            spec = dict(self.block_specs[name])
            kind = str(spec["kind"])
            in_dim = int(spec["input_dim"])
            if kind == "bag":
                if self.pool_mode == "mean_max":
                    pooler = MeanMaxPool()
                    pooled_dim = 2 * int(in_dim)
                elif self.pool_mode == "attention":
                    pooler = MaskedAttentionPool(in_dim=int(in_dim), attn_dim=int(max(1, attention_hidden_dim)), attn_dropout=float(attention_dropout))
                    pooled_dim = int(in_dim)
                elif self.pool_mode == "gem":
                    pooler = SignedGeMPool(in_dim=int(in_dim), p_init=float(gem_p_init.get(name, gem_p_init.get("default", 3.0))))
                    pooled_dim = int(in_dim)
                elif self.pool_mode == "netvlad":
                    pooler = NetVLADPool(in_dim=int(in_dim), clusters=int(netvlad_clusters))
                    pooled_dim = int(in_dim) * int(netvlad_clusters)
                else:
                    raise ValueError(f"unsupported pool_mode={self.pool_mode!r}")
                self.block_poolers[name] = pooler
            else:
                pooled_dim = int(in_dim)
            out_dim = int(bottleneck_dims.get(name, 0))
            if out_dim > 0 and out_dim != pooled_dim:
                self.block_projectors[name] = nn.Sequential(nn.LayerNorm(int(pooled_dim)), nn.Linear(int(pooled_dim), int(out_dim)), nn.GELU())
                total_in += int(out_dim)
            else:
                self.block_projectors[name] = nn.LayerNorm(int(pooled_dim))
                total_in += int(pooled_dim)
            if str(name) != "admin":
                self.remote_block_names.append(str(name))
        total_remote = int(sum(int(self.block_projectors[name][1].out_features) if isinstance(self.block_projectors[name], nn.Sequential) else self.block_specs[name]["input_dim"] if str(self.block_specs[name]["kind"]) == "dense" else (2 * int(self.block_specs[name]["input_dim"]) if self.pool_mode == "mean_max" else int(self.block_specs[name]["input_dim"]) if self.pool_mode != "netvlad" else int(self.block_specs[name]["input_dim"]) * int(netvlad_clusters)) for name in self.remote_block_names))
        if self.remote_gating and total_remote > 0:
            gate_hidden = int(max(8, min(128, total_remote // 2 if total_remote >= 16 else total_remote)))
            self.remote_gate = nn.Sequential(nn.LayerNorm(int(total_remote)), nn.Linear(int(total_remote), int(gate_hidden)), nn.GELU(), nn.Linear(int(gate_hidden), int(total_remote)))
        else:
            self.remote_gate = None
        self.net = nn.Sequential(nn.LayerNorm(int(total_in)), nn.Linear(int(total_in), int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), int(joint_dim)))
        self.ssl_projector = nn.Sequential(
            nn.LayerNorm(int(joint_dim)),
            nn.Linear(int(joint_dim), int(max(1, projector_hidden_dim))),
            nn.GELU(),
            nn.Linear(int(max(1, projector_hidden_dim)), int(max(1, projector_dim))),
        )

    def _subsample_mask(self, mask: torch.Tensor, keep_rate: float) -> torch.Tensor:
        keep_prob = float(min(max(keep_rate, 0.0), 1.0))
        if keep_prob >= 1.0:
            return mask
        sampled = torch.logical_and(mask, torch.rand(mask.shape, device=mask.device) < keep_prob)
        missing = torch.logical_and(torch.any(mask, dim=1), ~torch.any(sampled, dim=1))
        if torch.any(missing):
            fallback_idx = torch.argmax(mask.to(dtype=torch.int64), dim=1)
            sampled = sampled.clone()
            sampled[missing, fallback_idx[missing]] = True
        return sampled

    def _augment_pack(self, pack: dict[str, object]) -> dict[str, object]:
        blocks_out: dict[str, dict[str, object]] = {}
        for name in self.block_order:
            block = dict(pack["blocks"][name])
            if str(block["kind"]) == "bag":
                blocks_out[name] = {"kind": "bag", "x": block["x"], "mask": self._subsample_mask(block["mask"], float(self.bag_keep_rates.get(name, self.bag_keep_rates.get("default", 1.0))))}
            else:
                blocks_out[name] = {"kind": "dense", "x": block["x"]}
        return {"blocks": blocks_out, "block_order": list(pack["block_order"]), "block_specs": dict(pack["block_specs"])}

    def pool_blocks(self, pack: dict[str, object]) -> dict[str, torch.Tensor]:
        pooled: dict[str, torch.Tensor] = {}
        for name in self.block_order:
            block = dict(pack["blocks"][name])
            if str(block["kind"]) == "bag":
                pooled[name] = self.block_poolers[name](block["x"], block["mask"])
            else:
                pooled[name] = block["x"]
        return pooled

    def extract_block_features(self, pack: dict[str, object]) -> dict[str, torch.Tensor]:
        pooled = self.pool_blocks(pack)
        return {name: self.block_projectors[name](pooled[name]) for name in self.block_order}

    def assemble_features(self, pack: dict[str, object]) -> torch.Tensor:
        block_features = self.extract_block_features(pack)
        if self.remote_gate is not None and self.remote_block_names:
            remote_parts = [block_features[name] for name in self.remote_block_names if name in block_features]
            if remote_parts:
                remote_cat = torch.cat(remote_parts, dim=1)
                remote_cat = remote_cat * torch.sigmoid(self.remote_gate(remote_cat))
                split_sizes = [int(block_features[name].shape[1]) for name in self.remote_block_names if name in block_features]
                split_parts = torch.split(remote_cat, split_sizes, dim=1)
                remote_map = {name: part for name, part in zip([name for name in self.remote_block_names if name in block_features], split_parts)}
                return torch.cat([remote_map.get(name, block_features[name]) if name in remote_map else block_features[name] for name in self.block_order], dim=1)
        return torch.cat([block_features[name] for name in self.block_order], dim=1)

    def encode(self, pack: dict[str, object]) -> torch.Tensor:
        return F.normalize(self.net(self.assemble_features(pack)), dim=1)

    def forward(self, pack: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.assemble_features(pack)
        z_raw = self.net(h)
        z = F.normalize(z_raw, dim=1)
        if self.training:
            aug_pack = self._augment_pack(pack)
            h_aug = self.assemble_features(aug_pack)
            if self.dropout_p > 0.0:
                keep_prob = max(1e-6, 1.0 - float(self.dropout_p))
                keep = (torch.rand_like(h_aug) < keep_prob).to(dtype=h_aug.dtype)
                h_aug = (h_aug * keep) / keep_prob
        else:
            h_aug = h
        z_aug_raw = self.net(h_aug)
        z_aug = F.normalize(z_aug_raw, dim=1)
        y = self.ssl_projector(z_raw)
        y_aug = self.ssl_projector(z_aug_raw)
        return z, z_aug, y, y_aug


def resolve_gem_p_init_map(config: TopologyConfig) -> dict[str, float]:
    out = {"default": 3.0}
    for modality in config.modalities:
        block_cfg = config.block_cfg(modality)
        out[str(modality)] = float(block_cfg.gem_p_init)
    return out


def resolve_bag_keep_rate_map(config: TopologyConfig) -> dict[str, float]:
    out = {"default": 1.0}
    for modality in config.modalities:
        block_cfg = config.block_cfg(modality)
        out[str(modality)] = float(block_cfg.bag_keep_rate)
    return out


def build_graph_optimizer(model: nn.Module, *, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))


def build_soft_pre_adjacency_torch(z: torch.Tensor, *, support_mask: torch.Tensor, geo_penalty: torch.Tensor, tau_graph: float, beta_geo: float, geo_residual_graph: bool) -> torch.Tensor:
    dist_lat = torch.cdist(z, z, p=1.0)
    lat_aff = torch.exp(-(dist_lat / float(max(tau_graph, 1e-6))))
    lat_aff = torch.where(support_mask, lat_aff, torch.zeros_like(lat_aff))
    lat_aff.fill_diagonal_(0.0)
    if bool(geo_residual_graph):
        geo_score = -float(beta_geo) * geo_penalty
        geo_score = torch.where(support_mask, geo_score, torch.full_like(geo_score, -1e9))
        geo_score.fill_diagonal_(-1e9)
        geo_w = torch.softmax(geo_score, dim=1)
        geo_w = torch.where(torch.isfinite(geo_w), geo_w, torch.zeros_like(geo_w))
        geo_w.fill_diagonal_(0.0)
        return geo_w * (1.0 + lat_aff)
    score = -(dist_lat / float(max(tau_graph, 1e-6))) - float(beta_geo) * geo_penalty
    score = torch.where(support_mask, score, torch.full_like(score, -1e9))
    score.fill_diagonal_(-1e9)
    pre_w = torch.exp(score)
    pre_w = torch.where(torch.isfinite(pre_w), pre_w, torch.zeros_like(pre_w))
    pre_w.fill_diagonal_(0.0)
    return pre_w


def degree_penalty_loss(z: torch.Tensor, *, support_mask: torch.Tensor, geo_penalty: torch.Tensor, tau_graph: float, beta_geo: float, geo_residual_graph: bool) -> torch.Tensor:
    pre_w = build_soft_pre_adjacency_torch(z, support_mask=support_mask, geo_penalty=geo_penalty, tau_graph=tau_graph, beta_geo=beta_geo, geo_residual_graph=geo_residual_graph)
    deg = torch.sum(pre_w, dim=1)
    deg_mu = torch.clamp(torch.mean(deg), min=1e-6)
    deg_rel = deg / deg_mu
    return torch.mean((deg_rel - 1.0) ** 2)


def build_learned_adjacency(z: np.ndarray, *, support_mask: np.ndarray, geo_penalty: np.ndarray, tau_graph: float, beta_geo: float, final_row_topk: int, geo_residual_graph: bool = False, mutual_knn: bool = False) -> np.ndarray:
    z_arr = np.asarray(z, dtype=np.float64)
    support = np.asarray(support_mask, dtype=bool)
    geo = np.asarray(geo_penalty, dtype=np.float64)
    dist_lat = np.sum(np.abs(z_arr[:, None, :] - z_arr[None, :, :]), axis=-1)
    lat_aff = np.exp(-(dist_lat / float(max(tau_graph, 1e-6))))
    lat_aff[~np.isfinite(lat_aff)] = 0.0
    lat_aff[~support] = 0.0
    np.fill_diagonal(lat_aff, 0.0)
    if bool(geo_residual_graph):
        geo_score = -float(beta_geo) * geo
        geo_score[~support] = -np.inf
        np.fill_diagonal(geo_score, -np.inf)
        geo_row_max = np.max(np.where(np.isfinite(geo_score), geo_score, -1e9), axis=1, keepdims=True)
        geo_w = np.exp(np.where(np.isfinite(geo_score), geo_score - geo_row_max, -1e9))
        geo_w[~np.isfinite(geo_w)] = 0.0
        np.fill_diagonal(geo_w, 0.0)
        geo_rs = np.clip(geo_w.sum(axis=1, keepdims=True), 1e-9, None)
        geo_w = geo_w / geo_rs
        w = geo_w * (1.0 + lat_aff)
    else:
        score = -(dist_lat / float(max(tau_graph, 1e-6))) - float(beta_geo) * geo
        score[~support] = -np.inf
        np.fill_diagonal(score, -np.inf)
        row_max = np.max(np.where(np.isfinite(score), score, -1e9), axis=1, keepdims=True)
        w = np.exp(np.where(np.isfinite(score), score - row_max, -1e9))
        w[~np.isfinite(w)] = 0.0
        np.fill_diagonal(w, 0.0)
    if int(final_row_topk) > 0 and int(final_row_topk) < int(w.shape[0] - 1):
        k = int(final_row_topk)
        work = w.copy()
        np.fill_diagonal(work, -np.inf)
        idx = np.argpartition(work, kth=work.shape[1] - k, axis=1)[:, -k:]
        keep = np.zeros_like(work, dtype=bool)
        rows = np.arange(work.shape[0], dtype=np.int64)[:, None]
        keep[rows, idx] = True
        if bool(mutual_knn):
            keep = np.logical_and(keep, keep.T)
        w = np.where(keep, np.asarray(w, dtype=np.float64), 0.0)
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    rs = np.clip(w.sum(axis=1, keepdims=True), 1e-9, None)
    return w / rs


def extract_pool_stats(model: nn.Module) -> dict[str, float]:
    stats: dict[str, float] = {}
    for name, pooler in getattr(model, "block_poolers", {}).items():
        if isinstance(pooler, SignedGeMPool):
            p = 1.0 + F.softplus(pooler.raw_p.detach())
            stats[f"gem_p_{name}"] = float(torch.mean(p).cpu().item())
    return stats


def parquet_rows_for_family(input_parquet: Path, *, family_tag_name: str, source_year: int) -> pd.DataFrame:
    table = pq.read_table(
        str(input_parquet),
        columns=["fips", "item_index", "item_count", "embedding"],
        filters=[("family_tag", "=", str(family_tag_name)), ("source_year", "=", int(source_year))],
    )
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError(f"{input_parquet}: no rows for family={family_tag_name} source_year={int(source_year)}")
    frame["fips"] = frame["fips"].astype(str).str.strip().str.zfill(5)
    frame["item_index"] = pd.to_numeric(frame["item_index"], errors="raise").astype(np.int64)
    frame["item_count"] = pd.to_numeric(frame["item_count"], errors="raise").astype(np.int64)
    frame["embedding"] = frame["embedding"].apply(lambda x: np.asarray(list(x), dtype=np.float64))
    return frame.sort_values(["fips", "item_index"]).reset_index(drop=True)


def load_dense_block(modality_cfg: ModalityConfig, *, family_end_year: int, source_year: int) -> DenseBlockRows:
    fam_tag = family_tag(str(modality_cfg.family_tag_base), int(family_end_year))
    frame = parquet_rows_for_family(modality_cfg.input_parquet, family_tag_name=fam_tag, source_year=int(source_year))
    order = frame["fips"].drop_duplicates(keep="first").astype(str).tolist()
    x_rows: list[np.ndarray] = []
    for fid, part in frame.groupby("fips", sort=False):
        if int(part.shape[0]) != 1:
            raise ValueError(f"{modality_cfg.input_parquet}: dense block expected exactly one row per fips; got {part.shape[0]} for {fid}")
        x_rows.append(np.asarray(part["embedding"].iloc[0], dtype=np.float64))
    return DenseBlockRows(fips=np.asarray(order, dtype="U5"), x=np.vstack(x_rows).astype(np.float64, copy=False))


def load_bag_block(modality_cfg: ModalityConfig, *, family_end_year: int, source_year: int) -> BagBlockRows:
    fam_tag = family_tag(str(modality_cfg.family_tag_base), int(family_end_year))
    frame = parquet_rows_for_family(modality_cfg.input_parquet, family_tag_name=fam_tag, source_year=int(source_year))
    groups = list(frame.groupby("fips", sort=False))
    if not groups:
        raise ValueError(f"{modality_cfg.input_parquet}: no bag groups for family={fam_tag} source_year={int(source_year)}")
    fips = np.asarray([str(fid) for fid, _ in groups], dtype="U5")
    max_tiles = int(max(int(part.shape[0]) for _, part in groups))
    embed_dim = int(np.asarray(groups[0][1]["embedding"].iloc[0], dtype=np.float64).shape[0])
    x = np.zeros((int(len(groups)), int(max_tiles), int(embed_dim)), dtype=np.float64)
    mask = np.zeros((int(len(groups)), int(max_tiles)), dtype=bool)
    for row_idx, (_fid, part) in enumerate(groups):
        part = part.sort_values("item_index").reset_index(drop=True)
        emb = np.vstack([np.asarray(v, dtype=np.float64) for v in part["embedding"].tolist()]).astype(np.float64, copy=False)
        n = int(emb.shape[0])
        x[row_idx, :n, :] = emb
        mask[row_idx, :n] = True
    return BagBlockRows(fips=fips, x=x, mask=mask)


def load_modality_block(modality_cfg: ModalityConfig, *, family_end_year: int, source_year: int) -> DenseBlockRows | BagBlockRows:
    if str(modality_cfg.kind) == "dense":
        return load_dense_block(modality_cfg, family_end_year=int(family_end_year), source_year=int(source_year))
    return load_bag_block(modality_cfg, family_end_year=int(family_end_year), source_year=int(source_year))


def build_feature_pack(config: TopologyConfig, *, family_end_year: int, source_year: int) -> FeaturePack:
    coords_by_fips = load_county_coords(config.paths.geo_coords_path)
    active_modalities = [m for m in config.modalities if bool(config.block_cfg(m).enabled)]
    if not active_modalities:
        raise ValueError("no enabled graph modalities")
    raw_blocks: dict[str, DenseBlockRows | BagBlockRows] = {}
    common = set(str(k).zfill(5) for k in coords_by_fips.keys())
    base_order: np.ndarray | None = None
    for modality in active_modalities:
        block_cfg = config.block_cfg(modality)
        block_rows = load_modality_block(block_cfg, family_end_year=int(family_end_year), source_year=int(source_year))
        raw_blocks[str(modality)] = block_rows
        common &= set(np.asarray(block_rows.fips, dtype="U5").tolist())
        if base_order is None:
            base_order = np.asarray(block_rows.fips, dtype="U5")
    if base_order is None:
        raise RuntimeError("no modalities loaded for graph feature pack")
    sample_ids_all = np.asarray([fid for fid in np.asarray(base_order, dtype="U5").tolist() if fid in common], dtype="U5")
    if int(sample_ids_all.shape[0]) <= 1:
        raise ValueError("selected graph modalities leave too few counties after alignment")
    coords_all = np.asarray([coords_by_fips[str(fid)] for fid in sample_ids_all.tolist()], dtype=np.float64)
    active_mask = np.ones(int(sample_ids_all.shape[0]), dtype=bool)
    aligned_dense: dict[str, np.ndarray] = {}
    aligned_bags: dict[str, dict[str, np.ndarray]] = {}
    block_order: list[str] = []
    for modality in active_modalities:
        block_rows = raw_blocks[str(modality)]
        idx = {str(fid): i for i, fid in enumerate(np.asarray(block_rows.fips, dtype="U5").tolist())}
        rows = np.asarray([idx[str(fid)] for fid in sample_ids_all.tolist()], dtype=np.int64)
        block_order.append(str(modality))
        if isinstance(block_rows, DenseBlockRows):
            x = np.asarray(block_rows.x[rows], dtype=np.float64)
            aligned_dense[str(modality)] = x
            active_mask &= np.all(np.isfinite(x), axis=1)
        else:
            x = np.asarray(block_rows.x[rows], dtype=np.float64)
            mask = np.asarray(block_rows.mask[rows], dtype=bool)
            aligned_bags[str(modality)] = {"x": x, "mask": mask}
            active_mask &= np.asarray(mask.any(axis=1), dtype=bool)
    if int(np.count_nonzero(active_mask)) <= 1:
        raise ValueError("graph feature pack has too few active counties after support filtering")
    sample_ids = np.asarray(sample_ids_all[active_mask], dtype="U5")
    coords = np.asarray(coords_all[active_mask], dtype=np.float64)
    blocks: dict[str, dict[str, object]] = {}
    block_specs: dict[str, dict[str, object]] = {}
    block_dims: dict[str, int] = {}
    for modality in block_order:
        block_cfg = config.block_cfg(modality)
        if modality in aligned_dense:
            x = np.asarray(aligned_dense[modality][active_mask], dtype=np.float64)
            if int(config.graph.block_pca_dim) > 0:
                x = apply_block_pca(x, int(config.graph.block_pca_dim))
            x = standardize_array(x)
            blocks[str(modality)] = {"kind": "dense", "x": x}
            block_specs[str(modality)] = {"kind": "dense", "input_dim": int(x.shape[1])}
            block_dims[str(modality)] = int(x.shape[1])
        else:
            x = np.asarray(aligned_bags[modality]["x"][active_mask], dtype=np.float64)
            mask = np.asarray(aligned_bags[modality]["mask"][active_mask], dtype=bool)
            blocks[str(modality)] = {"kind": "bag", "x": x, "mask": mask}
            block_specs[str(modality)] = {"kind": "bag", "input_dim": int(x.shape[2])}
            block_dims[str(modality)] = int(x.shape[2])
    return FeaturePack(sample_ids=sample_ids, coords=coords, blocks=blocks, block_order=block_order, block_specs=block_specs, block_dims=block_dims)


def feature_pack_to_torch(pack: FeaturePack, device: torch.device) -> dict[str, object]:
    blocks_out: dict[str, dict[str, object]] = {}
    for name, block in pack.blocks.items():
        if str(block["kind"]) == "dense":
            blocks_out[name] = {"kind": "dense", "x": torch.as_tensor(np.asarray(block["x"], dtype=np.float64), dtype=torch.float32, device=device)}
        else:
            blocks_out[name] = {
                "kind": "bag",
                "x": torch.as_tensor(np.asarray(block["x"], dtype=np.float64), dtype=torch.float32, device=device),
                "mask": torch.as_tensor(np.asarray(block["mask"], dtype=bool), dtype=torch.bool, device=device),
            }
    return {"blocks": blocks_out, "block_order": list(pack.block_order), "block_specs": dict(pack.block_specs)}


def train_graph_slice(config: TopologyConfig, *, family_end_year: int, source_year: int) -> TrainedGraphArtifact:
    pack = build_feature_pack(config, family_end_year=int(family_end_year), source_year=int(source_year))
    graph_cfg = config.graph
    dist = pairwise_distance(np.asarray(pack.coords, dtype=np.float64))
    finite = dist[np.isfinite(dist)]
    med = float(np.median(finite)) if finite.size else 1.0
    if (not np.isfinite(med)) or med <= 1e-9:
        med = 1.0
    dist_norm = dist / med
    geo_penalty = np.log1p(float(graph_cfg.geo_gamma) * dist_norm).astype(np.float64)
    support_mask = build_support_mask(dist, support_k=int(graph_cfg.support_k))
    w_knn = knn_weight_matrix(dist, k=int(graph_cfg.knn_k), bandwidth_k=int(graph_cfg.knn_bandwidth_k))
    pos_rw = sample_random_walk_positives(w_knn, np.random.default_rng(int(graph_cfg.seed) + 1000 * int(family_end_year) + int(source_year)))
    if str(graph_cfg.device).strip().lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        req = str(graph_cfg.device).strip().lower()
        device = torch.device("cpu") if req.startswith("cuda") and (not torch.cuda.is_available()) else torch.device(str(graph_cfg.device))
    np.random.seed(int(graph_cfg.seed))
    torch.manual_seed(int(graph_cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(graph_cfg.seed))
    pack_torch = feature_pack_to_torch(pack, device=device)
    pos_idx = torch.as_tensor(pos_rw, dtype=torch.long, device=device)
    support_mask_t = torch.as_tensor(np.asarray(support_mask, dtype=bool), dtype=torch.bool, device=device)
    dist_norm_t = torch.as_tensor(np.asarray(dist_norm, dtype=np.float32), dtype=torch.float32, device=device)
    geo_penalty_t = torch.as_tensor(np.asarray(geo_penalty, dtype=np.float32), dtype=torch.float32, device=device)
    bottleneck_dims = {str(m): int(config.block_cfg(m).bottleneck_dim) for m in config.modalities if bool(config.block_cfg(m).enabled)}
    model = GraphEncoder(
        block_specs=dict(pack.block_specs),
        block_order=list(pack.block_order),
        hidden_dim=int(graph_cfg.hidden_dim),
        joint_dim=int(graph_cfg.joint_dim),
        dropout_p=float(graph_cfg.dropout),
        pool_mode=str(graph_cfg.pool_mode),
        bottleneck_dims=bottleneck_dims,
        attention_hidden_dim=int(graph_cfg.attention_hidden_dim),
        attention_dropout=float(graph_cfg.attention_dropout),
        gem_p_init=resolve_gem_p_init_map(config),
        netvlad_clusters=int(graph_cfg.netvlad_clusters),
        remote_gating=bool(graph_cfg.remote_gating),
        bag_keep_rates=resolve_bag_keep_rate_map(config),
        projector_hidden_dim=int(graph_cfg.projector_hidden_dim),
        projector_dim=int(graph_cfg.projector_dim),
    ).to(device)
    opt = build_graph_optimizer(model, lr=float(graph_cfg.lr), weight_decay=float(graph_cfg.weight_decay))
    best_loss = float("inf")
    best_state = None
    for epoch in range(int(graph_cfg.epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        z, z_aug, y_ssl, y_ssl_aug = model(pack_torch)
        if str(graph_cfg.graph_objective).strip().lower() == "barlow":
            loss_ssl = barlow_twins_loss(y_ssl, y_ssl_aug, offdiag_lambda=float(graph_cfg.barlow_lambda))
        else:
            if bool(graph_cfg.spatial_negative_mining):
                loss_ssl = spatial_nt_xent_loss(y_ssl, y_ssl_aug, temperature=float(graph_cfg.temperature), support_mask=support_mask_t, dist_norm=dist_norm_t)
            else:
                loss_ssl = nt_xent_loss(y_ssl, y_ssl_aug, temperature=float(graph_cfg.temperature))
        pull = 1.0 - torch.sum(z * z[pos_idx], dim=1)
        loss_pull = torch.mean(pull)
        loss_deg = degree_penalty_loss(z, support_mask=support_mask_t, geo_penalty=geo_penalty_t, tau_graph=float(graph_cfg.tau_graph), beta_geo=float(graph_cfg.beta_geo), geo_residual_graph=bool(graph_cfg.geo_residual_graph)) if bool(graph_cfg.degree_penalty) else torch.zeros((), dtype=z.dtype, device=z.device)
        loss = loss_ssl + float(graph_cfg.w_pull) * loss_pull + float(graph_cfg.degree_penalty_weight) * loss_deg
        loss.backward()
        opt.step()
        loss_val = float(loss.detach().cpu().item())
        if loss_val < best_loss:
            best_loss = float(loss_val)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        LOGGER.debug("[graph epoch %03d] family=%d source=%d loss=%.6f ssl=%.6f pull=%.6f deg=%.6f", epoch, int(family_end_year), int(source_year), loss_val, float(loss_ssl.detach().cpu()), float(loss_pull.detach().cpu()), float(loss_deg.detach().cpu()))
    if best_state is not None:
        model.load_state_dict(best_state)
    pool_stats = extract_pool_stats(model)
    model.eval()
    with torch.no_grad():
        z_final = np.asarray(model.encode(pack_torch).detach().cpu(), dtype=np.float64)
    w_learn = build_learned_adjacency(
        z_final,
        support_mask=support_mask,
        geo_penalty=geo_penalty,
        tau_graph=float(graph_cfg.tau_graph),
        beta_geo=float(graph_cfg.beta_geo),
        final_row_topk=int(graph_cfg.final_row_topk),
        geo_residual_graph=bool(graph_cfg.geo_residual_graph),
        mutual_knn=bool(graph_cfg.mutual_knn),
    )
    basis_row_topk = max(int(graph_cfg.support_k), int(graph_cfg.final_row_topk), 32)
    evals_learn, evecs_learn = build_moran_basis_fast(w_learn, top_k=int(graph_cfg.mem_top_k), row_topk=int(basis_row_topk))
    evals_knn, evecs_knn = build_moran_basis_fast(w_knn, top_k=int(graph_cfg.mem_top_k), row_topk=max(int(graph_cfg.knn_k), 32))
    return TrainedGraphArtifact(
        z=np.asarray(z_final, dtype=np.float64),
        fips=np.asarray(pack.sample_ids, dtype="U5"),
        coords=np.asarray(pack.coords, dtype=np.float64),
        w_learn=np.asarray(w_learn, dtype=np.float64),
        evals_learn=np.asarray(evals_learn, dtype=np.float64),
        evecs_learn=np.asarray(evecs_learn, dtype=np.float64),
        w_knn=np.asarray(w_knn, dtype=np.float64),
        evals_knn=np.asarray(evals_knn, dtype=np.float64),
        evecs_knn=np.asarray(evecs_knn, dtype=np.float64),
        block_dims=dict(pack.block_dims),
        pool_stats=dict(pool_stats),
        graph_loss=float(best_loss),
    )


def checkpoint_path(config: TopologyConfig, *, family_end_year: int, source_year: int) -> Path:
    tag = graph_tag(str(config.graph.graph_tag_base), int(family_end_year))
    return Path(config.paths.run_root) / tag / f"source_{int(source_year)}" / "best.pt"


def modality_set_label(config: TopologyConfig) -> str:
    return "+".join([m for m in config.modalities if bool(config.block_cfg(m).enabled)])


def save_checkpoint(config: TopologyConfig, *, family_end_year: int, source_year: int, artifact: TrainedGraphArtifact) -> None:
    path = checkpoint_path(config, family_end_year=int(family_end_year), source_year=int(source_year))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fips": np.asarray(artifact.fips, dtype="U5"),
            "coords": np.asarray(artifact.coords, dtype=np.float64),
            "z": np.asarray(artifact.z, dtype=np.float32),
            "graph_loss": float(artifact.graph_loss),
            "pool_stats": dict(artifact.pool_stats),
            "block_dims": dict(artifact.block_dims),
        },
        path,
    )


def write_artifact_tables(config: TopologyConfig, *, family_end_year: int, source_year: int, artifact: TrainedGraphArtifact, writer: TopologyParquetWriter) -> None:
    anchor_year = int(config.anchor_year)
    fam_label = family_label(anchor_year, int(family_end_year))
    g_tag = graph_tag(str(config.graph.graph_tag_base), int(family_end_year))
    src_split = source_split(family_end_year=int(family_end_year), source_year=int(source_year))
    src_suffix = source_suffix(family_end_year=int(family_end_year), source_year=int(source_year))
    mod_set = modality_set_label(config)
    writer.write_run(
        graph_tag_base=str(config.graph.graph_tag_base),
        graph_tag_name=g_tag,
        graph_kind="learned",
        family_start_year=anchor_year,
        family_end_year=int(family_end_year),
        family_label_name=fam_label,
        source_year=int(source_year),
        source_split_name=str(src_split),
        source_suffix_name=str(src_suffix),
        modality_set=mod_set,
        n_counties=int(artifact.fips.shape[0]),
        basis_dim=int(artifact.evals_learn.shape[0]),
        graph_loss=float(artifact.graph_loss),
        graph_cfg=config.graph,
        block_dims=dict(artifact.block_dims),
        pool_stats=dict(artifact.pool_stats),
    )
    writer.write_basis(
        graph_tag_base=str(config.graph.graph_tag_base),
        graph_tag_name=g_tag,
        graph_kind="learned",
        family_start_year=anchor_year,
        family_end_year=int(family_end_year),
        family_label_name=fam_label,
        source_year=int(source_year),
        source_split_name=str(src_split),
        source_suffix_name=str(src_suffix),
        fips=artifact.fips,
        evals=artifact.evals_learn,
        evecs=artifact.evecs_learn,
    )
    writer.write_edges(
        graph_tag_base=str(config.graph.graph_tag_base),
        graph_tag_name=g_tag,
        graph_kind="learned",
        family_start_year=anchor_year,
        family_end_year=int(family_end_year),
        family_label_name=fam_label,
        source_year=int(source_year),
        source_split_name=str(src_split),
        source_suffix_name=str(src_suffix),
        fips=artifact.fips,
        weights=artifact.w_learn,
    )
    if bool(config.graph.write_knn_reference):
        writer.write_run(
            graph_tag_base=str(config.graph.graph_tag_base),
            graph_tag_name=g_tag,
            graph_kind="knn",
            family_start_year=anchor_year,
            family_end_year=int(family_end_year),
            family_label_name=fam_label,
            source_year=int(source_year),
            source_split_name=str(src_split),
            source_suffix_name=str(src_suffix),
            modality_set=mod_set,
            n_counties=int(artifact.fips.shape[0]),
            basis_dim=int(artifact.evals_knn.shape[0]),
            graph_loss=float("nan"),
            graph_cfg=config.graph,
            block_dims=dict(artifact.block_dims),
            pool_stats={},
        )
        writer.write_basis(
            graph_tag_base=str(config.graph.graph_tag_base),
            graph_tag_name=g_tag,
            graph_kind="knn",
            family_start_year=anchor_year,
            family_end_year=int(family_end_year),
            family_label_name=fam_label,
            source_year=int(source_year),
            source_split_name=str(src_split),
            source_suffix_name=str(src_suffix),
            fips=artifact.fips,
            evals=artifact.evals_knn,
            evecs=artifact.evecs_knn,
        )
        writer.write_edges(
            graph_tag_base=str(config.graph.graph_tag_base),
            graph_tag_name=g_tag,
            graph_kind="knn",
            family_start_year=anchor_year,
            family_end_year=int(family_end_year),
            family_label_name=fam_label,
            source_year=int(source_year),
            source_split_name=str(src_split),
            source_suffix_name=str(src_suffix),
            fips=artifact.fips,
            weights=artifact.w_knn,
        )


def run(config: TopologyConfig, *, skip_existing: bool = False, family_end_year: int | None = None, source_year: int | None = None) -> None:
    family_years = [int(family_end_year)] if family_end_year is not None else config.years.values
    existing = read_existing_run_keys(config.paths.runs_parquet) if bool(skip_existing) else set()
    writer: TopologyParquetWriter | None = None
    try:
        for family_end in family_years:
            src_years = [int(source_year)] if source_year is not None else list(range(int(config.anchor_year), int(family_end) + 1))
            for src_year in src_years:
                g_tag = graph_tag(str(config.graph.graph_tag_base), int(family_end))
                learned_key = (str(g_tag), int(src_year), "learned")
                knn_key = (str(g_tag), int(src_year), "knn")
                if bool(skip_existing):
                    need_knn = bool(config.graph.write_knn_reference)
                    if learned_key in existing and ((not need_knn) or knn_key in existing):
                        LOGGER.info("[graph] skip family=%d source=%d existing parquet rows present", int(family_end), int(src_year))
                        continue
                LOGGER.info("[graph] family=%d source=%d modalities=%s", int(family_end), int(src_year), modality_set_label(config))
                artifact = train_graph_slice(config, family_end_year=int(family_end), source_year=int(src_year))
                save_checkpoint(config, family_end_year=int(family_end), source_year=int(src_year), artifact=artifact)
                if writer is None:
                    append = bool(skip_existing) and (config.paths.runs_parquet.exists() or config.paths.basis_parquet.exists() or config.paths.edges_parquet.exists())
                    writer = TopologyParquetWriter(config=config, append=append)
                write_artifact_tables(config, family_end_year=int(family_end), source_year=int(src_year), artifact=artifact, writer=writer)
                existing.add(learned_key)
                if bool(config.graph.write_knn_reference):
                    existing.add(knn_key)
    finally:
        if writer is not None:
            writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn county graph topologies from manifold parquet embeddings.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--family-end-year", type=int, default=None)
    parser.add_argument("--source-year", type=int, default=None)
    parser.add_argument("--skip", "--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(str(args.log_level))
    config = load_config(args.config)
    run(
        config,
        skip_existing=bool(args.skip_existing),
        family_end_year=None if args.family_end_year is None else int(args.family_end_year),
        source_year=None if args.source_year is None else int(args.source_year),
    )


if __name__ == "__main__":
    main()
