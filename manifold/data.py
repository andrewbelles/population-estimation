#!/usr/bin/env python3
#
# data.py  Andrew Belles  Mar 27th, 2026
#
# Dataset loading and parquet export utilities for manifold generation.
#

import csv
import logging
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, TensorDataset


LOGGER = logging.getLogger("manifold.data")


NONFEATURE_COLUMNS = {
    "fips",
    "year",
    "county_name",
    "state_abbr",
    "label",
    "label_level",
    "label_prev",
    "label_delta",
    "target_correction_log",
    "target_correction_level",
}


@dataclass(slots=True)
class AdminYearData:
    year: int
    fips: np.ndarray
    feature_names: list[str]
    x_raw: np.ndarray
    x_norm: torch.Tensor


@dataclass(slots=True)
class SpatialYearPack:
    year: int
    sample_ids: np.ndarray
    admin_x: np.ndarray
    dataset: Dataset
    tile_shape: tuple[int, int, int]
    coords: np.ndarray
    valid_coords: np.ndarray


class SharedEpochPermutationSampler(Sampler[int]):
    def __init__(self, n: int, *, seed: int = 0, shuffle: bool = True):
        self.n = int(n)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        if self.n <= 0:
            return iter([])
        if not self.shuffle:
            return iter(range(self.n))
        rng = np.random.default_rng(self.seed + self.epoch)
        return iter(rng.permutation(self.n).astype(np.int64, copy=False).tolist())

    def __len__(self) -> int:
        return self.n


class YearCyclePolicy:
    def __init__(self, years: Sequence[int], *, random_state: int = 0):
        yrs = np.asarray(sorted({int(y) for y in years}), dtype=np.int64)
        if yrs.size == 0:
            raise ValueError("at least one year is required")
        self.years = yrs
        self.rng = np.random.default_rng(int(random_state))
        self._perm = np.arange(self.years.size, dtype=np.int64)
        self._cursor = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._perm = self.rng.permutation(self.years.size).astype(np.int64, copy=False)
        self._cursor = 0

    def next_year(self) -> int:
        if self._cursor >= self._perm.size:
            self._reshuffle()
        y = int(self.years[self._perm[self._cursor]])
        self._cursor += 1
        return y


class ExpertZipLoader:
    def __init__(self, loaders: Mapping[str, DataLoader]):
        self.expert_ids = list(loaders.keys())
        self.loaders = {k: loaders[k] for k in self.expert_ids}

    def __iter__(self):
        iters = [iter(self.loaders[k]) for k in self.expert_ids]
        for batches in zip(*iters):
            yield {k: b for k, b in zip(self.expert_ids, batches)}

    def __len__(self) -> int:
        return min(len(self.loaders[k]) for k in self.expert_ids)

    def set_epoch(self, epoch: int) -> None:
        for loader in self.loaders.values():
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(int(epoch))


class MMapTileBagDataset(Dataset):
    def __init__(
        self,
        *,
        root_dir: Path,
        modality: str,
        tile_shape: tuple[int, int, int],
        patch_size: int = 32,
        keep_idx: np.ndarray | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.modality = str(modality)
        self.index_csv = self.root_dir / "index.csv"
        self.bin_path = self.root_dir / "dataset.bin"
        self.stats_bin_path = self.root_dir / "stats.bin"
        if not self.index_csv.exists() or not self.bin_path.exists():
            raise FileNotFoundError(f"missing dataset.bin/index.csv in {self.root_dir}")
        self.tile_shape = tuple(int(x) for x in tile_shape)
        self.tile_elems = int(np.prod(np.asarray(self.tile_shape, dtype=np.int64)))
        self.patch_size = int(patch_size)
        self.dtype = np.dtype(np.float32)
        self.stats_dtype = np.dtype(np.float32)
        self.image_bytes_per_tile = self.tile_elems * self.dtype.itemsize
        self.num_patches_per_tile = (self.tile_shape[1] // self.patch_size) ** 2
        self.stats_dim = int(self.tile_shape[0])
        self.stats_elems_per_tile = self.num_patches_per_tile * self.stats_dim
        self.has_stats = bool(self.stats_bin_path.exists())
        self.mmap = None
        self.stats_mmap = None

        fips, labels, offset_bytes, num_tiles = self._load_index()
        if keep_idx is not None:
            keep = np.asarray(keep_idx, dtype=np.int64).reshape(-1)
            fips = fips[keep]
            labels = labels[keep]
            offset_bytes = offset_bytes[keep]
            num_tiles = num_tiles[keep]
        self.fips = fips
        self.labels = labels
        self.offset_bytes = offset_bytes
        self.num_tiles = num_tiles

    def _load_index(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fips: list[str] = []
        labels: list[float] = []
        offsets: list[int] = []
        counts: list[int] = []
        with open(self.index_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fid = str(row.get("fips", "")).strip()
                if fid.isdigit():
                    fid = fid.zfill(5)
                label_raw = row.get("label")
                try:
                    label = float(label_raw) if label_raw not in (None, "") else float("nan")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid label={label_raw!r} in {self.index_csv}") from exc
                byte_offset = int(row["byte_offset"])
                num_tiles = int(row["num_tiles"])
                if byte_offset % self.dtype.itemsize != 0:
                    raise ValueError(f"byte_offset must align to dtype size in {self.index_csv}")
                fips.append(fid)
                labels.append(label)
                offsets.append(byte_offset)
                counts.append(num_tiles)
        return (
            np.asarray(fips, dtype="U5"),
            np.asarray(labels, dtype=np.float32),
            np.asarray(offsets, dtype=np.int64),
            np.asarray(counts, dtype=np.int64),
        )

    def ensure_mmaps(self) -> None:
        if self.mmap is None:
            self.mmap = np.memmap(self.bin_path, mode="r", dtype=self.dtype)
        if self.has_stats and self.stats_mmap is None:
            self.stats_mmap = np.memmap(self.stats_bin_path, mode="r", dtype=self.stats_dtype)

    def close(self) -> None:
        for name in ("mmap", "stats_mmap"):
            mm = getattr(self, name, None)
            if mm is None:
                continue
            try:
                if hasattr(mm, "_mmap") and mm._mmap is not None:
                    mm._mmap.close()
            finally:
                setattr(self, name, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["mmap"] = None
        state["stats_mmap"] = None
        return state

    def __len__(self) -> int:
        return int(self.fips.shape[0])

    def __getitem__(self, idx: int):
        self.ensure_mmaps()
        if self.mmap is None:
            raise ValueError("dataset mmap is not open")
        n_tiles = int(self.num_tiles[int(idx)])
        if n_tiles <= 0:
            tiles = np.zeros((0, *self.tile_shape), dtype=np.float32)
            stats = np.zeros((0, self.num_patches_per_tile, self.stats_dim), dtype=np.float32)
            return tiles, stats, np.float32(np.nan)
        byte_offset = int(self.offset_bytes[int(idx)])
        start_elem = byte_offset // self.dtype.itemsize
        end_elem = start_elem + n_tiles * self.tile_elems
        tiles = np.asarray(self.mmap[start_elem:end_elem], dtype=np.float32).reshape(n_tiles, *self.tile_shape)
        tile_start = byte_offset // self.image_bytes_per_tile
        stats_start = tile_start * self.stats_elems_per_tile
        stats_end = stats_start + n_tiles * self.stats_elems_per_tile
        if self.has_stats:
            if self.stats_mmap is None:
                raise ValueError("stats mmap is not open")
            stats = np.asarray(self.stats_mmap[stats_start:stats_end], dtype=np.float32).reshape(
                n_tiles, self.num_patches_per_tile, self.stats_dim
            )
        else:
            stats = np.zeros((n_tiles, self.num_patches_per_tile, self.stats_dim), dtype=np.float32)
        return tiles, stats, np.float32(np.nan)


class ParquetEmbeddingWriter:
    def __init__(self, *, output_path: Path, modality: str, embed_dim: int, append: bool = False):
        self.output_path = Path(output_path)
        self.modality = str(modality)
        self.embed_dim = int(embed_dim)
        self.append = bool(append)
        self.temp_path: Path | None = None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists() and (not self.append):
            self.output_path.unlink()
        self.schema = pa.schema(
            [
                ("modality", pa.string()),
                ("family_tag_base", pa.string()),
                ("family_tag", pa.string()),
                ("family_start_year", pa.int16()),
                ("family_end_year", pa.int16()),
                ("family_label", pa.string()),
                ("source_year", pa.int16()),
                ("source_split", pa.string()),
                ("source_suffix", pa.string()),
                ("is_eval_year", pa.bool_()),
                ("fips", pa.string()),
                ("item_index", pa.int32()),
                ("item_count", pa.int32()),
                ("embedding", pa.list_(pa.float32(), list_size=self.embed_dim)),
            ]
        )
        if self.output_path.exists() and self.append:
            existing = pq.ParquetFile(str(self.output_path))
            existing_schema = existing.schema_arrow
            if not existing_schema.equals(self.schema, check_metadata=False):
                raise ValueError(f"{self.output_path}: existing parquet schema does not match expected embedding schema")
            self.temp_path = self.output_path.with_suffix(f"{self.output_path.suffix}.tmp")
            if self.temp_path.exists():
                self.temp_path.unlink()
            self.writer = pq.ParquetWriter(str(self.temp_path), self.schema, compression="zstd")
            for row_group_idx in range(int(existing.num_row_groups)):
                self.writer.write_table(existing.read_row_group(row_group_idx))
        else:
            self.writer = pq.ParquetWriter(str(self.output_path), self.schema, compression="zstd")

    def _embedding_array(self, values: np.ndarray) -> pa.Array:
        vals = np.asarray(values, dtype=np.float32)
        flat = pa.array(vals.reshape(-1), type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(flat, list_size=self.embed_dim)

    def write_rows(
        self,
        *,
        family_tag_base: str,
        family_tag: str,
        family_start_year: int,
        family_end_year: int,
        family_label: str,
        source_year: int,
        source_split: str,
        source_suffix: str,
        is_eval_year: bool,
        fips: np.ndarray,
        item_index: np.ndarray,
        item_count: np.ndarray,
        embeddings: np.ndarray,
    ) -> None:
        if int(embeddings.shape[0]) <= 0:
            return
        fips_arr = np.asarray(fips).astype("U5")
        item_index_arr = np.asarray(item_index, dtype=np.int32)
        item_count_arr = np.asarray(item_count, dtype=np.int32)
        emb_arr = np.asarray(embeddings, dtype=np.float32)
        if int(fips_arr.shape[0]) != int(emb_arr.shape[0]):
            raise ValueError("fips/embedding row mismatch")
        if int(item_index_arr.shape[0]) != int(emb_arr.shape[0]) or int(item_count_arr.shape[0]) != int(emb_arr.shape[0]):
            raise ValueError("item index/count row mismatch")
        order = np.lexsort((item_index_arr.astype(np.int64, copy=False), fips_arr))
        fips_arr = fips_arr[order]
        item_index_arr = item_index_arr[order]
        item_count_arr = item_count_arr[order]
        emb_arr = emb_arr[order]
        n = int(emb_arr.shape[0])
        table = pa.Table.from_arrays(
            [
                pa.array([self.modality] * n, type=pa.string()),
                pa.array([str(family_tag_base)] * n, type=pa.string()),
                pa.array([str(family_tag)] * n, type=pa.string()),
                pa.array(np.full(n, int(family_start_year), dtype=np.int16)),
                pa.array(np.full(n, int(family_end_year), dtype=np.int16)),
                pa.array([str(family_label)] * n, type=pa.string()),
                pa.array(np.full(n, int(source_year), dtype=np.int16)),
                pa.array([str(source_split)] * n, type=pa.string()),
                pa.array([str(source_suffix)] * n, type=pa.string()),
                pa.array(np.full(n, bool(is_eval_year), dtype=bool)),
                pa.array(fips_arr.tolist(), type=pa.string()),
                pa.array(item_index_arr),
                pa.array(item_count_arr),
                self._embedding_array(emb_arr),
            ],
            schema=self.schema,
        )
        self.writer.write_table(table)

    def close(self) -> None:
        self.writer.close()
        if self.temp_path is not None:
            self.temp_path.replace(self.output_path)
            self.temp_path = None


def read_embedding_row_counts(output_path: Path, *, modality: str | None = None) -> dict[tuple[str, int], int]:
    path = Path(output_path)
    if not path.exists():
        return {}
    parquet = pq.ParquetFile(str(path))
    schema_names = set(parquet.schema_arrow.names)
    columns = ["family_tag", "source_year"]
    if modality is not None and "modality" in schema_names:
        columns = ["modality", *columns]
    counts: dict[tuple[str, int], int] = {}
    for row_group_idx in range(int(parquet.num_row_groups)):
        table = parquet.read_row_group(row_group_idx, columns=columns)
        frame = table.to_pandas()
        if frame.empty:
            continue
        if modality is not None and "modality" in frame.columns:
            frame = frame.loc[frame["modality"].astype(str) == str(modality), ["family_tag", "source_year"]]
        if frame.empty:
            continue
        grouped = frame.groupby(["family_tag", "source_year"], sort=False).size()
        for (family_tag, source_year), count in grouped.items():
            key = (str(family_tag), int(source_year))
            counts[key] = int(counts.get(key, 0) + int(count))
    return counts


def canon_fips_vec(arr: Any) -> np.ndarray:
    vals = np.asarray(arr).reshape(-1)
    out: list[str] = []
    for v in vals:
        s = str(v).strip().replace("'", "").replace('"', "")
        if s.isdigit():
            s = s.zfill(5)
        out.append(s)
    return np.asarray(out, dtype="U5")


def format_template(template: str, *, year: int) -> str:
    txt = str(template)
    if "{year" in txt:
        return txt.format(year=int(year))
    return txt


def set_seed(seed: int) -> None:
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def autocast_ctx(*, device: torch.device, enabled: bool, dtype: torch.dtype):
    if not bool(enabled):
        return nullcontext()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def dynamic_tile_collate(batch):
    tiles_list, stats_list, labels, num_tiles_list = [], [], [], []
    for item in batch:
        tiles, stats, label = item[:3]
        t = torch.from_numpy(np.asarray(tiles, dtype=np.float32).copy())
        s = torch.from_numpy(np.asarray(stats, dtype=np.float32).copy())
        tiles_list.append(t)
        stats_list.append(s)
        labels.append(label)
        num_tiles_list.append(int(t.shape[0]))
    if not tiles_list:
        raise ValueError("empty spatial batch")
    flat_inputs = torch.cat(tiles_list, dim=0).contiguous(memory_format=torch.channels_last)
    flat_stats = torch.cat(stats_list, dim=0).contiguous() if stats_list else torch.zeros((0, 1, 1), dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    sections = torch.tensor(num_tiles_list, dtype=torch.long)
    batch_idx = torch.repeat_interleave(torch.arange(len(batch)), sections)
    return flat_inputs, labels_t, batch_idx, flat_stats


def load_table(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def load_admin_frame(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = load_table(path)
    if "fips" not in df.columns:
        raise KeyError(f"{path} missing fips column")
    fips = canon_fips_vec(df["fips"].to_numpy())
    feature_names = [c for c in df.columns.tolist() if c not in NONFEATURE_COLUMNS]
    x = np.asarray(df.loc[:, feature_names], dtype=np.float32)
    return x, fips, feature_names


def build_admin_year_data(*, year: int, input_path: Path, mu: np.ndarray, sd: np.ndarray) -> AdminYearData:
    x_raw, fips, feature_names = load_admin_frame(input_path)
    x_norm = ((x_raw - mu) / sd).astype(np.float32, copy=False)
    return AdminYearData(
        year=int(year),
        fips=fips,
        feature_names=feature_names,
        x_raw=x_raw,
        x_norm=torch.from_numpy(x_norm),
    )


def load_county_coords(path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if str(path).lower().endswith(".shp"):
        gdf = gpd.read_file(path)
        if "GEOID" not in gdf.columns:
            raise ValueError(f"{path}: missing GEOID column")
        lat_col = "INTPTLAT" if "INTPTLAT" in gdf.columns else None
        lon_col = "INTPTLONG" if "INTPTLONG" in gdf.columns else None
        if lat_col is None or lon_col is None:
            gdf_proj = gdf.to_crs("EPSG:5070")
            cent_proj = gdf_proj.geometry.centroid
            cent_ll = gpd.GeoSeries(cent_proj, crs="EPSG:5070").to_crs("EPSG:4326")
            lat_vals = np.asarray(cent_ll.y, dtype=np.float64)
            lon_vals = np.asarray(cent_ll.x, dtype=np.float64)
        else:
            lat_vals = np.asarray(gdf[lat_col], dtype=np.float64)
            lon_vals = np.asarray(gdf[lon_col], dtype=np.float64)
        for fid, lat, lon in zip(np.asarray(gdf["GEOID"]).astype("U5"), lat_vals, lon_vals):
            if np.isfinite(lat) and np.isfinite(lon):
                out[str(fid).strip().zfill(5)] = (float(lat), float(lon))
        return out
    rows = np.genfromtxt(str(path), delimiter="\t", names=True, dtype=None, encoding="utf-8")
    if rows.ndim == 0:
        rows = np.asarray([rows], dtype=rows.dtype)
    for r in rows:
        f = str(r["GEOID"]).strip().zfill(5)
        lat = float(r["INTPTLAT"])
        lon = float(r["INTPTLONG"])
        if np.isfinite(lat) and np.isfinite(lon):
            out[f] = (lat, lon)
    return out


def coords_for_sample_ids(sample_ids: np.ndarray, coords_by_fips: Mapping[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    n = int(sample_ids.shape[0])
    coords = np.zeros((n, 2), dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)
    for i, fid in enumerate(np.asarray(sample_ids).astype("U5").tolist()):
        c = coords_by_fips.get(str(fid))
        if c is None:
            continue
        coords[i, 0] = float(c[0])
        coords[i, 1] = float(c[1])
        valid[i] = True
    return coords, valid


def build_spatial_year_pack(
    *,
    year: int,
    admin_input_path: Path,
    tensor_input_path: Path,
    modality: str,
    tile_shape: tuple[int, int, int],
    coords_by_fips: Mapping[str, tuple[float, float]],
) -> SpatialYearPack:
    admin_x, admin_fips, _feature_names = load_admin_frame(admin_input_path)
    admin_index = {fid: i for i, fid in enumerate(admin_fips.tolist())}
    probe_ds = MMapTileBagDataset(
        root_dir=Path(tensor_input_path),
        modality=str(modality),
        tile_shape=tile_shape,
        patch_size=int(tile_shape[1]),
        keep_idx=None,
    )
    # Match the prototype BYOL path: admin defines the year universe and
    # spatial bags are restricted to counties that align on FIPS.
    keep_rows = np.asarray([i for i, fid in enumerate(probe_ds.fips.tolist()) if fid in admin_index], dtype=np.int64)
    probe_ds.close()
    if keep_rows.size == 0:
        raise ValueError(f"no overlapping counties between {admin_input_path} and {tensor_input_path}")
    dataset = MMapTileBagDataset(
        root_dir=Path(tensor_input_path),
        modality=str(modality),
        tile_shape=tile_shape,
        patch_size=int(tile_shape[1]),
        keep_idx=keep_rows,
    )
    sample_ids = canon_fips_vec(dataset.fips)
    admin_rows = np.asarray([admin_index[fid] for fid in sample_ids.tolist()], dtype=np.int64)
    admin_x_aligned = np.asarray(admin_x[admin_rows], dtype=np.float32, copy=False)
    coords, valid = coords_for_sample_ids(sample_ids, coords_by_fips)
    return SpatialYearPack(
        year=int(year),
        sample_ids=sample_ids,
        admin_x=admin_x_aligned,
        dataset=dataset,
        tile_shape=tile_shape,
        coords=coords,
        valid_coords=valid,
    )


def expected_spatial_export_rows(
    *,
    admin_input_path: Path,
    tensor_input_path: Path,
    modality: str,
    tile_shape: tuple[int, int, int],
    max_tiles_per_bag: int,
) -> int:
    _admin_x, admin_fips, _feature_names = load_admin_frame(admin_input_path)
    admin_set = set(admin_fips.tolist())
    probe_ds = MMapTileBagDataset(
        root_dir=Path(tensor_input_path),
        modality=str(modality),
        tile_shape=tile_shape,
        patch_size=int(tile_shape[1]),
        keep_idx=None,
    )
    try:
        keep_mask = np.asarray([fid in admin_set for fid in probe_ds.fips.tolist()], dtype=bool)
        num_tiles = np.asarray(probe_ds.num_tiles[keep_mask], dtype=np.int64).reshape(-1)
    finally:
        probe_ds.close()
    if int(max_tiles_per_bag) > 0 and int(num_tiles.size) > 0:
        num_tiles = np.minimum(num_tiles, int(max_tiles_per_bag))
    num_tiles = np.maximum(num_tiles, 0)
    return int(np.sum(num_tiles, dtype=np.int64))


def build_viirs_radiance_probs(
    *,
    dataset: Dataset,
    weight_mode: str,
    active_threshold: float,
    weight_gamma: float,
    min_weight: float,
    clip_pctl: float,
) -> np.ndarray:
    base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
    if not isinstance(base_ds, MMapTileBagDataset):
        raise TypeError("radiance sampling requires an MMapTileBagDataset")
    n = int(len(base_ds))
    if n <= 0:
        raise ValueError("cannot build radiance weights: empty dataset")
    scores = np.zeros((n,), dtype=np.float64)
    num_tiles = np.asarray(base_ds.num_tiles, dtype=np.int64).reshape(-1)
    if bool(base_ds.has_stats):
        base_ds.ensure_mmaps()
        if base_ds.stats_mmap is None:
            raise RuntimeError("dataset has_stats=True but stats mmap is not available")
        thr = float(active_threshold)
        mode = str(weight_mode).strip().lower()
        for i in range(n):
            n_tiles_i = int(num_tiles[i])
            if n_tiles_i <= 0:
                continue
            tile_start = int(base_ds.offset_bytes[i] // max(1, base_ds.image_bytes_per_tile))
            st = int(tile_start * base_ds.stats_elems_per_tile)
            en = int(st + n_tiles_i * base_ds.stats_elems_per_tile)
            arr = np.asarray(base_ds.stats_mmap[st:en], dtype=np.float32).reshape(
                n_tiles_i, base_ds.num_patches_per_tile, base_ds.stats_dim
            )
            rad = np.asarray(arr[..., 0], dtype=np.float32)
            if mode == "active_count":
                score = float(np.count_nonzero(rad >= thr))
            else:
                score = float(np.maximum(rad - thr, 0.0).sum()) if thr > 0.0 else float(np.maximum(rad, 0.0).sum())
            scores[i] = max(0.0, score)
    else:
        scores = np.asarray(np.maximum(num_tiles, 0), dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.maximum(scores, 0.0)
    cp = float(clip_pctl)
    if 0.0 < cp < 100.0 and int(np.count_nonzero(scores > 0.0)) > 0:
        hi = float(np.percentile(scores, cp))
        if np.isfinite(hi) and hi > 0.0:
            scores = np.minimum(scores, hi)
    gamma = float(max(0.0, weight_gamma))
    if gamma != 1.0:
        scores = np.power(scores, gamma, dtype=np.float64)
    weights = scores + float(max(0.0, min_weight))
    sw = float(np.sum(weights))
    if not np.isfinite(sw) or sw <= 0.0:
        return np.full((n,), 1.0 / float(n), dtype=np.float64)
    return (weights / sw).astype(np.float64, copy=False)


def make_aligned_loaders(
    *,
    pack: SpatialYearPack,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int,
    shared_seed: int,
    subset_idx: np.ndarray | None = None,
) -> ExpertZipLoader:
    admin_x = torch.from_numpy(np.asarray(pack.admin_x, dtype=np.float32))
    node_ids = torch.from_numpy(np.arange(int(pack.sample_ids.shape[0]), dtype=np.int64))
    admin_ds: Dataset = TensorDataset(admin_x, node_ids)
    spatial_ds: Dataset = pack.dataset
    if subset_idx is not None:
        idx_list = np.asarray(subset_idx, dtype=np.int64).reshape(-1).tolist()
        admin_ds = Subset(admin_ds, idx_list)
        spatial_ds = Subset(spatial_ds, idx_list)
    shared_sampler: Sampler[int] | None = None
    if bool(shuffle):
        shared_sampler = SharedEpochPermutationSampler(len(admin_ds), seed=int(shared_seed), shuffle=True)
    common: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": False,
    }
    if shared_sampler is not None:
        common["sampler"] = shared_sampler
    if int(num_workers) > 0:
        common["persistent_workers"] = bool(persistent_workers)
        common["prefetch_factor"] = int(prefetch_factor)
    loaders = {
        "admin": DataLoader(admin_ds, **common),
        str(pack.dataset.modality): DataLoader(spatial_ds, collate_fn=dynamic_tile_collate, **common),
    }
    return ExpertZipLoader(loaders)
