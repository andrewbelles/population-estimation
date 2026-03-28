#!/usr/bin/env python3
#
# spatial_bags.py  Andrew Belles  Mar 27th, 2026
#
# Prototype-faithful spatial tile-bag materialization for SSL ingestion.
#

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import rasterio
import rasterio.mask
from rasterio.errors import WindowError
from scipy.ndimage import zoom as ndi_zoom
from shapely.geometry import mapping

from ingestion.common import ensure_dir, load_counties, materialize_gzip


LOGGER = logging.getLogger("ingestion.spatial_bags")

LEGACY_2020_EXCLUDED_FIPS = {
    "60010",
    "60020",
    "60050",
    "66010",
    "69100",
    "69110",
    "69120",
    "78010",
    "78020",
    "78030",
}
# The 2020 labeled prototype path excluded these non-PR territories while
# unlabeled 2021+ tensor builds retained them.


@dataclass(frozen=True, slots=True)
class SpatialBagHandler:
    modality: str
    preprocess: Callable[..., np.ndarray | None]


def bag_root_complete(root: Path, *, write_stats: bool) -> bool:
    root = Path(root)
    if not (root / "dataset.bin").exists():
        return False
    if not (root / "index.csv").exists():
        return False
    if bool(write_stats) and not (root / "stats.bin").exists():
        return False
    return True


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat_r = np.deg2rad(float(lat_deg))
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2.0 * lat_r) + 1.175 * np.cos(4.0 * lat_r) - 0.0023 * np.cos(6.0 * lat_r)
    m_per_deg_lon = 111412.84 * np.cos(lat_r) - 93.5 * np.cos(3.0 * lat_r) + 0.118 * np.cos(5.0 * lat_r)
    return float(m_per_deg_lon), float(m_per_deg_lat)


def meters_per_pixel_from_transform(transform, crs, *, height: int, width: int) -> tuple[float, float]:
    if int(width) <= 0 or int(height) <= 0:
        return 0.0, 0.0
    px_x = float(abs(transform.a))
    px_y = float(abs(transform.e))
    if px_x <= 0.0 or px_y <= 0.0:
        return 0.0, 0.0
    if crs is not None and bool(getattr(crs, "is_geographic", False)):
        lon, lat = rasterio.transform.xy(transform, float(height) * 0.5, float(width) * 0.5, offset="center")
        m_per_deg_lon, m_per_deg_lat = meters_per_degree(float(lat))
        return float(px_x * m_per_deg_lon), float(px_y * m_per_deg_lat)
    return float(px_x), float(px_y)


def resample_2d_for_target_mpp(
    data: np.ndarray,
    valid_mask: np.ndarray,
    *,
    current_x_mpp: float,
    current_y_mpp: float,
    target_x_mpp: float,
    target_y_mpp: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = data.shape
    if h <= 0 or w <= 0:
        return np.asarray(data, dtype=np.float32), np.asarray(valid_mask, dtype=bool)
    if not (
        np.isfinite(current_x_mpp)
        and np.isfinite(current_y_mpp)
        and np.isfinite(target_x_mpp)
        and np.isfinite(target_y_mpp)
    ):
        return np.asarray(data, dtype=np.float32), np.asarray(valid_mask, dtype=bool)
    if current_x_mpp <= 0.0 or current_y_mpp <= 0.0 or target_x_mpp <= 0.0 or target_y_mpp <= 0.0:
        return np.asarray(data, dtype=np.float32), np.asarray(valid_mask, dtype=bool)
    scale_x = float(target_x_mpp / current_x_mpp)
    scale_y = float(target_y_mpp / current_y_mpp)
    out_w = max(1, int(round(float(w) / scale_x)))
    out_h = max(1, int(round(float(h) / scale_y)))
    if out_h == h and out_w == w:
        return np.asarray(data, dtype=np.float32), np.asarray(valid_mask, dtype=bool)
    zoom_y = float(out_h) / float(h)
    zoom_x = float(out_w) / float(w)
    data_rs = ndi_zoom(np.asarray(data, dtype=np.float32), zoom=(zoom_y, zoom_x), order=1)
    mask_rs = ndi_zoom(np.asarray(valid_mask, dtype=np.uint8), zoom=(zoom_y, zoom_x), order=0) > 0
    return np.asarray(data_rs, dtype=np.float32), np.asarray(mask_rs, dtype=bool)


def tight_crop(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        if data.ndim == 3:
            return np.zeros((data.shape[0], 0, 0), dtype=data.dtype)
        return np.zeros((0, 0), dtype=data.dtype)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    if data.ndim == 3:
        return data[:, y0:y1, x0:x1]
    return data[y0:y1, x0:x1]


def iter_tiles(data: np.ndarray, *, tile_hw: tuple[int, int]) -> Iterator[np.ndarray]:
    ht, wt = int(tile_hw[0]), int(tile_hw[1])
    c, h, w = data.shape
    for y0 in range(0, h, ht):
        for x0 in range(0, w, wt):
            y1 = min(y0 + ht, h)
            x1 = min(x0 + wt, w)
            tile = data[:, y0:y1, x0:x1]
            if tile.shape[-2:] != (ht, wt):
                padded = np.zeros((c, ht, wt), dtype=data.dtype)
                padded[:, : tile.shape[1], : tile.shape[2]] = tile
                tile = padded
            if np.any(tile):
                yield tile


def tile_patch_stats(tile: np.ndarray, *, patch_size: int) -> np.ndarray:
    c, h, w = tile.shape
    p = int(patch_size)
    if h % p != 0 or w % p != 0:
        raise ValueError(f"tile shape {(h, w)} must be divisible by patch_size={p}")
    gh, gw = h // p, w // p
    patches = tile.reshape(c, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(gh * gw, c, p * p)
    return np.quantile(patches, 0.95, axis=-1).astype(np.float32, copy=False)


def preprocess_viirs(
    arr,
    *,
    transform,
    crs,
    tile_hw: tuple[int, int],
    tile_window_km: float,
    force_tight_crop: bool,
    nodata: float | None,
) -> np.ndarray | None:
    data = np.asarray(arr, dtype=np.float32)
    valid_mask = ~arr.mask if hasattr(arr, "mask") else np.isfinite(data)
    if nodata is not None:
        valid_mask &= data != float(nodata)
    if not np.any(valid_mask):
        return None
    if float(tile_window_km) > 0.0:
        target_x_mpp = (float(tile_window_km) * 1000.0) / float(tile_hw[1])
        target_y_mpp = (float(tile_window_km) * 1000.0) / float(tile_hw[0])
        cur_x_mpp, cur_y_mpp = meters_per_pixel_from_transform(
            transform,
            crs,
            height=int(data.shape[0]),
            width=int(data.shape[1]),
        )
        data, valid_mask = resample_2d_for_target_mpp(
            data,
            valid_mask,
            current_x_mpp=float(cur_x_mpp),
            current_y_mpp=float(cur_y_mpp),
            target_x_mpp=float(target_x_mpp),
            target_y_mpp=float(target_y_mpp),
        )
    data = np.log1p(np.maximum(data, 0.0)) / 9.0
    tile = data[None, ...]
    tile[:, ~valid_mask] = 0.0
    if bool(force_tight_crop):
        tile = tight_crop(tile, valid_mask)
    return np.asarray(tile, dtype=np.float32)


def preprocess_s5p(
    arr,
    *,
    transform,
    crs,
    tile_hw: tuple[int, int],
    tile_window_km: float,
    force_tight_crop: bool,
    nodata: float | None,
) -> np.ndarray | None:
    data = np.asarray(arr, dtype=np.float32)
    valid_mask = ~arr.mask if hasattr(arr, "mask") else np.isfinite(data)
    if nodata is not None:
        valid_mask &= data != float(nodata)
    valid_mask &= np.isfinite(data)
    if not np.any(valid_mask):
        return None
    if float(tile_window_km) > 0.0:
        target_x_mpp = (float(tile_window_km) * 1000.0) / float(tile_hw[1])
        target_y_mpp = (float(tile_window_km) * 1000.0) / float(tile_hw[0])
        cur_x_mpp, cur_y_mpp = meters_per_pixel_from_transform(
            transform,
            crs,
            height=int(data.shape[0]),
            width=int(data.shape[1]),
        )
        data, valid_mask = resample_2d_for_target_mpp(
            data,
            valid_mask,
            current_x_mpp=float(cur_x_mpp),
            current_y_mpp=float(cur_y_mpp),
            target_x_mpp=float(target_x_mpp),
            target_y_mpp=float(target_y_mpp),
        )
    vals = data[valid_mask]
    lo = float(np.quantile(vals, 0.01))
    hi = float(np.quantile(vals, 0.99))
    den = max(hi - lo, 1e-6)
    data = np.clip((data - lo) / den, 0.0, 1.0)
    tile = data[None, ...]
    tile[:, ~valid_mask] = 0.0
    if bool(force_tight_crop):
        tile = tight_crop(tile, valid_mask)
    return np.asarray(tile, dtype=np.float32)


SPATIAL_BAG_HANDLERS: dict[str, SpatialBagHandler] = {
    "viirs": SpatialBagHandler(modality="viirs", preprocess=preprocess_viirs),
    "s5p": SpatialBagHandler(modality="s5p", preprocess=preprocess_s5p),
}


def require_spatial_bag_handler(modality: str) -> SpatialBagHandler:
    key = str(modality).strip().lower()
    if key not in SPATIAL_BAG_HANDLERS:
        raise ValueError(f"unsupported spatial bag modality={modality!r}; known={sorted(SPATIAL_BAG_HANDLERS)}")
    return SPATIAL_BAG_HANDLERS[key]


def build_spatial_bag_dataset(
    *,
    raster_path: Path,
    county_shapefile: Path,
    out_root: Path,
    year: int,
    modality: str,
    temp_root: Path,
    tile_size: int,
    tile_window_km: float,
    write_stats: bool,
    allowed_fips: set[str] | None = None,
    all_touched: bool = False,
    force_tight_crop: bool = False,
) -> Path:
    handler = require_spatial_bag_handler(str(modality))
    mod_key = str(handler.modality)
    tile_hw = (int(tile_size), int(tile_size))
    tile_shape = (1, int(tile_size), int(tile_size))
    out_root = ensure_dir(Path(out_root))
    bin_path = out_root / "dataset.bin"
    idx_path = out_root / "index.csv"
    stats_path = out_root / "stats.bin"
    meta_path = out_root / "manifest.json"
    if stats_path.exists() and not bool(write_stats):
        stats_path.unlink()

    counties = load_counties(county_shapefile)
    bytes_per_tile = int(np.prod(np.asarray(tile_shape, dtype=np.int64))) * np.dtype(np.float32).itemsize
    bags_written = 0

    with materialize_gzip(raster_path, temp_root=temp_root) as local_raster:
        with rasterio.open(local_raster) as ds:
            county_view = counties.to_crs(ds.crs) if counties.crs != ds.crs else counties
            nodata = None if ds.nodata is None or not np.isfinite(float(ds.nodata)) else float(ds.nodata)
            with open(bin_path, "wb") as bin_f, open(idx_path, "w", newline="", encoding="utf-8") as idx_f:
                stats_f = open(stats_path, "wb") if bool(write_stats) else None
                try:
                    writer = csv.DictWriter(idx_f, fieldnames=["fips", "label", "byte_offset", "num_tiles"])
                    writer.writeheader()
                    byte_offset = 0
                    for row in county_view.itertuples(index=False):
                        fid = str(row.fips).strip().zfill(5)
                        if allowed_fips is not None and fid not in allowed_fips:
                            continue
                        if allowed_fips is None and int(year) <= 2020 and fid in LEGACY_2020_EXCLUDED_FIPS:
                            continue
                        try:
                            arr, out_transform = rasterio.mask.mask(
                                ds,
                                [mapping(row.geometry)],
                                crop=True,
                                filled=False,
                                all_touched=bool(all_touched),
                            )
                        except (ValueError, WindowError):
                            continue
                        if arr.size == 0:
                            continue
                        band0 = arr[0]
                        bag = handler.preprocess(
                            band0,
                            transform=out_transform,
                            crs=ds.crs,
                            tile_hw=tile_hw,
                            tile_window_km=float(tile_window_km),
                            force_tight_crop=bool(force_tight_crop),
                            nodata=nodata,
                        )
                        if bag is None or bag.size == 0:
                            continue
                        n_written = 0
                        for tile in iter_tiles(bag, tile_hw=tile_hw):
                            tile = np.asarray(tile, dtype=np.float32)
                            if not np.any(tile):
                                continue
                            bin_f.write(tile.tobytes(order="C"))
                            if stats_f is not None:
                                stats = tile_patch_stats(tile, patch_size=int(tile_size))
                                stats_f.write(np.asarray(stats, dtype=np.float32).tobytes(order="C"))
                            n_written += 1
                        if n_written <= 0:
                            continue
                        writer.writerow(
                            {
                                "fips": str(fid),
                                "label": float("nan"),
                                "byte_offset": int(byte_offset),
                                "num_tiles": int(n_written),
                            }
                        )
                        byte_offset += int(n_written) * int(bytes_per_tile)
                        bags_written += 1
                finally:
                    if stats_f is not None:
                        stats_f.close()

    payload = {
        "contract": "prototype_tile_bag_v1",
        "modality": str(mod_key),
        "year": int(year),
        "source_raster": str(raster_path),
        "tile_shape": list(tile_shape),
        "tile_window_km": float(tile_window_km),
        "write_stats": bool(write_stats),
        "bags_written": int(bags_written),
        "normalization": "viirs_log1p_div9" if mod_key == "viirs" else "s5p_quantile01_99_clip",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    LOGGER.debug("built %s bag root year=%d bags=%d out=%s", mod_key, int(year), int(bags_written), out_root)
    return out_root
