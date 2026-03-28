#!/usr/bin/env python3
#
# raster.py  Andrew Belles  Mar 27th, 2026
#
# Raster staging and county-tensor parquet extraction.
#

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from shapely.geometry import mapping

from ingestion.common import (
    affine_to_json, 
    ensure_dir, 
    gzip_copy, 
    load_counties, 
    materialize_gzip, 
    serialize_array, 
    stage_copy, 
    write_parquet
)


LOGGER = logging.getLogger("ingestion.raster")


def discover_source(source_dir: Path, patterns: list[str], *, year: int) -> Path:
    for pattern in patterns:
        matches = sorted(source_dir.glob(str(pattern).format(year=int(year))))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no source raster found in {source_dir} for year={year}; patterns={patterns}")


def normalize_canonical_source(
    source: Path,
    *,
    source_dir: Path,
    canonical_name: str,
    year: int,
    rewrite_source_dir: bool,
) -> Path:
    canonical_gz = source_dir / f"{canonical_name.format(year=int(year))}.gz"
    if canonical_gz.exists():
        return canonical_gz

    ensure_dir(canonical_gz.parent)
    if source.suffix == ".gz":
        shutil.copy2(source, canonical_gz)
        if rewrite_source_dir and source.resolve() != canonical_gz.resolve():
            source.unlink(missing_ok=True)
        return canonical_gz

    gzip_copy(source, canonical_gz)
    if rewrite_source_dir and source.resolve() != canonical_gz.resolve():
        source.unlink(missing_ok=True)
    return canonical_gz


def stage_raster(
    source: Path,
    *,
    raw_root: Path,
    subdir: str,
    preserve_name: bool,
    stage_compressed: bool,
    target_name: str | None = None,
) -> Path:
    raw_dir = ensure_dir(raw_root / subdir)
    base_name = target_name if target_name else source.name
    if stage_compressed:
        if not base_name.endswith(".gz"):
            base_name = f"{base_name}.gz"
        target = raw_dir / base_name
        if target.exists():
            return target
        if source.suffix == ".gz":
            return stage_copy(source, target)
        return gzip_copy(source, target)

    target = raw_dir / (source.name if preserve_name else (target_name or source.name))
    if target.exists():
        return target
    return stage_copy(source, target)


def build_county_tensor_parquet(
    *,
    raster_path: Path,
    county_shapefile: Path,
    out_path: Path,
    year: int,
    modality: str,
    temp_root: Path,
) -> Path:
    counties = load_counties(county_shapefile)
    records: list[dict[str, Any]] = []
    LOGGER.debug("extract tensors modality=%s year=%d raster=%s counties=%d", str(modality), int(year), raster_path, int(counties.shape[0]))

    with materialize_gzip(raster_path, temp_root=temp_root) as local_raster:
        with rasterio.open(local_raster) as ds:
            county_view = counties.to_crs(ds.crs) if counties.crs != ds.crs else counties
            nodata = ds.nodata
            for row in county_view.itertuples(index=False):
                geom = mapping(row.geometry)
                try:
                    arr, transform = rasterio.mask.mask(ds, [geom], crop=True, filled=True, nodata=nodata)
                except ValueError:
                    continue
                data = np.asarray(arr, dtype=np.float32)
                if data.size == 0 or data.shape[-1] == 0 or data.shape[-2] == 0:
                    continue
                if nodata is None or not np.isfinite(float(nodata)):
                    valid_pixels = int(np.isfinite(data).sum())
                else:
                    valid_pixels = int(np.count_nonzero(data != float(nodata)))
                if valid_pixels == 0:
                    continue
                left, bottom, right, top = rasterio.transform.array_bounds(data.shape[1], data.shape[2], transform)
                records.append(
                    {
                        "fips": str(row.fips),
                        "year": int(year),
                        "modality": str(modality),
                        "county_name": str(row.county_name),
                        "state_abbr": str(row.state_abbr),
                        "channels": int(data.shape[0]),
                        "height": int(data.shape[1]),
                        "width": int(data.shape[2]),
                        "valid_pixels": int(valid_pixels),
                        "affine_json": affine_to_json(transform),
                        "bounds_left": float(left),
                        "bounds_bottom": float(bottom),
                        "bounds_right": float(right),
                        "bounds_top": float(top),
                        "tensor_codec": "npy.float32.gz",
                        "tensor_blob": serialize_array(data),
                        "source_path": str(raster_path),
                    }
                )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError(f"no county tensors were extracted from {raster_path}")
    LOGGER.debug("tensor parquet rows=%d out=%s", int(df.shape[0]), out_path)
    return write_parquet(df, out_path)
