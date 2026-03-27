#!/usr/bin/env python3
#
# viirs.py  Andrew Belles  Mar 27th, 2026
#
# VIIRS raw staging and county tensor extraction.
#

from __future__ import annotations

import logging
from pathlib import Path

from ingestion.common import parquet_has_rows
from ingestion.config import IngestConfig
from ingestion.raster import build_county_tensor_parquet, discover_source, normalize_canonical_source, stage_raster
from ingestion.spatial_bags import bag_root_complete, build_spatial_bag_dataset


LOGGER = logging.getLogger("ingestion.viirs")


def resolve_staged_raster(config: IngestConfig, *, year: int) -> Path:
    source = discover_source(config.viirs.source_dir, config.viirs.source_globs, year=year)
    LOGGER.debug("year=%d source=%s", int(year), source)
    canonical = normalize_canonical_source(
        source,
        source_dir=config.viirs.source_dir,
        canonical_name=str(config.viirs.canonical_name),
        year=year,
        rewrite_source_dir=bool(config.viirs.rewrite_source_dir),
    )
    LOGGER.debug("year=%d canonical=%s", int(year), canonical)
    staged = stage_raster(
        canonical,
        raw_root=config.paths.raw_root,
        subdir=config.viirs.raw_subdir,
        preserve_name=False,
        stage_compressed=bool(config.viirs.stage_compressed),
        target_name=canonical.name,
    )
    return staged


def run(config: IngestConfig, *, skip_existing: bool = False) -> list[Path]:
    if not config.viirs.enabled:
        return []

    outputs = [config.paths.dataset_root / config.viirs.tensor_subdir / f"viirs_county_tensors_{year}.parquet" for year in config.years.values]
    if bool(skip_existing) and outputs and all(parquet_has_rows(path) for path in outputs):
        LOGGER.debug("skip existing viirs tensors=%s", [str(p) for p in outputs])
        return outputs

    outputs = []
    for year in config.years.values:
        staged = resolve_staged_raster(config, year=int(year))
        out_path = config.paths.dataset_root / config.viirs.tensor_subdir / f"viirs_county_tensors_{year}.parquet"
        LOGGER.debug("year=%d staged=%s out=%s", int(year), staged, out_path)
        outputs.append(
            build_county_tensor_parquet(
                raster_path=staged,
                county_shapefile=config.paths.county_shapefile,
                out_path=out_path,
                year=year,
                modality="viirs",
                temp_root=config.paths.temp_root,
            )
        )
    return outputs


def build_bags(config: IngestConfig, *, skip_existing: bool = False) -> list[Path]:
    if not config.viirs.enabled:
        return []
    outputs = [config.paths.dataset_root / config.viirs.bag_subdir / f"viirs_{year}" for year in config.years.values]
    if bool(skip_existing) and outputs and all(bag_root_complete(path, write_stats=bool(config.viirs.write_stats)) for path in outputs):
        LOGGER.debug("skip existing viirs bag roots=%s", [str(p) for p in outputs])
        return outputs
    built: list[Path] = []
    for year in config.years.values:
        staged = resolve_staged_raster(config, year=int(year))
        out_root = config.paths.dataset_root / config.viirs.bag_subdir / f"viirs_{year}"
        built.append(
            build_spatial_bag_dataset(
                raster_path=staged,
                county_shapefile=config.paths.county_shapefile,
                out_root=out_root,
                year=int(year),
                modality="viirs",
                temp_root=config.paths.temp_root,
                tile_size=int(config.viirs.tile_size),
                tile_window_km=float(config.viirs.tile_window_km),
                write_stats=bool(config.viirs.write_stats),
            )
        )
    return built
