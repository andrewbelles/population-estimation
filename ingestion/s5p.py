#!/usr/bin/env python3
#
# s5p.py  Andrew Belles  Mar 27th, 2026
#
# Sentinel-5P NO2 raw staging and county tensor extraction.
#

import logging
from pathlib import Path

from ingestion.common import parquet_has_rows
from ingestion.config import IngestConfig
from ingestion.raster import build_county_tensor_parquet, discover_source, stage_raster
from ingestion.spatial_bags import bag_root_complete, build_spatial_bag_dataset


LOGGER = logging.getLogger("ingestion.s5p")


def resolve_staged_raster(config: IngestConfig, *, year: int) -> Path:
    source = discover_source(config.s5p.source_dir, config.s5p.source_globs, year=year)
    LOGGER.debug("year=%d source=%s", int(year), source)
    staged = stage_raster(
        source,
        raw_root=config.paths.raw_root,
        subdir=config.s5p.raw_subdir,
        preserve_name=bool(config.s5p.preserve_name),
        stage_compressed=bool(config.s5p.stage_compressed),
        target_name=None,
    )
    return staged


def run(config: IngestConfig, *, skip_existing: bool = False) -> list[Path]:
    if not config.s5p.enabled:
        return []

    outputs = [config.paths.dataset_root / config.s5p.tensor_subdir / f"s5p_county_tensors_{year}.parquet" for year in config.years.values]
    if bool(skip_existing) and outputs and all(parquet_has_rows(path) for path in outputs):
        LOGGER.debug("skip existing s5p tensors=%s", [str(p) for p in outputs])
        return outputs

    outputs = []
    for year in config.years.values:
        staged = resolve_staged_raster(config, year=int(year))
        out_path = config.paths.dataset_root / config.s5p.tensor_subdir / f"s5p_county_tensors_{year}.parquet"
        LOGGER.debug("year=%d staged=%s out=%s", int(year), staged, out_path)
        outputs.append(
            build_county_tensor_parquet(
                raster_path=staged,
                county_shapefile=config.paths.county_shapefile,
                out_path=out_path,
                year=year,
                modality="s5p_no2",
                temp_root=config.paths.temp_root,
            )
        )
    return outputs


def build_bags(config: IngestConfig, *, skip_existing: bool = False) -> list[Path]:
    if not config.s5p.enabled:
        return []
    outputs = [config.paths.dataset_root / config.s5p.bag_subdir / f"s5p_{year}" for year in config.years.values]
    if bool(skip_existing) and outputs and all(bag_root_complete(path, write_stats=bool(config.s5p.write_stats)) for path in outputs):
        LOGGER.debug("skip existing s5p bag roots=%s", [str(p) for p in outputs])
        return outputs
    built: list[Path] = []
    for year in config.years.values:
        staged = resolve_staged_raster(config, year=int(year))
        out_root = config.paths.dataset_root / config.s5p.bag_subdir / f"s5p_{year}"
        built.append(
            build_spatial_bag_dataset(
                raster_path=staged,
                county_shapefile=config.paths.county_shapefile,
                out_root=out_root,
                year=int(year),
                modality="s5p",
                temp_root=config.paths.temp_root,
                tile_size=int(config.s5p.tile_size),
                tile_window_km=float(config.s5p.tile_window_km),
                write_stats=bool(config.s5p.write_stats),
            )
        )
    return built
