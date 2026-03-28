#!/usr/bin/env python3
#
# config.py  Andrew Belles  Mar 27th, 2026
#
# YAML contract loader for the ingestion stage.
#

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class YearRange:
    start: int
    end: int

    @property
    def values(self) -> list[int]:
        if int(self.end) < int(self.start):
            raise ValueError(f"invalid year range: start={self.start} end={self.end}")
        return list(range(int(self.start), int(self.end) + 1))


@dataclass(slots=True)
class PathsConfig:
    county_shapefile: Path
    raw_root: Path
    dataset_root: Path
    metadata_root: Path
    temp_root: Path


@dataclass(slots=True)
class RasterConfig:
    enabled: bool
    source_dir: Path
    source_globs: list[str]
    raw_subdir: str
    tensor_subdir: str
    bag_subdir: str
    tile_size: int
    tile_window_km: float
    write_stats: bool
    canonical_name: str | None = None
    rewrite_source_dir: bool = False
    preserve_name: bool = False
    stage_compressed: bool = True


@dataclass(slots=True)
class USPSConfig:
    enabled: bool
    source_dir: Path
    raw_subdir: str
    zip_glob: str
    gpkg_template: str
    table_path: Path
    tracts_root: Path | None = None
    tracts_year: int = 2023
    tracts_download_url_template: str = "https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{statefp}_tract.zip"


@dataclass(slots=True)
class PEPConfig:
    enabled: bool
    census_2020_csv: Path
    census_2023_csv: Path
    census_2024_csv: Path
    census_2020_truth_csv: Path
    intercensal_state_split_glob: str
    table_path: Path
    pep_2020_anchor: str = "with_resid"


@dataclass(slots=True)
class LAUSConfig:
    enabled: bool
    raw_subdir: str
    table_path: Path
    data_path: Path
    download_url: str
    download_base_url: str
    state_shard_dir: Path
    seasonal_code: str
    area_type_code: str
    urate_code: str
    unemp_code: str
    emp_code: str
    lf_code: str
    chunksize: int


@dataclass(slots=True)
class HousingConfig:
    enabled: bool
    raw_subdir: str
    table_path: Path
    source_mode: str
    inventory_url: str
    hotness_url: str
    inventory_csv: Path
    hotness_csv: Path


@dataclass(slots=True)
class AdminConfig:
    enabled: bool
    merge_path: Path
    yearly_dir: Path | None
    laus: LAUSConfig
    housing: HousingConfig


@dataclass(slots=True)
class IngestConfig:
    years: YearRange
    paths: PathsConfig
    viirs: RasterConfig
    s5p: RasterConfig
    usps: USPSConfig
    pep: PEPConfig
    admin: AdminConfig


def _as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


def _require(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise KeyError(f"missing required config key: {key}")
    return section[key]


def _parse_raster_cfg(section: dict[str, Any]) -> RasterConfig:
    return RasterConfig(
        enabled=bool(section.get("enabled", True)),
        source_dir=_as_path(_require(section, "source_dir")),
        source_globs=[str(x) for x in _require(section, "source_globs")],
        raw_subdir=str(section.get("raw_subdir", "")),
        tensor_subdir=str(section.get("tensor_subdir", "")),
        bag_subdir=str(section.get("bag_subdir", "")),
        tile_size=int(section.get("tile_size", 32)),
        tile_window_km=float(section.get("tile_window_km", 24.0)),
        write_stats=bool(section.get("write_stats", True)),
        canonical_name=None if section.get("canonical_name") in (None, "") else str(section.get("canonical_name")),
        rewrite_source_dir=bool(section.get("rewrite_source_dir", False)),
        preserve_name=bool(section.get("preserve_name", False)),
        stage_compressed=bool(section.get("stage_compressed", True)),
    )


def _parse_usps_cfg(section: dict[str, Any]) -> USPSConfig:
    return USPSConfig(
        enabled=bool(section.get("enabled", True)),
        source_dir=_as_path(_require(section, "source_dir")),
        raw_subdir=str(section.get("raw_subdir", "usps")),
        zip_glob=str(section.get("zip_glob", "*.zip")),
        gpkg_template=str(_require(section, "gpkg_template")),
        table_path=_as_path(_require(section, "table_path")),
        tracts_root=None if section.get("tracts_root") in (None, "") else _as_path(section.get("tracts_root")),
        tracts_year=int(section.get("tracts_year", 2023)),
        tracts_download_url_template=str(
            section.get(
                "tracts_download_url_template",
                "https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{statefp}_tract.zip",
            )
        ),
    )


def _parse_pep_cfg(section: dict[str, Any]) -> PEPConfig:
    return PEPConfig(
        enabled=bool(section.get("enabled", True)),
        census_2020_csv=_as_path(_require(section, "census_2020_csv")),
        census_2023_csv=_as_path(_require(section, "census_2023_csv")),
        census_2024_csv=_as_path(_require(section, "census_2024_csv")),
        census_2020_truth_csv=_as_path(_require(section, "census_2020_truth_csv")),
        intercensal_state_split_glob=str(_require(section, "intercensal_state_split_glob")),
        table_path=_as_path(_require(section, "table_path")),
        pep_2020_anchor=str(section.get("pep_2020_anchor", "with_resid")),
    )


def _parse_laus_cfg(section: dict[str, Any]) -> LAUSConfig:
    return LAUSConfig(
        enabled=bool(section.get("enabled", True)),
        raw_subdir=str(section.get("raw_subdir", "admin/laus")),
        table_path=_as_path(_require(section, "table_path")),
        data_path=_as_path(_require(section, "data_path")),
        download_url=str(_require(section, "download_url")),
        download_base_url=str(_require(section, "download_base_url")),
        state_shard_dir=_as_path(_require(section, "state_shard_dir")),
        seasonal_code=str(section.get("seasonal_code", "U")),
        area_type_code=str(section.get("area_type_code", "CN")),
        urate_code=str(section.get("urate_code", "03")),
        unemp_code=str(section.get("unemp_code", "04")),
        emp_code=str(section.get("emp_code", "05")),
        lf_code=str(section.get("lf_code", "06")),
        chunksize=int(section.get("chunksize", 1_000_000)),
    )


def _parse_housing_cfg(section: dict[str, Any]) -> HousingConfig:
    return HousingConfig(
        enabled=bool(section.get("enabled", True)),
        raw_subdir=str(section.get("raw_subdir", "admin/housing")),
        table_path=_as_path(_require(section, "table_path")),
        source_mode=str(section.get("source_mode", "realtor")),
        inventory_url=str(_require(section, "inventory_url")),
        hotness_url=str(_require(section, "hotness_url")),
        inventory_csv=_as_path(_require(section, "inventory_csv")),
        hotness_csv=_as_path(_require(section, "hotness_csv")),
    )


def _parse_admin_cfg(section: dict[str, Any]) -> AdminConfig:
    return AdminConfig(
        enabled=bool(section.get("enabled", True)),
        merge_path=_as_path(_require(section, "merge_path")),
        yearly_dir=None if section.get("yearly_dir") in (None, "") else _as_path(section.get("yearly_dir")),
        laus=_parse_laus_cfg(dict(_require(section, "laus"))),
        housing=_parse_housing_cfg(dict(_require(section, "housing"))),
    )


def load_config(path: str | Path) -> IngestConfig:
    config_path = _as_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {config_path}")

    years_raw = dict(_require(raw, "years"))
    paths_raw = dict(_require(raw, "paths"))

    years = YearRange(start=int(_require(years_raw, "start")), end=int(_require(years_raw, "end")))
    paths = PathsConfig(
        county_shapefile=_as_path(_require(paths_raw, "county_shapefile")),
        raw_root=_as_path(_require(paths_raw, "raw_root")),
        dataset_root=_as_path(_require(paths_raw, "dataset_root")),
        metadata_root=_as_path(_require(paths_raw, "metadata_root")),
        temp_root=_as_path(paths_raw.get("temp_root", Path("/tmp") / "topographic_ingestion")),
    )

    return IngestConfig(
        years=years,
        paths=paths,
        viirs=_parse_raster_cfg(dict(_require(raw, "viirs"))),
        s5p=_parse_raster_cfg(dict(_require(raw, "s5p"))),
        usps=_parse_usps_cfg(dict(_require(raw, "usps"))),
        pep=_parse_pep_cfg(dict(_require(raw, "pep"))),
        admin=_parse_admin_cfg(dict(_require(raw, "admin"))),
    )
