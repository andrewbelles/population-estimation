#!/usr/bin/env python3
#
# usps.py  Andrew Belles  Mar 27th, 2026
#
# USPS raw staging plus exact county scalar aggregation for the admin pathway.
#

import functools
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ingestion.common import ensure_dir, parquet_has_rows, stage_copy, write_parquet
from ingestion.config import IngestConfig


AREA_CRS = "EPSG:5070"
LOGGER = logging.getLogger("ingestion.usps")
USPS_FEATURE_COLS = [
    "usps_flux_rate",
    "usps_comm_ratio",
    "usps_inst_ratio",
    "usps_coverage_ratio",
    "usps_log_density_land",
    "usps_address_hhi",
    "usps_log_density_iqr",
    "usps_total_res",
    "usps_residency_velocity",
    "usps_b2r_ratio",
    "usps_vacancy_aging_ratio",
    "usps_dormancy_index",
]


def _sum_if_exists(group: pd.DataFrame, cols: list[str]) -> float:
    val = 0.0
    used = False
    for col in cols:
        if col in group.columns:
            used = True
            val += float(pd.to_numeric(group[col], errors="coerce").fillna(0.0).sum())
    return float(val if used else np.nan)


def _zip_candidates_for_year(config: IngestConfig, *, year: int) -> list[Path]:
    roots = [config.usps.source_dir, config.paths.raw_root / config.usps.raw_subdir]
    hits: list[Path] = []
    seen: set[str] = set()
    year_token = str(int(year))
    for root in roots:
        if not Path(root).exists():
            continue
        for path in sorted(Path(root).glob(config.usps.zip_glob)):
            if year_token not in path.name:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            hits.append(path)
    return hits


def _year_assets_available(config: IngestConfig, *, year: int) -> bool:
    gpkg_name = config.usps.gpkg_template.format(year=int(year))
    gpkg_candidates = [
        config.usps.source_dir / gpkg_name,
        config.paths.raw_root / config.usps.raw_subdir / gpkg_name,
    ]
    return any(path.exists() for path in gpkg_candidates) or bool(_zip_candidates_for_year(config, year=int(year)))


def _aggregation_years(config: IngestConfig) -> list[int]:
    years = {int(y) for y in config.years.values}
    if not years:
        return []
    min_year = min(years)
    prev_year = int(min_year) - 1
    if _year_assets_available(config, year=int(prev_year)):
        years.add(int(prev_year))
    return sorted(years)


def _stage_raw_inputs(config: IngestConfig) -> list[Path]:
    raw_dir = ensure_dir(config.paths.raw_root / config.usps.raw_subdir)
    staged: list[Path] = []
    for path in sorted(config.usps.source_dir.glob(config.usps.zip_glob)):
        staged.append(stage_copy(path, raw_dir / path.name))
    for year in _aggregation_years(config):
        gpkg_name = config.usps.gpkg_template.format(year=int(year))
        gpkg_path = config.usps.source_dir / gpkg_name
        if gpkg_path.exists():
            staged.append(stage_copy(gpkg_path, raw_dir / gpkg_path.name))
    LOGGER.debug("staged usps raw files=%d", len(staged))
    return staged


def _effective_tracts_root(config: IngestConfig) -> Path:
    if config.usps.tracts_root is not None:
        return Path(config.usps.tracts_root).expanduser()
    return config.paths.metadata_root / "geography" / "tract_shapefiles"


def _tract_zip_path(root: Path, *, year: int, state_fips: str) -> Path:
    return Path(root) / f"tl_{int(year)}_{str(state_fips).zfill(2)}_tract.zip"


def _tract_shapefile_path(root: Path, *, year: int, state_fips: str) -> Path:
    return Path(root) / f"tl_{int(year)}_{str(state_fips).zfill(2)}_tract.shp"


def _download_tract_zip(config: IngestConfig, *, year: int, state_fips: str, zip_path: Path) -> Path:
    url = str(config.usps.tracts_download_url_template).format(year=int(year), statefp=str(state_fips).zfill(2))
    ensure_dir(zip_path.parent)
    LOGGER.info("download census tracts state=%s year=%d -> %s", str(state_fips).zfill(2), int(year), zip_path)
    with urllib.request.urlopen(url, timeout=120) as src, open(zip_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return zip_path


def _ensure_tract_shapefiles(config: IngestConfig, *, state_fips: list[str]) -> Path:
    root = ensure_dir(_effective_tracts_root(config))
    year = int(config.usps.tracts_year)
    for state in sorted({str(s).zfill(2) for s in state_fips if str(s).strip()}):
        shp_path = _tract_shapefile_path(root, year=year, state_fips=state)
        if shp_path.exists():
            continue
        zip_path = _tract_zip_path(root, year=year, state_fips=state)
        if not zip_path.exists():
            _download_tract_zip(config, year=year, state_fips=state, zip_path=zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        if not shp_path.exists():
            raise FileNotFoundError(f"expected extracted tract shapefile missing after unzip: {shp_path}")
    return root


def _tract_root_covers_states(root: Path, *, year: int, state_fips: list[str]) -> bool:
    need = {str(s).zfill(2) for s in state_fips if str(s).strip()}
    if not need:
        return False
    return all(_tract_shapefile_path(root, year=int(year), state_fips=state).exists() for state in sorted(need))


def _find_tract_files(root: Path) -> list[Path]:
    return sorted(Path(root).rglob("*tract*.shp"))


@functools.lru_cache(maxsize=2)
def _load_tract_master_cached(root_str: str) -> gpd.GeoDataFrame:
    root = Path(root_str).expanduser()
    tract_files = _find_tract_files(root)
    if not tract_files:
        raise FileNotFoundError(f"no tract shapefiles found under {root}")
    tracts = gpd.GeoDataFrame(
        pd.concat([gpd.read_file(path)[["GEOID", "geometry"]] for path in tract_files], ignore_index=True)
    )
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    tracts = tracts.drop_duplicates(subset=["GEOID"], keep="first").reset_index(drop=True)
    return tracts


@functools.lru_cache(maxsize=8)
def _load_tract_master_from_gpkg_cached(path_str: str) -> gpd.GeoDataFrame:
    path = Path(path_str).expanduser()
    gdf = gpd.read_file(path)[["GEOID", "geometry"]].copy()
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(11)
    gdf = gdf.drop_duplicates(subset=["GEOID"], keep="first").reset_index(drop=True)
    return gdf


def _geometry_seed_candidates(config: IngestConfig, *, exclude_year: int | None = None) -> list[Path]:
    roots = [config.paths.raw_root / config.usps.raw_subdir, config.usps.source_dir]
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not Path(root).exists():
            continue
        for path in sorted(Path(root).glob("usps_master_tracts_*.gpkg")):
            if exclude_year is not None and str(int(exclude_year)) in path.stem:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    return out


def _load_tract_master(config: IngestConfig, *, target_year: int, state_fips_required: list[str]) -> gpd.GeoDataFrame:
    root = _effective_tracts_root(config)
    if _tract_root_covers_states(root, year=int(config.usps.tracts_year), state_fips=state_fips_required):
        tract_files = _find_tract_files(root)
        if tract_files:
            return _load_tract_master_cached(str(root))
    seed_candidates = _geometry_seed_candidates(config, exclude_year=int(target_year))
    if not seed_candidates:
        root = _ensure_tract_shapefiles(config, state_fips=state_fips_required)
        tract_files = _find_tract_files(root)
        if tract_files:
            return _load_tract_master_cached(str(root))
    for gpkg_path in _geometry_seed_candidates(config, exclude_year=int(target_year)):
        if gpkg_path.exists():
            LOGGER.debug("use existing usps gpkg geometry seed target_year=%d source=%s", int(target_year), gpkg_path)
            return _load_tract_master_from_gpkg_cached(str(gpkg_path.resolve()))
    if not _tract_root_covers_states(root, year=int(config.usps.tracts_year), state_fips=state_fips_required):
        root = _ensure_tract_shapefiles(config, state_fips=state_fips_required)
        tract_files = _find_tract_files(root)
        if tract_files:
            return _load_tract_master_cached(str(root))
    raise FileNotFoundError(
        "missing tract geometry source for USPS ZIP conversion after tract download and GPKG-seed fallback"
    )


def _load_usps_attrs(path: Path) -> pd.DataFrame:
    gdf = gpd.read_file(path)
    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    df.columns = [str(c).lower() for c in df.columns]
    if "geoid" not in df.columns:
        raise ValueError(f"{path}: missing USPS geoid column")
    df["geoid"] = df["geoid"].astype(str).str.zfill(11)
    df = df.rename(columns={"geoid": "GEOID"})

    required_cols = [
        "GEOID",
        "ams_res",
        "ams_bus",
        "ams_oth",
        "res_vac",
        "bus_vac",
        "oth_vac",
        "nostat_res",
        "nostat_bus",
        "nostat_oth",
    ]
    optional_cols = [
        "vac_3_res",
        "vac_3_6_r",
        "vac_6_12r",
        "vac_12_24r",
        "vac_24_36r",
        "vac_36_res",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing USPS DBF columns {missing}")

    keep_cols = [*required_cols, *[c for c in optional_cols if c in df.columns]]
    df = df.loc[:, keep_cols].copy()
    for col in keep_cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=required_cols[1:], how="all").reset_index(drop=True)


def _compute_usps_channels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_res"] = out["ams_res"].fillna(0.0) + out["res_vac"].fillna(0.0) + out["nostat_res"].fillna(0.0)
    out["total_business"] = out["ams_bus"].fillna(0.0) + out["bus_vac"].fillna(0.0) + out["nostat_bus"].fillna(0.0)
    out["total_other"] = out["ams_oth"].fillna(0.0) + out["oth_vac"].fillna(0.0) + out["nostat_oth"].fillna(0.0)
    out["total_addresses"] = out["total_res"] + out["total_business"] + out["total_other"]
    out = out.loc[out["total_addresses"] > 0.0].copy()

    out["comm_ratio"] = out["total_business"] / np.maximum(out["total_addresses"], 1.0)
    total_nostat = out["nostat_res"].fillna(0.0) + out["nostat_bus"].fillna(0.0) + out["nostat_oth"].fillna(0.0)
    out["flux_rate"] = total_nostat / np.maximum(out["total_addresses"], 1.0)

    short_cols = [c for c in ["vac_3_res"] if c in out.columns]
    long_cols = [c for c in ["vac_3_6_r", "vac_6_12r", "vac_12_24r", "vac_24_36r", "vac_36_res"] if c in out.columns]
    if short_cols:
        vac_short = np.zeros(len(out), dtype=np.float64)
        for col in short_cols:
            vac_short += pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        vac_short = pd.to_numeric(out["res_vac"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if long_cols:
        vac_long = np.zeros(len(out), dtype=np.float64)
        for col in long_cols:
            vac_long += pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        res_vac = pd.to_numeric(out["res_vac"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        vac_long = np.clip(res_vac - vac_short, a_min=0.0, a_max=None)

    out["vac_short_res"] = vac_short
    out["vac_long_res"] = vac_long
    return out


def _extract_usps_dbf(zip_path: Path, *, year: int, extract_root: Path) -> Path:
    out_dir = ensure_dir(extract_root / str(int(year)))
    with zipfile.ZipFile(zip_path, "r") as zf:
        dbf_members = [name for name in zf.namelist() if (not name.endswith("/")) and name.lower().endswith(".dbf")]
        if not dbf_members:
            raise RuntimeError(f"{zip_path}: USPS ZIP has no DBF member")
        member = sorted(dbf_members)[0]
        dbf_path = out_dir / Path(member).name
        if not dbf_path.exists():
            with zf.open(member, "r") as src, open(dbf_path, "wb") as dst:
                dst.write(src.read())
    return dbf_path


def _build_usps_gpkg(config: IngestConfig, dbf_path: Path, *, year: int, out_path: Path) -> Path:
    usps_raw = _load_usps_attrs(dbf_path)
    state_fips = sorted({str(g)[:2] for g in usps_raw["GEOID"].astype(str).tolist() if str(g)})
    tracts = _load_tract_master(config, target_year=int(year), state_fips_required=state_fips)
    usps = _compute_usps_channels(usps_raw)
    merged = tracts.merge(
        usps[
            [
                "GEOID",
                "flux_rate",
                "comm_ratio",
                "total_res",
                "total_other",
                "total_business",
                "total_addresses",
                "res_vac",
                "nostat_res",
                "vac_short_res",
                "vac_long_res",
            ]
        ],
        on="GEOID",
        how="inner",
    )
    ensure_dir(out_path.parent)
    merged.to_file(out_path, driver="GPKG")
    return out_path


def _ensure_gpkg_for_year(config: IngestConfig, *, year: int) -> Path | None:
    raw_dir = ensure_dir(config.paths.raw_root / config.usps.raw_subdir)
    gpkg_path = raw_dir / config.usps.gpkg_template.format(year=int(year))
    if gpkg_path.exists():
        return gpkg_path

    source_gpkg = config.usps.source_dir / config.usps.gpkg_template.format(year=int(year))
    if source_gpkg.exists():
        return stage_copy(source_gpkg, gpkg_path)

    zip_candidates = _zip_candidates_for_year(config, year=int(year))
    if not zip_candidates:
        return None
    zip_path = zip_candidates[0]
    staged_zip = stage_copy(zip_path, raw_dir / zip_path.name)
    dbf_path = _extract_usps_dbf(staged_zip, year=int(year), extract_root=raw_dir / "_extracted")
    LOGGER.debug("build usps gpkg from zip year=%d zip=%s dbf=%s out=%s", int(year), staged_zip, dbf_path, gpkg_path)
    return _build_usps_gpkg(config, dbf_path, year=int(year), out_path=gpkg_path)


def _load_county_area_map(county_shapefile: Path) -> dict[str, float]:
    counties = gpd.read_file(county_shapefile)
    if "GEOID" not in counties.columns or "ALAND" not in counties.columns:
        raise ValueError(f"{county_shapefile}: missing GEOID or ALAND")
    out: dict[str, float] = {}
    for _, row in counties.iterrows():
        out[str(row["GEOID"]).strip().zfill(5)] = float(row["ALAND"]) / 1e6
    return out


def _aggregate_one_year(gpkg_path: Path, county_shapefile: Path, *, year: int) -> pd.DataFrame:
    LOGGER.debug("aggregate usps year=%d source=%s", int(year), gpkg_path)
    gdf = gpd.read_file(gpkg_path)
    gdf.columns = [str(c).lower() for c in gdf.columns]

    required = ["geoid", "total_addresses", "total_business", "total_other", "flux_rate"]
    missing = [col for col in required if col not in gdf.columns]
    if missing:
        raise ValueError(f"{gpkg_path}: missing USPS columns {missing}")

    gdf["geoid"] = gdf["geoid"].astype(str).str.zfill(11)
    gdf["fips"] = gdf["geoid"].str[:5]
    if gdf.crs is None or str(gdf.crs) != AREA_CRS:
        gdf = gdf.to_crs(AREA_CRS)
    gdf["tract_area_sqkm"] = gdf.geometry.area / 1e6
    gdf = gdf.loc[pd.to_numeric(gdf["tract_area_sqkm"], errors="coerce") > 1e-6].copy()

    county_area_map = _load_county_area_map(county_shapefile)

    def agg(group: pd.DataFrame) -> pd.Series:
        fips = str(group.name).strip().zfill(5)
        s_addr = float(pd.to_numeric(group["total_addresses"], errors="coerce").fillna(0.0).sum())
        s_bus = float(pd.to_numeric(group["total_business"], errors="coerce").fillna(0.0).sum())
        s_oth = float(pd.to_numeric(group["total_other"], errors="coerce").fillna(0.0).sum())
        if "total_res" in group.columns:
            s_res = float(pd.to_numeric(group["total_res"], errors="coerce").fillna(0.0).sum())
        else:
            s_res = float(max(s_addr - s_bus - s_oth, 0.0))

        s_covered_area = float(pd.to_numeric(group["tract_area_sqkm"], errors="coerce").fillna(0.0).sum())
        s_land_area = float(county_area_map.get(fips, s_covered_area))
        s_land_area = max(s_land_area, 1e-6)
        s_covered_area = max(s_covered_area, 1e-6)
        s_addr_safe = max(s_addr, 1.0)
        out: dict[str, float] = {}

        if s_addr > 0.0:
            out["usps_flux_rate"] = float(
                np.average(
                    pd.to_numeric(group["flux_rate"], errors="coerce").fillna(0.0),
                    weights=pd.to_numeric(group["total_addresses"], errors="coerce").fillna(0.0),
                )
            )
            out["usps_comm_ratio"] = float(s_bus / s_addr_safe)
            out["usps_inst_ratio"] = float(s_oth / s_addr_safe)
        else:
            out["usps_flux_rate"] = 0.0
            out["usps_comm_ratio"] = 0.0
            out["usps_inst_ratio"] = 0.0

        out["usps_coverage_ratio"] = float(min(s_covered_area / s_land_area, 1.05))
        out["usps_log_density_land"] = float(np.log1p(s_addr_safe / s_land_area))
        shares = pd.to_numeric(group["total_addresses"], errors="coerce").fillna(0.0) / s_addr_safe
        out["usps_address_hhi"] = float(np.square(shares).sum())

        local_dens = pd.to_numeric(group["total_addresses"], errors="coerce").fillna(0.0) / np.maximum(
            pd.to_numeric(group["tract_area_sqkm"], errors="coerce").fillna(np.nan),
            1e-6,
        )
        local_log = np.log1p(local_dens.to_numpy(dtype=np.float64))
        if local_log.shape[0] > 1:
            q75, q25 = np.percentile(local_log, [75, 25])
            out["usps_log_density_iqr"] = float(q75 - q25)
        else:
            out["usps_log_density_iqr"] = 0.0

        s_res_safe = max(s_res, 1.0)
        out["usps_total_res"] = float(s_res)
        out["usps_residency_velocity"] = 0.0
        out["usps_b2r_ratio"] = float(s_bus / s_res_safe)

        s_vac_short_res = _sum_if_exists(group, ["vac_short_res", "vac_3_res"])
        s_vac_long_res = _sum_if_exists(group, ["vac_long_res", "vac_3_6_r", "vac_6_12r", "vac_12_24r", "vac_24_36r", "vac_36_res"])
        if (not np.isfinite(s_vac_short_res)) and ("res_vac" in group.columns):
            s_vac_short_res = float(pd.to_numeric(group["res_vac"], errors="coerce").fillna(0.0).sum())
        if not np.isfinite(s_vac_short_res):
            s_vac_short_res = 0.0
        if not np.isfinite(s_vac_long_res):
            if "res_vac" in group.columns:
                s_res_vac = float(pd.to_numeric(group["res_vac"], errors="coerce").fillna(0.0).sum())
                s_vac_long_res = max(s_res_vac - s_vac_short_res, 0.0)
            else:
                s_vac_long_res = 0.0
        out["usps_vacancy_aging_ratio"] = float(s_vac_short_res / max(s_vac_long_res, 1.0))

        s_nostat_res = _sum_if_exists(group, ["nostat_res"])
        if not np.isfinite(s_nostat_res):
            s_nostat_res = 0.0
        out["usps_dormancy_index"] = float(s_nostat_res / s_res_safe)
        return pd.Series(out)

    df = gdf.groupby("fips", sort=False).apply(agg, include_groups=False).reset_index()
    df["year"] = int(year)
    df = df.loc[:, ["fips", "year", *USPS_FEATURE_COLS]].copy()
    LOGGER.debug("aggregated usps year=%d rows=%d", int(year), int(df.shape[0]))
    return df.sort_values(["year", "fips"]).reset_index(drop=True)


def _apply_residency_velocity(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["year", "fips"]).reset_index(drop=True).copy()
    out["usps_residency_velocity"] = 0.0
    by_year = {int(year): frame.copy() for year, frame in out.groupby("year", sort=True)}
    years_sorted = sorted(by_year)
    prev_maps_from_panel: dict[int, dict[str, float]] = {}
    for prev_year in years_sorted:
        part = by_year[prev_year]
        prev_maps_from_panel[int(prev_year)] = {
            str(f): float(v)
            for f, v in zip(
                np.asarray(part["fips"], dtype="U8").tolist(),
                pd.to_numeric(part["usps_total_res"], errors="coerce").fillna(0.0).tolist(),
            )
        }
    for year in years_sorted:
        prev_year = int(year) - 1
        prev_map = prev_maps_from_panel.get(int(prev_year), {})
        if not prev_map:
            continue
        mask = np.asarray(out["year"], dtype=np.int64) == int(year)
        cur = pd.to_numeric(out.loc[mask, "usps_total_res"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        prev = np.asarray([prev_map.get(str(f), np.nan) for f in out.loc[mask, "fips"].astype(str).tolist()], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            vel = (cur - prev) / np.maximum(prev, 1.0)
        vel[~np.isfinite(vel)] = 0.0
        out.loc[mask, "usps_residency_velocity"] = vel
    return out


def run(config: IngestConfig, *, skip_existing: bool = False) -> tuple[list[Path], Path | None]:
    if not config.usps.enabled:
        return [], None
    if bool(skip_existing) and parquet_has_rows(config.usps.table_path):
        LOGGER.debug("skip existing usps table=%s", config.usps.table_path)
        raw_dir = config.paths.raw_root / config.usps.raw_subdir
        staged = sorted(p for p in raw_dir.iterdir() if p.is_file()) if raw_dir.exists() else []
        return staged, config.usps.table_path

    staged = _stage_raw_inputs(config)
    frames: list[pd.DataFrame] = []
    use_years = _aggregation_years(config)
    for year in use_years:
        gpkg_path = _ensure_gpkg_for_year(config, year=int(year))
        if gpkg_path is not None and gpkg_path.exists():
            frames.append(_aggregate_one_year(gpkg_path, config.paths.county_shapefile, year=int(year)))
    if not frames:
        LOGGER.debug("no usps yearly frames produced")
        return staged, None
    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = _apply_residency_velocity(merged)
    target_years = {int(y) for y in config.years.values}
    merged = merged.loc[merged["year"].astype(int).isin(target_years)].copy()
    merged = merged.sort_values(["year", "fips"]).reset_index(drop=True)
    write_parquet(merged, config.usps.table_path)
    LOGGER.debug("usps merged rows=%d out=%s", int(merged.shape[0]), config.usps.table_path)
    return staged, config.usps.table_path
