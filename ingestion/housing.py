#!/usr/bin/env python3
#
# housing.py  Andrew Belles  Mar 27th, 2026
#
# Realtor housing download and county-year feature construction.
#

import logging
import re
import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ingestion.common import STATE_ABBR_BY_FIPS, load_counties, parquet_has_rows, write_parquet
from ingestion.config import IngestConfig


LOGGER = logging.getLogger("ingestion.housing")


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
        "Accept": "text/csv,text/plain,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    req = Request(str(url), headers=headers)
    try:
        with urlopen(req, timeout=180) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        if shutil.which("curl") is None:
            raise RuntimeError(f"failed to download {url}: {exc}") from exc
        cmd = [
            "curl", "-fL", "--retry", "4", "--retry-delay", "2", "--retry-connrefused",
            "-A", "Mozilla/5.0 (X11; Linux x86_64)", str(url), "-o", str(out_path),
        ]
        subprocess.run(cmd, check=True)


def _norm_county_name(value: str) -> str:
    text = re.sub(r"[^\w\s]", " ", str(value).strip().lower())
    for suffix in (" county", " parish", " borough", " census area", " municipality", " city and borough", " city", " municipio"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return re.sub(r"\s+", " ", text).strip()


def _build_county_map(counties_path: Path) -> dict[tuple[str, str], str]:
    counties = load_counties(counties_path)
    out: dict[tuple[str, str], str] = {}
    for row in counties.itertuples(index=False):
        st = STATE_ABBR_BY_FIPS.get(str(row.state_fips).zfill(2), "")
        if st:
            out[(st, _norm_county_name(row.county_name))] = str(row.fips).zfill(5)
    return out


def _pick_col(df: pd.DataFrame, candidates: list[str], *, required: bool) -> str | None:
    low = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        col = low.get(str(candidate).strip().lower())
        if col is not None:
            return col
    if required:
        raise ValueError(f"missing required column; candidates={candidates}")
    return None


def _to_month_start(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    dt = pd.to_datetime(raw, format="%Y%m", errors="coerce")
    miss = dt.isna()
    if bool(miss.any()):
        dt = dt.where(~miss, pd.to_datetime(raw.where(miss), format="%Y-%m", errors="coerce"))
    miss = dt.isna()
    if bool(miss.any()):
        dt = dt.where(~miss, pd.to_datetime(raw.where(miss), errors="coerce"))
    return dt.dt.to_period("M").dt.to_timestamp()


def _finalize_monthly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fips"] = out["fips"].astype(str).str.strip().str.zfill(5)
    out["date"] = _to_month_start(out["date"])
    out = out.dropna(subset=["fips", "date"]).copy()
    out["year"] = out["date"].dt.year.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    return out


def _coerce_metric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load_realtor_monthly(path: Path, counties_path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    date_col = _pick_col(df, ["month_date_yyyymm", "month_date", "date", "month"], required=True)
    fips_col = _pick_col(df, ["county_fips", "fips", "geoid"], required=False)
    if fips_col is not None:
        fips = df[fips_col].astype(str).str.strip().str.zfill(5)
    else:
        county_map = _build_county_map(counties_path)
        state_col = _pick_col(df, ["state_id", "state", "state_code"], required=True)
        county_col = _pick_col(df, ["county_name", "county"], required=True)
        fips = [county_map.get((str(st).strip().upper(), _norm_county_name(ct)), "") for st, ct in zip(df[state_col], df[county_col])]
    out = pd.DataFrame({"fips": fips, "date": df[date_col]})
    metric_map = {
        "active_inventory": ["active_listing_count", "active_inventory"],
        "new_listings": ["new_listing_count", "new_listings"],
        "pending": ["pending_listing_count", "pending"],
        "days_on_market": ["median_days_on_market", "days_on_market"],
        "median_list_price": ["median_listing_price", "median_list_price"],
        "demand_score": ["demand_score", "hotness_score"],
    }
    for out_name, candidates in metric_map.items():
        col = _pick_col(df, candidates, required=False)
        if col is not None:
            out[out_name] = df[col]
    out = _finalize_monthly(out)
    out = out.loc[out["fips"].astype(str).str.len() == 5].copy()
    return _coerce_metric_cols(out, [c for c in metric_map if c in out.columns])


def _load_realtor_hotness_monthly(path: Path, counties_path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    date_col = _pick_col(df, ["month_date_yyyymm", "month_date", "date", "month"], required=True)
    demand_col = _pick_col(df, ["demand_score", "ldp_unique_viewers_per_property_vs_us", "ldp_unique_viewers_per_property", "hotness_score"], required=False)
    if demand_col is None:
        return pd.DataFrame(columns=["fips", "date", "demand_score"])
    fips_col = _pick_col(df, ["county_fips", "fips", "geoid"], required=False)
    if fips_col is not None:
        fips = df[fips_col].astype(str).str.strip().str.zfill(5)
    else:
        county_map = _build_county_map(counties_path)
        state_col = _pick_col(df, ["state_id", "state", "state_code"], required=True)
        county_col = _pick_col(df, ["county_name", "county"], required=True)
        fips = [county_map.get((str(st).strip().upper(), _norm_county_name(ct)), "") for st, ct in zip(df[state_col], df[county_col])]
    out = pd.DataFrame({"fips": fips, "date": df[date_col], "demand_score": df[demand_col]})
    out = _finalize_monthly(out)
    out = out.loc[out["fips"].astype(str).str.len() == 5].copy()
    out["demand_score"] = pd.to_numeric(out["demand_score"], errors="coerce")
    return out.groupby(["fips", "date"], as_index=False).agg(demand_score=("demand_score", "mean"))


def _build_housing_features(monthly: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    x = monthly.loc[monthly["year"].isin(years)].copy()
    grouped = x.groupby(["fips", "year"], as_index=False)
    out = grouped.agg(
        housing_active_inv=("active_inventory", "mean"),
        housing_new_listings=("new_listings", "sum"),
        housing_pending=("pending", "sum"),
        housing_dom=("days_on_market", "mean"),
        housing_med_list_price=("median_list_price", "mean"),
        housing_demand_score=("demand_score", "mean"),
    )
    nov = x.loc[x["month"] == 11, ["fips", "year", "median_list_price", "days_on_market"]].rename(columns={"median_list_price": "price_nov", "days_on_market": "dom_nov"})
    dec = x.loc[x["month"] == 12, ["fips", "year", "median_list_price", "days_on_market"]].rename(columns={"median_list_price": "price_dec", "days_on_market": "dom_dec"})
    price = dec.merge(nov, on=["fips", "year"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        price["housing_price_mom"] = (pd.to_numeric(price["price_dec"], errors="coerce") / np.maximum(pd.to_numeric(price["price_nov"], errors="coerce"), 1.0)) - 1.0
    price["housing_dom_mom"] = pd.to_numeric(price["dom_dec"], errors="coerce") - pd.to_numeric(price["dom_nov"], errors="coerce")
    price_prev = price.rename(columns={"year": "year_prev", "price_dec": "price_prev_dec"})
    yoy = price.merge(price_prev, left_on=["fips", "year"], right_on=["fips", "year_prev"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        yoy["housing_price_yoy"] = (pd.to_numeric(yoy["price_dec"], errors="coerce") / np.maximum(pd.to_numeric(yoy["price_prev_dec"], errors="coerce"), 1.0)) - 1.0
    out = out.merge(price[["fips", "year", "housing_price_mom", "housing_dom_mom"]], on=["fips", "year"], how="left")
    out = out.merge(yoy[["fips", "year", "housing_price_yoy"]], on=["fips", "year"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["housing_turnover"] = (pd.to_numeric(out["housing_new_listings"], errors="coerce") + pd.to_numeric(out["housing_pending"], errors="coerce")) / np.maximum(pd.to_numeric(out["housing_active_inv"], errors="coerce"), 1.0)
        out["housing_absorption"] = pd.to_numeric(out["housing_pending"], errors="coerce") / np.maximum(pd.to_numeric(out["housing_active_inv"], errors="coerce"), 1.0)
    feat_cols = [c for c in out.columns if c.startswith("housing_")]
    for col in feat_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        med = out[col].median(skipna=True)
        fill = float(med) if np.isfinite(med) else 0.0
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(fill)
    return out.sort_values(["year", "fips"]).reset_index(drop=True)


def run(config: IngestConfig, *, skip_existing: bool = False) -> Path | None:
    cfg = config.admin.housing
    if not cfg.enabled:
        return None
    if bool(skip_existing) and parquet_has_rows(cfg.table_path):
        LOGGER.debug("skip existing housing table=%s", cfg.table_path)
        return cfg.table_path
    if str(cfg.source_mode).lower() != "realtor":
        raise ValueError(f"unsupported housing source_mode={cfg.source_mode}; ingestion currently supports realtor")
    _download_file(cfg.inventory_url, cfg.inventory_csv)
    _download_file(cfg.hotness_url, cfg.hotness_csv)
    LOGGER.debug("housing sources inventory=%s hotness=%s", cfg.inventory_csv, cfg.hotness_csv)
    monthly = _load_realtor_monthly(cfg.inventory_csv, config.paths.county_shapefile)
    hotness = _load_realtor_hotness_monthly(cfg.hotness_csv, config.paths.county_shapefile)
    if not hotness.empty:
        monthly = monthly.merge(hotness, on=["fips", "date"], how="left")
    features = _build_housing_features(monthly, config.years.values)
    write_parquet(features, cfg.table_path)
    LOGGER.debug("housing features rows=%d out=%s", int(features.shape[0]), cfg.table_path)
    return cfg.table_path
