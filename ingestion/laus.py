#!/usr/bin/env python3
#
# laus.py  Andrew Belles  Mar 27th, 2026
#
# BLS LAUS download and county-year feature construction.
#

import logging
import re
import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ingestion.common import ensure_dir, parquet_has_rows, write_parquet
from ingestion.config import IngestConfig


LOGGER = logging.getLogger("ingestion.laus")


LAUS_REGEX_LEGACY = re.compile(
    r"^LAU(?P<seasonal>[A-Z0-9])(?P<area_type>[A-Z0-9]{2})(?P<state>\d{2})(?P<county>\d{3})(?P<area>\d{6})(?P<measure>\d{2})$"
)
LAUS_REGEX_COUNTY = re.compile(
    r"^LAU(?P<area_type>[A-Z0-9]{2})(?P<state>\d{2})(?P<county>\d{3})(?P<area>\d{7})(?P<measure>\d{3})$"
)


def _download_via_urllib(url: str, out_path: Path) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
        "Accept": "text/plain,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    req = Request(url, headers=headers)
    with urlopen(req, timeout=180) as resp, open(out_path, "wb") as f:
        f.write(resp.read())


def _download_via_curl(url: str, out_path: Path) -> None:
    if shutil.which("curl") is None:
        raise RuntimeError("curl not available")
    cmd = [
        "curl", "-fL", "--retry", "4", "--retry-delay", "2", "--retry-connrefused",
        "-A", "Mozilla/5.0 (X11; Linux x86_64)", url, "-o", str(out_path),
    ]
    import subprocess
    subprocess.run(cmd, check=True)


def _url_candidates(primary: str) -> list[str]:
    p = str(primary).strip()
    out = [p] if p else []
    if p.endswith("la.data.64.County"):
        out.append(p[:-len("la.data.64.County")] + "la.data.64.county")
    if p.endswith("la.data.64.county"):
        out.append(p[:-len("la.data.64.county")] + "la.data.64.County")
    return list(dict.fromkeys(out))


def _download_index_html(base_url: str) -> str:
    req = Request(str(base_url), headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,*/*;q=0.8"})
    with urlopen(req, timeout=120) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore")


def _discover_state_shard_urls(base_url: str) -> list[str]:
    html = _download_index_html(base_url)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    pat = re.compile(r"^la\.data\.(?:[7-9]|[1-5][0-9])\.[A-Za-z0-9]+$")
    urls = [base_url.rstrip("/") + "/" + str(h).strip() for h in hrefs if pat.match(str(h).strip())]
    return sorted(dict.fromkeys(urls))


def _download_state_shards(base_url: str, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    urls = _discover_state_shard_urls(base_url)
    if not urls:
        raise RuntimeError(f"no LAUS state shard links discovered at {base_url}")
    ok = 0
    for url in urls:
        dst = out_dir / str(url).rsplit("/", 1)[-1]
        try:
            _maybe_download(url, dst)
            ok += 1
        except Exception:
            continue
    if ok == 0:
        raise RuntimeError("LAUS state shard fallback failed")
    return out_dir


def _maybe_download(url: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    if out_path.exists():
        return
    errs: list[str] = []
    for candidate in _url_candidates(url):
        try:
            _download_via_urllib(candidate, out_path)
            return
        except HTTPError as exc:
            errs.append(f"{candidate} http={exc.code}")
            if int(exc.code) == 403:
                try:
                    _download_via_curl(candidate, out_path)
                    return
                except Exception as curl_exc:
                    errs.append(f"{candidate} curl={curl_exc}")
        except URLError as exc:
            errs.append(f"{candidate} url={exc}")
            try:
                _download_via_curl(candidate, out_path)
                return
            except Exception as curl_exc:
                errs.append(f"{candidate} curl={curl_exc}")
        except Exception as exc:
            errs.append(f"{candidate} err={exc}")
    raise RuntimeError("; ".join(errs))


def _extract_month(period: pd.Series) -> pd.Series:
    out = period.astype(str).str.extract(r"^M(\d{2})$", expand=False)
    return pd.to_numeric(out, errors="coerce")


def _normalize_measure_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace(r"^0+", "", regex=True)
    return s.replace("", "0")


def _load_monthly(*, laus_data_path: Path | list[Path], years: list[int], seasonal_code: str, area_type_code: str, measure_codes: set[str], chunksize: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    need_cols = {"series_id", "year", "period", "value"}
    if isinstance(laus_data_path, list):
        files = [Path(p) for p in laus_data_path]
    else:
        p = Path(laus_data_path)
        files = sorted(p.glob("la.data.*.*")) if p.is_dir() else [p]
    files = [f for f in files if f.exists() and f.is_file()]
    if not files:
        raise FileNotFoundError("no LAUS source files found")

    allow_measure = {str(int(str(x))) for x in sorted(measure_codes)}
    for src in files:
        for chunk in pd.read_csv(src, sep="\t", dtype=str, chunksize=int(chunksize)):
            chunk.columns = [str(c).strip().lower() for c in chunk.columns]
            if not need_cols.issubset(set(chunk.columns)):
                raise ValueError(f"{src}: expected columns {sorted(need_cols)}")
            year_num = pd.to_numeric(chunk["year"], errors="coerce")
            chunk = chunk.loc[year_num.isin([float(y) for y in years])].copy()
            if chunk.empty:
                continue
            sid = chunk["series_id"].astype(str).str.strip()
            ex_legacy = sid.str.extract(LAUS_REGEX_LEGACY)
            ex_county = sid.str.extract(LAUS_REGEX_COUNTY)
            ex = ex_legacy.copy()
            for col in ("area_type", "state", "county", "area", "measure"):
                ex[col] = ex[col].where(ex[col].notna(), ex_county[col])
            if "seasonal" not in ex.columns:
                ex["seasonal"] = np.nan
            keep = ex["area_type"].eq(str(area_type_code)) & _normalize_measure_code(ex["measure"]).isin(allow_measure)
            if ex["seasonal"].notna().any():
                keep = keep & ex["seasonal"].eq(str(seasonal_code))
            if not keep.any():
                continue
            chunk = chunk.loc[keep].copy()
            ex = ex.loc[keep].copy()
            month = _extract_month(chunk["period"])
            chunk = chunk.loc[month.notna()].copy()
            ex = ex.loc[month.notna()].copy()
            month = month.loc[month.notna()].astype(int)
            if chunk.empty:
                continue
            chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce").astype(int)
            chunk["month"] = month.to_numpy(dtype=np.int64)
            chunk["fips"] = (ex["state"].astype(str) + ex["county"].astype(str)).str.zfill(5).to_numpy(dtype=str)
            chunk["measure_code"] = _normalize_measure_code(ex["measure"]).to_numpy(dtype=str)
            chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
            chunk = chunk.dropna(subset=["value"])
            if not chunk.empty:
                frames.append(chunk[["fips", "year", "month", "measure_code", "value"]].copy())
    if not frames:
        raise ValueError("no LAUS county rows matched requested filters")
    out = pd.concat(frames, axis=0, ignore_index=True)
    return out.drop_duplicates(subset=["fips", "year", "month", "measure_code"], keep="last")


def _series_by_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    norm_code = str(int(str(code)))
    x = df.loc[df["measure_code"].astype(str) == norm_code, ["fips", "year", "month", "value"]].copy()
    return x.rename(columns={"value": f"m_{code}"})


def _month_value_frame(df: pd.DataFrame, *, month: int, col: str) -> pd.DataFrame:
    x = df.loc[df["month"] == int(month), ["fips", "year", col]].copy()
    return x.rename(columns={col: f"{col}_m{int(month):02d}"})


def _yoy(df_dec: pd.DataFrame, *, col_dec: str, out_col: str, pct: bool) -> pd.DataFrame:
    cur = df_dec.rename(columns={col_dec: "_cur"})
    prv = df_dec.rename(columns={"year": "year_prev", col_dec: "_prev"})
    merged = cur.merge(prv, left_on=["fips", "year"], right_on=["fips", "year_prev"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        if pct:
            values = (pd.to_numeric(merged["_cur"], errors="coerce") - pd.to_numeric(merged["_prev"], errors="coerce")) / np.maximum(np.abs(pd.to_numeric(merged["_prev"], errors="coerce")), 1.0)
        else:
            values = pd.to_numeric(merged["_cur"], errors="coerce") - pd.to_numeric(merged["_prev"], errors="coerce")
    out = merged[["fips", "year"]].copy()
    out[out_col] = np.asarray(values, dtype=np.float64)
    return out.replace([np.inf, -np.inf], np.nan)


def _build_features(*, monthly: pd.DataFrame, urate_code: str, unemp_code: str, emp_code: str, lf_code: str) -> pd.DataFrame:
    ur = _series_by_code(monthly, urate_code)
    un = _series_by_code(monthly, unemp_code)
    em = _series_by_code(monthly, emp_code)
    lf = _series_by_code(monthly, lf_code)

    ur_agg = ur.groupby(["fips", "year"], as_index=False).agg(
        laus_urate_mean=(f"m_{urate_code}", "mean"),
        laus_urate_std=(f"m_{urate_code}", "std"),
    )
    ur_q1 = ur.loc[ur["month"].isin([1, 2, 3])].groupby(["fips", "year"], as_index=False).agg(q1=(f"m_{urate_code}", "mean"))
    ur_q4 = ur.loc[ur["month"].isin([10, 11, 12])].groupby(["fips", "year"], as_index=False).agg(q4=(f"m_{urate_code}", "mean"))
    ur_q = ur_q1.merge(ur_q4, on=["fips", "year"], how="outer")
    ur_q["laus_urate_q4_minus_q1"] = pd.to_numeric(ur_q["q4"], errors="coerce") - pd.to_numeric(ur_q["q1"], errors="coerce")
    ur_q = ur_q[["fips", "year", "laus_urate_q4_minus_q1"]]

    ur_dec = _month_value_frame(ur, month=12, col=f"m_{urate_code}")
    em_dec = _month_value_frame(em, month=12, col=f"m_{emp_code}")
    lf_dec = _month_value_frame(lf, month=12, col=f"m_{lf_code}")
    un_dec = _month_value_frame(un, month=12, col=f"m_{unemp_code}")

    ur_yoy = _yoy(ur_dec, col_dec=f"m_{urate_code}_m12", out_col="laus_urate_dec_yoy", pct=False)
    em_yoy = _yoy(em_dec, col_dec=f"m_{emp_code}_m12", out_col="laus_emp_dec_yoy", pct=True)
    lf_yoy = _yoy(lf_dec, col_dec=f"m_{lf_code}_m12", out_col="laus_lf_dec_yoy", pct=True)

    dec_join = un_dec.merge(lf_dec, on=["fips", "year"], how="outer")
    with np.errstate(divide="ignore", invalid="ignore"):
        dec_join["laus_unemp_to_lf_dec"] = pd.to_numeric(dec_join[f"m_{unemp_code}_m12"], errors="coerce") / np.maximum(pd.to_numeric(dec_join[f"m_{lf_code}_m12"], errors="coerce"), 1.0)
    dec_ratio = dec_join[["fips", "year", "laus_unemp_to_lf_dec"]].copy()

    out = ur_agg.merge(ur_q, on=["fips", "year"], how="outer")
    out = out.merge(ur_yoy, on=["fips", "year"], how="outer")
    out = out.merge(em_yoy, on=["fips", "year"], how="outer")
    out = out.merge(lf_yoy, on=["fips", "year"], how="outer")
    out = out.merge(dec_ratio, on=["fips", "year"], how="outer")
    feat_cols = [c for c in out.columns if c.startswith("laus_")]
    for col in feat_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def run(config: IngestConfig, *, skip_existing: bool = False) -> Path | None:
    cfg = config.admin.laus
    if not cfg.enabled:
        return None
    if bool(skip_existing) and parquet_has_rows(cfg.table_path):
        LOGGER.debug("skip existing laus table=%s", cfg.table_path)
        return cfg.table_path
    data_path = cfg.data_path
    if not data_path.exists():
        try:
            _maybe_download(cfg.download_url, data_path)
            laus_input: Path | list[Path] = data_path
            LOGGER.debug("laus source downloaded=%s", data_path)
        except Exception:
            laus_input = _download_state_shards(cfg.download_base_url, cfg.state_shard_dir)
            LOGGER.debug("laus source fallback shards=%s", laus_input)
    else:
        laus_input = data_path
        LOGGER.debug("laus source existing=%s", data_path)
    monthly = _load_monthly(
        laus_data_path=laus_input,
        years=config.years.values,
        seasonal_code=cfg.seasonal_code,
        area_type_code=cfg.area_type_code,
        measure_codes={cfg.urate_code, cfg.unemp_code, cfg.emp_code, cfg.lf_code},
        chunksize=cfg.chunksize,
    )
    features = _build_features(
        monthly=monthly,
        urate_code=cfg.urate_code,
        unemp_code=cfg.unemp_code,
        emp_code=cfg.emp_code,
        lf_code=cfg.lf_code,
    )
    features = features.loc[features["year"].isin(config.years.values)].sort_values(["year", "fips"]).reset_index(drop=True)
    write_parquet(features, cfg.table_path)
    LOGGER.debug("laus features rows=%d out=%s", int(features.shape[0]), cfg.table_path)
    return cfg.table_path
