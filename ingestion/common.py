#!/usr/bin/env python3
#
# common.py  Andrew Belles  Mar 27th, 2026
#
# Shared filesystem, county lookup, and parquet helpers for ingestion.
#

import gzip
import io
import json
import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator

import geopandas as gpd
import numpy as np
import pandas as pd

import pyarrow  # noqa: F401
import pyarrow.parquet as pq


LOGGER = logging.getLogger("ingestion.common")


STATE_ABBR_BY_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
    "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
    "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY", "72": "PR",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def parquet_has_rows(path: Path) -> bool:
    try:
        if not Path(path).exists():
            return False
        pf = pq.ParquetFile(path)
        return int(pf.metadata.num_rows) > 0
    except Exception as exc:
        LOGGER.debug("invalid parquet path=%s err=%s", path, exc)
        return False


def stage_copy(source: Path, target: Path) -> Path:
    ensure_dir(target.parent)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def materialize_source(target: Path, candidates: list[Path]) -> Path:
    tgt = Path(target).expanduser()
    if tgt.exists():
        return tgt
    tried: list[str] = []
    for cand in candidates:
        src = Path(cand).expanduser()
        tried.append(str(src))
        if src.exists():
            LOGGER.debug("stage source fallback src=%s target=%s", src, tgt)
            return stage_copy(src, tgt)
    raise FileNotFoundError(f"missing source for target={tgt}; candidates={tried}")


def gzip_copy(source: Path, target_gz: Path) -> Path:
    ensure_dir(target_gz.parent)
    with open(source, "rb") as src, gzip.open(target_gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return target_gz


@contextmanager
def materialize_gzip(path: Path, *, temp_root: Path) -> Iterator[Path]:
    suffix = "".join(path.suffixes[:-1]) if path.suffix == ".gz" else path.suffix
    ensure_dir(temp_root)
    with NamedTemporaryFile(suffix=suffix or ".tmp", dir=temp_root, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy2(path, tmp_path)
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


def serialize_array(array: np.ndarray) -> bytes:
    payload = io.BytesIO()
    np.save(payload, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return gzip.compress(payload.getvalue())


def affine_to_json(transform) -> str:
    vals = [float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)]
    return json.dumps(vals)


def load_counties(path: Path) -> gpd.GeoDataFrame:
    counties = gpd.read_file(path)
    need = {"GEOID", "STATEFP", "NAME", "geometry"}
    missing = sorted(need.difference(counties.columns))
    if missing:
        raise ValueError(f"county shapefile missing columns: {missing}")
    counties = counties.loc[:, ["GEOID", "STATEFP", "NAME", "geometry"]].copy()
    counties["fips"] = counties["GEOID"].astype(str).str.zfill(5)
    counties["state_fips"] = counties["STATEFP"].astype(str).str.zfill(2)
    counties["state_abbr"] = counties["state_fips"].map(STATE_ABBR_BY_FIPS).fillna("")
    counties["county_name"] = counties["NAME"].astype(str).str.strip()
    return counties
