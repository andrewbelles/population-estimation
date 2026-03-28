#!/usr/bin/env python3
#
# pep.py  Andrew Belles  Mar 27th, 2026
#
# County-year PEP component assembly for 2020-2024.
#

from __future__ import annotations

import glob
import logging
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.common import load_counties, materialize_source, parquet_has_rows, stage_copy, write_parquet
from ingestion.config import IngestConfig


LOGGER = logging.getLogger("ingestion.pep")


def _county_only(df: pd.DataFrame) -> pd.DataFrame:
    state = df["STATE"].astype(str).str.strip().str.zfill(2)
    county = df["COUNTY"].astype(str).str.strip().str.zfill(3)
    out = df.copy()
    out["fips"] = state + county
    out = out[(out["fips"] != "00000") & (~out["fips"].str.endswith("000"))].copy()
    return out.drop_duplicates(subset=["fips"], keep="first")


def _legacy_pep_candidates(year: int) -> list[Path]:
    if int(year) == 2020:
        return [
            Path("data/census/co-est2020-alldata.csv"),
            Path("data/pep/co-est2020-alldata.csv"),
            Path("data/census/co-est2023-alldata.csv"),
            Path("data/census/co-est2024-alldata.csv"),
        ]
    if int(year) <= 2023:
        return [Path("data/census/co-est2023-alldata.csv"), Path("data/census/co-est2024-alldata.csv")]
    return [Path("data/census/co-est2024-alldata.csv")]


def _truth_2020_candidates() -> list[Path]:
    return [Path("data/census/census2020_county_pl.csv")]


def _intercensal_split_candidates() -> list[str]:
    return [
        "data/census/intercensal/county/co-est2020int-pop-*.xlsx",
    ]


def _pick_source(config: IngestConfig, year: int) -> Path:
    if year == 2020:
        primary = config.pep.census_2020_csv
        fallbacks = [config.pep.census_2023_csv, config.pep.census_2024_csv, *_legacy_pep_candidates(year)]
        return materialize_source(primary, [primary, *fallbacks])
    if year <= 2023:
        primary = config.pep.census_2023_csv
        return materialize_source(primary, [primary, config.pep.census_2024_csv, *_legacy_pep_candidates(year)])
    primary = config.pep.census_2024_csv
    return materialize_source(primary, [primary, *_legacy_pep_candidates(year)])


def _materialize_intercensal_paths(config: IngestConfig) -> list[Path]:
    target_glob = str(config.pep.intercensal_state_split_glob).strip()
    staged = [Path(p).expanduser() for p in sorted(glob.glob(target_glob))]
    if staged:
        return staged
    target_dir = Path(target_glob).expanduser().parent
    for pattern in _intercensal_split_candidates():
        matches = [Path(p).expanduser() for p in sorted(glob.glob(pattern))]
        if not matches:
            continue
        staged = [stage_copy(src, target_dir / src.name) for src in matches]
        LOGGER.debug("staged refined intercensal split files=%d target_dir=%s", len(staged), target_dir)
        return staged
    raise FileNotFoundError(
        "missing refined intercensal county files for 2020 supervision; "
        f"target_glob={target_glob}"
    )


def _normalize_name(text: str) -> str:
    out = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    out = out.lower().strip().replace("&", " and ")
    out = re.sub(r"[^a-z0-9, ]+", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def _county_name_keys(county_name: str, state_name: str) -> list[str]:
    county = _normalize_name(county_name)
    state = _normalize_name(state_name)
    out = [f"{county}, {state}"]
    trimmed = re.sub(
        r"\s+(county|parish|borough|census area|municipality|city and borough|municipio|city)$",
        "",
        county,
    ).strip()
    if trimmed and trimmed != county:
        out.append(f"{trimmed}, {state}")
    return out


def _county_name_to_fips_map(config: IngestConfig) -> dict[str, str]:
    ref_path = materialize_source(
        config.pep.census_2020_csv,
        [config.pep.census_2020_csv, *_legacy_pep_candidates(2020)],
    )
    df = pd.read_csv(ref_path, dtype=str, encoding="latin-1")
    required = {"STATE", "COUNTY", "STNAME", "CTYNAME"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{ref_path}: missing columns required for county name crosswalk: {missing}")
    state = df["STATE"].astype(str).str.strip().str.zfill(2)
    county = df["COUNTY"].astype(str).str.strip().str.zfill(3)
    keep = (state != "00") & (county != "000")
    part = df.loc[keep, ["STNAME", "CTYNAME"]].copy()
    part["state"] = state.loc[keep].to_numpy(dtype="U2")
    part["county"] = county.loc[keep].to_numpy(dtype="U3")
    out: dict[str, str] = {}
    for _, row in part.iterrows():
        fips = f"{str(row['state']).zfill(2)}{str(row['county']).zfill(3)}"
        for key in _county_name_keys(str(row["CTYNAME"]), str(row["STNAME"])):
            out.setdefault(str(key), str(fips))
    return out


def _xlsx_col_to_idx(ref: str) -> int:
    letters = "".join(ch for ch in str(ref) if ch.isalpha()).upper()
    if not letters:
        return -1
    out = 0
    for ch in letters:
        out = (out * 26) + (ord(ch) - ord("A") + 1)
    return int(out - 1)


def _read_xlsx_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path, mode="r") as zf:
        names = set(zf.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sroot.findall(".//{*}si"):
                shared.append("".join((t.text or "") for t in si.findall(".//{*}t")))
        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in names:
            candidates = sorted(n for n in names if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
            if not candidates:
                raise ValueError(f"no worksheet XML found in {path}")
            sheet_name = candidates[0]
        root = ET.fromstring(zf.read(sheet_name))
        out_rows: list[list[str]] = []
        for row in root.findall(".//{*}sheetData/{*}row"):
            cells: dict[int, str] = {}
            max_col = -1
            for cell in row.findall("{*}c"):
                cref = str(cell.attrib.get("r", ""))
                cidx = _xlsx_col_to_idx(cref) if cref else (max_col + 1)
                ctyp = str(cell.attrib.get("t", "")).strip().lower()
                val = ""
                if ctyp == "s":
                    vnode = cell.find("{*}v")
                    if vnode is not None and vnode.text is not None:
                        sidx = int(float(str(vnode.text).strip()))
                        if 0 <= sidx < len(shared):
                            val = shared[sidx]
                elif ctyp == "inlinestr":
                    inode = cell.find("{*}is")
                    if inode is not None:
                        val = "".join((t.text or "") for t in inode.findall(".//{*}t"))
                else:
                    vnode = cell.find("{*}v")
                    if vnode is not None and vnode.text is not None:
                        val = str(vnode.text)
                if cidx < 0:
                    continue
                cells[int(cidx)] = str(val)
                max_col = max(max_col, int(cidx))
            if max_col < 0:
                out_rows.append([])
                continue
            out_rows.append([cells.get(i, "") for i in range(int(max_col + 1))])
    return out_rows


def _load_refined_intercensal_series_from_xlsx(path: Path, *, target_year: int, name_to_fips: dict[str, str]) -> pd.Series:
    rows = _read_xlsx_rows(path)
    geo_idx = None
    for i, row in enumerate(rows[:80]):
        c0 = str(row[0]).strip().lower() if row else ""
        if c0 == "geographic area":
            geo_idx = int(i)
            break
    if geo_idx is None or geo_idx + 1 >= len(rows):
        raise ValueError(f"unable to locate geographic/year headers in {path}")
    year_row = rows[geo_idx + 1]
    year_col = None
    for j, raw in enumerate(year_row):
        txt = str(raw).strip()
        if not txt:
            continue
        try:
            year_val = int(float(txt))
        except Exception:
            continue
        if int(year_val) == int(target_year):
            year_col = int(j)
            break
    if year_col is None:
        raise KeyError(f"{path}: missing refined intercensal column for year={int(target_year)}")

    out: dict[str, float] = {}
    for row in rows[geo_idx + 2:]:
        if not row:
            continue
        area = str(row[0]).strip()
        if not area:
            continue
        low = area.lower()
        if low.startswith("source:") or low.startswith("note:"):
            break
        if not area.startswith(".") or "," not in area:
            continue
        county_name, state_name = [x.strip() for x in area.lstrip(".").rsplit(",", 1)]
        if year_col >= len(row):
            continue
        vtxt = str(row[year_col]).strip().replace(",", "")
        if not vtxt:
            continue
        try:
            pop = float(vtxt)
        except Exception:
            continue
        if (not np.isfinite(pop)) or pop <= 0.0:
            continue
        fips = None
        for key in _county_name_keys(county_name, state_name):
            fips = name_to_fips.get(key)
            if fips is not None:
                break
        if fips is None:
            continue
        out[str(fips)] = float(pop)
    if not out:
        raise ValueError(f"no county rows parsed from refined intercensal file: {path}")
    series = pd.Series(out, dtype=np.float64)
    series.index = series.index.astype(str)
    series = series[~series.index.duplicated(keep="first")]
    return series.sort_index()


def _load_refined_intercensal_series(config: IngestConfig, *, target_year: int) -> pd.Series:
    name_to_fips = _county_name_to_fips_map(config)
    parts = [
        _load_refined_intercensal_series_from_xlsx(path, target_year=int(target_year), name_to_fips=name_to_fips)
        for path in _materialize_intercensal_paths(config)
    ]
    if len(parts) == 1:
        return parts[0]
    out = pd.concat(parts, axis=0)
    out = out[~out.index.duplicated(keep="first")]
    return out.sort_index()


def _load_2020_truth(config: IngestConfig) -> pd.DataFrame:
    source = materialize_source(
        config.pep.census_2020_truth_csv,
        [config.pep.census_2020_truth_csv, *_truth_2020_candidates()],
    )
    df = pd.read_csv(source, dtype=str, encoding="latin-1")
    required = {"fips", "CENSUS2020POP"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{source}: missing required truth columns: {missing}")
    out = pd.DataFrame(
        {
            "fips": df["fips"].astype(str).str.strip().str.zfill(5),
            "label_level": pd.to_numeric(df["CENSUS2020POP"], errors="coerce"),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["fips", "label_level"])
    out = out[out["label_level"] > 0.0].drop_duplicates(subset=["fips"], keep="first")
    return out.reset_index(drop=True)


def _load_2020_anchor(config: IngestConfig) -> pd.DataFrame:
    source = _pick_source(config, 2020)
    LOGGER.debug("year=2020 pep anchor source=%s", source)
    df = _county_only(pd.read_csv(source, encoding="latin-1", dtype=str))
    prev = pd.to_numeric(df.get("POPESTIMATE2019"), errors="coerce")
    births = pd.to_numeric(df.get("BIRTHS2020"), errors="coerce")
    deaths = pd.to_numeric(df.get("DEATHS2020"), errors="coerce")
    dom = pd.to_numeric(df.get("DOMESTICMIG2020"), errors="coerce")
    intl = pd.to_numeric(df.get("INTERNATIONALMIG2020"), errors="coerce")
    with_resid = pd.to_numeric(df.get("POPESTIMATE2020"), errors="coerce")
    no_resid = prev + births - deaths + dom + intl
    anchor_mode = str(config.pep.pep_2020_anchor).strip().lower()
    if anchor_mode == "with_resid":
        anchor_level = with_resid
    elif anchor_mode == "no_resid":
        anchor_level = no_resid
    else:
        raise ValueError(f"unsupported pep_2020_anchor={config.pep.pep_2020_anchor!r}; expected with_resid or no_resid")
    out = pd.DataFrame({"fips": df["fips"].astype(str).str.zfill(5), "anchor_level": anchor_level})
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["fips", "anchor_level"])
    out = out[out["anchor_level"] > 0.0].drop_duplicates(subset=["fips"], keep="first")
    return out.reset_index(drop=True)


def _load_2020_supervision(config: IngestConfig) -> pd.DataFrame:
    truth = _load_2020_truth(config)
    prev = _load_refined_intercensal_series(config, target_year=2019).rename("label_prev").reset_index()
    prev = prev.rename(columns={"index": "fips"})
    anchor = _load_2020_anchor(config)
    out = truth.merge(prev, on="fips", how="inner").merge(anchor, on="fips", how="inner")
    out["year"] = 2020
    out["label_level"] = pd.to_numeric(out["label_level"], errors="coerce")
    out["label_prev"] = pd.to_numeric(out["label_prev"], errors="coerce")
    out["anchor_level"] = pd.to_numeric(out["anchor_level"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["label_level", "label_prev", "anchor_level"])
    out = out[(out["label_level"] > 0.0) & (out["label_prev"] > 0.0) & (out["anchor_level"] > 0.0)].copy()
    out["label"] = np.log(np.asarray(out["label_level"], dtype=np.float64))
    out["label_delta"] = np.asarray(out["label_level"], dtype=np.float64) - np.asarray(out["label_prev"], dtype=np.float64)
    out["target_correction_level"] = np.asarray(out["label_level"], dtype=np.float64) - np.asarray(out["anchor_level"], dtype=np.float64)
    out["target_correction_log"] = out["label"] - np.log(np.asarray(out["anchor_level"], dtype=np.float64))
    cols = [
        "fips",
        "year",
        "label",
        "label_level",
        "label_prev",
        "label_delta",
        "target_correction_log",
        "target_correction_level",
    ]
    return out.loc[:, cols].sort_values("fips").reset_index(drop=True)


def _extract_year_frame(config: IngestConfig, counties: pd.DataFrame, *, year: int) -> pd.DataFrame:
    source = _pick_source(config, year)
    LOGGER.debug("year=%d pep source=%s", int(year), source)
    df = _county_only(pd.read_csv(source, encoding="latin-1", dtype=str))

    pop_col = f"POPESTIMATE{int(year)}"
    prev_col = f"POPESTIMATE{int(year) - 1}" if int(year) > 2020 else "ESTIMATESBASE2020"
    births_col = f"BIRTHS{int(year)}"
    deaths_col = f"DEATHS{int(year)}"
    dom_col = f"DOMESTICMIG{int(year)}"
    intl_col = f"INTERNATIONALMIG{int(year)}"
    resid_col = f"RESIDUAL{int(year)}"

    out = pd.DataFrame(
        {
            "fips": df["fips"].astype(str).str.zfill(5),
            "year": int(year),
            "pep_population": pd.to_numeric(df.get(pop_col), errors="coerce"),
            "pep_population_prev": pd.to_numeric(df.get(prev_col), errors="coerce"),
            "pep_births": pd.to_numeric(df.get(births_col), errors="coerce"),
            "pep_deaths": pd.to_numeric(df.get(deaths_col), errors="coerce"),
            "pep_domestic_migration": pd.to_numeric(df.get(dom_col), errors="coerce"),
            "pep_international_migration": pd.to_numeric(df.get(intl_col), errors="coerce"),
            "pep_residual": pd.to_numeric(df.get(resid_col), errors="coerce"),
        }
    )
    out["pep_net_migration"] = out[["pep_domestic_migration", "pep_international_migration"]].sum(axis=1, min_count=1)
    out = counties.merge(out, on="fips", how="right")
    return out


def run(config: IngestConfig, *, skip_existing: bool = False) -> Path | None:
    if not config.pep.enabled:
        return None
    if bool(skip_existing) and parquet_has_rows(config.pep.table_path):
        LOGGER.debug("skip existing pep table=%s", config.pep.table_path)
        return config.pep.table_path
    counties = load_counties(config.paths.county_shapefile).loc[:, ["fips", "county_name", "state_abbr"]].copy()
    frames = [_extract_year_frame(config, counties, year=year) for year in config.years.values]
    out = pd.concat(frames, axis=0, ignore_index=True)
    sup_2020 = _load_2020_supervision(config)
    if not sup_2020.empty:
        LOGGER.debug("merged strict 2020 supervision rows=%d", int(sup_2020.shape[0]))
        out = out.merge(sup_2020, on=["fips", "year"], how="left")
    write_parquet(out, config.pep.table_path)
    LOGGER.debug("pep panel rows=%d out=%s", int(out.shape[0]), config.pep.table_path)
    return config.pep.table_path
