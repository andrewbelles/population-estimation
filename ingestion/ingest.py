#!/usr/bin/env python3
#
# ingest.py  Andrew Belles  Mar 27th, 2026
#
# Thin YAML-driven orchestrator for the ingestion stage.
#

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion import admin, pep, s5p, usps, viirs
from ingestion.common import ensure_dir, parquet_has_rows, write_parquet
from ingestion.config import IngestConfig, load_config


LOGGER = logging.getLogger("ingestion")
LABEL_COLS = ["label", "label_level", "label_prev", "label_delta"]
LAUS_FEATURES = [
    "laus_urate_mean",
    "laus_urate_std",
    "laus_urate_q4_minus_q1",
    "laus_urate_dec_yoy",
    "laus_emp_dec_yoy",
    "laus_lf_dec_yoy",
    "laus_unemp_to_lf_dec",
]
HOUSING_FEATURES = [
    "housing_active_inv",
    "housing_new_listings",
    "housing_pending",
    "housing_dom",
    "housing_med_list_price",
    "housing_demand_score",
    "housing_price_mom",
    "housing_dom_mom",
    "housing_price_yoy",
    "housing_turnover",
    "housing_absorption",
]


def _merge_year_features(base: pd.DataFrame, ext: pd.DataFrame | None, feat_cols: list[str]) -> pd.DataFrame:
    out = base.copy()
    if ext is not None and not ext.empty:
        keep_cols = [c for c in ["fips", *feat_cols] if c in ext.columns]
        if keep_cols:
            out = out.merge(ext.loc[:, keep_cols].copy(), on="fips", how="left")
    for col in feat_cols:
        if col not in out.columns:
            out[col] = np.nan
        vals = pd.to_numeric(out[col], errors="coerce")
        med = vals.median(skipna=True)
        fill = float(med) if np.isfinite(med) else 0.0
        out[col] = vals.replace([np.inf, -np.inf], np.nan).fillna(fill).astype(np.float64)
    return out


def _label_year_frame(base: pd.DataFrame, pep_sup: pd.DataFrame | None, *, year: int) -> pd.DataFrame:
    out = base.copy()
    for col in LABEL_COLS:
        if col not in out.columns:
            out[col] = np.nan
    if pep_sup is None or pep_sup.empty:
        return out
    sup = pep_sup.loc[pep_sup["year"].astype(int) == int(year), ["fips", "year", "target_correction_log", "label_level", "label_prev", "label_delta"]].copy()
    if sup.empty:
        return out
    sup = sup.rename(columns={"target_correction_log": "label"})
    out = out.merge(sup.drop(columns="year"), on="fips", how="left", suffixes=("", "_sup"))
    for col in LABEL_COLS:
        sup_col = f"{col}_sup"
        if sup_col in out.columns:
            out[col] = pd.to_numeric(out[sup_col], errors="coerce")
            out = out.drop(columns=sup_col)
    return out


def _merge_admin_panel(
    config: IngestConfig,
    *,
    pep_path: Path | None,
    usps_path: Path | None,
    admin_paths: dict[str, Path],
    skip_existing: bool = False,
) -> Path:
    if usps_path is None or (not usps_path.exists()):
        raise RuntimeError("USPS feature panel is required to build the admin scalar dataset")
    if bool(skip_existing) and parquet_has_rows(config.admin.merge_path):
        yearly_ok = True
        if config.admin.yearly_dir is not None:
            yearly_ok = all(
                parquet_has_rows(config.admin.yearly_dir / f"usps_scalar_{int(year)}.parquet")
                for year in config.years.values
            )
        if bool(yearly_ok):
            LOGGER.debug("skip existing admin scalar dataset=%s", config.admin.merge_path)
            return config.admin.merge_path

    usps_panel = pd.read_parquet(usps_path)
    pep_sup = pd.read_parquet(pep_path) if pep_path is not None and pep_path.exists() else None
    laus_panel = pd.read_parquet(admin_paths["laus"]) if "laus" in admin_paths else None
    housing_panel = pd.read_parquet(admin_paths["housing"]) if "housing" in admin_paths else None
    year_frames: list[pd.DataFrame] = []

    for year in config.years.values:
        base = usps_panel.loc[usps_panel["year"].astype(int) == int(year), ["fips", "year", *usps.USPS_FEATURE_COLS]].copy()
        if base.empty:
            continue
        base = _label_year_frame(base, pep_sup, year=int(year))
        laus_year = None if laus_panel is None else laus_panel.loc[laus_panel["year"].astype(int) == int(year)].copy()
        housing_year = None if housing_panel is None else housing_panel.loc[housing_panel["year"].astype(int) == int(year)].copy()
        base = _merge_year_features(base, laus_year, LAUS_FEATURES)
        base = _merge_year_features(base, housing_year, HOUSING_FEATURES)
        base = base.loc[:, ["fips", "year", *LABEL_COLS, *usps.USPS_FEATURE_COLS, *LAUS_FEATURES, *HOUSING_FEATURES]].copy()
        base = base.sort_values("fips").reset_index(drop=True)
        year_frames.append(base)
        if config.admin.yearly_dir is not None:
            yearly = base.drop(columns="year")
            write_parquet(yearly, config.admin.yearly_dir / f"usps_scalar_{int(year)}.parquet")

    if not year_frames:
        raise RuntimeError("no admin year frames were produced")
    frame = pd.concat(year_frames, axis=0, ignore_index=True)
    write_parquet(frame, config.admin.merge_path)
    return config.admin.merge_path


def _write_run_record(config: IngestConfig, *, outputs: dict[str, object]) -> Path:
    run_root = ensure_dir(config.paths.metadata_root / "runs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record_path = run_root / f"ingest_{stamp}.json"
    payload = {
        "created_at_utc": stamp,
        "years": config.years.values,
        "outputs": outputs,
    }
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return record_path


def configure_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).strip().upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"invalid log level: {level_name}")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    handler = logging.StreamHandler(stream=sys.stdout)
    if level <= logging.DEBUG:
        handler.setFormatter(logging.Formatter("[%(levelname)s %(name)s] %(message)s"))
    else:
        handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    root.addHandler(handler)
    for noisy in ("pyogrio", "geopandas", "fiona", "rasterio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _run_stage(idx: int, total: int, label: str, fn, *args, **kwargs):
    LOGGER.info("[%d/%d] %s", int(idx), int(total), str(label))
    result = fn(*args, **kwargs)
    LOGGER.debug("stage complete: %s", str(label))
    return result


def run(config: IngestConfig, *, skip_existing: bool = False) -> dict[str, object]:
    ensure_dir(config.paths.raw_root)
    ensure_dir(config.paths.dataset_root)
    ensure_dir(config.paths.metadata_root)
    ensure_dir(config.paths.temp_root)

    LOGGER.debug("ingestion start years=%s..%s", config.years.start, config.years.end)

    total_stages = 8
    viirs_outputs = _run_stage(1, total_stages, "VIIRS tensors", viirs.run, config, skip_existing=bool(skip_existing))
    LOGGER.debug("viirs tensors=%d outputs=%s", len(viirs_outputs), [str(p) for p in viirs_outputs])
    s5p_outputs = _run_stage(2, total_stages, "S5P tensors", s5p.run, config, skip_existing=bool(skip_existing))
    LOGGER.debug("s5p tensors=%d outputs=%s", len(s5p_outputs), [str(p) for p in s5p_outputs])
    usps_raw, usps_table = _run_stage(3, total_stages, "USPS features", usps.run, config, skip_existing=bool(skip_existing))
    LOGGER.debug("usps staged=%d table=%s", len(usps_raw), usps_table)
    pep_table = _run_stage(4, total_stages, "PEP panel", pep.run, config, skip_existing=bool(skip_existing))
    LOGGER.debug("pep table=%s", pep_table)
    admin_tables = _run_stage(5, total_stages, "Admin features", admin.run, config, skip_existing=bool(skip_existing))
    LOGGER.debug("admin tables=%s", {k: str(v) for k, v in admin_tables.items()})
    merged_panel = _run_stage(
        6,
        total_stages,
        "Admin scalar dataset",
        _merge_admin_panel,
        config,
        pep_path=pep_table,
        usps_path=usps_table,
        admin_paths=admin_tables,
        skip_existing=bool(skip_existing),
    )
    LOGGER.debug("merged admin panel=%s", merged_panel)
    viirs_bags = _run_stage(7, total_stages, "VIIRS tile bags", viirs.build_bags, config, skip_existing=bool(skip_existing))
    LOGGER.debug("viirs bag roots=%d outputs=%s", len(viirs_bags), [str(p) for p in viirs_bags])
    s5p_bags = _run_stage(8, total_stages, "S5P tile bags", s5p.build_bags, config, skip_existing=bool(skip_existing))
    LOGGER.debug("s5p bag roots=%d outputs=%s", len(s5p_bags), [str(p) for p in s5p_bags])

    outputs: dict[str, object] = {
        "viirs_tensors": [str(p) for p in viirs_outputs],
        "s5p_tensors": [str(p) for p in s5p_outputs],
        "viirs_bags": [str(p) for p in viirs_bags],
        "s5p_bags": [str(p) for p in s5p_bags],
        "usps_raw": [str(p) for p in usps_raw],
        "usps_table": None if usps_table is None else str(usps_table),
        "pep_table": None if pep_table is None else str(pep_table),
        "admin_tables": {k: str(v) for k, v in admin_tables.items()},
        "admin_panel": str(merged_panel),
    }
    outputs["run_record"] = str(_write_run_record(config, outputs=outputs))
    LOGGER.debug("ingestion done run_record=%s", outputs["run_record"])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ingestion stage from YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = load_config(args.config)
    run(config, skip_existing=bool(args.skip_existing))


if __name__ == "__main__":
    main()
