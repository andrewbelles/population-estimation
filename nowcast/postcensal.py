#!/usr/bin/env python3
#
# postcensal.py  Andrew Belles  Mar 27th, 2026
#
# Roll-forward postcensal nowcast evaluation anchored on strict 2020 truth.
#

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nowcast.common import (
    YearSlice,
    align_basis_to_reference,
    build_state_group_splits,
    compute_community_ari,
    compute_grassmann_sqdist,
    finite_mean,
    fit_capped_huber_delta,
    fit_linear_huber_state,
    fit_predict,
    fit_predict_residual,
    load_modality_block,
    load_pep_year,
    predict_linear_huber_state,
    load_topology_rows,
    mape_pop_pct,
)
from nowcast.config import DownstreamModelConfig, NowcastConfig, load_config


LOGGER = logging.getLogger("nowcast.postcensal")


@dataclass(slots=True)
class GraphReference:
    fips: np.ndarray
    basis: np.ndarray


def setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout)


def compose_feature_matrix(direct_blocks: dict[str, np.ndarray], mem_x: np.ndarray, *, direct_order: list[str], use_mem: bool) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in direct_order:
        if key in direct_blocks:
            parts.append(np.asarray(direct_blocks[key], dtype=np.float64))
    if bool(use_mem) and int(mem_x.shape[1]) > 0:
        parts.append(np.asarray(mem_x, dtype=np.float64))
    if not parts:
        return np.zeros((mem_x.shape[0], 0), dtype=np.float64) if mem_x.ndim == 2 else np.zeros((0, 0), dtype=np.float64)
    return np.concatenate(parts, axis=1)


def build_year_slice(
    *,
    config: NowcastConfig,
    family_year: int,
    source_year: int,
    reference_basis: GraphReference | None,
) -> tuple[YearSlice, GraphReference | None]:
    pep = load_pep_year(config.paths.pep_parquet, year=int(source_year))
    direct_blocks_raw: dict[str, object] = {}
    direct_order: list[str] = []
    for modality in config.evaluation.postcensal_direct_modalities:
        block_cfg = config.block_cfg(modality)
        if not bool(block_cfg.enabled):
            continue
        direct_blocks_raw[str(modality)] = load_modality_block(
            block_cfg,
            family_end_year=int(family_year),
            source_year=int(source_year),
            pool_mode=str(config.evaluation.tile_pool_mode),
        )
        direct_order.append(str(modality))
    mem_rows = None
    current_ref = reference_basis
    if bool(config.evaluation.postcensal_use_mem) and bool(config.graph.enabled):
        mem_rows = load_topology_rows(
            basis_parquet=config.paths.topology_basis_parquet,
            runs_parquet=config.paths.topology_runs_parquet,
            graph_tag_base=config.graph.graph_tag_base,
            graph_kind=config.graph.graph_kind,
            family_end_year=int(family_year),
            source_year=int(source_year),
            top_k=config.graph.mem_top_k,
        )
        mem_x = np.asarray(mem_rows.x, dtype=np.float64)
        if reference_basis is not None and int(source_year) != int(config.anchor_year):
            mem_x, align_corr = align_basis_to_reference(
                ref_fips=np.asarray(reference_basis.fips, dtype="U5"),
                ref_basis=np.asarray(reference_basis.basis, dtype=np.float64),
                cur_fips=np.asarray(mem_rows.fips, dtype="U5"),
                cur_basis=np.asarray(mem_x, dtype=np.float64),
            )
        else:
            align_corr = float("nan")
        mem_rows = dataclass_replace_topology(mem_rows, x=mem_x)
        if int(source_year) == int(config.anchor_year):
            current_ref = GraphReference(fips=np.asarray(mem_rows.fips, dtype="U5"), basis=np.asarray(mem_x, dtype=np.float64))
    else:
        align_corr = float("nan")

    common = set(pep["fips"].astype(str).tolist())
    for block in direct_blocks_raw.values():
        common &= set(np.asarray(block.fips, dtype="U5").tolist())
    if mem_rows is not None:
        common &= set(np.asarray(mem_rows.fips, dtype="U5").tolist())
    sample_ids = np.asarray([f for f in pep["fips"].astype(str).tolist() if f in common], dtype="U5")
    if sample_ids.size <= 1:
        raise ValueError(f"insufficient aligned counties for family={family_year} source={source_year}")
    pep_idx = {str(f): i for i, f in enumerate(pep["fips"].astype(str).tolist())}
    pep_rows = np.asarray([pep_idx[str(f)] for f in sample_ids.tolist()], dtype=np.int64)
    pep_part = pep.iloc[pep_rows].reset_index(drop=True)
    if int(source_year) == int(config.anchor_year):
        keep = np.isfinite(np.asarray(pep_part["y_log"], dtype=np.float64)) & np.isfinite(np.asarray(pep_part["pep_log"], dtype=np.float64))
    else:
        keep = np.isfinite(np.asarray(pep_part["pep_log"], dtype=np.float64))
    if np.count_nonzero(keep) <= 1:
        raise ValueError(f"insufficient finite counties for family={family_year} source={source_year}")
    pep_part = pep_part.loc[keep].reset_index(drop=True)
    sample_ids = canon_series(pep_part["fips"])

    direct_blocks: dict[str, np.ndarray] = {}
    for key, block in direct_blocks_raw.items():
        idx = {str(f): i for i, f in enumerate(np.asarray(block.fips, dtype="U5").tolist())}
        rows = np.asarray([idx[str(f)] for f in sample_ids.tolist()], dtype=np.int64)
        direct_blocks[str(key)] = np.asarray(block.x[rows], dtype=np.float64)
    if mem_rows is not None:
        idx = {str(f): i for i, f in enumerate(np.asarray(mem_rows.fips, dtype="U5").tolist())}
        rows = np.asarray([idx[str(f)] for f in sample_ids.tolist()], dtype=np.int64)
        mem_x = np.asarray(mem_rows.x[rows], dtype=np.float64)
        graph_tag = str(mem_rows.graph_tag)
        graph_kind = str(mem_rows.graph_kind)
        graph_loss = float(mem_rows.graph_loss)
        graph_counties = int(mem_rows.graph_counties)
    else:
        mem_x = np.zeros((sample_ids.shape[0], 0), dtype=np.float64)
        graph_tag = ""
        graph_kind = ""
        graph_loss = float("nan")
        graph_counties = int(sample_ids.shape[0])

    x = compose_feature_matrix(
        direct_blocks,
        mem_x,
        direct_order=direct_order,
        use_mem=bool(config.evaluation.postcensal_use_mem),
    )
    if int(x.shape[1]) <= 0:
        raise ValueError("postcensal feature matrix has zero columns")
    return (
        YearSlice(
            family_year=int(family_year),
            source_year=int(source_year),
            sample_ids=sample_ids,
            states=np.asarray([str(f)[:2] for f in sample_ids.tolist()], dtype="U2"),
            has_truth=bool(int(source_year) == int(config.anchor_year)),
            y_log=np.asarray(pep_part["y_log"], dtype=np.float64),
            y_level=np.asarray(pep_part["y_level"], dtype=np.float64),
            pep_log=np.asarray(pep_part["pep_log"], dtype=np.float64),
            pep_level=np.asarray(pep_part["pep_population"], dtype=np.float64),
            raw_pep_log=np.asarray(pep_part["pep_log"], dtype=np.float64),
            raw_pep_level=np.asarray(pep_part["pep_population"], dtype=np.float64),
            p_t_minus_1=np.asarray(pep_part["p_t_minus_1"], dtype=np.float64),
            births=np.asarray(pep_part["births"], dtype=np.float64),
            deaths=np.asarray(pep_part["deaths"], dtype=np.float64),
            migration=np.asarray(pep_part["migration"], dtype=np.float64),
            residual=np.asarray(pep_part["residual"], dtype=np.float64),
            direct_blocks=direct_blocks,
            mem_x=mem_x,
            x=x,
            graph_tag=graph_tag,
            graph_kind=graph_kind,
            graph_loss=graph_loss,
            graph_counties=graph_counties,
            basis_align_mean_abs_corr=float(align_corr),
        ),
        current_ref,
    )


def canon_series(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.strip().str.zfill(5).to_numpy(dtype="U5")


def dataclass_replace_topology(rows, *, x: np.ndarray):
    return type(rows)(
        fips=np.asarray(rows.fips, dtype="U5"),
        x=np.asarray(x, dtype=np.float64),
        graph_tag=str(rows.graph_tag),
        graph_kind=str(rows.graph_kind),
        graph_loss=float(rows.graph_loss),
        graph_counties=int(rows.graph_counties),
    )


def predict_2020_oof(
    *,
    year_slice: YearSlice,
    model_cfg: DownstreamModelConfig,
    config: NowcastConfig,
) -> tuple[np.ndarray, np.ndarray]:
    splits = build_state_group_splits(
        year_slice.sample_ids,
        n_splits=config.evaluation.n_splits,
        strategy=config.evaluation.fold_strategy,
        region_level=config.evaluation.fold_region_level,
    )
    pred = np.full(year_slice.sample_ids.shape[0], np.nan, dtype=np.float64)
    sigma = np.full(year_slice.sample_ids.shape[0], np.nan, dtype=np.float64)
    resid = np.asarray(year_slice.y_log - year_slice.pep_log, dtype=np.float64)
    full_prediction = bool(config.evaluation.postcensal_full_prediction)
    for split in splits:
        tr_idx = np.asarray(split.train_idx, dtype=np.int64)
        te_idx = np.asarray(split.test_idx, dtype=np.int64)
        target_tr = np.asarray(year_slice.y_log[tr_idx], dtype=np.float64) if full_prediction else np.asarray(resid[tr_idx], dtype=np.float64)
        mu_raw, sig_raw, _ = fit_predict(
            model_cfg=model_cfg,
            Xtr=np.asarray(year_slice.x[tr_idx], dtype=np.float64),
            ytr=target_tr,
            Xte=np.asarray(year_slice.x[te_idx], dtype=np.float64),
            seed=int(config.evaluation.seed) + int(split.fold_id),
        )
        pred[te_idx] = np.asarray(mu_raw, dtype=np.float64) if full_prediction else np.asarray(year_slice.pep_log[te_idx], dtype=np.float64) + np.asarray(mu_raw, dtype=np.float64)
        sigma[te_idx] = np.asarray(sig_raw, dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        raise RuntimeError("strict 2020 OOF prediction left NaNs")
    if not np.all(np.isfinite(sigma)):
        raise RuntimeError("strict 2020 OOF sigma left NaNs")
    return pred, sigma


def rolling_pseudo_target_log(
    *,
    year_slice: YearSlice,
    prev_corr_adjusted_level_by_fips: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    prev_level = np.asarray([prev_corr_adjusted_level_by_fips.get(str(f), np.nan) for f in year_slice.sample_ids.tolist()], dtype=np.float64)
    births = np.asarray(year_slice.births, dtype=np.float64)
    deaths = np.asarray(year_slice.deaths, dtype=np.float64)
    migration = np.asarray(year_slice.migration, dtype=np.float64)
    residual = np.asarray(year_slice.residual, dtype=np.float64)
    pseudo_level = np.full(year_slice.sample_ids.shape[0], np.nan, dtype=np.float64)
    keep = np.isfinite(prev_level) & np.isfinite(births) & np.isfinite(deaths) & np.isfinite(migration) & np.isfinite(residual)
    pseudo_level[keep] = prev_level[keep] + births[keep] - deaths[keep] + migration[keep] + residual[keep]
    pseudo_log = np.full(year_slice.sample_ids.shape[0], np.nan, dtype=np.float64)
    pos = keep & (pseudo_level > 0.0)
    pseudo_log[pos] = np.log(np.clip(pseudo_level[pos], 1e-9, None))
    return pseudo_log, pseudo_level


def predict_nowcast_year(
    *,
    train_slice: YearSlice,
    test_slice: YearSlice,
    model_cfg: DownstreamModelConfig,
    config: NowcastConfig,
    seed: int,
    pseudo_target_log: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not bool(train_slice.has_truth):
        raise ValueError("train_slice must carry labeled 2020 truth")
    Xtr = np.asarray(train_slice.x, dtype=np.float64)
    Xte = np.asarray(test_slice.x, dtype=np.float64)
    full_prediction = bool(config.evaluation.postcensal_full_prediction)
    if full_prediction:
        target_tr = np.asarray(train_slice.y_log, dtype=np.float64)
    else:
        target_tr = np.asarray(train_slice.y_log - train_slice.pep_log, dtype=np.float64)
    if bool(model_cfg.rolling_online_update) and str(model_cfg.model) == "huber":
        if bool(model_cfg.huber_kernelize):
            raise ValueError("rolling_online_update with capped weight drift only supports non-kernelized huber")
        base_state = fit_linear_huber_state(Xtr=Xtr, ytr=target_tr, model_cfg=model_cfg)
        mu = predict_linear_huber_state(state=base_state, Xte=Xte)
        sigma = np.full(Xte.shape[0], float(base_state.sigma_), dtype=np.float64)
        base_pred = np.asarray(mu, dtype=np.float64) if full_prediction else np.asarray(test_slice.pep_log, dtype=np.float64) + np.asarray(mu, dtype=np.float64)
        if pseudo_target_log is None:
            return base_pred, sigma
        pseudo = np.asarray(pseudo_target_log, dtype=np.float64)
        target_anchor = np.asarray(base_pred, dtype=np.float64)
        keep = np.isfinite(pseudo) & np.isfinite(target_anchor) & np.isfinite(Xte).all(axis=1)
        if int(np.count_nonzero(keep)) <= 1:
            return base_pred, sigma
        delta_target = np.asarray(pseudo[keep] - target_anchor[keep], dtype=np.float64)
        delta_mu, delta_sigma = fit_capped_huber_delta(
            base_state=base_state,
            Xcur=np.asarray(Xte[keep], dtype=np.float64),
            delta_target=delta_target,
            model_cfg=model_cfg,
        )
        pred_total = np.asarray(base_pred, dtype=np.float64)
        pred_total[keep] = pred_total[keep] + np.asarray(delta_mu, dtype=np.float64)
        sigma_total = np.asarray(sigma, dtype=np.float64)
        sigma_total[keep] = np.sqrt(np.square(sigma_total[keep]) + float(delta_sigma) ** 2)
        return pred_total, sigma_total
    mu, sigma, _ = fit_predict(model_cfg=model_cfg, Xtr=Xtr, ytr=target_tr, Xte=Xte, seed=int(seed) + 10000 * int(test_slice.source_year))
    base_pred = np.asarray(mu, dtype=np.float64) if full_prediction else np.asarray(test_slice.pep_log, dtype=np.float64) + np.asarray(mu, dtype=np.float64)
    if not bool(model_cfg.rolling_online_update) or pseudo_target_log is None:
        return base_pred, np.asarray(sigma, dtype=np.float64)
    pseudo = np.asarray(pseudo_target_log, dtype=np.float64)
    keep = np.isfinite(pseudo) & np.isfinite(base_pred) & np.isfinite(Xte).all(axis=1)
    if int(np.count_nonzero(keep)) <= 1:
        return base_pred, np.asarray(sigma, dtype=np.float64)
    delta_target = np.asarray(pseudo[keep] - base_pred[keep], dtype=np.float64)
    rolling_cfg = model_cfg if not bool(model_cfg.rolling_online_update) else rolling_cfg_from(model_cfg)
    delta_mu, delta_sigma = fit_predict_residual(
        model_cfg=rolling_cfg,
        Xtr=np.asarray(Xte[keep], dtype=np.float64),
        resid_tr=delta_target,
        Xte=Xte,
        seed=int(seed) + 10000 * int(test_slice.source_year) + 1,
    )
    pred_total = np.asarray(base_pred, dtype=np.float64) + np.asarray(delta_mu, dtype=np.float64)
    sigma_total = np.sqrt(np.square(np.asarray(sigma, dtype=np.float64)) + np.square(np.asarray(delta_sigma, dtype=np.float64)))
    return pred_total, sigma_total


def rolling_cfg_from(model_cfg: DownstreamModelConfig) -> DownstreamModelConfig:
    from nowcast.common import rolling_regularized_model_cfg

    return rolling_regularized_model_cfg(model_cfg)


def build_trajectory_rows(
    *,
    year_slice: YearSlice,
    pred_log: np.ndarray,
    pred_sigma_log: np.ndarray,
    fit_mode: str,
) -> pd.DataFrame:
    pred_level = np.exp(np.asarray(pred_log, dtype=np.float64))
    true_level = np.asarray(year_slice.y_level, dtype=np.float64)
    pep_level = np.asarray(year_slice.pep_level, dtype=np.float64)
    has_truth = np.isfinite(true_level)
    pep_ape = np.full(true_level.shape[0], np.nan, dtype=np.float64)
    pred_ape = np.full(true_level.shape[0], np.nan, dtype=np.float64)
    if np.any(has_truth):
        pep_ape[has_truth] = np.abs(pep_level[has_truth] - true_level[has_truth]) / np.clip(np.abs(true_level[has_truth]), 1e-9, None) * 100.0
        pred_ape[has_truth] = np.abs(pred_level[has_truth] - true_level[has_truth]) / np.clip(np.abs(true_level[has_truth]), 1e-9, None) * 100.0
    return pd.DataFrame(
        {
            "family_year": int(year_slice.family_year),
            "year": int(year_slice.source_year),
            "fips": np.asarray(year_slice.sample_ids, dtype="U5"),
            "state": np.asarray([str(f)[:2] for f in year_slice.sample_ids.tolist()], dtype="U2"),
            "fit_mode": str(fit_mode),
            "has_truth": bool(year_slice.has_truth),
            "graph_tag": str(year_slice.graph_tag),
            "graph_kind": str(year_slice.graph_kind),
            "graph_counties": int(year_slice.graph_counties),
            "graph_train_loss": float(year_slice.graph_loss),
            "basis_align_mean_abs_corr": float(year_slice.basis_align_mean_abs_corr),
            "truth_log": np.asarray(year_slice.y_log, dtype=np.float64),
            "truth_level": true_level,
            "raw_pep_log": np.asarray(year_slice.raw_pep_log, dtype=np.float64),
            "raw_pep_level": np.asarray(year_slice.raw_pep_level, dtype=np.float64),
            "p_t_minus_1": np.asarray(year_slice.p_t_minus_1, dtype=np.float64),
            "births": np.asarray(year_slice.births, dtype=np.float64),
            "deaths": np.asarray(year_slice.deaths, dtype=np.float64),
            "migration": np.asarray(year_slice.migration, dtype=np.float64),
            "residual_component": np.asarray(year_slice.residual, dtype=np.float64),
            "pep_log": np.asarray(year_slice.pep_log, dtype=np.float64),
            "pep_level": pep_level,
            "corrected_log": np.asarray(pred_log, dtype=np.float64),
            "corrected_sigma_log": np.asarray(pred_sigma_log, dtype=np.float64),
            "corrected_level": pred_level,
            "resid_true": np.asarray(year_slice.y_log - year_slice.pep_log, dtype=np.float64),
            "resid_pred": np.asarray(pred_log - year_slice.pep_log, dtype=np.float64),
            "pep_ape_pct": pep_ape,
            "corrected_ape_pct": pred_ape,
            "delta_ape_pct": pred_ape - pep_ape,
        }
    )


def summarize_year_metrics(traj: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, part in traj.groupby("year", sort=True):
        pep_mape = float(np.nanmean(np.asarray(part["pep_ape_pct"], dtype=np.float64))) if np.any(np.isfinite(np.asarray(part["pep_ape_pct"], dtype=np.float64))) else float("nan")
        corrected_mape = float(np.nanmean(np.asarray(part["corrected_ape_pct"], dtype=np.float64))) if np.any(np.isfinite(np.asarray(part["corrected_ape_pct"], dtype=np.float64))) else float("nan")
        rows.append(
            {
                "year": int(year),
                "fit_mode": str(part["fit_mode"].iloc[0]),
                "has_truth": bool(np.asarray(part["has_truth"], dtype=bool).any()),
                "graph_tag": str(part["graph_tag"].iloc[0]),
                "graph_kind": str(part["graph_kind"].iloc[0]),
                "n_counties": int(part.shape[0]),
                "pep_mape_pct": pep_mape,
                "corrected_mape_pct": corrected_mape,
                "delta_mape_pct": corrected_mape - pep_mape,
                "graph_train_loss": finite_mean(np.asarray(part["graph_train_loss"], dtype=np.float64)),
                "basis_align_mean_abs_corr": finite_mean(np.asarray(part["basis_align_mean_abs_corr"], dtype=np.float64)),
            }
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def summarize_counties(traj: pd.DataFrame) -> pd.DataFrame:
    return (
        traj.groupby("fips", as_index=False)
        .agg(
            state=("state", "first"),
            n_years=("year", "size"),
            n_labeled_years=("has_truth", "sum"),
            pep_ape_pct_mean=("pep_ape_pct", "mean"),
            corrected_ape_pct_mean=("corrected_ape_pct", "mean"),
            delta_ape_pct_mean=("delta_ape_pct", "mean"),
            pep_ape_pct_max=("pep_ape_pct", "max"),
            corrected_ape_pct_max=("corrected_ape_pct", "max"),
            delta_ape_pct_max=("delta_ape_pct", "max"),
        )
        .sort_values(["delta_ape_pct_mean", "fips"], ascending=[True, True])
        .reset_index(drop=True)
    )


def write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run_postcensal(config: NowcastConfig, *, model_key: str | None = None) -> dict[str, object]:
    model_cfg = config.downstream.model_cfg(model_key)
    all_rows: list[pd.DataFrame] = []
    year_meta: list[dict[str, object]] = []
    prev_corr_adjusted_level_by_fips: dict[str, float] = {}
    strict_ref_basis: GraphReference | None = None
    seed = int(config.evaluation.seed)

    for family_year in config.years.values:
        LOGGER.info("family_year=%d years=%d..%d", int(family_year), int(config.anchor_year), int(family_year))
        slices: dict[int, YearSlice] = {}
        current_ref_basis: GraphReference | None = strict_ref_basis
        current_artifact_basis: GraphReference | None = None
        for source_year in range(int(config.anchor_year), int(family_year) + 1):
            year_slice, current_ref_basis = build_year_slice(
                config=config,
                family_year=int(family_year),
                source_year=int(source_year),
                reference_basis=current_ref_basis if int(source_year) != int(config.anchor_year) else strict_ref_basis,
            )
            slices[int(source_year)] = year_slice
            if int(source_year) == int(config.anchor_year) and strict_ref_basis is None and year_slice.mem_x.shape[1] > 0:
                strict_ref_basis = GraphReference(fips=np.asarray(year_slice.sample_ids, dtype="U5"), basis=np.asarray(year_slice.mem_x, dtype=np.float64))
            if int(source_year) == int(family_year) and year_slice.mem_x.shape[1] > 0:
                current_artifact_basis = GraphReference(fips=np.asarray(year_slice.sample_ids, dtype="U5"), basis=np.asarray(year_slice.mem_x, dtype=np.float64))
            LOGGER.info(
                "aligned family=%d source=%d n=%d graph_n=%d align_corr=%.4f",
                int(family_year),
                int(source_year),
                int(year_slice.sample_ids.shape[0]),
                int(year_slice.graph_counties),
                float(year_slice.basis_align_mean_abs_corr),
            )

        current = slices[int(family_year)]
        grassmann_sqdist = float("nan")
        community_ari = float("nan")
        topo_common_n = 0
        if strict_ref_basis is not None and current_artifact_basis is not None:
            grassmann_sqdist, topo_common_n = compute_grassmann_sqdist(
                ref_fips=np.asarray(strict_ref_basis.fips, dtype="U5"),
                ref_basis=np.asarray(strict_ref_basis.basis, dtype=np.float64),
                cur_fips=np.asarray(current_artifact_basis.fips, dtype="U5"),
                cur_basis=np.asarray(current_artifact_basis.basis, dtype=np.float64),
                k=int(config.graph.mem_top_k),
            )
            community_ari, topo_common_n_ari = compute_community_ari(
                ref_fips=np.asarray(strict_ref_basis.fips, dtype="U5"),
                ref_basis=np.asarray(strict_ref_basis.basis, dtype=np.float64),
                cur_fips=np.asarray(current_artifact_basis.fips, dtype="U5"),
                cur_basis=np.asarray(current_artifact_basis.basis, dtype=np.float64),
                k=int(config.graph.mem_top_k),
                n_clusters=int(max(2, min(config.graph.mem_top_k, 15))),
            )
            topo_common_n = int(min(topo_common_n, topo_common_n_ari)) if topo_common_n > 0 else int(topo_common_n_ari)

        if int(family_year) == int(config.anchor_year):
            pred_log, pred_sigma_log = predict_2020_oof(year_slice=current, model_cfg=model_cfg, config=config)
            fit_mode = "strict_2020_oof"
        else:
            pseudo_target_log, pseudo_target_level = rolling_pseudo_target_log(
                year_slice=current,
                prev_corr_adjusted_level_by_fips=prev_corr_adjusted_level_by_fips,
            )
            pred_log, pred_sigma_log = predict_nowcast_year(
                train_slice=slices[int(config.anchor_year)],
                test_slice=current,
                model_cfg=model_cfg,
                config=config,
                seed=seed,
                pseudo_target_log=pseudo_target_log,
            )
            fit_mode = (
                "2020_anchor_nowcast_online_capped"
                if bool(model_cfg.rolling_online_update) and str(model_cfg.model) == "huber"
                else "2020_anchor_nowcast_online"
                if bool(model_cfg.rolling_online_update)
                else "2020_anchor_nowcast"
            )

        part = build_trajectory_rows(year_slice=current, pred_log=pred_log, pred_sigma_log=pred_sigma_log, fit_mode=fit_mode)
        if int(family_year) == int(config.anchor_year):
            prev_corr_adjusted_level_by_fips = {
                str(f): float(v)
                for f, v in zip(np.asarray(part["fips"], dtype="U5").tolist(), np.asarray(part["truth_level"], dtype=np.float64).tolist())
                if np.isfinite(float(v))
            }
        else:
            corr_adjusted = np.full(part.shape[0], np.nan, dtype=np.float64)
            resid_pred = np.asarray(part["resid_pred"], dtype=np.float64)
            keep = np.isfinite(pseudo_target_level) & np.isfinite(resid_pred) & (np.asarray(pseudo_target_level, dtype=np.float64) > 0.0)
            corr_adjusted[keep] = np.exp(np.log(np.asarray(pseudo_target_level, dtype=np.float64)[keep]) + resid_pred[keep])
            prev_corr_adjusted_level_by_fips = {
                str(f): float(v)
                for f, v in zip(np.asarray(part["fips"], dtype="U5").tolist(), corr_adjusted.tolist())
                if np.isfinite(float(v))
            }
        all_rows.append(part)
        pep_mape = mape_pop_pct(np.asarray(part.loc[np.isfinite(part["truth_log"]), "truth_log"], dtype=np.float64), np.asarray(part.loc[np.isfinite(part["truth_log"]), "pep_log"], dtype=np.float64)) if bool(current.has_truth) else float("nan")
        corrected_mape = mape_pop_pct(np.asarray(part.loc[np.isfinite(part["truth_log"]), "truth_log"], dtype=np.float64), np.asarray(part.loc[np.isfinite(part["truth_log"]), "corrected_log"], dtype=np.float64)) if bool(current.has_truth) else float("nan")
        year_meta.append(
            {
                "family_year": int(family_year),
                "prediction_year": int(family_year),
                "fit_mode": fit_mode,
                "has_truth": bool(current.has_truth),
                "graph_tag": str(current.graph_tag),
                "graph_kind": str(current.graph_kind),
                "n_counties": int(part.shape[0]),
                "pep_mape_pct": pep_mape,
                "corrected_mape_pct": corrected_mape,
                "delta_mape_pct": corrected_mape - pep_mape,
                "graph_loss": float(current.graph_loss),
                "basis_align_mean_abs_corr": float(current.basis_align_mean_abs_corr),
                "grassmann_sqdist": float(grassmann_sqdist),
                "community_ari": float(community_ari),
                "topology_common_counties": int(topo_common_n),
            }
        )
    traj = pd.concat(all_rows, axis=0, ignore_index=True)
    year_metrics = summarize_year_metrics(traj)
    if year_meta:
        year_meta_df = pd.DataFrame(year_meta)
        year_metrics = year_metrics.merge(
            year_meta_df.loc[:, ["prediction_year", "community_ari", "grassmann_sqdist", "topology_common_counties"]].rename(columns={"prediction_year": "year"}),
            on="year",
            how="left",
        )
    county_summary = summarize_counties(traj)
    topology_transfer = year_metrics.loc[:, ["year", "fit_mode", "has_truth", "graph_tag", "graph_kind", "graph_train_loss", "basis_align_mean_abs_corr", "community_ari", "grassmann_sqdist", "topology_common_counties"]].copy()
    summary = {
        "experiment_name": "postcensal_nowcast",
        "model_key": str(model_key if model_key is not None else config.downstream.selected),
        "start_year": int(config.anchor_year),
        "end_year": int(config.years.end),
        "years_evaluated_with_truth": int(np.count_nonzero(np.asarray(year_metrics["has_truth"], dtype=bool))),
        "years_nowcasted_total": int(year_metrics.shape[0]),
        "mean_pep_mape_pct": finite_mean(np.asarray(year_metrics["pep_mape_pct"], dtype=np.float64)),
        "mean_corrected_mape_pct": finite_mean(np.asarray(year_metrics["corrected_mape_pct"], dtype=np.float64)),
        "mean_delta_mape_pct": finite_mean(np.asarray(year_metrics["delta_mape_pct"], dtype=np.float64)),
    }
    return {
        "trajectory": traj,
        "year_metrics": year_metrics,
        "county_summary": county_summary,
        "topology_transfer": topology_transfer,
        "summary": summary,
    }


def persist_postcensal(result: dict[str, object], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(result["trajectory"], output_dir / "county_trajectory.parquet")
    write_frame(result["year_metrics"], output_dir / "year_metrics.parquet")
    write_frame(result["county_summary"], output_dir / "county_summary.parquet")
    write_frame(result["topology_transfer"], output_dir / "topology_transfer.parquet")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(result["summary"], fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll-forward postcensal nowcast evaluation using parquet-native manifold and graph artifacts.")
    parser.add_argument("--config", type=Path, default=Path("configs/nowcast/config.nowcast.yaml"))
    parser.add_argument("--model-key", type=str, default="", help="override downstream.selected")
    parser.add_argument("--skip", action=argparse.BooleanOptionalAction, default=False, help="skip if year_metrics parquet already exists")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)
    output_dir = config.paths.outputs.postcensal_dir
    summary_path = output_dir / "year_metrics.parquet"
    if bool(args.skip) and summary_path.exists():
        LOGGER.info("skip requested and existing postcensal outputs found at %s", summary_path)
        return
    result = run_postcensal(config, model_key=str(args.model_key).strip() or None)
    persist_postcensal(result, output_dir=output_dir)
    LOGGER.info("wrote postcensal outputs to %s", output_dir)


if __name__ == "__main__":
    main()
