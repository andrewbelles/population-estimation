#!/usr/bin/env python3
#
# censal.py  Andrew Belles  Mar 27th, 2026
#
# Strict 2020 censal evaluation against PEP using parquet-native manifold and graph outputs.
#

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nowcast.common import (
    FoldSplit,
    POP_STRATA_LABELS,
    align_rows,
    apply_block_pca,
    assign_population_strata,
    build_state_group_splits,
    compute_topology_leakage_proxy,
    fit_predict,
    gaussian_crps,
    load_county_display_lookup,
    load_modality_block,
    load_pep_year,
    load_topology_edges,
    load_topology_rows,
    mape_pop_pct,
)
from nowcast.config import NowcastConfig, load_config


LOGGER = logging.getLogger("nowcast.censal")


@dataclass(slots=True)
class StrictInputs:
    truth_pep: pd.DataFrame
    aligned: dict[str, np.ndarray]
    direct_keys: list[str]
    mem_available: bool
    graph_tag: str | None
    graph_kind: str | None
    graph_edges: pd.DataFrame


@dataclass(slots=True)
class StrictResult:
    summary_df: pd.DataFrame
    fold_df: pd.DataFrame
    state_df: pd.DataFrame
    pop_df: pd.DataFrame
    pop_compare_df: pd.DataFrame
    coverage_df: pd.DataFrame
    abs_df: pd.DataFrame
    summary: dict[str, object]


def setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout)


def strict_feature_specs(direct_keys: list[str], *, mem_available: bool, requested: list[str]) -> list[tuple[str, list[str], bool, bool]]:
    specs: list[tuple[str, list[str], bool, bool]] = []
    allow = {str(x).strip().lower() for x in requested}
    admin_keys = [str(k).strip().lower() for k in direct_keys if str(k).strip().lower() == "admin"]
    if mem_available and "mem" in allow:
        specs.append(("mem", [], True, False))
    if direct_keys:
        if "embeddings" in allow:
            specs.append(("embeddings", direct_keys, False, False))
        if "embeddings_only" in allow:
            specs.append(("embeddings_only", direct_keys, False, True))
        if mem_available and admin_keys and "embeddings_mem" in allow:
            specs.append(("embeddings_mem", admin_keys, True, False))
        if mem_available and admin_keys and "embeddings_mem_only" in allow:
            specs.append(("embeddings_mem_only", admin_keys, True, True))
    return specs


def build_strict_inputs(config: NowcastConfig) -> StrictInputs:
    strict_year = int(config.evaluation.strict_year)
    truth_pep = load_pep_year(config.paths.pep_parquet, year=strict_year)
    truth_pep = truth_pep.loc[np.isfinite(np.asarray(truth_pep["y_log"], dtype=np.float64)) & np.isfinite(np.asarray(truth_pep["pep_log"], dtype=np.float64))].copy()
    if truth_pep.empty:
        raise ValueError(f"no strict truth rows available for year={strict_year}")
    direct_blocks = {}
    direct_keys: list[str] = []
    for modality in config.downstream.strict_direct_modalities:
        block_cfg = config.block_cfg(modality)
        if not bool(block_cfg.enabled):
            continue
        direct_blocks[str(modality)] = load_modality_block(
            block_cfg,
            family_end_year=strict_year,
            source_year=strict_year,
            pool_mode=str(config.evaluation.tile_pool_mode),
        )
        direct_keys.append(str(modality))
    mem_block = None
    graph_edges = pd.DataFrame(columns=["src_fips", "dst_fips", "edge_weight"])
    if bool(config.graph.enabled):
        mem_block = load_topology_rows(
            basis_parquet=config.paths.topology_basis_parquet,
            runs_parquet=config.paths.topology_runs_parquet,
            graph_tag_base=config.graph.graph_tag_base,
            graph_kind=config.graph.graph_kind,
            family_end_year=strict_year,
            source_year=strict_year,
            top_k=config.graph.mem_top_k,
        )
        graph_edges = load_topology_edges(
            edges_parquet=config.paths.topology_edges_parquet,
            graph_tag_name=mem_block.graph_tag,
            graph_kind=mem_block.graph_kind,
            source_year=strict_year,
        )
    aligned = align_rows(truth_pep=truth_pep, direct_blocks=direct_blocks, mem_block=mem_block)
    return StrictInputs(
        truth_pep=truth_pep,
        aligned=aligned,
        direct_keys=direct_keys,
        mem_available=mem_block is not None,
        graph_tag=mem_block.graph_tag if mem_block is not None else None,
        graph_kind=mem_block.graph_kind if mem_block is not None else None,
        graph_edges=graph_edges,
    )


def write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def evaluate_strict(config: NowcastConfig, *, model_key: str | None = None) -> StrictResult:
    model_cfg = config.downstream.model_cfg(model_key)
    county_lookup = load_county_display_lookup(config.paths.county_shapefile)
    inputs = build_strict_inputs(config)
    aligned = inputs.aligned
    sample_ids = np.asarray(aligned["sample_ids"], dtype="U5")
    y_log = np.asarray(aligned["y_log"], dtype=np.float64)
    y_level = np.asarray(aligned["y_level"], dtype=np.float64)
    pep_log = np.asarray(aligned["pep_log"], dtype=np.float64)
    pep_level = np.asarray(aligned["pep_level"], dtype=np.float64)
    true_resid = y_log - pep_log

    direct_blocks = {k: np.asarray(aligned[k], dtype=np.float64) for k in inputs.direct_keys}
    block_masks = {k: np.asarray(aligned[f"{k}_mask"], dtype=bool) for k in inputs.direct_keys}
    Xmem_full = np.asarray(aligned.get("mem", np.zeros((sample_ids.shape[0], 0), dtype=np.float64)), dtype=np.float64)
    mem_mask = np.asarray(aligned.get("mem_mask", np.zeros(sample_ids.shape[0], dtype=bool)), dtype=bool)
    if len(direct_blocks) <= 0 and Xmem_full.shape[1] <= 0:
        raise ValueError("no active features: enable at least one direct modality or graph MEM")

    splits = build_state_group_splits(
        sample_ids,
        n_splits=config.evaluation.n_splits,
        strategy=config.evaluation.fold_strategy,
        region_level=config.evaluation.fold_region_level,
    )
    LOGGER.info(
        "strict year=%d n=%d states=%d direct=%s mem_dim=%d fold_strategy=%s",
        int(config.evaluation.strict_year),
        int(sample_ids.shape[0]),
        int(np.unique(np.asarray([str(f)[:2] for f in sample_ids.tolist()], dtype="U2")).shape[0]),
        ",".join(inputs.direct_keys) if inputs.direct_keys else "none",
        int(Xmem_full.shape[1]),
        str(config.evaluation.fold_strategy),
    )

    pred_mu = {
        "pep": np.asarray(pep_log, dtype=np.float64).copy(),
        "mem": np.full(sample_ids.shape[0], np.nan, dtype=np.float64),
        "embeddings": np.full(sample_ids.shape[0], np.nan, dtype=np.float64),
        "embeddings_mem": np.full(sample_ids.shape[0], np.nan, dtype=np.float64),
        "embeddings_only": np.full(sample_ids.shape[0], np.nan, dtype=np.float64),
        "embeddings_mem_only": np.full(sample_ids.shape[0], np.nan, dtype=np.float64),
    }
    pred_sigma = {k: np.full(sample_ids.shape[0], np.nan, dtype=np.float64) for k in pred_mu}
    pred_sigma["pep"] = np.full(sample_ids.shape[0], max(float(np.std(true_resid, ddof=0)), 1e-6), dtype=np.float64)
    fold_by_sample = np.full(sample_ids.shape[0], -1, dtype=np.int64)
    fold_states_by_sample = np.full(sample_ids.shape[0], "", dtype=object)

    fold_rows: list[dict[str, object]] = []
    specs = strict_feature_specs(inputs.direct_keys, mem_available=inputs.mem_available, requested=config.downstream.strict_feature_specs)
    mem_models = {"mem", "embeddings_mem", "embeddings_mem_only"}

    for split in splits:
        tr_idx = np.asarray(split.train_idx, dtype=np.int64)
        te_idx = np.asarray(split.test_idx, dtype=np.int64)
        fold_by_sample[te_idx] = int(split.fold_id)
        fold_states_by_sample[te_idx] = ",".join(split.heldout_states)
        LOGGER.info(
            "fold=%d/%d train=%d test=%d heldout_states=%s heldout_regions=%s",
            int(split.fold_id),
            int(len(splits)),
            int(tr_idx.shape[0]),
            int(te_idx.shape[0]),
            ",".join(split.heldout_states),
            ",".join(split.heldout_regions),
        )
        leakage_proxy = compute_topology_leakage_proxy(
            edges=inputs.graph_edges,
            sample_ids=sample_ids,
            test_idx=te_idx,
            mode=config.analysis.leakage_proxy_mode,
        )
        yte_fold = np.asarray(y_log[te_idx], dtype=np.float64)
        pep_te_fold = np.asarray(pep_log[te_idx], dtype=np.float64)
        for model_id, direct_keys, use_mem_block, full_prediction in specs:
            train_mask = np.ones(tr_idx.shape[0], dtype=bool)
            test_mask = np.ones(te_idx.shape[0], dtype=bool)
            for key in direct_keys:
                train_mask &= np.asarray(block_masks[key][tr_idx], dtype=bool)
                test_mask &= np.asarray(block_masks[key][te_idx], dtype=bool)
            if use_mem_block:
                train_mask &= np.asarray(mem_mask[tr_idx], dtype=bool)
                test_mask &= np.asarray(mem_mask[te_idx], dtype=bool)
            tr_sub = np.asarray(tr_idx[train_mask], dtype=np.int64)
            te_sub = np.asarray(te_idx[test_mask], dtype=np.int64)
            if tr_sub.shape[0] <= 1 or te_sub.shape[0] <= 0:
                LOGGER.debug("skip fold=%d model=%s train_active=%d test_active=%d", int(split.fold_id), model_id, int(tr_sub.shape[0]), int(te_sub.shape[0]))
                continue
            blocks_tr = {k: np.asarray(direct_blocks[k][tr_sub], dtype=np.float64) for k in direct_keys}
            blocks_te = {k: np.asarray(direct_blocks[k][te_sub], dtype=np.float64) for k in direct_keys}
            Xemb_tr, Xemb_te, _diag = apply_block_pca(
                blocks_tr=blocks_tr,
                blocks_te=blocks_te,
                reduce=bool(config.evaluation.model_pca_reduce),
                dim=int(config.evaluation.model_pca_dim),
                mode=str(config.evaluation.model_pca_mode),
            )
            Xmem_tr = np.asarray(Xmem_full[tr_sub], dtype=np.float64) if use_mem_block else np.zeros((tr_sub.shape[0], 0), dtype=np.float64)
            Xmem_te = np.asarray(Xmem_full[te_sub], dtype=np.float64) if use_mem_block else np.zeros((te_sub.shape[0], 0), dtype=np.float64)
            if Xemb_tr.shape[1] > 0 and Xmem_tr.shape[1] > 0:
                Ftr = np.concatenate([Xemb_tr, Xmem_tr], axis=1)
                Fte = np.concatenate([Xemb_te, Xmem_te], axis=1)
            elif Xemb_tr.shape[1] > 0:
                Ftr, Fte = Xemb_tr, Xemb_te
            elif Xmem_tr.shape[1] > 0:
                Ftr, Fte = Xmem_tr, Xmem_te
            else:
                continue
            ytr = np.asarray(y_log[tr_sub], dtype=np.float64)
            yte = np.asarray(y_log[te_sub], dtype=np.float64)
            pep_tr = np.asarray(pep_log[tr_sub], dtype=np.float64)
            pep_te = np.asarray(pep_log[te_sub], dtype=np.float64)
            resid_tr = ytr - pep_tr
            target_tr = ytr if full_prediction else resid_tr
            train_fit_mask = np.isfinite(Ftr).all(axis=1) & np.isfinite(target_tr)
            test_fit_mask = np.isfinite(Fte).all(axis=1) & np.isfinite(yte) & np.isfinite(pep_te)
            tr_fit = np.asarray(np.flatnonzero(train_fit_mask), dtype=np.int64)
            te_fit = np.asarray(np.flatnonzero(test_fit_mask), dtype=np.int64)
            if tr_fit.shape[0] <= 1 or te_fit.shape[0] <= 0:
                LOGGER.debug("skip fold=%d model=%s train_fit=%d test_fit=%d", int(split.fold_id), model_id, int(tr_fit.shape[0]), int(te_fit.shape[0]))
                continue
            Ftr_fit = np.asarray(Ftr[tr_fit], dtype=np.float64)
            Fte_fit = np.asarray(Fte[te_fit], dtype=np.float64)
            target_tr_fit = np.asarray(target_tr[tr_fit], dtype=np.float64)
            yte_fit = np.asarray(yte[te_fit], dtype=np.float64)
            pep_te_fit = np.asarray(pep_te[te_fit], dtype=np.float64)
            te_pred_idx = np.asarray(te_sub[te_fit], dtype=np.int64)
            mu_raw, sig_raw, feat_dim = fit_predict(
                model_cfg=model_cfg,
                Xtr=Ftr_fit,
                ytr=target_tr_fit,
                Xte=Fte_fit,
                seed=int(config.evaluation.seed) + 1000 + int(split.fold_id),
            )
            mu = np.asarray(mu_raw, dtype=np.float64) if full_prediction else np.asarray(pep_te_fit + mu_raw, dtype=np.float64)
            pred_mu[model_id][te_pred_idx] = mu
            pred_sigma[model_id][te_pred_idx] = np.asarray(sig_raw, dtype=np.float64)
            fold_rows.append(
                {
                    "fold": int(split.fold_id),
                    "model": model_id,
                    "n_train": int(tr_fit.shape[0]),
                    "n_test": int(te_fit.shape[0]),
                    "n_states": int(len(split.heldout_states)),
                    "heldout_states": ",".join(split.heldout_states),
                    "heldout_regions": ",".join(split.heldout_regions),
                    "heldout_divisions": ",".join(split.heldout_divisions),
                    "feature_dim": int(feat_dim),
                    "graph_tag": str(inputs.graph_tag or ""),
                    "graph_kind": str(inputs.graph_kind or ""),
                    "topology_leakage_proxy": float(leakage_proxy) if model_id in mem_models else 0.0,
                    "mae_log": float(np.mean(np.abs(yte_fit - mu))),
                    "mape_pop_pct": mape_pop_pct(yte_fit, mu),
                    "crps": gaussian_crps(yte_fit, mu, np.asarray(sig_raw, dtype=np.float64)),
                    "residual_corr_pearson": float(np.corrcoef((yte_fit - pep_te_fit), (mu - pep_te_fit))[0, 1]) if te_fit.shape[0] > 1 else np.nan,
                    "residual_corr_spearman": float(spearmanr((yte_fit - pep_te_fit), (mu - pep_te_fit)).statistic) if te_fit.shape[0] > 1 else np.nan,
                }
            )
        fold_rows.append(
            {
                "fold": int(split.fold_id),
                "model": "pep",
                "n_train": int(tr_idx.shape[0]),
                "n_test": int(te_idx.shape[0]),
                "n_states": int(len(split.heldout_states)),
                "heldout_states": ",".join(split.heldout_states),
                "heldout_regions": ",".join(split.heldout_regions),
                "heldout_divisions": ",".join(split.heldout_divisions),
                "feature_dim": 0,
                "graph_tag": str(inputs.graph_tag or ""),
                "graph_kind": str(inputs.graph_kind or ""),
                "topology_leakage_proxy": 0.0,
                "mae_log": float(np.mean(np.abs(yte_fold - pep_te_fold))),
                "mape_pop_pct": mape_pop_pct(yte_fold, pep_te_fold),
                "crps": gaussian_crps(yte_fold, pep_te_fold, pred_sigma["pep"][te_idx]),
                "residual_corr_pearson": 0.0,
                "residual_corr_spearman": 0.0,
            }
        )

    fold_df = pd.DataFrame(fold_rows).sort_values(["model", "fold"]).reset_index(drop=True)
    pep_fold = fold_df.loc[fold_df["model"] == "pep", ["fold", "mape_pop_pct"]].rename(columns={"mape_pop_pct": "pep_mape_pop_pct"})
    fold_df = fold_df.merge(pep_fold, on="fold", how="left")
    fold_df["relative_error_improvement_frac"] = (
        (np.asarray(fold_df["pep_mape_pop_pct"], dtype=np.float64) - np.asarray(fold_df["mape_pop_pct"], dtype=np.float64))
        / np.clip(np.asarray(fold_df["pep_mape_pop_pct"], dtype=np.float64), 1e-9, None)
    )
    fold_df.loc[fold_df["model"] == "pep", "relative_error_improvement_frac"] = 0.0
    positive_improvement = np.maximum(np.asarray(fold_df["relative_error_improvement_frac"], dtype=np.float64), 0.0)
    fold_df["attributable_relative_improvement_frac"] = positive_improvement * np.asarray(fold_df["topology_leakage_proxy"], dtype=np.float64)
    fold_df["adjusted_relative_improvement_frac"] = np.asarray(fold_df["relative_error_improvement_frac"], dtype=np.float64) - np.asarray(fold_df["attributable_relative_improvement_frac"], dtype=np.float64)
    fold_df["relative_error_improvement_pct"] = np.asarray(fold_df["relative_error_improvement_frac"], dtype=np.float64) * 100.0
    fold_df["attributable_relative_improvement_pct"] = np.asarray(fold_df["attributable_relative_improvement_frac"], dtype=np.float64) * 100.0
    fold_df["adjusted_relative_improvement_pct"] = np.asarray(fold_df["adjusted_relative_improvement_frac"], dtype=np.float64) * 100.0
    fold_df["adjusted_mape_pop_pct"] = np.asarray(fold_df["pep_mape_pop_pct"], dtype=np.float64) * (1.0 - np.asarray(fold_df["adjusted_relative_improvement_frac"], dtype=np.float64))
    fold_df.loc[fold_df["model"] == "pep", "adjusted_mape_pop_pct"] = np.asarray(fold_df.loc[fold_df["model"] == "pep", "mape_pop_pct"], dtype=np.float64)

    summary_rows: list[dict[str, object]] = []
    for mid, mu in pred_mu.items():
        sig = np.asarray(pred_sigma[mid], dtype=np.float64)
        valid = np.isfinite(mu) & np.isfinite(sig)
        corr = mu[valid] - pep_log[valid]
        pear = float(np.corrcoef(true_resid[valid], corr)[0, 1]) if np.count_nonzero(valid) > 1 and mid != "pep" else 0.0 if mid == "pep" else np.nan
        spear = float(spearmanr(true_resid[valid], corr).statistic) if np.count_nonzero(valid) > 1 and mid != "pep" else 0.0 if mid == "pep" else np.nan
        rows_mid = fold_df.loc[fold_df["model"] == mid].copy()
        summary_rows.append(
            {
                "model": mid,
                "mae_log_mean": float(rows_mid["mae_log"].mean()),
                "mae_log_std": float(rows_mid["mae_log"].std(ddof=0)),
                "mape_pop_pct_mean": float(rows_mid["mape_pop_pct"].mean()),
                "mape_pop_pct_std": float(rows_mid["mape_pop_pct"].std(ddof=0)),
                "adjusted_mape_pop_pct_mean": float(rows_mid["adjusted_mape_pop_pct"].mean()),
                "adjusted_mape_pop_pct_std": float(rows_mid["adjusted_mape_pop_pct"].std(ddof=0)),
                "relative_error_improvement_pct_mean": float(rows_mid["relative_error_improvement_pct"].mean()),
                "attributable_relative_improvement_pct_mean": float(rows_mid["attributable_relative_improvement_pct"].mean()),
                "adjusted_relative_improvement_pct_mean": float(rows_mid["adjusted_relative_improvement_pct"].mean()),
                "topology_leakage_proxy_mean": float(rows_mid["topology_leakage_proxy"].mean()),
                "crps_mean": float(rows_mid["crps"].mean()),
                "crps_std": float(rows_mid["crps"].std(ddof=0)),
                "residual_corr_pearson": pear,
                "residual_corr_spearman": spear,
                "n_eval": int(np.count_nonzero(valid)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("mape_pop_pct_mean").reset_index(drop=True)

    long_rows: list[dict[str, object]] = []
    for mid, mu in pred_mu.items():
        sig = np.asarray(pred_sigma[mid], dtype=np.float64)
        for i, f in enumerate(sample_ids.tolist()):
            if not np.isfinite(mu[i]) or not np.isfinite(sig[i]):
                continue
            long_rows.append(
                {
                    "fips": f,
                    "state": str(f)[:2],
                    "fold": int(fold_by_sample[i]),
                    "heldout_states": str(fold_states_by_sample[i]),
                    "y_log": float(y_log[i]),
                    "y_level": float(y_level[i]),
                    "pep_log": float(pep_log[i]),
                    "pep_level": float(pep_level[i]),
                    "model": mid,
                    "pred_log": float(mu[i]),
                    "pred_level": float(np.exp(mu[i])),
                    "sigma": float(sig[i]),
                    "true_resid_log": float(true_resid[i]),
                    "pred_correction_log": float(mu[i] - pep_log[i]),
                    "abs_err_log": float(abs(y_log[i] - mu[i])),
                    "ape_pop_pct": float(abs(np.exp(mu[i]) - y_level[i]) / max(abs(y_level[i]), 1e-9) * 100.0),
                }
            )
    long_df = pd.DataFrame(long_rows)
    long_df["pop_stratum"] = assign_population_strata(np.asarray(long_df["y_level"], dtype=np.float64))
    county_meta = (
        long_df[["fips", "state", "y_level", "pop_stratum"]]
        .drop_duplicates(subset=["fips"])
        .merge(county_lookup[["fips", "state", "county", "state_abbr", "region", "division", "aland_sqkm"]], on=["fips", "state"], how="left")
    )
    fold_leakage = fold_df[["fold", "model", "topology_leakage_proxy", "relative_error_improvement_pct", "adjusted_relative_improvement_pct"]].drop_duplicates()
    long_df = long_df.merge(fold_leakage, on=["fold", "model"], how="left")

    state_rows = []
    for (mid, state), part in long_df.groupby(["model", "state"], sort=True):
        state_rows.append(
            {
                "model": mid,
                "state": state,
                "n": int(part.shape[0]),
                "mae_log": float(part["abs_err_log"].mean()),
                "mape_pop_pct": float(part["ape_pop_pct"].mean()),
            }
        )
    state_df = pd.DataFrame(state_rows).sort_values(["model", "state"]).reset_index(drop=True)

    pop_rows = []
    for (mid, pop_stratum), part in long_df.groupby(["model", "pop_stratum"], observed=True, sort=False):
        pop_rows.append(
            {
                "model": mid,
                "pop_stratum": str(pop_stratum),
                "n": int(part["fips"].nunique()),
                "mae_log": float(part["abs_err_log"].mean()),
                "mape_pop_pct": float(part["ape_pop_pct"].mean()),
            }
        )
    pop_df = pd.DataFrame(pop_rows)
    if not pop_df.empty:
        pop_df["pop_stratum"] = pd.Categorical(pop_df["pop_stratum"].astype(str), categories=list(POP_STRATA_LABELS), ordered=True)
        pop_df = pop_df.sort_values(["pop_stratum", "model"]).reset_index(drop=True)

    coverage_df = (
        county_meta.groupby("pop_stratum", observed=True, sort=False)
        .agg(n_counties=("fips", "nunique"), population_total=("y_level", "sum"))
        .reset_index()
        if not county_meta.empty
        else pd.DataFrame(columns=["pop_stratum", "n_counties", "population_total"])
    )
    if not coverage_df.empty:
        coverage_df["pop_stratum"] = pd.Categorical(coverage_df["pop_stratum"], categories=list(POP_STRATA_LABELS), ordered=True)
        coverage_df = coverage_df.sort_values("pop_stratum").reset_index(drop=True)
        n_total = float(max(1, county_meta["fips"].nunique()))
        pop_total = float(max(1e-9, np.asarray(county_meta["y_level"], dtype=np.float64).sum()))
        coverage_df["county_share_pct"] = np.asarray(coverage_df["n_counties"], dtype=np.float64) / n_total * 100.0
        coverage_df["population_share_pct"] = np.asarray(coverage_df["population_total"], dtype=np.float64) / pop_total * 100.0

    pep_mape = float(summary_df.loc[summary_df["model"] == "pep", "mape_pop_pct_mean"].iloc[0])
    contenders = summary_df.loc[summary_df["model"] != "pep"].copy()
    best_model = str(contenders.iloc[0]["model"]) if not contenders.empty else "pep"
    best_mape = float(contenders.iloc[0]["mape_pop_pct_mean"]) if not contenders.empty else pep_mape
    delta_best = best_mape - pep_mape

    pop_cmp = pop_df.pivot(index="pop_stratum", columns="model", values="mape_pop_pct") if not pop_df.empty else pd.DataFrame()
    pop_counts = (
        pop_df[pop_df["model"] == "pep"][["pop_stratum", "n"]].rename(columns={"n": "n_counties"}).set_index("pop_stratum")
        if not pop_df.empty and "pep" in pop_df["model"].tolist()
        else pd.DataFrame()
    )
    if best_model in pop_cmp.columns and "pep" in pop_cmp.columns:
        pop_compare_df = pop_cmp.reset_index().merge(pop_counts, on="pop_stratum", how="left")
        pop_compare_df["delta_vs_pep"] = pop_compare_df[best_model] - pop_compare_df["pep"]
        pop_compare_df = pop_compare_df[["pop_stratum", "n_counties", "pep", best_model, "delta_vs_pep"]]
    else:
        pop_compare_df = pd.DataFrame(columns=["pop_stratum", "n_counties", "pep", best_model, "delta_vs_pep"])

    summary = {
        "strict_year": int(config.evaluation.strict_year),
        "model_key": str(model_key if model_key is not None else config.downstream.selected),
        "best_model": best_model,
        "pep_mape_pop_pct_mean": pep_mape,
        "best_mape_pop_pct_mean": best_mape,
        "delta_best_vs_pep_pct": delta_best,
        "graph_tag": str(inputs.graph_tag or ""),
        "graph_kind": str(inputs.graph_kind or ""),
        "fold_strategy": str(config.evaluation.fold_strategy),
        "n_eval_counties": int(long_df["fips"].nunique()),
    }

    return StrictResult(
        summary_df=summary_df,
        fold_df=fold_df,
        state_df=state_df,
        pop_df=pop_df,
        pop_compare_df=pop_compare_df,
        coverage_df=coverage_df,
        abs_df=long_df,
        summary=summary,
    )


def persist_strict(result: StrictResult, *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(result.summary_df, output_dir / "summary.parquet")
    write_frame(result.fold_df, output_dir / "fold_metrics.parquet")
    write_frame(result.state_df, output_dir / "state_metrics.parquet")
    write_frame(result.pop_df, output_dir / "population_strata.parquet")
    write_frame(result.abs_df, output_dir / "abs_errors.parquet")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(result.summary, fh, indent=2)


def log_strict_summary(result: StrictResult) -> None:
    cols = [
        "model",
        "mape_pop_pct_mean",
        "adjusted_mape_pop_pct_mean",
        "relative_error_improvement_pct_mean",
        "adjusted_relative_improvement_pct_mean",
        "crps_mean",
        "n_eval",
    ]
    keep = [c for c in cols if c in result.summary_df.columns]
    if not keep:
        return
    table = result.summary_df.loc[:, keep].copy()
    for col in table.columns:
        if col == "model":
            continue
        table[col] = pd.to_numeric(table[col], errors="coerce")
    LOGGER.info("strict summary\n%s", table.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict 2020 censal evaluation against PEP using parquet-native artifacts.")
    parser.add_argument("--config", type=Path, default=Path("configs/nowcast/config.nowcast.yaml"))
    parser.add_argument("--model-key", type=str, default="", help="override downstream.selected")
    parser.add_argument("--skip", action=argparse.BooleanOptionalAction, default=False, help="skip if summary parquet already exists")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)
    output_dir = config.paths.outputs.censal_dir
    summary_path = output_dir / "summary.parquet"
    if bool(args.skip) and summary_path.exists():
        LOGGER.info("skip requested and existing strict summary found at %s", summary_path)
        return
    result = evaluate_strict(config, model_key=str(args.model_key).strip() or None)
    persist_strict(result, output_dir=output_dir)
    log_strict_summary(result)
    LOGGER.info("wrote strict outputs to %s", output_dir)


if __name__ == "__main__":
    main()
