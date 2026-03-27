#!/usr/bin/env python3
#
# embeddings.py  Andrew Belles  Mar 27th, 2026
#
# Main interface for manifold embedding generation across admin, VIIRS, and S5P modalities.
#

import argparse
import copy
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from manifold.config import AdminConfig, EmbeddingsConfig, SpatialConfig, load_config
from manifold.data import (
    AdminYearData,
    ParquetEmbeddingWriter,
    SpatialYearPack,
    YearCyclePolicy,
    autocast_ctx,
    build_admin_year_data,
    build_spatial_year_pack,
    build_viirs_radiance_probs,
    expected_spatial_export_rows,
    format_template,
    load_admin_frame,
    load_county_coords,
    make_aligned_loaders,
    read_embedding_row_counts,
    set_seed,
)
from manifold.models import LightTabularDAE, SpatialBYOLGeoModel
from manifold.objectives import (
    apply_feature_dropout,
    apply_swap_noise,
    augment_geometric,
    byol_loss,
    ema_update,
    geo_rank_loss_batch,
    linear_hsic_penalty,
    mask_patches,
    masked_recon_loss,
    tau_cosine,
)
from manifold.optim import build_admin_optimizers, build_spatial_optimizers


LOGGER = logging.getLogger("manifold.embeddings")


@dataclass(slots=True)
class AdminEpochStats:
    loss: float
    loss_rec: float
    loss_mask: float
    n_steps: int
    year: int
    dt: float


@dataclass(slots=True)
class SpatialEpochStats:
    loss_total: float
    loss_byol: float
    loss_hsic: float
    loss_geo: float
    geo_hard: float
    geo_mask: float
    n_steps: int
    year: int
    dt: float


def setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout)


def family_label(anchor_year: int, family_end_year: int) -> str:
    if int(anchor_year) == int(family_end_year):
        return str(int(anchor_year))
    return f"{int(anchor_year)}->{int(family_end_year)}"


def family_tag(family_tag_base: str, family_end_year: int) -> str:
    return f"{str(family_tag_base)}_y{int(family_end_year)}_nowcast"


def source_split(*, family_end_year: int, source_year: int) -> str:
    return "eval" if int(source_year) == int(family_end_year) else "pqval"


def source_suffix(*, family_end_year: int, source_year: int) -> str:
    split = source_split(family_end_year=int(family_end_year), source_year=int(source_year))
    if split == "eval":
        return str(int(source_year))
    return f"pqval_{int(source_year)}"


def admin_expected_rows_by_year(cfg: EmbeddingsConfig, admin_cfg: AdminConfig, *, family_end_year: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for year in range(int(cfg.anchor_year), int(family_end_year) + 1):
        path = Path(format_template(admin_cfg.input_template, year=int(year)))
        _x, fips, _names = load_admin_frame(path)
        out[int(year)] = int(fips.shape[0])
    return out


def spatial_expected_rows_by_year(cfg: EmbeddingsConfig, spatial_cfg: SpatialConfig, *, family_end_year: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for year in range(int(cfg.anchor_year), int(family_end_year) + 1):
        out[int(year)] = expected_spatial_export_rows(
            admin_input_path=Path(format_template(cfg.admin.input_template, year=int(year))),
            tensor_input_path=Path(format_template(spatial_cfg.input_template, year=int(year))),
            modality=str(spatial_cfg.modality),
            tile_shape=spatial_cfg.tile_shape,
            max_tiles_per_bag=int(spatial_cfg.max_tiles_per_bag),
        )
    return out


def resolve_skip_source_years(
    *,
    existing_counts: dict[tuple[str, int], int],
    family_tag_name: str,
    expected_rows_by_year: dict[int, int],
    modality: str,
) -> set[int]:
    skip_years: set[int] = set()
    mismatches: list[tuple[int, int, int]] = []
    for source_year, expected_rows in sorted(expected_rows_by_year.items()):
        observed_rows = int(existing_counts.get((str(family_tag_name), int(source_year)), 0))
        if observed_rows <= 0:
            continue
        if observed_rows == int(expected_rows):
            skip_years.add(int(source_year))
            continue
        mismatches.append((int(source_year), observed_rows, int(expected_rows)))
    if mismatches:
        parts = ", ".join(f"year={year} observed={obs} expected={exp}" for year, obs, exp in mismatches)
        raise RuntimeError(
            f"[{modality}] parquet row-count mismatch for family={family_tag_name}: {parts}. "
            "Refusing --skip because partial rows cannot be repaired safely."
        )
    return skip_years


def spatial_stem(cfg: SpatialConfig) -> tuple[int, int]:
    if cfg.stem_kernel_size is not None and cfg.stem_stride is not None:
        return int(cfg.stem_kernel_size), int(cfg.stem_stride)
    if bool(cfg.small_tile_cifar_stem) and str(cfg.modality) in ("viirs", "s5p"):
        return 3, 1
    return 2, 2


def spatial_loader_settings(cfg: SpatialConfig) -> tuple[bool, int, bool, int]:
    pin_memory = bool(cfg.pin_memory)
    num_workers = int(cfg.num_workers)
    persistent_workers = bool(cfg.persistent_workers)
    prefetch_factor = int(cfg.prefetch_factor)
    if bool(cfg.flatten) and bool(cfg.radiance_sampling):
        if num_workers > 0:
            LOGGER.info(
                "[%s] loader safety override for flatten+radiance: "
                "num_workers=0 pin_memory=False persistent_workers=False prefetch_factor=1",
                cfg.modality,
            )
        return False, 0, False, 1
    if persistent_workers and num_workers > 0:
        LOGGER.debug("[%s] disabling persistent_workers for epoch-scoped loaders", cfg.modality)
    return pin_memory, num_workers, False, prefetch_factor


def cap_tiles_per_bag(
    tiles: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    n_bags: int,
    max_tiles_per_bag: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = int(max_tiles_per_bag)
    if k <= 0 or int(tiles.shape[0]) == 0:
        return tiles, batch_idx
    keep_chunks: list[torch.Tensor] = []
    for b in range(int(n_bags)):
        sel = torch.nonzero(batch_idx == int(b), as_tuple=False).squeeze(1)
        if int(sel.numel()) == 0:
            continue
        if int(sel.numel()) > k:
            perm = torch.randperm(int(sel.numel()), device=sel.device)[:k]
            sel = sel[perm]
        keep_chunks.append(sel)
    if not keep_chunks:
        return tiles[:0], batch_idx[:0]
    keep = torch.cat(keep_chunks, dim=0)
    keep, _ = torch.sort(keep)
    return tiles.index_select(0, keep), batch_idx.index_select(0, keep)


def cap_tiles_per_bag_uniform(
    tiles: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    n_bags: int,
    max_tiles_per_bag: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = int(max_tiles_per_bag)
    if k <= 0 or int(tiles.shape[0]) == 0:
        return tiles, batch_idx
    keep_chunks: list[torch.Tensor] = []
    for b in range(int(n_bags)):
        sel = torch.nonzero(batch_idx == int(b), as_tuple=False).squeeze(1)
        n = int(sel.numel())
        if n == 0:
            continue
        if n > k:
            lin = torch.linspace(0, n - 1, steps=k, device=sel.device)
            take = torch.round(lin).long().clamp_min(0).clamp_max(n - 1)
            sel = sel.index_select(0, take)
        keep_chunks.append(sel)
    if not keep_chunks:
        return tiles[:0], batch_idx[:0]
    keep = torch.cat(keep_chunks, dim=0)
    keep, _ = torch.sort(keep)
    return tiles.index_select(0, keep), batch_idx.index_select(0, keep)


def train_admin_epoch(
    *,
    data: AdminYearData,
    model: LightTabularDAE,
    optimizers: list[torch.optim.Optimizer],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    batch_size: int,
    swap_noise_prob: float,
    feature_dropout_prob: float,
    mask_loss_weight: float,
    grad_clip: float,
) -> AdminEpochStats:
    model.train()
    x = data.x_norm
    n = int(x.shape[0])
    order = np.random.permutation(n)
    total = 0.0
    total_rec = 0.0
    total_mask = 0.0
    steps = 0
    t0 = time.perf_counter()
    bs = int(max(1, batch_size))
    for s in range(0, n, bs):
        idx = order[s : s + bs]
        xb = x[idx].to(device=device, non_blocking=True)
        xn, m_swap = apply_swap_noise(xb, float(swap_noise_prob))
        xn, m_drop = apply_feature_dropout(xn, float(feature_dropout_prob))
        m_tgt = torch.clamp(m_swap + m_drop, min=0.0, max=1.0)
        with autocast_ctx(device=device, enabled=bool(use_amp), dtype=amp_dtype):
            _z, xh, m_logit = model(xn)
            loss_rec = F.mse_loss(xh, xb)
            loss_mask = F.binary_cross_entropy_with_logits(m_logit, m_tgt)
            loss = loss_rec + float(mask_loss_weight) * loss_mask
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss).item()):
            continue
        loss.backward()
        if float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        for opt in optimizers:
            opt.step()
        total += float(loss.detach().cpu())
        total_rec += float(loss_rec.detach().cpu())
        total_mask += float(loss_mask.detach().cpu())
        steps += 1
    den = float(max(1, steps))
    return AdminEpochStats(
        loss=float(total / den),
        loss_rec=float(total_rec / den),
        loss_mask=float(total_mask / den),
        n_steps=int(steps),
        year=int(data.year),
        dt=float(time.perf_counter() - t0),
    )


@torch.no_grad()
def export_admin_year(
    *,
    year_data: AdminYearData,
    model: LightTabularDAE,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    x = year_data.x_norm
    n = int(x.shape[0])
    out: list[np.ndarray] = []
    bs = int(max(1, batch_size))
    for s in range(0, n, bs):
        xb = x[s : s + bs].to(device=device, non_blocking=True)
        z = model.encode(xb)
        out.append(np.asarray(z.detach().cpu(), dtype=np.float32))
    return np.vstack(out).astype(np.float32, copy=False)


def build_spatial_model(cfg: SpatialConfig) -> SpatialBYOLGeoModel:
    stem_kernel, stem_stride = spatial_stem(cfg)
    return SpatialBYOLGeoModel(
        in_channels=int(cfg.tile_shape[0]),
        tile_shape=cfg.tile_shape,
        embed_dim=int(cfg.embed_dim),
        proj_dim=int(cfg.proj_dim),
        widths=cfg.spatial_widths,
        depths=cfg.spatial_depths,
        stem_kernel_size=int(stem_kernel),
        stem_stride=int(stem_stride),
        dw_kernel_size=int(cfg.dw_kernel_size),
        attn_hidden=int(cfg.attn_hidden),
        attn_dropout=float(cfg.attn_dropout),
        ssl_objective=str(cfg.ssl_objective),
        mae_decoder_hidden=int(cfg.mae_decoder_hidden),
    )


def train_spatial_epoch(
    *,
    pack: SpatialYearPack,
    loader,
    model: SpatialBYOLGeoModel,
    opts: list[torch.optim.Optimizer],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    cfg: SpatialConfig,
    tau_base: float,
    tau_final: float,
    global_step_start: int,
    total_steps: int,
) -> tuple[SpatialEpochStats, int]:
    model.train()
    loss_tot = 0.0
    loss_by = 0.0
    loss_hs = 0.0
    loss_geo = 0.0
    hard_geo = 0.0
    mask_geo = 0.0
    steps = 0
    t0 = time.perf_counter()
    step = int(global_step_start)
    year_coords_t = torch.from_numpy(pack.coords).to(device=device, non_blocking=True)
    year_valid_t = torch.from_numpy(pack.valid_coords.astype(np.bool_)).to(device=device, non_blocking=True)
    for batch_map in loader:
        admin_b = batch_map["admin"]
        n_bags = int(admin_b[0].shape[0])
        node_ids = admin_b[1].long().to(device=device, non_blocking=True)
        spatial_key = str(cfg.modality)
        mod_b = batch_map[spatial_key]
        x_tiles_cpu = mod_b[0]
        batch_idx_cpu = mod_b[2].long()
        x_tiles_cpu, batch_idx_cpu = cap_tiles_per_bag(
            x_tiles_cpu,
            batch_idx_cpu,
            n_bags=n_bags,
            max_tiles_per_bag=int(cfg.max_tiles_per_bag),
        )
        if int(x_tiles_cpu.shape[0]) <= 0:
            continue
        x_tiles = x_tiles_cpu.to(device=device, non_blocking=True)
        batch_idx = batch_idx_cpu.to(device=device, non_blocking=True)
        with autocast_ctx(device=device, enabled=bool(use_amp), dtype=amp_dtype):
            if str(cfg.ssl_objective).strip().lower() == "masked_ae":
                target_tiles = augment_geometric(x_tiles, crop_scale_min=float(cfg.crop_scale_min), crop_scale_max=float(cfg.crop_scale_max))
                masked_tiles, patch_mask = mask_patches(target_tiles, mask_ratio=float(cfg.mae_mask_ratio), patch_size=int(cfg.mae_patch_size))
                z_geo, _p_mae, _q_mae = model.encode_online(masked_tiles)
                recon_tiles = model.reconstruct_online(z_geo)
                l_byol = masked_recon_loss(recon_tiles, target_tiles, loss_kind=str(cfg.mae_loss_kind), mask=patch_mask)
            else:
                v1 = augment_geometric(x_tiles, crop_scale_min=float(cfg.crop_scale_min), crop_scale_max=float(cfg.crop_scale_max))
                v2 = augment_geometric(x_tiles, crop_scale_min=float(cfg.crop_scale_min), crop_scale_max=float(cfg.crop_scale_max))
                z1, _p1, q1 = model.encode_online(v1)
                z2, _p2, q2 = model.encode_online(v2)
                with torch.no_grad():
                    _tz1, tp1 = model.encode_target(v1)
                    _tz2, tp2 = model.encode_target(v2)
                l_byol = 0.5 * (byol_loss(q1, tp2.detach()) + byol_loss(q2, tp1.detach())) if bool(cfg.use_byol_contrastive) else torch.zeros((), device=q1.device, dtype=q1.dtype)
                z_geo = 0.5 * (z1 + z2)
            if float(cfg.w_geo) > 0.0:
                bag_coords = year_coords_t.index_select(0, node_ids)
                bag_valid = year_valid_t.index_select(0, node_ids)
                tile_coords = bag_coords.index_select(0, batch_idx)
                tile_valid = bag_valid.index_select(0, batch_idx)
                tile_county_ids = node_ids.index_select(0, batch_idx)
                l_geo, l_geo_hard, geo_mask = geo_rank_loss_batch(
                    z_county=z_geo,
                    coords_batch=tile_coords,
                    valid_coords=tile_valid,
                    county_ids=tile_county_ids,
                    k=int(cfg.geo_rank_k),
                    dmax_km=float(cfg.geo_rank_dmax_km),
                    softrank_strength=float(cfg.geo_rank_softrank_strength),
                    normalize_rank=bool(cfg.geo_rank_normalize_rank),
                    neighbor_backend=str(cfg.geo_rank_neighbor_backend),
                    same_county_eps_km=float(cfg.geo_rank_same_county_eps_km),
                )
            else:
                l_geo = torch.zeros((), device=l_byol.device, dtype=l_byol.dtype)
                l_geo_hard = torch.zeros((), device=l_byol.device, dtype=l_byol.dtype)
                geo_mask = 0.0
            if float(cfg.w_hsic) > 0.0:
                admin_x = torch.as_tensor(admin_b[0], dtype=torch.float32, device=device)
                admin_x = torch.nan_to_num(admin_x, nan=0.0, posinf=0.0, neginf=0.0)
                admin_mu = admin_x.mean(dim=0, keepdim=True)
                admin_sd = torch.clamp(admin_x.std(dim=0, unbiased=False, keepdim=True), min=1e-6)
                admin_x = (admin_x - admin_mu) / admin_sd
                tile_admin = admin_x.index_select(0, batch_idx)
                l_hsic = linear_hsic_penalty(z_geo, tile_admin)
            else:
                l_hsic = torch.zeros((), device=l_byol.device, dtype=l_byol.dtype)
            loss = float(cfg.w_byol) * l_byol + float(cfg.w_geo) * l_geo + float(cfg.w_hsic) * l_hsic
        for o in opts:
            o.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss).item()):
            continue
        loss.backward()
        if float(cfg.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
        for o in opts:
            o.step()
        if str(cfg.ssl_objective).strip().lower() != "masked_ae":
            tau = tau_cosine(step=int(step), total_steps=int(total_steps), tau_base=float(tau_base), tau_final=float(tau_final))
            ema_update(model, tau=tau)
        step += 1
        loss_tot += float(loss.detach().cpu())
        loss_by += float(l_byol.detach().cpu())
        loss_hs += float(l_hsic.detach().cpu())
        loss_geo += float(l_geo.detach().cpu())
        hard_geo += float(l_geo_hard.detach().cpu())
        mask_geo += float(geo_mask)
        steps += 1
    den = float(max(1, steps))
    return SpatialEpochStats(
        loss_total=float(loss_tot / den),
        loss_byol=float(loss_by / den),
        loss_hsic=float(loss_hs / den),
        loss_geo=float(loss_geo / den),
        geo_hard=float(hard_geo / den),
        geo_mask=float(mask_geo / den),
        n_steps=int(steps),
        year=int(pack.year),
        dt=float(time.perf_counter() - t0),
    ), int(step)


@torch.no_grad()
def export_spatial_year(
    *,
    pack: SpatialYearPack,
    model: SpatialBYOLGeoModel,
    device: torch.device,
    batch_size: int,
    max_tiles_per_bag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = make_aligned_loaders(
        pack=pack,
        batch_size=int(batch_size),
        shuffle=False,
        pin_memory=False,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=2,
        shared_seed=0,
        subset_idx=None,
    )
    z_all: list[np.ndarray] = []
    tile_fips_all: list[np.ndarray] = []
    counts_all: list[np.ndarray] = []
    cursor = 0
    for admin_b, mod_b in zip(loader.loaders["admin"], loader.loaders[str(pack.dataset.modality)]):
        n_bags = int(admin_b[0].shape[0])
        if n_bags <= 0:
            continue
        lo = int(cursor)
        hi = int(cursor + n_bags)
        batch_fips = pack.sample_ids[lo:hi]
        cursor = hi
        x_tiles_cpu = mod_b[0]
        batch_idx_cpu = mod_b[2].long()
        x_tiles_cpu, batch_idx_cpu = cap_tiles_per_bag_uniform(
            x_tiles_cpu,
            batch_idx_cpu,
            n_bags=n_bags,
            max_tiles_per_bag=int(max_tiles_per_bag),
        )
        if int(x_tiles_cpu.shape[0]) == 0:
            counts = np.zeros((n_bags,), dtype=np.int64)
        else:
            x_tiles = x_tiles_cpu.to(device=device, non_blocking=True)
            z_tile = model.online_encoder(x_tiles)
            z_tile = F.normalize(z_tile, dim=1)
            z_np = np.asarray(z_tile.detach().cpu(), dtype=np.float32)
            bidx = np.asarray(batch_idx_cpu.cpu(), dtype=np.int64)
            z_all.append(z_np)
            tile_fips_all.append(batch_fips[bidx].astype("U5", copy=False))
            counts = np.bincount(bidx, minlength=n_bags).astype(np.int64, copy=False)
        counts_all.append(counts)
    if int(cursor) != int(pack.sample_ids.shape[0]):
        raise RuntimeError(f"year={int(pack.year)} export cursor mismatch with sample ids")
    d = int(model.online_encoder.to_embed.out_channels)
    z_year = np.concatenate(z_all, axis=0).astype(np.float32, copy=False) if z_all else np.zeros((0, d), np.float32)
    tile_fips = np.concatenate(tile_fips_all, axis=0).astype("U5", copy=False) if tile_fips_all else np.zeros((0,), dtype="U5")
    counts = np.concatenate(counts_all, axis=0).astype(np.int64, copy=False) if counts_all else np.zeros((int(pack.sample_ids.shape[0]),), dtype=np.int64)
    return z_year, tile_fips, counts


def train_admin_family(
    cfg: EmbeddingsConfig,
    admin_cfg: AdminConfig,
    *,
    family_end_year: int,
    writer: ParquetEmbeddingWriter,
    skip_source_years: set[int] | None = None,
) -> None:
    anchor_year = int(cfg.anchor_year)
    fam_years = list(range(anchor_year, int(family_end_year) + 1))
    fam_label = family_label(anchor_year, int(family_end_year))
    fam_tag = family_tag(str(admin_cfg.family_tag_base), int(family_end_year))
    skip_years = {int(y) for y in (skip_source_years or set())}
    LOGGER.info("[admin] family=%s years=%s", fam_tag, fam_years)
    set_seed(int(admin_cfg.seed))
    req_device = str(admin_cfg.device).strip().lower()
    device = torch.device("cpu") if req_device.startswith("cuda") and (not torch.cuda.is_available()) else torch.device(str(admin_cfg.device))
    amp_dtype = torch.bfloat16 if str(admin_cfg.amp_dtype).lower() == "bfloat16" else torch.float16
    raw_blocks = []
    for year in fam_years:
        path = Path(format_template(admin_cfg.input_template, year=int(year)))
        x_raw, _fips, _names = load_admin_frame(path)
        raw_blocks.append(x_raw)
    x_train = np.concatenate(raw_blocks, axis=0)
    mu = np.mean(x_train, axis=0, keepdims=True)
    sd = np.std(x_train, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    year_data: dict[int, AdminYearData] = {}
    for year in fam_years:
        year_data[int(year)] = build_admin_year_data(
            year=int(year),
            input_path=Path(format_template(admin_cfg.input_template, year=int(year))),
            mu=mu,
            sd=sd,
        )
    in_dim = int(year_data[int(fam_years[0])].x_norm.shape[1])
    model = LightTabularDAE(in_dim=int(in_dim), embed_dim=int(admin_cfg.embed_dim), hidden_dim=int(admin_cfg.hidden_dim)).to(device)
    opts, optim_info = build_admin_optimizers(
        model=model,
        adamw_lr=float(admin_cfg.lr),
        muon_lr=float(admin_cfg.muon_lr),
        weight_decay=float(admin_cfg.weight_decay),
        muon_momentum=float(admin_cfg.muon_momentum),
        muon_optimizer_spec=str(admin_cfg.muon_optimizer_spec),
    )
    LOGGER.debug("[admin] family=%s optimizer muon_params=%d adamw_params=%d", fam_tag, optim_info["muon_param_count"], optim_info["adamw_param_count"])
    policy = YearCyclePolicy(fam_years, random_state=int(admin_cfg.seed))
    best_loss = float("inf")
    best_state = None
    for ep in range(int(admin_cfg.epochs)):
        year = int(policy.next_year())
        st = train_admin_epoch(
            data=year_data[year],
            model=model,
            optimizers=opts,
            device=device,
            use_amp=bool(admin_cfg.use_amp),
            amp_dtype=amp_dtype,
            batch_size=int(admin_cfg.batch_size),
            swap_noise_prob=float(admin_cfg.swap_noise_prob),
            feature_dropout_prob=float(admin_cfg.feature_dropout_prob),
            mask_loss_weight=float(admin_cfg.mask_loss_weight),
            grad_clip=float(admin_cfg.grad_clip),
        )
        if st.loss < best_loss:
            best_loss = float(st.loss)
            best_state = copy.deepcopy(model.state_dict())
        LOGGER.debug("[admin epoch %03d] family=%s year=%d loss=%.6f rec=%.6f mask=%.6f", ep, fam_tag, st.year, st.loss, st.loss_rec, st.loss_mask)
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_dir = Path(cfg.paths.run_root) / "admin" / fam_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}, ckpt_dir / "best.pt")
    for source_year in fam_years:
        if int(source_year) in skip_years:
            LOGGER.info("[admin] family=%s skip existing source_year=%d", fam_tag, int(source_year))
            continue
        z = export_admin_year(year_data=year_data[int(source_year)], model=model, device=device, batch_size=int(admin_cfg.batch_size))
        src_split = source_split(family_end_year=int(family_end_year), source_year=int(source_year))
        src_suffix = source_suffix(family_end_year=int(family_end_year), source_year=int(source_year))
        writer.write_rows(
            family_tag_base=str(admin_cfg.family_tag_base),
            family_tag=fam_tag,
            family_start_year=anchor_year,
            family_end_year=int(family_end_year),
            family_label=fam_label,
            source_year=int(source_year),
            source_split=str(src_split),
            source_suffix=str(src_suffix),
            is_eval_year=bool(int(source_year) == int(family_end_year)),
            fips=year_data[int(source_year)].fips,
            item_index=np.zeros((int(z.shape[0]),), dtype=np.int32),
            item_count=np.ones((int(z.shape[0]),), dtype=np.int32),
            embeddings=z,
        )


def _aggregate_spatial_epoch(stats: Sequence[SpatialEpochStats], *, family_end_year: int) -> SpatialEpochStats:
    steps = int(sum(int(s.n_steps) for s in stats))
    den = float(max(1, steps))
    return SpatialEpochStats(
        loss_total=float(sum(float(s.loss_total) * float(s.n_steps) for s in stats) / den),
        loss_byol=float(sum(float(s.loss_byol) * float(s.n_steps) for s in stats) / den),
        loss_hsic=float(sum(float(s.loss_hsic) * float(s.n_steps) for s in stats) / den),
        loss_geo=float(sum(float(s.loss_geo) * float(s.n_steps) for s in stats) / den),
        geo_hard=float(sum(float(s.geo_hard) * float(s.n_steps) for s in stats) / den),
        geo_mask=float(sum(float(s.geo_mask) * float(s.n_steps) for s in stats) / den),
        n_steps=steps,
        year=int(family_end_year),
        dt=float(sum(float(s.dt) for s in stats)),
    )


def train_spatial_family(
    cfg: EmbeddingsConfig,
    spatial_cfg: SpatialConfig,
    *,
    family_end_year: int,
    writer: ParquetEmbeddingWriter,
    skip_source_years: set[int] | None = None,
) -> None:
    anchor_year = int(cfg.anchor_year)
    fam_years = list(range(anchor_year, int(family_end_year) + 1))
    fam_label = family_label(anchor_year, int(family_end_year))
    fam_tag = family_tag(str(spatial_cfg.family_tag_base), int(family_end_year))
    skip_years = {int(y) for y in (skip_source_years or set())}
    LOGGER.info("[%s] family=%s years=%s", spatial_cfg.modality, fam_tag, fam_years)
    set_seed(int(spatial_cfg.seed))
    req_device = str(spatial_cfg.device).strip().lower()
    device = torch.device("cpu") if req_device.startswith("cuda") and (not torch.cuda.is_available()) else torch.device(str(spatial_cfg.device))
    amp_dtype = torch.bfloat16 if str(spatial_cfg.amp_dtype).lower() == "bfloat16" else torch.float16
    coords_by_fips = load_county_coords(Path(cfg.paths.geo_coords_path))
    year_packs: dict[int, SpatialYearPack] = {}
    for year in fam_years:
        year_packs[int(year)] = build_spatial_year_pack(
            year=int(year),
            admin_input_path=Path(format_template(cfg.admin.input_template, year=int(year))),
            tensor_input_path=Path(format_template(spatial_cfg.input_template, year=int(year))),
            modality=str(spatial_cfg.modality),
            tile_shape=spatial_cfg.tile_shape,
            coords_by_fips=coords_by_fips,
        )
    model = build_spatial_model(spatial_cfg).to(device)
    adamw_lr = float(spatial_cfg.adamw_lr) if spatial_cfg.adamw_lr is not None else float(spatial_cfg.lr)
    muon_lr = float(spatial_cfg.muon_lr) if spatial_cfg.muon_lr is not None else float(spatial_cfg.lr)
    opts, opt_info = build_spatial_optimizers(
        model=model,
        optimizer_mode=str(spatial_cfg.optimizer_mode),
        adamw_lr=float(adamw_lr),
        muon_lr=float(muon_lr),
        weight_decay=float(spatial_cfg.weight_decay),
        muon_momentum=float(spatial_cfg.muon_momentum),
        muon_optimizer_spec=str(spatial_cfg.muon_optimizer_spec),
    )
    LOGGER.debug("[%s] family=%s optimizer=%s muon_params=%d adamw_params=%d", spatial_cfg.modality, fam_tag, opt_info["mode"], opt_info["muon_params"], opt_info["adamw_params"])
    total_steps = 0
    for year in fam_years:
        n_draw = int(spatial_cfg.radiance_samples_per_epoch) if int(spatial_cfg.radiance_samples_per_epoch) > 0 else int(year_packs[int(year)].sample_ids.shape[0])
        total_steps += int(math.ceil(float(n_draw) / float(max(1, int(spatial_cfg.batch_size)))))
    total_steps = int(max(1, int(spatial_cfg.epochs) * int(total_steps)))
    policy = YearCyclePolicy(fam_years, random_state=int(spatial_cfg.seed))
    loader_pin_memory, loader_num_workers, loader_persistent, loader_prefetch = spatial_loader_settings(spatial_cfg)
    LOGGER.debug(
        "[%s] family=%s loader pin_memory=%s num_workers=%d persistent_workers=%s prefetch_factor=%d",
        spatial_cfg.modality,
        fam_tag,
        str(bool(loader_pin_memory)).lower(),
        int(loader_num_workers),
        str(bool(loader_persistent)).lower(),
        int(loader_prefetch),
    )
    best_loss = float("inf")
    best_state = None
    global_step = 0
    for ep in range(int(spatial_cfg.epochs)):
        w_geo_ep = 0.0 if int(ep) < int(spatial_cfg.byol_only_epochs) else float(spatial_cfg.w_geo)
        epoch_stats: list[SpatialEpochStats] = []
        year_iter = fam_years if bool(spatial_cfg.flatten) else [int(policy.next_year())]
        for year in year_iter:
            pack = year_packs[int(year)]
            subset_idx = None
            if bool(spatial_cfg.radiance_sampling) and str(spatial_cfg.modality) == "viirs":
                probs = build_viirs_radiance_probs(
                    dataset=pack.dataset,
                    weight_mode=str(spatial_cfg.radiance_weight_mode),
                    active_threshold=float(spatial_cfg.radiance_active_threshold),
                    weight_gamma=float(spatial_cfg.radiance_weight_gamma),
                    min_weight=float(spatial_cfg.radiance_min_weight),
                    clip_pctl=float(spatial_cfg.radiance_clip_pctl),
                )
                n_draw = int(spatial_cfg.radiance_samples_per_epoch) if int(spatial_cfg.radiance_samples_per_epoch) > 0 else int(probs.shape[0])
                rng = np.random.default_rng(int(spatial_cfg.seed) + int(ep) * 1009 + int(year) * 1000003)
                subset_idx = rng.choice(np.arange(int(probs.shape[0]), dtype=np.int64), size=int(max(1, n_draw)), replace=True, p=probs).astype(np.int64, copy=False)
            loader = make_aligned_loaders(
                pack=pack,
                batch_size=int(spatial_cfg.batch_size),
                shuffle=True,
                pin_memory=bool(loader_pin_memory),
                num_workers=int(loader_num_workers),
                persistent_workers=bool(loader_persistent),
                prefetch_factor=int(loader_prefetch),
                shared_seed=int(spatial_cfg.seed) + int(year) * 10007 + int(ep),
                subset_idx=subset_idx,
            )
            loader.set_epoch(ep)
            cfg_ep = copy.copy(spatial_cfg)
            cfg_ep.w_geo = float(w_geo_ep)
            st, global_step = train_spatial_epoch(
                pack=pack,
                loader=loader,
                model=model,
                opts=opts,
                device=device,
                use_amp=bool(spatial_cfg.use_amp),
                amp_dtype=amp_dtype,
                cfg=cfg_ep,
                tau_base=float(spatial_cfg.tau_base),
                tau_final=float(spatial_cfg.tau_final),
                global_step_start=int(global_step),
                total_steps=int(total_steps),
            )
            epoch_stats.append(st)
        st_agg = _aggregate_spatial_epoch(epoch_stats, family_end_year=int(family_end_year))
        if st_agg.loss_total < best_loss:
            best_loss = float(st_agg.loss_total)
            best_state = copy.deepcopy(model.state_dict())
        LOGGER.debug("[%s epoch %03d] family=%s loss=%.6f byol=%.6f hsic=%.6f geo=%.6f", spatial_cfg.modality, ep, fam_tag, st_agg.loss_total, st_agg.loss_byol, st_agg.loss_hsic, st_agg.loss_geo)
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_dir = Path(cfg.paths.run_root) / str(spatial_cfg.modality) / fam_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "best_loss": float(best_loss)}, ckpt_dir / "best.pt")
    for source_year in fam_years:
        if int(source_year) in skip_years:
            LOGGER.info("[%s] family=%s skip existing source_year=%d", spatial_cfg.modality, fam_tag, int(source_year))
            continue
        pack = year_packs[int(source_year)]
        z_year, tile_fips, counts = export_spatial_year(
            pack=pack,
            model=model,
            device=device,
            batch_size=int(spatial_cfg.batch_size),
            max_tiles_per_bag=int(spatial_cfg.max_tiles_per_bag),
        )
        item_index = np.concatenate([np.arange(int(c), dtype=np.int32) for c in counts.tolist() if int(c) > 0], axis=0) if int(np.sum(counts)) > 0 else np.zeros((0,), dtype=np.int32)
        item_count = np.concatenate([np.full(int(c), int(c), dtype=np.int32) for c in counts.tolist() if int(c) > 0], axis=0) if int(np.sum(counts)) > 0 else np.zeros((0,), dtype=np.int32)
        src_split = source_split(family_end_year=int(family_end_year), source_year=int(source_year))
        src_suffix = source_suffix(family_end_year=int(family_end_year), source_year=int(source_year))
        writer.write_rows(
            family_tag_base=str(spatial_cfg.family_tag_base),
            family_tag=fam_tag,
            family_start_year=anchor_year,
            family_end_year=int(family_end_year),
            family_label=fam_label,
            source_year=int(source_year),
            source_split=str(src_split),
            source_suffix=str(src_suffix),
            is_eval_year=bool(int(source_year) == int(family_end_year)),
            fips=tile_fips,
            item_index=item_index,
            item_count=item_count,
            embeddings=z_year,
        )


def run(config: EmbeddingsConfig, *, skip_existing: bool = False) -> None:
    LOGGER.info("start years=%s modalities=%s skip_existing=%s", config.years, config.modalities, str(bool(skip_existing)).lower())
    writers: list[ParquetEmbeddingWriter] = []
    try:
        if "admin" in config.modalities and bool(config.admin.enabled):
            writer: ParquetEmbeddingWriter | None = None
            existing_counts = read_embedding_row_counts(config.admin.output_parquet, modality="admin") if bool(skip_existing) else {}
            for family_end in config.years:
                fam_tag = family_tag(str(config.admin.family_tag_base), int(family_end))
                skip_years: set[int] = set()
                expected_rows: dict[int, int] | None = None
                if bool(skip_existing):
                    expected_rows = admin_expected_rows_by_year(config, config.admin, family_end_year=int(family_end))
                    skip_years = resolve_skip_source_years(
                        existing_counts=existing_counts,
                        family_tag_name=fam_tag,
                        expected_rows_by_year=expected_rows,
                        modality="admin",
                    )
                    if len(skip_years) == len(expected_rows):
                        LOGGER.info("[admin] family=%s skip all source years already present in %s", fam_tag, config.admin.output_parquet)
                        continue
                if writer is None:
                    writer = ParquetEmbeddingWriter(
                        output_path=config.admin.output_parquet,
                        modality="admin",
                        embed_dim=int(config.admin.embed_dim),
                        append=bool(skip_existing) and Path(config.admin.output_parquet).exists(),
                    )
                    writers.append(writer)
                train_admin_family(config, config.admin, family_end_year=int(family_end), writer=writer, skip_source_years=skip_years)
                if bool(skip_existing) and expected_rows is not None:
                    for source_year, row_count in expected_rows.items():
                        existing_counts[(str(fam_tag), int(source_year))] = int(row_count)
        for modality in config.modalities:
            if modality == "admin":
                continue
            spatial_cfg = config.spatial_cfg(modality)
            if not bool(spatial_cfg.enabled):
                continue
            writer = None
            existing_counts = read_embedding_row_counts(spatial_cfg.output_parquet, modality=str(modality)) if bool(skip_existing) else {}
            for family_end in config.years:
                fam_tag = family_tag(str(spatial_cfg.family_tag_base), int(family_end))
                skip_years = set()
                expected_rows: dict[int, int] | None = None
                if bool(skip_existing):
                    expected_rows = spatial_expected_rows_by_year(config, spatial_cfg, family_end_year=int(family_end))
                    skip_years = resolve_skip_source_years(
                        existing_counts=existing_counts,
                        family_tag_name=fam_tag,
                        expected_rows_by_year=expected_rows,
                        modality=str(modality),
                    )
                    if len(skip_years) == len(expected_rows):
                        LOGGER.info("[%s] family=%s skip all source years already present in %s", modality, fam_tag, spatial_cfg.output_parquet)
                        continue
                if writer is None:
                    writer = ParquetEmbeddingWriter(
                        output_path=spatial_cfg.output_parquet,
                        modality=str(modality),
                        embed_dim=int(spatial_cfg.embed_dim),
                        append=bool(skip_existing) and Path(spatial_cfg.output_parquet).exists(),
                    )
                    writers.append(writer)
                train_spatial_family(config, spatial_cfg, family_end_year=int(family_end), writer=writer, skip_source_years=skip_years)
                if bool(skip_existing) and expected_rows is not None:
                    for source_year, row_count in expected_rows.items():
                        existing_counts[(str(fam_tag), int(source_year))] = int(row_count)
    finally:
        for writer in writers:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SSL manifold embeddings to parquet.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--skip", "--skip-existing", dest="skip_existing", action="store_true")
    args = parser.parse_args()
    setup_logging(str(args.log_level))
    config = load_config(args.config)
    run(config, skip_existing=bool(args.skip_existing))


if __name__ == "__main__":
    main()
