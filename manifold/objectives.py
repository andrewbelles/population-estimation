#!/usr/bin/env python3
#
# objectives.py  Andrew Belles  Mar 27th, 2026
#
# SSL objectives and augmentations for manifold embedding generation.
#

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def augment_geometric(
    x: torch.Tensor,
    *,
    crop_scale_min: float,
    crop_scale_max: float,
) -> torch.Tensor:
    b = int(x.shape[0])
    if b <= 0:
        return x
    _b, _c, h, w = x.shape
    area = float(h * w)
    s_min = float(max(1e-3, min(crop_scale_min, crop_scale_max)))
    s_max = float(min(1.0, max(crop_scale_min, crop_scale_max)))
    crops: list[torch.Tensor] = []
    for i in range(b):
        s = float(torch.empty((), device=x.device).uniform_(s_min, s_max).item())
        side = int(max(1, round(math.sqrt(max(1.0, s * area)))))
        ch = int(min(h, side))
        cw = int(min(w, side))
        top = int(torch.randint(0, h - ch + 1, (1,), device=x.device).item()) if h > ch else 0
        left = int(torch.randint(0, w - cw + 1, (1,), device=x.device).item()) if w > cw else 0
        crop = x[i : i + 1, :, top : top + ch, left : left + cw]
        if ch != h or cw != w:
            crop = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)
        crops.append(crop)
    y = torch.cat(crops, dim=0)
    p_h = torch.rand((b,), device=y.device) < 0.5
    p_v = torch.rand((b,), device=y.device) < 0.5
    if bool(torch.any(p_h)):
        y[p_h] = torch.flip(y[p_h], dims=(3,))
    if bool(torch.any(p_v)):
        y[p_v] = torch.flip(y[p_v], dims=(2,))
    k = torch.randint(low=0, high=4, size=(b,), device=y.device)
    out = y.clone()
    for i in range(b):
        kk = int(k[i].item())
        if kk != 0:
            out[i] = torch.rot90(y[i], k=kk, dims=(1, 2))
    return out


def byol_loss(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    qn = F.normalize(q, dim=1)
    tn = F.normalize(t, dim=1)
    return torch.mean(2.0 - 2.0 * torch.sum(qn * tn, dim=1))


def mask_patches(
    x: torch.Tensor,
    *,
    mask_ratio: float,
    patch_size: int,
    fill_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if float(mask_ratio) <= 0.0:
        return x, torch.zeros((int(x.shape[0]), 1, int(x.shape[2]), int(x.shape[3])), device=x.device, dtype=torch.bool)
    b, _c, h, w = [int(v) for v in x.shape]
    ps = int(patch_size)
    if (h % ps) != 0 or (w % ps) != 0:
        raise ValueError(f"patch_size={ps} must divide tile size {(h, w)}")
    gh = int(h // ps)
    gw = int(w // ps)
    n_patches = int(gh * gw)
    n_mask = int(max(1, round(float(mask_ratio) * float(n_patches))))
    n_mask = min(n_patches, n_mask)
    mask_grid = torch.zeros((b, n_patches), device=x.device, dtype=torch.bool)
    for i in range(b):
        perm = torch.randperm(n_patches, device=x.device)[:n_mask]
        mask_grid[i, perm] = True
    mask = mask_grid.view(b, gh, gw).repeat_interleave(ps, dim=1).repeat_interleave(ps, dim=2).unsqueeze(1)
    x_masked = x.clone().masked_fill(mask.expand_as(x), float(fill_value))
    return x_masked, mask


def masked_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_kind: str,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kind = str(loss_kind).strip().lower()
    pred_f = pred.float()
    target_f = target.float()
    if mask is not None:
        mask_f = mask.float()
        while mask_f.ndim < pred_f.ndim:
            mask_f = mask_f.unsqueeze(1)
        mask_f = mask_f.expand_as(pred_f)
    else:
        mask_f = None
    if kind == "l1":
        diff = torch.abs(pred_f - target_f)
    elif kind == "mse":
        diff = torch.square(pred_f - target_f)
    else:
        raise ValueError(f"unsupported masked reconstruction loss={loss_kind!r}")
    if mask_f is None:
        return diff.mean().to(dtype=pred.dtype)
    den = torch.clamp(mask_f.sum(), min=1.0)
    return (diff * mask_f).sum().div(den).to(dtype=pred.dtype)


def linear_hsic_penalty(
    z: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float = 1e-6,
    clamp_val: float = 1e6,
) -> torch.Tensor:
    n = int(z.shape[0])
    if n <= 1:
        return z.new_zeros(())
    zf = torch.nan_to_num(z.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bf = torch.nan_to_num(b.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    zc = zf - zf.mean(dim=0, keepdim=True)
    bc = bf - bf.mean(dim=0, keepdim=True)
    zc = zc / torch.clamp(zc.std(dim=0, unbiased=False, keepdim=True), min=float(eps))
    bc = bc / torch.clamp(bc.std(dim=0, unbiased=False, keepdim=True), min=float(eps))
    c = (zc.transpose(0, 1) @ bc) / float(max(1, n - 1))
    out = torch.mean(c * c)
    out = torch.nan_to_num(out, nan=0.0, posinf=float(clamp_val), neginf=0.0)
    return out.to(dtype=z.dtype)


@torch.no_grad()
def ema_update(model, tau: float) -> None:
    t = float(tau)
    for po, pt in zip(model.online_encoder.parameters(), model.target_encoder.parameters()):
        pt.data.mul_(t).add_(po.data, alpha=1.0 - t)
    for po, pt in zip(model.online_projector.parameters(), model.target_projector.parameters()):
        pt.data.mul_(t).add_(po.data, alpha=1.0 - t)


def haversine_torch(coords_deg: torch.Tensor) -> torch.Tensor:
    lat = torch.deg2rad(coords_deg[:, 0]).unsqueeze(1)
    lon = torch.deg2rad(coords_deg[:, 1]).unsqueeze(1)
    dlat = lat.transpose(0, 1) - lat
    dlon = lon.transpose(0, 1) - lon
    a = torch.sin(dlat * 0.5) ** 2 + torch.cos(lat) * torch.cos(lat.transpose(0, 1)) * torch.sin(dlon * 0.5) ** 2
    a = torch.clamp(a, 0.0, 1.0)
    return 2.0 * 6371.0088 * torch.arcsin(torch.sqrt(a))


def hard_rank(values: torch.Tensor, *, descending: bool) -> torch.Tensor:
    order = torch.argsort(values, dim=1, descending=descending)
    ranks = torch.empty_like(order, dtype=torch.float32)
    pos = torch.arange(values.shape[1], device=values.device, dtype=torch.float32).view(1, -1) + 1.0
    ranks.scatter_(1, order, pos.expand_as(order))
    return ranks


def geo_knn_ball_tree(
    *,
    coords_batch: torch.Tensor,
    valid_coords: torch.Tensor,
    county_ids: torch.Tensor,
    k: int,
    dmax_km: float,
    same_county_eps_km: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from sklearn.neighbors import BallTree

    b = int(coords_batch.shape[0])
    k_eff = int(max(1, min(int(k), max(1, b - 1))))
    nn_idx = torch.zeros((b, k_eff), dtype=torch.long, device=coords_batch.device)
    nn_dist = torch.full((b, k_eff), float("inf"), dtype=torch.float32, device=coords_batch.device)
    nn_valid = torch.zeros((b, k_eff), dtype=torch.bool, device=coords_batch.device)
    if b <= 1:
        return nn_idx, nn_dist, nn_valid
    valid_idx = torch.nonzero(valid_coords.bool(), as_tuple=False).squeeze(1)
    n_valid = int(valid_idx.numel())
    if n_valid <= 1:
        return nn_idx, nn_dist, nn_valid
    coords_np = coords_batch.index_select(0, valid_idx).detach().cpu().numpy().astype(np.float64, copy=False)
    county_np = county_ids.index_select(0, valid_idx).detach().cpu().numpy().astype(np.int64, copy=False)
    coords_rad = np.deg2rad(coords_np)
    tree = BallTree(coords_rad, metric="haversine")
    k_query = int(min(n_valid, max(k_eff + 8, k_eff * 2)))
    dist_rad, ind = tree.query(coords_rad, k=k_query, return_distance=True, dualtree=True)
    dist_km = dist_rad * 6371.0088
    valid_idx_np = valid_idx.detach().cpu().numpy().astype(np.int64, copy=False)
    eps = float(max(0.0, same_county_eps_km))
    for i_local in range(n_valid):
        i_global = int(valid_idx_np[i_local])
        filled = 0
        for jpos in range(k_query):
            j_local = int(ind[i_local, jpos])
            j_global = int(valid_idx_np[j_local])
            if j_global == i_global:
                continue
            d = float(dist_km[i_local, jpos])
            if county_np[i_local] == county_np[j_local] and d < eps:
                d = eps
            if float(dmax_km) > 0.0 and d > float(dmax_km):
                continue
            nn_idx[i_global, filled] = j_global
            nn_dist[i_global, filled] = float(d)
            nn_valid[i_global, filled] = True
            filled += 1
            if filled >= k_eff:
                break
    return nn_idx, nn_dist, nn_valid


def geo_rank_loss_batch(
    *,
    z_county: torch.Tensor,
    coords_batch: torch.Tensor,
    valid_coords: torch.Tensor,
    county_ids: torch.Tensor,
    k: int,
    dmax_km: float,
    softrank_strength: float,
    normalize_rank: bool,
    neighbor_backend: str,
    same_county_eps_km: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    b = int(z_county.shape[0])
    if b <= 2:
        z0 = torch.zeros((), device=z_county.device, dtype=z_county.dtype)
        return z0, z0, 0.0
    valid_vec = valid_coords.bool()
    if int(valid_vec.sum().item()) < 3:
        z0 = torch.zeros((), device=z_county.device, dtype=z_county.dtype)
        return z0, z0, float(valid_vec.float().mean().item())
    import torchsort

    z = F.normalize(z_county, dim=1)
    k_eff = int(max(1, min(int(k), b - 1)))
    backend = str(neighbor_backend).strip().lower()
    if backend == "ball_tree":
        nn_idx, d_sel, v_sel = geo_knn_ball_tree(
            coords_batch=coords_batch.to(dtype=torch.float32),
            valid_coords=valid_vec,
            county_ids=county_ids,
            k=int(k_eff),
            dmax_km=float(dmax_km),
            same_county_eps_km=float(same_county_eps_km),
        )
    elif backend == "dense":
        d_geo = haversine_torch(coords_batch.to(dtype=torch.float32))
        eye = torch.eye(b, device=z_county.device, dtype=torch.bool)
        pair_valid = valid_vec.unsqueeze(1) & valid_vec.unsqueeze(0) & (~eye)
        same_county = (county_ids.view(-1, 1) == county_ids.view(1, -1)) & (~eye)
        if float(same_county_eps_km) > 0.0:
            d_geo = torch.where(same_county, torch.clamp_min(d_geo, float(same_county_eps_km)), d_geo)
        if float(dmax_km) > 0.0:
            pair_valid = pair_valid & (d_geo <= float(dmax_km))
        if int(pair_valid.sum().item()) <= 0:
            z0 = torch.zeros((), device=z_county.device, dtype=z_county.dtype)
            return z0, z0, float(pair_valid.float().mean().item())
        d_work = torch.where(pair_valid, d_geo, torch.full_like(d_geo, float("inf")))
        nn_idx = torch.topk(d_work, k=k_eff, dim=1, largest=False).indices
        d_sel = torch.gather(d_geo, dim=1, index=nn_idx)
        v_sel = torch.gather(pair_valid, dim=1, index=nn_idx)
    else:
        raise ValueError(f"unsupported geo-rank neighbor backend: {neighbor_backend!r}")

    safe_idx = torch.clamp(nn_idx, min=0, max=max(0, b - 1))
    nbr = z.index_select(0, safe_idx.reshape(-1)).reshape(b, k_eff, -1)
    s_sel = torch.sum(z.unsqueeze(1) * nbr, dim=2)
    row_valid = v_sel.sum(dim=1) >= 2
    if int(row_valid.sum().item()) == 0:
        z0 = torch.zeros((), device=z_county.device, dtype=z_county.dtype)
        return z0, z0, float(v_sel.float().mean().item())

    d_fill = torch.where(v_sel, d_sel, torch.full_like(d_sel, 1.0e9))
    s_fill = torch.where(v_sel, s_sel, torch.full_like(s_sel, -1.0e9))
    r_s = torchsort.soft_rank(-s_fill.to(dtype=torch.float32), regularization="l2", regularization_strength=float(max(1e-6, softrank_strength)))
    r_d = torchsort.soft_rank(d_fill.to(dtype=torch.float32), regularization="l2", regularization_strength=float(max(1e-6, softrank_strength)))
    m = v_sel.to(dtype=torch.float32)
    if bool(normalize_rank):
        k_valid = torch.clamp(m.sum(dim=1, keepdim=True), min=1.0)
        denom = torch.clamp(k_valid - 1.0, min=1.0)
        r_s = (r_s - 1.0) / denom
        r_d = (r_d - 1.0) / denom
    err = (r_s - r_d) ** 2
    den = torch.clamp(m.sum(dim=1), min=1.0)
    soft = ((err * m).sum(dim=1) / den)[row_valid]
    loss = soft.mean().to(dtype=z_county.dtype)
    with torch.no_grad():
        r_sh = hard_rank(s_fill, descending=True)
        r_dh = hard_rank(d_fill, descending=False)
        if bool(normalize_rank):
            k_valid = torch.clamp(m.sum(dim=1, keepdim=True), min=1.0)
            denom = torch.clamp(k_valid - 1.0, min=1.0)
            r_sh = (r_sh - 1.0) / denom
            r_dh = (r_dh - 1.0) / denom
        err_h = (r_sh - r_dh) ** 2
        hard = ((err_h * m).sum(dim=1) / den)[row_valid].mean().to(dtype=z_county.dtype)
    return loss, hard, float(m.mean().item())


def tau_cosine(*, step: int, total_steps: int, tau_base: float, tau_final: float) -> float:
    if total_steps <= 1:
        return float(tau_final)
    p = float(step) / float(max(1, total_steps - 1))
    p = min(max(p, 0.0), 1.0)
    c = 0.5 * (1.0 + np.cos(np.pi * p))
    return float(tau_final - (tau_final - tau_base) * c)


def apply_swap_noise(x: torch.Tensor, prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    p = float(prob)
    if p <= 0.0 or int(x.shape[0]) <= 1:
        return x, torch.zeros_like(x)
    perm = torch.randperm(int(x.shape[0]), device=x.device)
    x_perm = x.index_select(0, perm)
    mask = torch.rand_like(x) < p
    out = torch.where(mask, x_perm, x)
    return out, mask.to(dtype=x.dtype)


def apply_feature_dropout(x: torch.Tensor, prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    p = float(prob)
    if p <= 0.0:
        return x, torch.zeros_like(x)
    keep = torch.rand_like(x) >= p
    out = x * keep.to(dtype=x.dtype)
    mask = (~keep).to(dtype=x.dtype)
    return out, mask
