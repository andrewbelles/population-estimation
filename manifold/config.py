#!/usr/bin/env python3
#
# config.py  Andrew Belles  Mar 27th, 2026
#
# YAML contract loader for manifold embedding generation.
#

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RESERVED_MODALITY_KEYS = {"years", "modalities", "paths", "admin"}


@dataclass(slots=True)
class PathsConfig:
    geo_coords_path: Path
    run_root: Path


@dataclass(slots=True)
class AdminConfig:
    enabled: bool
    input_template: str
    output_parquet: Path
    family_tag_base: str
    epochs: int
    batch_size: int
    embed_dim: int
    hidden_dim: int
    swap_noise_prob: float
    feature_dropout_prob: float
    mask_loss_weight: float
    lr: float
    muon_lr: float
    muon_momentum: float
    muon_optimizer_spec: str
    weight_decay: float
    grad_clip: float
    seed: int
    device: str
    use_amp: bool
    amp_dtype: str


@dataclass(slots=True)
class SpatialConfig:
    enabled: bool
    modality: str
    input_template: str
    output_parquet: Path
    family_tag_base: str
    epochs: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    max_tiles_per_bag: int
    embed_dim: int
    proj_dim: int
    tile_shape: tuple[int, int, int]
    spatial_widths: tuple[int, int, int]
    spatial_depths: tuple[int, int, int]
    stem_kernel_size: int | None
    stem_stride: int | None
    dw_kernel_size: int
    small_tile_cifar_stem: bool
    crop_scale_min: float
    crop_scale_max: float
    attn_hidden: int
    attn_dropout: float
    ssl_objective: str
    mae_mask_ratio: float
    mae_patch_size: int
    mae_decoder_hidden: int
    mae_loss_kind: str
    w_byol: float
    w_geo: float
    w_hsic: float
    use_byol_contrastive: bool
    byol_only_epochs: int
    geo_rank_k: int
    geo_rank_dmax_km: float
    geo_rank_softrank_strength: float
    geo_rank_normalize_rank: bool
    geo_rank_neighbor_backend: str
    geo_rank_same_county_eps_km: float
    optimizer_mode: str
    lr: float
    adamw_lr: float | None
    muon_lr: float | None
    muon_momentum: float
    muon_optimizer_spec: str
    weight_decay: float
    tau_base: float
    tau_final: float
    grad_clip: float
    seed: int
    device: str
    use_amp: bool
    amp_dtype: str
    flatten: bool
    radiance_sampling: bool
    radiance_weight_mode: str
    radiance_active_threshold: float
    radiance_weight_gamma: float
    radiance_min_weight: float
    radiance_clip_pctl: float
    radiance_samples_per_epoch: int
    tile_window_km: float


@dataclass(slots=True)
class EmbeddingsConfig:
    years: list[int]
    modalities: list[str]
    paths: PathsConfig
    admin: AdminConfig
    spatial: dict[str, SpatialConfig]

    @property
    def anchor_year(self) -> int:
        return int(self.years[0])

    def spatial_cfg(self, modality: str) -> SpatialConfig:
        key = str(modality).strip().lower()
        if key not in self.spatial:
            raise KeyError(f"unknown spatial modality={modality!r}; known={sorted(self.spatial)}")
        return self.spatial[key]


def _as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


def _require(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise KeyError(f"missing required config key: {key}")
    return section[key]


def _parse_int_tuple(section: dict[str, Any], key: str, expected_len: int) -> tuple[int, ...]:
    raw = list(_require(section, key))
    vals = tuple(int(v) for v in raw)
    if len(vals) != int(expected_len):
        raise ValueError(f"{key} must have length {expected_len}; got {vals}")
    return vals


def _parse_paths(section: dict[str, Any]) -> PathsConfig:
    return PathsConfig(
        geo_coords_path=_as_path(_require(section, "geo_coords_path")),
        run_root=_as_path(_require(section, "run_root")),
    )


def _parse_admin(section: dict[str, Any]) -> AdminConfig:
    return AdminConfig(
        enabled=bool(section.get("enabled", True)),
        input_template=str(_require(section, "input_template")),
        output_parquet=_as_path(_require(section, "output_parquet")),
        family_tag_base=str(section.get("family_tag_base", "tabdae_vime8")),
        epochs=int(section.get("epochs", 120)),
        batch_size=int(section.get("batch_size", 256)),
        embed_dim=int(section.get("embed_dim", 8)),
        hidden_dim=int(section.get("hidden_dim", 128)),
        swap_noise_prob=float(section.get("swap_noise_prob", 0.30)),
        feature_dropout_prob=float(section.get("feature_dropout_prob", 0.30)),
        mask_loss_weight=float(section.get("mask_loss_weight", 1.0)),
        lr=float(section.get("lr", 1e-3)),
        muon_lr=float(section.get("muon_lr", 1e-3)),
        muon_momentum=float(section.get("muon_momentum", 0.95)),
        muon_optimizer_spec=str(section.get("muon_optimizer_spec", "muon:Muon")),
        weight_decay=float(section.get("weight_decay", 1e-6)),
        grad_clip=float(section.get("grad_clip", 1.0)),
        seed=int(section.get("seed", 42)),
        device=str(section.get("device", "cuda")),
        use_amp=bool(section.get("use_amp", True)),
        amp_dtype=str(section.get("amp_dtype", "bfloat16")),
    )


def _parse_spatial(section: dict[str, Any], modality: str) -> SpatialConfig:
    return SpatialConfig(
        enabled=bool(section.get("enabled", True)),
        modality=str(modality),
        input_template=str(_require(section, "input_template")),
        output_parquet=_as_path(_require(section, "output_parquet")),
        family_tag_base=str(_require(section, "family_tag_base")),
        epochs=int(section.get("epochs", 20)),
        batch_size=int(section.get("batch_size", 16)),
        num_workers=int(section.get("num_workers", 4)),
        pin_memory=bool(section.get("pin_memory", True)),
        persistent_workers=bool(section.get("persistent_workers", True)),
        prefetch_factor=int(section.get("prefetch_factor", 2)),
        max_tiles_per_bag=int(section.get("max_tiles_per_bag", 32)),
        embed_dim=int(section.get("embed_dim", 64)),
        proj_dim=int(section.get("proj_dim", 128)),
        tile_shape=tuple(int(v) for v in section.get("tile_shape", [1, 32, 32])),
        spatial_widths=_parse_int_tuple(section, "spatial_widths", 3),
        spatial_depths=_parse_int_tuple(section, "spatial_depths", 3),
        stem_kernel_size=None if section.get("stem_kernel_size") in (None, "") else int(section.get("stem_kernel_size")),
        stem_stride=None if section.get("stem_stride") in (None, "") else int(section.get("stem_stride")),
        dw_kernel_size=int(section.get("dw_kernel_size", 5)),
        small_tile_cifar_stem=bool(section.get("small_tile_cifar_stem", True)),
        crop_scale_min=float(section.get("crop_scale_min", 0.7)),
        crop_scale_max=float(section.get("crop_scale_max", 1.0)),
        attn_hidden=int(section.get("attn_hidden", 128)),
        attn_dropout=float(section.get("attn_dropout", 0.0)),
        ssl_objective=str(section.get("ssl_objective", "byol")),
        mae_mask_ratio=float(section.get("mae_mask_ratio", 0.4)),
        mae_patch_size=int(section.get("mae_patch_size", 4)),
        mae_decoder_hidden=int(section.get("mae_decoder_hidden", 256)),
        mae_loss_kind=str(section.get("mae_loss_kind", "l1")),
        w_byol=float(section.get("w_byol", 1.0)),
        w_geo=float(section.get("w_geo", 0.1)),
        w_hsic=float(section.get("w_hsic", 0.0)),
        use_byol_contrastive=bool(section.get("use_byol_contrastive", True)),
        byol_only_epochs=int(section.get("byol_only_epochs", 0)),
        geo_rank_k=int(section.get("geo_rank_k", 32)),
        geo_rank_dmax_km=float(section.get("geo_rank_dmax_km", 125.0)),
        geo_rank_softrank_strength=float(section.get("geo_rank_softrank_strength", 1e-3)),
        geo_rank_normalize_rank=bool(section.get("geo_rank_normalize_rank", False)),
        geo_rank_neighbor_backend=str(section.get("geo_rank_neighbor_backend", "ball_tree")),
        geo_rank_same_county_eps_km=float(section.get("geo_rank_same_county_eps_km", 1e-6)),
        optimizer_mode=str(section.get("optimizer_mode", "muon_conv_linear")),
        lr=float(section.get("lr", 1e-4)),
        adamw_lr=None if section.get("adamw_lr") in (None, "") else float(section.get("adamw_lr")),
        muon_lr=None if section.get("muon_lr") in (None, "") else float(section.get("muon_lr")),
        muon_momentum=float(section.get("muon_momentum", 0.95)),
        muon_optimizer_spec=str(section.get("muon_optimizer_spec", "muon:Muon")),
        weight_decay=float(section.get("weight_decay", 1e-6)),
        tau_base=float(section.get("tau_base", 0.99)),
        tau_final=float(section.get("tau_final", 1.0)),
        grad_clip=float(section.get("grad_clip", 1.0)),
        seed=int(section.get("seed", 42)),
        device=str(section.get("device", "cuda")),
        use_amp=bool(section.get("use_amp", True)),
        amp_dtype=str(section.get("amp_dtype", "bfloat16")),
        flatten=bool(section.get("flatten", True)),
        radiance_sampling=bool(section.get("radiance_sampling", modality == "viirs")),
        radiance_weight_mode=str(section.get("radiance_weight_mode", "active_count")),
        radiance_active_threshold=float(section.get("radiance_active_threshold", 0.03)),
        radiance_weight_gamma=float(section.get("radiance_weight_gamma", 1.5)),
        radiance_min_weight=float(section.get("radiance_min_weight", 1e-6)),
        radiance_clip_pctl=float(section.get("radiance_clip_pctl", 99.5)),
        radiance_samples_per_epoch=int(section.get("radiance_samples_per_epoch", 0)),
        tile_window_km=float(section.get("tile_window_km", 24.0)),
    )


def load_config(path: str | Path) -> EmbeddingsConfig:
    with open(_as_path(path), "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("manifold config must be a mapping")
    years = [int(y) for y in list(_require(raw, "years"))]
    years = sorted(set(years))
    if not years:
        raise ValueError("years must be non-empty")
    expected = list(range(int(years[0]), int(years[-1]) + 1))
    if years != expected:
        raise ValueError(f"years must be contiguous; got {years}")
    modalities = [str(m).strip().lower() for m in list(_require(raw, "modalities"))]
    if not modalities:
        raise ValueError("modalities must be non-empty")
    spatial: dict[str, SpatialConfig] = {}
    for modality in modalities:
        if modality == "admin":
            continue
        if modality in RESERVED_MODALITY_KEYS:
            raise ValueError(f"reserved modality name: {modality!r}")
        spatial[modality] = _parse_spatial(dict(_require(raw, modality)), modality)
    return EmbeddingsConfig(
        years=years,
        modalities=modalities,
        paths=_parse_paths(dict(_require(raw, "paths"))),
        admin=_parse_admin(dict(_require(raw, "admin"))),
        spatial=spatial,
    )
