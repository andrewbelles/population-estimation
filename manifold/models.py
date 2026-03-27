#!/usr/bin/env python3
#
# models.py  Andrew Belles  Mar 27th, 2026
#
# Model components for manifold embedding generation.
#

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, int(dim)))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, int(dim)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / gx.mean(dim=-1, keepdim=True).clamp_min(self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, dw_kernel_size: int = 5):
        super().__init__()
        d = int(dim)
        k = int(dw_kernel_size)
        if k <= 0 or (k % 2) == 0:
            raise ValueError(f"dw_kernel_size must be positive odd integer, got {dw_kernel_size}")
        p = int(k // 2)
        self.dw = nn.Conv2d(d, d, kernel_size=k, padding=p, groups=d, bias=True)
        self.ln = nn.LayerNorm(d, eps=1e-6)
        self.fc1 = nn.Linear(d, int(mlp_ratio * d))
        self.act = nn.GELU()
        self.grn = GRN(int(mlp_ratio * d))
        self.fc2 = nn.Linear(int(mlp_ratio * d), d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x + r


class SpatialConvNeXtV2Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int = 128,
        widths: tuple[int, int, int] = (32, 64, 128),
        depths: tuple[int, int, int] = (2, 2, 2),
        stem_kernel_size: int = 2,
        stem_stride: int = 2,
        dw_kernel_size: int = 5,
    ):
        super().__init__()
        w0, w1, w2 = [int(x) for x in widths]
        d0, d1, d2 = [int(x) for x in depths]
        self.stem = nn.Conv2d(int(in_channels), w0, kernel_size=int(stem_kernel_size), stride=int(stem_stride))
        self.stage0 = nn.Sequential(*[ConvNeXtV2Block(w0, dw_kernel_size=int(dw_kernel_size)) for _ in range(d0)])
        self.down1 = nn.Conv2d(w0, w1, kernel_size=2, stride=2)
        self.stage1 = nn.Sequential(*[ConvNeXtV2Block(w1, dw_kernel_size=int(dw_kernel_size)) for _ in range(d1)])
        self.down2 = nn.Conv2d(w1, w2, kernel_size=2, stride=2)
        self.stage2 = nn.Sequential(*[ConvNeXtV2Block(w2, dw_kernel_size=int(dw_kernel_size)) for _ in range(d2)])
        self.to_embed = nn.Conv2d(w2, int(embed_dim), kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stage0(h)
        h = self.down1(h)
        h = self.stage1(h)
        h = self.down2(h)
        h = self.stage2(h)
        h = self.to_embed(h)
        return F.adaptive_avg_pool2d(h, output_size=1).flatten(1)


class SpatialBYOLGeoModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        tile_shape: tuple[int, int, int],
        embed_dim: int,
        proj_dim: int,
        widths: tuple[int, int, int],
        depths: tuple[int, int, int],
        stem_kernel_size: int,
        stem_stride: int,
        dw_kernel_size: int,
        attn_hidden: int,
        attn_dropout: float,
        ssl_objective: str = "byol",
        mae_decoder_hidden: int = 256,
    ):
        super().__init__()
        del attn_hidden, attn_dropout
        d = int(embed_dim)
        p = int(proj_dim)
        self.ssl_objective = str(ssl_objective)
        self.tile_shape = tuple(int(x) for x in tile_shape)
        self.online_encoder = SpatialConvNeXtV2Encoder(
            in_channels=int(in_channels),
            embed_dim=d,
            widths=widths,
            depths=depths,
            stem_kernel_size=int(stem_kernel_size),
            stem_stride=int(stem_stride),
            dw_kernel_size=int(dw_kernel_size),
        )
        self.online_projector = nn.Sequential(
            nn.Linear(d, p),
            nn.BatchNorm1d(p, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Linear(p, p),
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(p, p),
            nn.BatchNorm1d(p, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Linear(p, p),
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p_t in self.target_encoder.parameters():
            p_t.requires_grad = False
        for p_t in self.target_projector.parameters():
            p_t.requires_grad = False
        flat_dim = int(np.prod(np.asarray(self.tile_shape, dtype=np.int64)))
        self.online_decoder = nn.Sequential(
            nn.Linear(d, int(mae_decoder_hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(mae_decoder_hidden), flat_dim),
        )

    def encode_online(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.online_encoder(x)
        p = self.online_projector(z)
        q = self.online_predictor(p)
        return z, p, q

    def reconstruct_online(self, z: torch.Tensor) -> torch.Tensor:
        rec = self.online_decoder(z)
        return rec.view(int(z.shape[0]), *self.tile_shape)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.target_encoder(x)
        p = self.target_projector(z)
        return z, p


class LightTabularDAE(nn.Module):
    def __init__(self, in_dim: int, *, embed_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        c = int(in_dim)
        d = int(embed_dim)
        h = int(hidden_dim)
        h2 = int(max(4, 2 * h))
        self.enc_fc1 = nn.Linear(c, h2)
        self.enc_bn1 = nn.BatchNorm1d(h2)
        self.enc_fc2 = nn.Linear(h2, h)
        self.enc_bn2 = nn.BatchNorm1d(h)
        self.enc_fc3 = nn.Linear(h, d)
        self.dec_fc1 = nn.Linear(d, h)
        self.dec_bn1 = nn.BatchNorm1d(h)
        self.dec_fc2 = nn.Linear(h, h2)
        self.dec_bn2 = nn.BatchNorm1d(h2)
        self.dec_fc3 = nn.Linear(h2, c)
        self.mask_fc1 = nn.Linear(d, h)
        self.mask_bn1 = nn.BatchNorm1d(h)
        self.mask_fc2 = nn.Linear(h, c)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.enc_bn1(self.enc_fc1(x)))
        z = F.gelu(self.enc_bn2(self.enc_fc2(h)))
        return self.enc_fc3(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.dec_bn1(self.dec_fc1(z)))
        h = F.gelu(self.dec_bn2(self.dec_fc2(h)))
        return self.dec_fc3(h)

    def mask_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.mask_bn1(self.mask_fc1(z)))
        return self.mask_fc2(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z), self.mask_logits(z)
