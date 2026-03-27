#!/usr/bin/env python3
#
# optim.py  Andrew Belles  Mar 27th, 2026
#
# Optimizer construction for manifold embedding models.
#

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn as nn


def resolve_muon_cls(spec: str):
    txt = str(spec).strip()
    if ":" in txt:
        mod_name, cls_name = txt.split(":", 1)
    else:
        mod_name, cls_name = txt, "Muon"
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


class LocalMuonMat(torch.optim.Optimizer):
    def __init__(self, params, *, muon_update_fn, lr=1e-3, weight_decay=0.0, momentum=0.95):
        defaults = dict(lr=float(lr), weight_decay=float(weight_decay), momentum=float(momentum))
        super().__init__(params, defaults)
        self._muon_update_fn = muon_update_fn

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            beta = float(group["momentum"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                upd = self._muon_update_fn(g, buf, beta=beta)
                if p.ndim == 4 and upd.ndim == 2:
                    upd = upd.view_as(p)
                if upd.shape != p.shape:
                    raise RuntimeError(f"Muon update shape mismatch for param {tuple(p.shape)} vs update {tuple(upd.shape)}")
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(upd, alpha=-lr)
        return loss


class LocalMuon(torch.optim.Optimizer):
    def __init__(self, params, *, muon_update_fn, lr=1e-3, weight_decay=0.0, momentum=0.95):
        defaults = dict(lr=float(lr), weight_decay=float(weight_decay), momentum=float(momentum))
        super().__init__(params, defaults)
        self._muon_update_fn = muon_update_fn

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            beta = float(group["momentum"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                upd = self._muon_update_fn(g, buf, beta=beta)
                if upd.shape != p.shape:
                    raise RuntimeError(f"Muon update shape mismatch for param {tuple(p.shape)} vs update {tuple(upd.shape)}")
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(upd, alpha=-lr)
        return loss


def _collect_trainable_params(module: nn.Module) -> list[nn.Parameter]:
    out: list[nn.Parameter] = []
    seen: set[int] = set()
    for p in module.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def _find_spatial_muon_params(module: nn.Module) -> list[nn.Parameter]:
    out: list[nn.Parameter] = []
    seen: set[int] = set()
    for _name, mod in module.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            w = getattr(mod, "weight", None)
            if isinstance(w, nn.Parameter) and w.requires_grad and id(w) not in seen:
                seen.add(id(w))
                out.append(w)
    return out


def _find_admin_muon_params(module: nn.Module) -> list[nn.Parameter]:
    out: list[nn.Parameter] = []
    seen: set[int] = set()
    for _name, mod in module.named_modules():
        if isinstance(mod, nn.Linear):
            w = getattr(mod, "weight", None)
            if isinstance(w, nn.Parameter) and w.requires_grad and id(w) not in seen:
                seen.add(id(w))
                out.append(w)
    return out


def build_spatial_optimizers(
    *,
    model: nn.Module,
    optimizer_mode: str,
    adamw_lr: float,
    muon_lr: float,
    weight_decay: float,
    muon_momentum: float,
    muon_optimizer_spec: str,
) -> tuple[list[torch.optim.Optimizer], dict[str, Any]]:
    mode = str(optimizer_mode).strip().lower()
    all_params = _collect_trainable_params(model)
    if mode == "adamw":
        opt = torch.optim.AdamW(all_params, lr=float(adamw_lr), weight_decay=float(weight_decay))
        return [opt], {"mode": "adamw", "muon_tensors": 0, "muon_params": 0, "adamw_params": int(sum(p.numel() for p in all_params))}
    if mode != "muon_conv_linear":
        raise ValueError(f"unsupported optimizer-mode: {mode!r}")
    muon_params = _find_spatial_muon_params(model)
    muon_ids = {id(p) for p in muon_params}
    adamw_params = [p for p in all_params if id(p) not in muon_ids]
    muon_cls = resolve_muon_cls(str(muon_optimizer_spec))
    muon_mod = importlib.import_module(muon_cls.__module__)
    muon_update_fn = getattr(muon_mod, "muon_update", None)
    if muon_update_fn is None:
        raise RuntimeError(f"muon module {muon_mod.__name__!r} does not expose muon_update")
    muon_opt = LocalMuonMat(muon_params, muon_update_fn=muon_update_fn, lr=float(muon_lr), weight_decay=0.0, momentum=float(muon_momentum))
    opts: list[torch.optim.Optimizer] = [muon_opt]
    if adamw_params:
        opts.append(torch.optim.AdamW(adamw_params, lr=float(adamw_lr), weight_decay=float(weight_decay)))
    return opts, {
        "mode": mode,
        "muon_tensors": int(len(muon_params)),
        "muon_params": int(sum(p.numel() for p in muon_params)),
        "adamw_params": int(sum(p.numel() for p in adamw_params)),
    }


def build_admin_optimizers(
    *,
    model: nn.Module,
    adamw_lr: float,
    muon_lr: float,
    weight_decay: float,
    muon_momentum: float,
    muon_optimizer_spec: str,
) -> tuple[list[torch.optim.Optimizer], dict[str, int]]:
    all_params = _collect_trainable_params(model)
    muon_params = _find_admin_muon_params(model)
    muon_ids = {id(p) for p in muon_params}
    adamw_params = [p for p in all_params if id(p) not in muon_ids]
    muon_cls = resolve_muon_cls(str(muon_optimizer_spec))
    muon_mod = importlib.import_module(muon_cls.__module__)
    muon_update_fn = getattr(muon_mod, "muon_update", None)
    if muon_update_fn is None:
        raise RuntimeError(f"muon module {muon_mod.__name__!r} does not expose muon_update")
    muon_opt = LocalMuon(muon_params, muon_update_fn=muon_update_fn, lr=float(muon_lr), weight_decay=0.0, momentum=float(muon_momentum))
    opts: list[torch.optim.Optimizer] = [muon_opt]
    if adamw_params:
        opts.append(torch.optim.AdamW(adamw_params, lr=float(adamw_lr), weight_decay=float(weight_decay)))
    return opts, {
        "muon_tensor_count": int(len(muon_params)),
        "muon_param_count": int(sum(p.numel() for p in muon_params)),
        "adamw_param_count": int(sum(p.numel() for p in adamw_params)),
    }
