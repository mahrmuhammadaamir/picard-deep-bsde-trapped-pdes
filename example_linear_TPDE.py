from __future__ import annotations

import argparse
import copy
import math
import os
from dataclasses import asdict, dataclass, field
from math import lgamma
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time

try:
    from scipy.special import hyp2f1
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def terminal_g(x: torch.Tensor) -> torch.Tensor:
    """Dimension-normalized bounded radial initial condition."""
    scaled_r2 = x.square().sum(dim=1, keepdim=True) / float(x.shape[1])
    return (1.0 + torch.sin(scaled_r2)) / (1.0 + scaled_r2)


def mittag_leffler(alpha: float, z: float, tol: float = 1e-15) -> float:
    """Evaluate E_alpha(z) by its convergent power series (test-scale use)."""
    total = 1.0
    for n in range(1, 10000):
        magnitude = math.exp(n * math.log(abs(z)) - lgamma(alpha * n + 1.0)) if z else 0.0
        term = magnitude if z > 0.0 or n % 2 == 0 else -magnitude
        total_new = total + term
        if n > 5 and abs(term) <= tol * max(1.0, abs(total_new)):
            return total_new
        total = total_new
    raise RuntimeError("Mittag-Leffler series did not converge")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "mps":
        ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not ok:
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def yosida_moment(k: int, alpha: float, epsilon: float) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return 1.0
    beta_ratio = math.exp(
        log_beta(k - alpha, alpha) - log_beta(1.0 - alpha, alpha)
    )
    if epsilon >= 1.0:
        hypergeom = 1.0
    else:
        if not HAVE_SCIPY:
            raise ImportError("epsilon < 1 requires scipy")
        hypergeom = float(hyp2f1(k - 1, k - alpha, k, 1.0 - epsilon))
    value = beta_ratio * hypergeom
    if not math.isfinite(value) or value <= 0.0:
        raise FloatingPointError(f"invalid Yosida moment m_{k}={value}")
    return value


def build_moments(alpha: float, epsilon: float, max_k: int) -> List[float]:
    return [yosida_moment(k, alpha, epsilon) for k in range(1, max_k + 1)]


def choose_K(
    alpha: float,
    epsilon: float,
    ratio_tol: float,
    min_K: int,
    max_K: int,
) -> Tuple[int, float, bool]:
    moments = build_moments(alpha, epsilon, max_K + 1)
    previous: Optional[float] = None
    for k in range(1, max_K + 1):
        rho_k = moments[k] / moments[k - 1]
        if previous is not None and k >= min_K and abs(rho_k - previous) < ratio_tol:
            return k, rho_k, False
        previous = rho_k
    return max_K, float(previous), True


@dataclass
class YosidaScalars:
    K: int
    moments: List[float]
    rho: List[float]
    delta: List[float]

    @classmethod
    def build(cls, alpha: float, epsilon: float, K: int) -> "YosidaScalars":
        moments = build_moments(alpha, epsilon, K + 1)
        rho = [moments[j + 1] / moments[j] for j in range(K)]
        return cls(K, moments, rho, [1.0 + r for r in rho])


@dataclass
class Config:
    device: str = "auto"
    alpha: float = 0.5
    epsilon: float = 0.3
    d: int = 50
    T: float = 0.5
    x0: float = 0.0
    # Independent MC estimate and its sampling standard error.
    reference: Optional[float] = 0.72273583
    reference_se: float = 0.00021554

    ratio_tol: float = 1e-3
    min_K: int = 3
    max_K: int = 30
    K_override: Optional[int] = 5

    N: int = 100
    batch_size: int = 256
    validation_batch_size: int = 1024
    regression_paths: int = 1024
    collocation_size: int = 512

    picard_iterations: int = 30
    relaxation: float = 0.4
    picard_tol: float = 1e-3

    # Continue from the checkpoint saved by the previous run.
    resume: bool = False

    # Saved-history plotting controls.  plot_until=None uses every saved
    # Picard iteration.  plot_only=True loads the checkpoint, creates figures
    # and reports the selected iteration without performing more training.
    plot_until: Optional[int] = None
    plot_only: bool = False

    bsde_epochs_first: int = 6000
    bsde_epochs_later: int = 4000
    value_epochs: int = 4000
    auxiliary_epochs: int = 4000
    value_batch_size: int = 1024
    auxiliary_batch_size: int = 1024

    lr_bsde: float = 2e-4
    lr_value: float = 2e-4
    lr_auxiliary: float = 2e-4
    hidden: int = 128
    depth: int = 3
    field_hidden: int = 256
    field_depth: int = 4

    # Stage-2 value-field losses.  Theta and Z are frozen during this stage.
    # The local residual is divided by dt before weighting.
    lambda_field: float = 1.0
    lambda_local_bsde: float = 2.0
    lambda_initial: float = 1.0
    # Auxiliary target fit plus discrete-residual fidelity.  The latter keeps
    # the intended under-relaxation and becomes the original residual at the
    # Picard fixed point.
    lambda_aux_residual: float = 0.05

    antithetic: bool = True
    grad_clip: float = 10.0
    ema_decay: float = 0.999
    seed: int = 1234
    print_every: int = 250
    out_dir: str = "results_d50_normalized_radial_time_autodiff1"
    show_plots: bool = False

    K: int = field(init=False, default=0)
    final_rho: float = field(init=False, default=0.0)
    clipped: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0,1)")
        if not 0.0 < self.epsilon <= 1.0:
            raise ValueError("epsilon must be in (0,1]")
        if self.d < 1 or self.N < 2 or self.batch_size < 1:
            raise ValueError("d and batch_size must be positive; N must be >= 2")
        if self.value_epochs < 1 or self.auxiliary_epochs < 1:
            raise ValueError("value_epochs and auxiliary_epochs must be positive")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation must be in (0,1]")
        if self.plot_until is not None:
            if self.plot_until < 1:
                raise ValueError("plot_until must be >= 1 or None")
            self.plot_until = int(self.plot_until)
        if self.plot_only and not self.resume:
            raise ValueError("plot_only=True requires resume=True")
        if self.reference is None:
            raise ValueError("this benchmark requires an independent MC reference")
        if self.reference_se <= 0.0:
            raise ValueError("reference_se must be positive")
        if self.K_override is None:
            self.K, self.final_rho, self.clipped = choose_K(
                self.alpha, self.epsilon, self.ratio_tol, self.min_K, self.max_K
            )
        else:
            if self.K_override < 2:
                raise ValueError("K_override must be >= 2")
            self.K = int(self.K_override)
            moments = build_moments(self.alpha, self.epsilon, self.K + 1)
            self.final_rho = moments[-1] / moments[-2]


class FNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = []
        if depth <= 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers += [nn.Linear(input_dim, hidden), nn.Softsign()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.Softsign()]
            layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def zero_last_layer(self) -> None:
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AuxiliaryField(nn.Module):
    """A_2,...,A_K as a field of (tau,x), with A(T,x)=0 exactly."""

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.net = FNN(
            cfg.d + 1, cfg.K - 1, cfg.field_hidden, cfg.field_depth
        ).to(device)
        self.net.zero_last_layer()

    def forward(self, tau: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tau):
            tau_col = torch.full(
                (x.shape[0], 1), float(tau) / self.cfg.T,
                dtype=x.dtype, device=x.device,
            )
            factor = 1.0 - float(tau) / self.cfg.T
        else:
            tau_col = tau.reshape(-1, 1).to(dtype=x.dtype, device=x.device)
            if tau_col.shape[0] == 1 and x.shape[0] != 1:
                tau_col = tau_col.expand(x.shape[0], 1)
            tau_col = tau_col / self.cfg.T
            factor = 1.0 - tau_col
        return factor * self.net(torch.cat([tau_col, x], dim=1))


class ValueField(nn.Module):
    """Physical value U(tau,x), with U(T,x)=g(x) exactly."""

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.net = FNN(
            cfg.d + 1, 1, cfg.field_hidden, cfg.field_depth
        ).to(device)

    def forward(self, tau: torch.Tensor | float, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tau):
            tau_col = torch.full(
                (x.shape[0], 1), float(tau) / self.cfg.T,
                dtype=x.dtype, device=x.device,
            )
            factor = 1.0 - float(tau) / self.cfg.T
        else:
            tau_col = tau.reshape(-1, 1).to(dtype=x.dtype, device=x.device)
            if tau_col.shape[0] == 1 and x.shape[0] != 1:
                tau_col = tau_col.expand(x.shape[0], 1)
            tau_col = tau_col / self.cfg.T
            factor = 1.0 - tau_col
        correction = factor * self.net(torch.cat([tau_col, x], dim=1))
        return terminal_g(x) + correction


class PhysicalDeepBSDE(nn.Module):
    """One scalar physical BSDE; auxiliary fields are frozen in its driver."""

    def __init__(self, cfg: Config, delta_1: float, rho_1: float, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.delta_1 = delta_1
        self.rho_1 = rho_1
        self.brownian_scale = math.sqrt(2.0 * delta_1)
        self.theta = nn.Parameter(torch.tensor([0.0], device=device))
        self.register_buffer("theta_ema", torch.tensor([0.0], device=device))
        self.register_buffer("ema_initialized", torch.tensor(False, device=device))
        self.z_nets = nn.ModuleList([
            FNN(cfg.d, cfg.d, cfg.hidden, cfg.depth) for _ in range(cfg.N)
        ]).to(device)

    def _sample_dw(self, batch: int, sqrt_dt: float, device: torch.device) -> torch.Tensor:
        if self.cfg.antithetic and batch % 2 == 0:
            half = torch.randn(batch // 2, self.cfg.d, device=device) * sqrt_dt
            return torch.cat([half, -half], dim=0)
        return torch.randn(batch, self.cfg.d, device=device) * sqrt_dt

    def rollout(
        self,
        batch_size: int,
        auxiliary: AuxiliaryField,
        value_field: Optional[ValueField] = None,
        increments: Optional[torch.Tensor] = None,
        keep_paths: bool = False,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        device = self.theta.device
        dt = cfg.T / cfg.N
        sqrt_dt = math.sqrt(dt)
        x = torch.full((batch_size, cfg.d), cfg.x0, device=device)
        y = self.theta.expand(batch_size)
        x_path: List[torch.Tensor] = [x]
        y_path: List[torch.Tensor] = [y]
        field_losses: List[torch.Tensor] = []
        local_losses: List[torch.Tensor] = []

        if value_field is None:
            initial_loss = torch.zeros((), device=device)
        else:
            initial_field = value_field(0.0, x).squeeze(1)
            initial_loss = (initial_field - self.theta.detach()).square().mean()

        for n in range(cfg.N):
            dW = increments[n] if increments is not None else self._sample_dw(
                batch_size, sqrt_dt, device
            )
            with torch.no_grad():
                a2 = auxiliary(n * dt, x)[:, 0]
            z = self.z_nets[n](x)
            martingale = (z * dW).sum(dim=1)
            y_next = y + self.rho_1 * a2 * dt + martingale
            x_next = x + self.brownian_scale * dW
            if value_field is not None:
                u_now = value_field(n * dt, x).squeeze(1)
                u_next = value_field((n + 1) * dt, x_next).squeeze(1)

                # During stage 2 theta and Z are frozen.  U must fit both the
                # propagated Deep-BSDE path values and their one-step identity.
                local_residual = (
                    u_next - u_now - self.rho_1 * a2 * dt - martingale.detach()
                )
                local_losses.append(local_residual.square().mean() / dt)
                field_losses.append((y.detach() - u_now).square().mean())

            y, x = y_next, x_next
            if keep_paths:
                x_path.append(x)
                y_path.append(y)

        target = terminal_g(x).squeeze(1)
        terminal_mse = (y - target).square().mean()
        if value_field is None:
            field_mse = torch.zeros((), device=device)
            local_bsde_mse = torch.zeros((), device=device)
            loss = terminal_mse
        else:
            field_mse = torch.stack(field_losses).mean()
            local_bsde_mse = torch.stack(local_losses).mean()
            loss = (
                cfg.lambda_field * field_mse
                + cfg.lambda_local_bsde * local_bsde_mse
                + cfg.lambda_initial * initial_loss
            )
        result = {
            "loss": loss,
            "terminal_mse": terminal_mse,
            "field_mse": field_mse,
            "local_bsde_mse": local_bsde_mse,
            "initial_mse": initial_loss,
            "x_terminal": x,
            "y_terminal": y,
        }
        if keep_paths:
            result["x_path"] = torch.stack(x_path, dim=0)
            result["y_path"] = torch.stack(y_path, dim=0)
        return result


def fixed_increments(cfg: Config, batch: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    sqrt_dt = math.sqrt(cfg.T / cfg.N)
    if cfg.antithetic and batch % 2 == 0:
        half = torch.randn(cfg.N, batch // 2, cfg.d, generator=gen) * sqrt_dt
        values = torch.cat([half, -half], dim=1)
    else:
        values = torch.randn(cfg.N, batch, cfg.d, generator=gen) * sqrt_dt
    return values.to(device)


def train_physical_bsde(
    cfg: Config,
    model: PhysicalDeepBSDE,
    auxiliary: AuxiliaryField,
    epochs: int,
    validation_dw: torch.Tensor,
    picard_index: int,
) -> Dict[str, float]:
    # The Picard driver changes between outer iterations, so the reported EMA
    # must describe the current physical solve rather than average incompatible
    # Picard problems.  The trainable theta and Z networks remain warm-started.
    with torch.no_grad():
        model.theta_ema.copy_(model.theta)
        model.ema_initialized.fill_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_bsde)
    for epoch in range(1, epochs + 1):
        out = model.rollout(cfg.batch_size, auxiliary)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        with torch.no_grad():
            if not bool(model.ema_initialized.item()):
                model.theta_ema.copy_(model.theta)
                model.ema_initialized.fill_(True)
            else:
                model.theta_ema.mul_(cfg.ema_decay).add_(
                    model.theta, alpha=1.0 - cfg.ema_decay
                )
        if epoch == 1 or epoch % cfg.print_every == 0 or epoch == epochs:
            with torch.no_grad():
                val = model.rollout(
                    cfg.validation_batch_size, auxiliary, increments=validation_dw
                )
            print(
                f"  Picard {picard_index:02d} stage-1 epoch {epoch:5d}/{epochs} | "
                f"terminal={out['terminal_mse'].item():.3e} "
                f"val_terminal={val['terminal_mse'].item():.3e} "
                f"u={model.theta.item():+.7f} ema={model.theta_ema.item():+.7f}"
            )
    with torch.no_grad():
        val = model.rollout(
            cfg.validation_batch_size, auxiliary, increments=validation_dw
        )
    return {
        "train_terminal": float(out["terminal_mse"].item()),
        "validation_terminal": float(val["terminal_mse"].item()),
    }


def train_value_field_stage(
    cfg: Config,
    model: PhysicalDeepBSDE,
    auxiliary: AuxiliaryField,
    value_field: ValueField,
    validation_dw: torch.Tensor,
    picard_index: int,
) -> Dict[str, float]:
    """Stage 2: freeze the genuine Deep BSDE and train only U(tau,x)."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(value_field.parameters(), lr=cfg.lr_value)
    try:
        for epoch in range(1, cfg.value_epochs + 1):
            out = model.rollout(cfg.batch_size, auxiliary, value_field)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(value_field.parameters(), cfg.grad_clip)
            optimizer.step()
            if epoch == 1 or epoch % cfg.print_every == 0 or epoch == cfg.value_epochs:
                with torch.no_grad():
                    val = model.rollout(
                        cfg.validation_batch_size, auxiliary, value_field,
                        increments=validation_dw,
                    )
                print(
                    f"  Picard {picard_index:02d} stage-2 epoch "
                    f"{epoch:5d}/{cfg.value_epochs} | "
                    f"local={out['local_bsde_mse'].item():.3e} "
                    f"field={out['field_mse'].item():.3e} "
                    f"initial={out['initial_mse'].item():.3e} "
                    f"val_local={val['local_bsde_mse'].item():.3e} "
                    f"val_field={val['field_mse'].item():.3e}"
                )
        with torch.no_grad():
            val = model.rollout(
                cfg.validation_batch_size, auxiliary, value_field,
                increments=validation_dw,
            )
    finally:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    return {
        "train_local": float(out["local_bsde_mse"].item()),
        "train_field": float(out["field_mse"].item()),
        "train_initial": float(out["initial_mse"].item()),
        "validation_local": float(val["local_bsde_mse"].item()),
        "validation_field": float(val["field_mse"].item()),
        "validation_initial": float(val["initial_mse"].item()),
    }


def collect_path_data(
    cfg: Config,
    model: PhysicalDeepBSDE,
    auxiliary: AuxiliaryField,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    increments = fixed_increments(cfg, cfg.regression_paths, seed, device)
    with torch.no_grad():
        out = model.rollout(
            cfg.regression_paths, auxiliary, increments=increments,
            keep_paths=True
        )
    x_path = out["x_path"].detach()
    return x_path.reshape(-1, cfg.d)


def build_auxiliary_targets(
    cfg: Config,
    scalars: YosidaScalars,
    old_auxiliary: AuxiliaryField,
    value_field: ValueField,
    x_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Build relaxed A targets on a fixed-x tensor product grid."""
    device = x_points.device
    dt = cfg.T / cfg.N
    tau_grid = torch.linspace(0.0, cfg.T, cfg.N + 1, device=device)
    values_u: List[torch.Tensor] = []
    derivatives_u: List[torch.Tensor] = []
    old_values: List[torch.Tensor] = []
    point_count = x_points.shape[0]

    # Differentiate the learned field only with respect to time.  This avoids
    # the 1/dt amplification of field errors produced by finite differences
    # and still requires no spatial derivative or Hessian.
    for tau_value in tau_grid:
        tau_batch = (
            tau_value.detach()
            .expand(point_count, 1)
            .clone()
            .requires_grad_(True)
        )
        u_value = value_field(tau_batch, x_points)
        du_value = torch.autograd.grad(
            outputs=u_value.sum(),
            inputs=tau_batch,
            create_graph=False,
            retain_graph=False,
        )[0]
        values_u.append(u_value.detach().squeeze(1))
        derivatives_u.append(du_value.detach().squeeze(1))
        with torch.no_grad():
            old_values.append(old_auxiliary(tau_value, x_points))

    u_grid = torch.stack(values_u, dim=0)             # (N+1,M)
    du = torch.stack(derivatives_u, dim=0)            # (N+1,M)
    old_grid = torch.stack(old_values, dim=0)         # (N+1,M,K-1)

    rho1 = scalars.rho[0]
    delta1 = scalars.delta[0]
    # L_n is evaluated with the frozen A_2^m, exactly as in the physical solve.
    laplacian = (rho1 * old_grid[:, :, 0] - du) / delta1

    kaux = cfg.K - 1
    B = torch.zeros(kaux, kaux, device=device)
    for j in range(kaux - 1):
        B[j, j + 1] = scalars.rho[j + 1]
    B[-1, -1] = scalars.rho[-1]
    delta_aux = torch.tensor(scalars.delta[1:], device=device)
    inverse = torch.linalg.inv(torch.eye(kaux, device=device) + dt * B)

    raw = torch.zeros_like(old_grid)
    # raw[N]=0 is the exact terminal condition.
    for n in range(cfg.N - 1, -1, -1):
        rhs = raw[n + 1] + dt * laplacian[n, :, None] * delta_aux[None, :]
        raw[n] = rhs @ inverse.T

    relaxed = (1.0 - cfg.relaxation) * old_grid + cfg.relaxation * raw
    relaxed[-1].zero_()
    change = torch.linalg.vector_norm(raw - old_grid) / (
        torch.linalg.vector_norm(raw) + 1e-12
    )

    # Residual of the raw backward-Euler solve before neural fitting.
    raw_residual = (
        (raw[1:] - raw[:-1]) / dt
        - torch.einsum("ij,nmj->nmi", B, raw[:-1])
        + laplacian[:-1, :, None] * delta_aux[None, None, :]
    )
    diagnostics = {
        "picard_aux_change_raw": float(change.item()),
        "laplacian_rms": float(torch.sqrt(laplacian.square().mean()).item()),
        "raw_aux_residual_rms": float(torch.sqrt(raw_residual.square().mean()).item()),
    }
    return tau_grid, relaxed, laplacian.detach(), diagnostics


def fit_auxiliary_field(
    cfg: Config,
    scalars: YosidaScalars,
    auxiliary: AuxiliaryField,
    tau_grid: torch.Tensor,
    x_points: torch.Tensor,
    targets: torch.Tensor,
    laplacian: torch.Tensor,
) -> Dict[str, float]:
    """Fit relaxed targets and their discrete temporal residual."""
    dt = cfg.T / cfg.N
    device = x_points.device
    point_count = x_points.shape[0]
    B = torch.zeros(cfg.K - 1, cfg.K - 1, device=device)
    for j in range(cfg.K - 2):
        B[j, j + 1] = scalars.rho[j + 1]
    B[-1, -1] = scalars.rho[-1]
    delta_aux = torch.tensor(scalars.delta[1:], device=device)
    optimizer = torch.optim.Adam(auxiliary.parameters(), lr=cfg.lr_auxiliary)
    final_value = float("nan")
    final_residual = float("nan")
    for _ in range(cfg.auxiliary_epochs):
        batch = min(cfg.auxiliary_batch_size, cfg.N * point_count)
        n_idx = torch.randint(0, cfg.N, (batch,), device=device)
        x_idx = torch.randint(0, point_count, (batch,), device=device)
        xb = x_points[x_idx]
        tau0 = tau_grid[n_idx]
        tau1 = tau_grid[n_idx + 1]
        a0 = auxiliary(tau0, xb)
        a1 = auxiliary(tau1, xb)

        target0 = targets[n_idx, x_idx]
        target1 = targets[n_idx + 1, x_idx]
        value_loss = 0.5 * (
            (a0 - target0).square().mean()
            + (a1 - target1).square().mean()
        )
        lap = laplacian[n_idx, x_idx]
        residual = (
            (a1 - a0) / dt
            - a0 @ B.T
            + lap[:, None] * delta_aux[None, :]
        )
        # Preserve under-relaxation: match the discrete residual of the
        # relaxed target rather than forcing the network directly to the raw
        # (unrelaxed) auxiliary solution.  At a Picard fixed point this target
        # residual itself tends to zero.
        target_residual = (
            (target1 - target0) / dt
            - target0 @ B.T
            + lap[:, None] * delta_aux[None, :]
        )
        residual_error = residual - target_residual
        residual_loss = residual_error.square().mean()
        loss = value_loss + cfg.lambda_aux_residual * residual_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(auxiliary.parameters(), cfg.grad_clip)
        optimizer.step()
        final_value = float(value_loss.item())
        final_residual = float(torch.sqrt(residual_loss).item())
    return {
        "value_mse": final_value,
        "training_residual_rms": final_residual,
    }


def fitted_auxiliary_residual(
    cfg: Config,
    scalars: YosidaScalars,
    auxiliary: AuxiliaryField,
    old_auxiliary: AuxiliaryField,
    value_field: ValueField,
    x_points: torch.Tensor,
) -> float:
    device = x_points.device
    dt = cfg.T / cfg.N
    rho1, delta1 = scalars.rho[0], scalars.delta[0]
    delta_aux = torch.tensor(scalars.delta[1:], device=device)
    B = torch.zeros(cfg.K - 1, cfg.K - 1, device=device)
    for j in range(cfg.K - 2):
        B[j, j + 1] = scalars.rho[j + 1]
    B[-1, -1] = scalars.rho[-1]
    residuals: List[torch.Tensor] = []
    point_count = x_points.shape[0]
    for n in range(cfg.N):
        t0, t1 = n * dt, (n + 1) * dt
        with torch.no_grad():
            a0, a1 = auxiliary(t0, x_points), auxiliary(t1, x_points)
            old_a2 = old_auxiliary(t0, x_points)[:, 0]

        tau_batch = torch.full(
            (point_count, 1), t0, dtype=x_points.dtype,
            device=device, requires_grad=True,
        )
        u0 = value_field(tau_batch, x_points)
        du = torch.autograd.grad(
            outputs=u0.sum(), inputs=tau_batch,
            create_graph=False, retain_graph=False,
        )[0].detach().squeeze(1)
        lap = (rho1 * old_a2 - du) / delta1
        res = (a1 - a0) / dt - a0 @ B.T + lap[:, None] * delta_aux
        residuals.append(res)
    return float(torch.sqrt(torch.cat(residuals).square().mean()).item())


def save_history(cfg: Config, history: Dict[str, List[float]]) -> None:
    available = len(history.get("u", []))
    if available == 0:
        raise ValueError("The checkpoint history contains no Picard solutions")
    requested = available if cfg.plot_until is None else cfg.plot_until
    count = min(requested, available)
    if requested > available:
        print(
            f"plot_until={requested} requested, but only {available} saved "
            f"iterations exist; plotting all {available}."
        )
    x = np.arange(1, count + 1)
    print(f"Plotting saved Picard iterations 1 through {count}.")

    for key, label, logscale in [
        ("relative_error", "Relative error", True),
    ]:
        if key not in history or len(history[key]) < count:
            print(f"Skipping unavailable/incomplete diagnostic: {key}")
            continue
        plt.figure(figsize=(7, 4))
        values = np.asarray(history[key][:count], dtype=float)
        if logscale:
            plt.semilogy(x, np.maximum(values, 1e-16), marker="o")
        else:
            plt.plot(x, values, marker="o")
        plt.xlabel("Picard iteration")
        plt.ylabel(label)
        plt.xticks(np.arange(1, count + 1, max(1, math.ceil(count / 15))))
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(
            os.path.join(cfg.out_dir, f"{key}_first_{count}.png"), dpi=220
        )
        plt.close()

    u_plot = np.asarray(history["u"][:count], dtype=float)
    ema_plot = np.asarray(history["u_ema"][:count], dtype=float)
    plt.figure(figsize=(7, 4))
    plt.plot(x, u_plot, marker="o", linewidth=2, label=r"$u$")
    plt.plot(x, ema_plot, marker="o", linewidth=2, label=r"$u$ EMA")
    plt.axhline(
        cfg.reference, color="tab:green", linestyle="--", linewidth=2,
        label="MC solution",
    )
    plt.xticks(x)
    plt.xlabel("Picard iteration")
    plt.ylabel(r"$u(T,x_0)$")
    plt.xticks(np.arange(1, count + 1, max(1, math.ceil(count / 15))))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(cfg.out_dir, f"picard_solution_first_{count}.png"),
        dpi=220,
    )
    plt.close()

    # A zoomed companion containing the same three curves and no confidence
    # band.  This makes small late-iteration differences visible.
    all_values = np.concatenate([
        u_plot, ema_plot, np.asarray([cfg.reference], dtype=float)
    ])
    lower, upper = float(all_values.min()), float(all_values.max())
    margin = 0.08 * max(upper - lower, 1e-8)
    plt.figure(figsize=(7, 4))
    plt.plot(x, u_plot, marker="o", linewidth=2, label=r"$u$")
    plt.plot(x, ema_plot, marker="o", linewidth=2, label=r"$u$ EMA")
    plt.axhline(
        cfg.reference, color="tab:green", linestyle="--", linewidth=2,
        label="MC solution",
    )
    plt.ylim(lower - margin, upper + margin)
    plt.xticks(x)
    plt.xlabel("Picard iteration")
    plt.ylabel(r"$u(T,x_0)$ (zoomed)")
    plt.xticks(np.arange(1, count + 1, max(1, math.ceil(count / 15))))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            cfg.out_dir, f"picard_solution_first_{count}_zoomed.png"
        ),
        dpi=220,
    )
    plt.close()


def train(cfg: Config) -> Dict[str, List[float]]:
    os.makedirs(cfg.out_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        cfg.out_dir,
        "checkpoint.pt",
    )

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    scalars = YosidaScalars.build(cfg.alpha, cfg.epsilon, cfg.K)
    print("\n" + "=" * 76)
    print("Picard-coupled normalized-Yosida Deep BSDE: radial MC benchmark")
    print(f"device={device}, d={cfg.d}, alpha={cfg.alpha}, epsilon={cfg.epsilon}")
    print(f"K={cfg.K}, N={cfg.N}, delta_1={scalars.delta[0]:.8f}")
    print(f"Picard iterations={cfg.picard_iterations}, relaxation={cfg.relaxation}")
    ci_half_width = 1.96 * cfg.reference_se
    print(f"independent MC reference = {cfg.reference:.10f}")
    print(f"MC standard error        = {cfg.reference_se:.10f}")
    print(
        f"MC 95% confidence interval= "
        f"[{cfg.reference-ci_half_width:.10f}, {cfg.reference+ci_half_width:.10f}]"
    )
    print("=" * 76)

    auxiliary = AuxiliaryField(cfg, device)
    physical = PhysicalDeepBSDE(
        cfg, scalars.delta[0], scalars.rho[0], device
    )
    value_field = ValueField(cfg, device)
    validation_dw = fixed_increments(
        cfg, cfg.validation_batch_size, cfg.seed + 999, device
    )
    history: Dict[str, List[float]] = {key: [] for key in [
        "u", "u_ema", "relative_error", "mc_z_score", "inside_mc_ci",
        "train_loss", "validation_loss",
        "physical_local_loss", "physical_field_loss", "auxiliary_fit_loss",
        "auxiliary_training_residual", "picard_aux_change",
        "laplacian_rms", "raw_aux_residual", "fitted_aux_residual",
    ]}

    previous_u: Optional[float] = None
    start_picard = 0

    if cfg.resume and os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")

        # weights_only=False is required because the checkpoint also contains
        # configuration dictionaries and training history.
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
            )

        # Check that the saved neural-network dimensions match the current run.
        saved_config = checkpoint.get("config", {})

        for name in ["d", "alpha", "epsilon", "T", "N", "K"]:
            if name in saved_config:
                old_value = saved_config[name]
                new_value = getattr(cfg, name)

                if old_value != new_value:
                    raise ValueError(
                        f"Checkpoint mismatch for {name}: "
                        f"saved={old_value}, current={new_value}"
                    )

        physical.load_state_dict(checkpoint["physical_state"])
        value_field.load_state_dict(checkpoint["value_state"])
        auxiliary.load_state_dict(checkpoint["auxiliary_state"])

        history = checkpoint["history"]
        start_picard = int(checkpoint["picard_iteration"])

        if history["u"]:
            previous_u = float(history["u"][-1])

        print(
            f"Checkpoint loaded successfully.\n"
            f"Completed Picard iterations: {start_picard}\n"
            f"Requested total iterations: {cfg.picard_iterations}"
        )

    elif cfg.resume:
        if cfg.plot_only:
            raise FileNotFoundError(
                f"plot_only requested, but no checkpoint exists at "
                f"{checkpoint_path}"
            )
        print(
            f"\nNo checkpoint found at {checkpoint_path}.\n"
            "Training will start from Picard iteration 1."
        )

    if cfg.plot_only:
        save_history(cfg, history)
        report_iteration = min(
            len(history["u"]),
            len(history["u"]) if cfg.plot_until is None else cfg.plot_until,
        )
        j = report_iteration - 1
        raw_error = abs(history["u"][j] - cfg.reference) / max(
            abs(cfg.reference), 1e-12
        )
        ema_error = abs(history["u_ema"][j] - cfg.reference) / max(
            abs(cfg.reference), 1e-12
        )
        print(f"\nSAVED REPORT AT PICARD ITERATION {report_iteration}")
        print(f"u(T,x0)              = {history['u'][j]:.10f}")
        print(f"u(T,x0), EMA         = {history['u_ema'][j]:.10f}")
        print(f"MC solution          = {cfg.reference:.10f}")
        print(f"relative MC diff, raw= {100*raw_error:.5f}%")
        print(f"relative MC diff, EMA= {100*ema_error:.5f}%")
        if "fitted_aux_residual" in history:
            print(
                f"auxiliary residual   = "
                f"{history['fitted_aux_residual'][j]:.6e}"
            )
        return history

    if start_picard >= cfg.picard_iterations:
        print(
            f"\nThe checkpoint already contains {start_picard} iterations, "
            f"which is not smaller than picard_iterations="
            f"{cfg.picard_iterations}."
        )
        print("Increase picard_iterations to continue.")
    else:
        print(
            f"\nStarting at Picard iteration "
            f"{start_picard + 1}/{cfg.picard_iterations}."
        )

    for m in range(start_picard, cfg.picard_iterations):
        print(f"\n--- OUTER PICARD ITERATION {m + 1}/{cfg.picard_iterations} ---")
        old_auxiliary = copy.deepcopy(auxiliary).to(device).eval()
        for p in old_auxiliary.parameters():
            p.requires_grad_(False)

        epochs = cfg.bsde_epochs_first if m == 0 else cfg.bsde_epochs_later
        physical_metrics = train_physical_bsde(
            cfg, physical, old_auxiliary, epochs, validation_dw, m + 1
        )
        value_metrics = train_value_field_stage(
            cfg, physical, old_auxiliary, value_field, validation_dw, m + 1
        )
        x_data = collect_path_data(
            cfg, physical, old_auxiliary, device, cfg.seed + 10000
        )

        # Select spatial points from the actual Brownian cloud, then keep each
        # selected x fixed throughout the auxiliary backward solve.
        count = x_data.shape[0]
        point_count = min(cfg.collocation_size, count)
        idx = torch.linspace(
            0, count - 1, steps=point_count, device=device
        ).round().long()
        x_points = x_data[idx].detach()
        tau_grid, targets, laplacian, diagnostics = build_auxiliary_targets(
            cfg, scalars, old_auxiliary, value_field, x_points
        )
        auxiliary_metrics = fit_auxiliary_field(
            cfg, scalars, auxiliary, tau_grid, x_points, targets, laplacian
        )
        fitted_residual = fitted_auxiliary_residual(
            cfg, scalars, auxiliary, old_auxiliary, value_field, x_points
        )

        u = float(physical.theta.item())
        u_ema = float(physical.theta_ema.item())
        relative_error = abs(u_ema - cfg.reference) / max(abs(cfg.reference), 1e-12)
        mc_z_score = (u_ema - cfg.reference) / cfg.reference_se
        inside_mc_ci = float(
            cfg.reference - ci_half_width <= u_ema <= cfg.reference + ci_half_width
        )
        u_change = (
            float("inf") if previous_u is None
            else abs(u - previous_u) / (abs(u) + 1e-12)
        )
        previous_u = u

        values = {
            "u": u,
            "u_ema": u_ema,
            "relative_error": relative_error,
            "mc_z_score": mc_z_score,
            "inside_mc_ci": inside_mc_ci,
            "train_loss": physical_metrics["train_terminal"],
            "validation_loss": physical_metrics["validation_terminal"],
            "physical_local_loss": value_metrics["validation_local"],
            "physical_field_loss": value_metrics["validation_field"],
            "auxiliary_fit_loss": auxiliary_metrics["value_mse"],
            "auxiliary_training_residual": auxiliary_metrics["training_residual_rms"],
            "picard_aux_change": diagnostics["picard_aux_change_raw"],
            "laplacian_rms": diagnostics["laplacian_rms"],
            "raw_aux_residual": diagnostics["raw_aux_residual_rms"],
            "fitted_aux_residual": fitted_residual,
        }
        for key, value in values.items():
            history[key].append(value)
        print(
            f"Picard {m + 1:02d} summary | u={u:+.8f} ema={u_ema:+.8f} "
            f"MC_rel_diff={100*relative_error:.3f}% "
            f"MC_z={mc_z_score:+.2f} inside_CI={bool(inside_mc_ci)} "
            f"u_change={u_change:.3e}\n"
            f"  BSDE terminal train/val={physical_metrics['train_terminal']:.3e}/"
            f"{physical_metrics['validation_terminal']:.3e} "
            f"local={value_metrics['validation_local']:.3e} "
            f"field={value_metrics['validation_field']:.3e} "
            f"initial={value_metrics['validation_initial']:.3e}\n"
            f"  aux_fit={auxiliary_metrics['value_mse']:.3e} "
            f"aux_res_fit_error={auxiliary_metrics['training_residual_rms']:.3e}\n"
            f"  Picard_A_change={diagnostics['picard_aux_change_raw']:.3e} "
            f"|L|_rms={diagnostics['laplacian_rms']:.3e} "
            f"raw_A_res={diagnostics['raw_aux_residual_rms']:.3e} "
            f"fitted_A_res={fitted_residual:.3e}"
        )

        checkpoint = {
            "picard_iteration": m + 1,
            "physical_state": physical.state_dict(),
            "value_state": value_field.state_dict(),
            "auxiliary_state": auxiliary.state_dict(),
            "history": history,
            "config": asdict(cfg),
            "scalars": asdict(scalars),
            "mc_reference": cfg.reference,
            "mc_reference_se": cfg.reference_se,
        }
        torch.save(checkpoint, os.path.join(cfg.out_dir, "checkpoint.pt"))

        if (
            m >= 1
            and u_change < cfg.picard_tol
            and diagnostics["picard_aux_change_raw"] < cfg.picard_tol
        ):
            print("Picard stopping tolerances reached.")
            break

    save_history(cfg, history)
    print("\nFINAL REPORT")
    print(f"u(T,x0)              = {history['u'][-1]:.10f}")
    print(f"u(T,x0), EMA         = {history['u_ema'][-1]:.10f}")
    print(f"MC reference         = {cfg.reference:.10f}")
    print(f"MC standard error    = {cfg.reference_se:.10f}")
    print(
        f"MC 95% CI            = [{cfg.reference-ci_half_width:.10f}, "
        f"{cfg.reference+ci_half_width:.10f}]"
    )
    raw_relative_error = abs(history["u"][-1] - cfg.reference) / max(
        abs(cfg.reference), 1e-12
    )
    raw_z_score = (history["u"][-1] - cfg.reference) / cfg.reference_se
    raw_inside = (
        cfg.reference - ci_half_width
        <= history["u"][-1]
        <= cfg.reference + ci_half_width
    )
    print(f"relative MC diff, raw= {100*raw_relative_error:.5f}%")
    print(f"MC z-score, raw      = {raw_z_score:+.5f}")
    print(f"raw inside MC 95% CI = {raw_inside}")
    print(f"relative MC diff, EMA= {100*history['relative_error'][-1]:.5f}%")
    print(f"MC z-score, EMA      = {history['mc_z_score'][-1]:+.5f}")
    print(f"EMA inside MC 95% CI = {bool(history['inside_mc_ci'][-1])}")
    print("Note: MC z-scores use MC uncertainty only; estimate Deep-BSDE")
    print("      uncertainty from independent training seeds.")
    print(f"auxiliary residual   = {history['fitted_aux_residual'][-1]:.6e}")
    return history


def parse_args(argv: Optional[List[str]] = None) -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=defaults.device)
    for name, typ in [
        ("alpha", float), ("epsilon", float), ("d", int), ("T", float),
        ("x0", float), ("reference", float), ("reference_se", float),
        ("K_override", int), ("plot_until", int),
        ("N", int), ("batch_size", int), ("validation_batch_size", int),
        ("regression_paths", int), ("collocation_size", int),
        ("picard_iterations", int), ("relaxation", float),
        ("picard_tol", float), ("bsde_epochs_first", int),
        ("bsde_epochs_later", int), ("value_epochs", int),
        ("auxiliary_epochs", int), ("value_batch_size", int),
        ("auxiliary_batch_size", int), ("lr_bsde", float),
        ("lr_value", float), ("lr_auxiliary", float), ("hidden", int),
        ("depth", int), ("field_hidden", int), ("field_depth", int),
        ("lambda_field", float), ("lambda_local_bsde", float),
        ("lambda_initial", float), ("lambda_aux_residual", float),
        ("grad_clip", float), ("ema_decay", float), ("seed", int),
        ("print_every", int), ("out_dir", str),
    ]:
        parser.add_argument(f"--{name}", type=typ, default=getattr(defaults, name))
    parser.add_argument("--no_antithetic", action="store_true")
    parser.add_argument("--show_plots", action="store_true")
    parser.add_argument("--plot_only", action="store_true", default=defaults.plot_only)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no_resume", dest="resume", action="store_false")
    parser.set_defaults(resume=defaults.resume)
    args, unknown = parser.parse_known_args(argv)
    if unknown and argv is None:
        print(f"[ignoring notebook arguments: {unknown}]")
    values = vars(args)
    values["antithetic"] = not values.pop("no_antithetic")
    return Config(**values)


def run(**overrides) -> Dict[str, List[float]]:
    return train(Config(**overrides))

def synchronize_device(device: torch.device) -> None:
    """Synchronize asynchronous GPU operations for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
def main() -> None:
    cfg = parse_args()
    device = get_device(cfg.device)

    synchronize_device(device)
    start_time = time.perf_counter()

    train(cfg)

    synchronize_device(device)
    elapsed_seconds = time.perf_counter() - start_time

    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\nCOMPUTATIONAL TIME")
    print(f"total seconds = {elapsed_seconds:.2f}")
    print(
        f"wall time     = "
        f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
    )


if __name__ == "__main__":
    main()
