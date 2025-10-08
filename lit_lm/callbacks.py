import copy
from argparse import Namespace
import tempfile
from ema_pytorch import EMA

import torch
from torch import nn
from pytorch_lightning import Callback
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning.loggers import WandbLogger
import wandb


def normalize_lists(obj):
    """
    Recursively replace all Python lists and tuples with dictionaries
    keyed by index.

    This ensures every element in a sequence is exposed as an individual
    key-value pair (e.g., "betas.0", "betas.1") when flattened later,
    instead of being hidden inside a JSON array.
    """
    if isinstance(obj, (list, tuple)):
        # Replace sequence with dict {"0": val0, "1": val1, ...}
        return {str(i): normalize_lists(v) for i, v in enumerate(obj)}
    elif isinstance(obj, dict):
        # Recurse into nested dictionaries
        return {k: normalize_lists(v) for k, v in obj.items()}
    else:
        # Base case: return unchanged scalar or object
        return obj


def ns_to_dict(obj):
    """
    Recursively convert argparse.Namespace objects into plain Python dicts.

    Handles:
      - Namespace → dict
      - dict → dict (recursively processed)
      - list/tuple → list of dicts
      - scalars → unchanged
    """
    if isinstance(obj, Namespace):
        return {k: ns_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: ns_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ns_to_dict(v) for v in obj]
    else:
        return obj


class LoggerSaveConfigCallback(SaveConfigCallback):
    """
    Custom callback to log Hydra/LightningCLI config into Weights & Biases (W&B).

    Workflow:
      1. Convert config (Namespace/DictConfig) into a plain dict.
      2. Normalize all sequences (lists/tuples) into index-keyed dicts.
      3. Flatten nested dicts into dot-separated keys.
      4. Push flattened config into W&B for searchability.
      5. Save full unflattened config as a YAML artifact for reproducibility.
    """

    def save_config(self, trainer, pl_module, stage: str) -> None:
        if isinstance(trainer.logger, WandbLogger):
            # Step 1: normalize Namespace → dict
            cfg_dict = ns_to_dict(self.config)

            # Step 2: replace all lists/tuples with index-keyed dicts
            cfg_dict = normalize_lists(cfg_dict)

            # Step 3: flatten nested dicts into dotted keys
            def _flatten(d, parent_key="", sep="."):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(_flatten(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)

            flat_cfg = _flatten(cfg_dict)

            # Step 4: update W&B config with all flattened keys
            trainer.logger.experiment.config.update(flat_cfg, allow_val_change=True)

            # Step 5: save full config as a temporary artifact
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as tmp:
                tmp.write(str(cfg_dict))
                tmp_path = tmp.name

            artifact = wandb.Artifact(name="experiment-config", type="config")
            artifact.add_file(tmp_path)
            trainer.logger.experiment.log_artifact(artifact)


class BEMACallback(Callback):
    """
    Bias-Corrected Exponential Moving Average (BEMA) Callback for PyTorch Lightning.

    Maintains a moving average of model parameters, optionally with bias correction
    anchored to the initialization. Implements the update scheme described in
    the BEMA paper (arXiv:2508.00180), with polynomially decaying schedules.

    Update rule (when bias_power is not None):
        α_t = (ρ + γ t)^(-η)
        β_t = (ρ + γ t)^(-κ)
        μ_EMA ← (1 - β_t) μ_EMA + β_t θ_t
        μ ← α_t (θ_t - θ_0) + μ_EMA

    If bias_power=None, the update reduces to standard EMA:
        β_t = (ρ + γ t)^(-κ)
        μ ← (1 - β_t) μ + β_t θ_t

    Args:
        ema_power (float): κ, power exponent for EMA decay. Default = 0.6 (paper).
        bias_power (float or None): η, power exponent for bias correction.
            If None, disables bias correction (plain EMA). Default = 0.4 (paper).
        multiplier (float): γ, multiplier applied to time t. Default = 1.0.
        lag (float): ρ, lag term to stabilize denominators. Default = 0.0.
        burn_in (int): τ, number of steps before applying BEMA. Default = 0.
        frequency (int): φ, only update every φ steps. Default = 1 (update every step).
        apply_on_validation (bool): If True, swaps in averaged weights at validation.
    """

    def __init__(
        self,
        ema_power: float = 0.6,
        bias_power: float | None = 0.4,
        multiplier: float = 1.0,
        lag: float = 0.0,
        burn_in: int = 0,
        frequency: int = 1,
        apply_on_validation: bool = True,
    ):
        super().__init__()
        self.ema_power = ema_power
        self.bias_power = bias_power
        self.multiplier = multiplier
        self.lag = lag
        self.burn_in = burn_in
        self.frequency = frequency
        self.apply_on_validation = apply_on_validation

        # State
        self.step_count = 0
        self.theta0 = None
        self.mu_ema = {}
        self.mu = {}
        self._online_params = None

    def on_fit_start(self, trainer, pl_module):
        state = pl_module.state_dict()
        self.theta0 = {k: v.detach().clone() for k, v in state.items()}
        self.mu_ema = {k: v.detach().clone() for k, v in state.items()}
        self.mu = {k: v.detach().clone() for k, v in state.items()}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step_count += 1
        t = self.step_count
        state_dict = pl_module.state_dict()

        if t <= self.burn_in:
            self.theta0 = {k: v.detach().clone() for k, v in state_dict.items()}
            self.mu_ema = {k: v.detach().clone() for k, v in state_dict.items()}
            self.mu = {k: v.detach().clone() for k, v in state_dict.items()}
            return

        if (t - self.burn_in) % self.frequency != 0:
            return

        beta_t = (self.lag + self.multiplier * t) ** (-self.ema_power)
        if self.bias_power is None:
            alpha_t = None
        else:
            alpha_t = (self.lag + self.multiplier * t) ** (-self.bias_power)

        with torch.no_grad():
            for k, p in state_dict.items():
                p_t = p.detach()

                # Always update EMA
                self.mu_ema[k].mul_(1 - beta_t).add_(beta_t * p_t)

                if alpha_t is None:
                    # Plain EMA
                    self.mu[k] = self.mu_ema[k]
                else:
                    # Bias-corrected EMA
                    self.mu[k] = alpha_t * (p_t - self.theta0[k]) + self.mu_ema[k]

    def on_validation_start(self, trainer, pl_module):
        if not self.apply_on_validation:
            return
        self._online_params = {
            k: v.detach().clone() for k, v in pl_module.state_dict().items()
        }
        new_state = {k: v for k, v in self.mu.items()}
        pl_module.load_state_dict(new_state, strict=False)

    def on_validation_end(self, trainer, pl_module):
        if not self.apply_on_validation or self._online_params is None:
            return
        pl_module.load_state_dict(self._online_params, strict=False)
        self._online_params = None

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "theta0": self.theta0,
            "mu_ema": self.mu_ema,
            "mu": self.mu,
        }

    def load_state_dict(self, state_dict):
        self.step_count = state_dict["step_count"]
        self.theta0 = state_dict["theta0"]
        self.mu_ema = state_dict["mu_ema"]
        self.mu = state_dict["mu"]


class EMACallback(Callback):
    def __init__(self, beta: float = 0.999):
        super().__init__()
        self.beta = beta
        self.ema_state = None
        self.online_state = None  # for restoring after validation

    def on_fit_start(self, trainer, pl_module):
        # initialize EMA copy of parameters
        self.ema_state = {
            name: p.detach().clone()
            for name, p in pl_module.named_parameters()
            if p.requires_grad
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # EMA update rule
        for name, p in pl_module.named_parameters():
            if not p.requires_grad:
                continue
            self.ema_state[name].mul_(self.beta).add_(p.detach(), alpha=1 - self.beta)

    def on_validation_start(self, trainer, pl_module):
        # store online weights
        self.online_state = {
            name: p.detach().clone()
            for name, p in pl_module.named_parameters()
            if p.requires_grad
        }
        # swap in EMA weights
        for name, p in pl_module.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.ema_state[name])

    def on_validation_end(self, trainer, pl_module):
        # restore online weights
        for name, p in pl_module.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.online_state[name])
        self.online_state = None

    def state_dict(self):
        return {"ema_state": self.ema_state}

    def load_state_dict(self, state_dict):
        self.ema_state = state_dict["ema_state"]
