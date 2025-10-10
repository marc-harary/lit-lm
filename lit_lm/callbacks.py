import copy
from argparse import Namespace
import tempfile

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
    Bias-Corrected Exponential Moving Average (BEMA) for model parameters.

    Features:
      • Standard EMA (always).
      • Bias correction term α_t(θ_t - θ0), optionally preconditioned with Adam's second moments.
      • Burn-in τ, update frequency φ, polynomial decay schedules for EMA/bias.
      • Stores shadow copies of EMA and BEMA weights for validation/eval.

    Args:
        ema_power (float): κ exponent for EMA decay schedule.
        bias_power (float|None): η exponent for bias correction. If None → pure EMA.
        multiplier (float): γ scaling of schedule.
        lag (float): ρ lag offset.
        burnin (int): τ burn-in steps before applying stabilization.
        frequency (int): φ update frequency.
        use_adam_precond (bool): whether to use Adam's exp_avg_sq for bias correction preconditioning.
        eps (float): epsilon for Adam preconditioner stability.
    """

    def __init__(
        self,
        ema_power: float = 0.6,
        bias_power: float | None = 0.4,
        multiplier: float = 1.0,
        lag: float = 0.0,
        burnin: int = 0,
        frequency: int = 1,
        use_adam_precond: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.ema_power = ema_power
        self.bias_power = bias_power
        self.multiplier = multiplier
        self.lag = lag
        self.burnin = burnin
        self.frequency = frequency
        self.use_adam_precond = use_adam_precond
        self.eps = eps

        # Internal state
        self.mu = {}  # bias-corrected EMA weights
        self.mu_ema = {}  # standard EMA weights
        self.theta0 = {}  # initial weights
        self.t = 0  # step counter

    def on_fit_start(self, trainer, pl_module):
        """Initialize buffers with model parameters."""
        for name, p in pl_module.named_parameters():
            if not p.requires_grad:
                continue
            self.mu[name] = p.detach().clone().float()
            self.mu_ema[name] = p.detach().clone().float()
            self.theta0[name] = p.detach().clone().float()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA and BEMA after each batch."""
        self.t += 1
        t = self.t

        # Skip until burn-in
        if t <= self.burnin:
            for name, p in pl_module.named_parameters():
                if not p.requires_grad:
                    continue
                self.mu[name] = p.detach().clone().float()
                self.mu_ema[name] = p.detach().clone().float()
                self.theta0[name] = p.detach().clone().float()
            return

        # Update only every φ steps
        if (t - self.burnin) % self.frequency != 0:
            return

        beta_t = (self.lag + self.multiplier * t) ** (-self.ema_power)
        alpha_t = (
            None
            if self.bias_power is None
            else (self.lag + self.multiplier * t) ** (-self.bias_power)
        )

        # Grab optimizer state if Adam preconditioner enabled
        opt = trainer.optimizers[0] if self.use_adam_precond else None
        beta2 = None
        if opt is not None:
            beta2 = opt.param_groups[0]["betas"][1]  # assumes all groups share beta2

        with torch.no_grad():
            for name, p in pl_module.named_parameters():
                if not p.requires_grad:
                    continue

                p_t = p.detach()
                # EMA update
                self.mu_ema[name].mul_(1 - beta_t).add_(p_t, alpha=beta_t)

                # If no bias correction, fallback to EMA
                if alpha_t is None:
                    self.mu[name] = self.mu_ema[name]
                    continue

                # Optional Adam-preconditioned correction
                if self.use_adam_precond and name in opt.state[p]:
                    st = opt.state[p]
                    if "exp_avg_sq" in st:
                        v = st["exp_avg_sq"]
                        denom = max(1.0 - (beta2**t), 1e-8)
                        v_hat = v / denom
                        scale = v_hat.sqrt().add_(self.eps)
                        corr = (p_t - self.theta0[name]) / scale
                    else:
                        corr = p_t - self.theta0[name]
                else:
                    corr = p_t - self.theta0[name]

                self.mu[name] = self.mu_ema[name] + alpha_t * corr

    def on_validation_start(self, trainer, pl_module):
        """Swap in BEMA weights for evaluation."""
        self.backup = {}
        with torch.no_grad():
            for name, p in pl_module.named_parameters():
                if not p.requires_grad:
                    continue
                self.backup[name] = p.detach().clone()
                p.copy_(self.mu[name])

    def on_validation_end(self, trainer, pl_module):
        """Restore original weights after validation."""
        with torch.no_grad():
            for name, p in pl_module.named_parameters():
                if not p.requires_grad:
                    continue
                p.copy_(self.backup[name])
        self.backup = None

    def state_dict(self):
        return {
            "t": self.t,
            "mu": self.mu,
            "mu_ema": self.mu_ema,
            "theta0": self.theta0,
        }

    def load_state_dict(self, state_dict):
        self.t = state_dict["t"]
        self.mu = state_dict["mu"]
        self.mu_ema = state_dict["mu_ema"]
        self.theta0 = state_dict["theta0"]
