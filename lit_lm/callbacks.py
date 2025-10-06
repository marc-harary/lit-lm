import copy
from argparse import Namespace

import torch
from pytorch_lightning import Callback
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning.loggers import WandbLogger
import wandb


def normalize_lists(obj):
    """
    Recursively replace all Python lists with dictionaries keyed by index.

    This ensures that every list element becomes accessible as an individual
    key-value pair (e.g., "betas.0", "betas.1") when flattened later, so that
    config values are searchable in W&B rather than hidden in opaque JSON blobs.
    """
    if isinstance(obj, list):
        # Replace list with a dict where keys are string indices ("0","1",...)
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

    Handles nested Namespaces, dicts, and lists/tuples. Scalars are returned as-is.
    This normalizes Hydra/LightningCLI configs into a uniform dictionary structure
    before applying list normalization and flattening.
    """
    if isinstance(obj, Namespace):
        # Convert Namespace attributes to dict entries
        return {k: ns_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        # Recurse into dictionaries
        return {k: ns_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recurse into each element of list/tuple
        return [ns_to_dict(v) for v in obj]
    else:
        # Base case: return unchanged
        return obj


class LoggerSaveConfigCallback(SaveConfigCallback):
    """
    Custom callback to log Hydra/LightningCLI config into Weights & Biases (W&B).

    Steps:
      1. Convert config (Namespace/DictConfig) into a clean Python dict.
      2. Normalize lists into dicts with numeric keys.
      3. Flatten nested dicts into dotted-key strings.
      4. Push flattened config to W&B so all params are searchable.
      5. Save the full unflattened config as a YAML artifact for reproducibility.
    """

    def save_config(self, trainer, pl_module, stage: str) -> None:
        if isinstance(trainer.logger, WandbLogger):
            # Step 1: recursively convert Namespace to dict
            cfg_dict = ns_to_dict(self.config)

            # Step 2: normalize lists everywhere (e.g., callbacks, betas)
            cfg_dict = normalize_lists(cfg_dict)

            # Step 3: flatten nested dictionaries into dot-separated keys
            def _flatten(d, parent_key="", sep="."):
                items = []
                for k, v in d.items():
                    # Build new dotted key path
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        # Recurse deeper
                        items.extend(_flatten(v, new_key, sep=sep).items())
                    else:
                        # Leaf node: append key-value pair
                        items.append((new_key, v))
                return dict(items)

            flat_cfg = _flatten(cfg_dict)

            # Step 4: update W&B run config with flattened key-value pairs
            trainer.logger.experiment.config.update(flat_cfg, allow_val_change=True)

            # Step 5: also save the full unflattened config to disk as artifact
            with open("config_dump.yaml", "w") as f:
                f.write(str(cfg_dict))
            artifact = wandb.Artifact(name="experiment-config", type="config")
            artifact.add_file("config_dump.yaml")
            trainer.logger.experiment.log_artifact(artifact)


class BEMACallback(Callback):
    """
    Bias-Corrected Exponential Moving Average (BEMA)
    with optional curvature scaling from Adam's exp_avg_sq.
    """

    def __init__(
        self,
        update_freq: int = 100,
        ema_power: float = 0.5,
        eta_power: float = 0.2,
        update_after: int = 0,
        scaling_lag: int = 10,
        ema_gamma: float = 1.0,
        min_ema_multiplier: float = 0.0,
        use_adam_curvature: bool = True,
        disable: bool = False,
    ):
        super().__init__()
        # hyperparameters
        self.update_freq = update_freq
        self.ema_power = ema_power
        self.eta_power = eta_power
        self.update_after = update_after or 0
        self.scaling_lag = scaling_lag
        self.ema_gamma = ema_gamma
        self.min_ema_multiplier = min_ema_multiplier
        self.use_adam_curvature = use_adam_curvature
        self.disable = disable

        # state
        self.initialized = False
        self.param_names = []
        self.thetat_params = []
        self.theta0_params = []
        self.ema_params = []
        self.running_model = None

    # -----------------------------
    # decay schedules
    # -----------------------------
    def _ema_beta(self, t: int) -> float:
        if self.ema_power < 0:
            return 1.0
        beta = (1 + self.ema_gamma * t) ** (-self.ema_power)
        return max(beta, self.min_ema_multiplier)

    def _bema_alpha(self, t: int) -> float:
        if self.eta_power < 0:
            return 0.0
        return (1 + self.ema_gamma * t) ** (-self.eta_power)

    # -----------------------------
    # Lightning hooks
    # -----------------------------
    @torch.no_grad()
    def on_fit_start(self, trainer, pl_module):
        if self.disable:
            return

        device = pl_module.device
        self.running_model = copy.deepcopy(pl_module).to(device)

        for name, p in pl_module.named_parameters():
            if not p.requires_grad:
                continue
            self.param_names.append(name)
            self.thetat_params.append(p)

            theta0_buf = p.data.detach().clone().to(device)
            ema_buf = theta0_buf.clone()
            self.theta0_params.append(theta0_buf)
            self.ema_params.append(ema_buf)

        self.initialized = False

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.disable:
            return

        step = trainer.global_step
        if step is None:
            return

        device = pl_module.device

        # init snapshot θ₀ and EMA
        if not self.initialized and step >= self.update_after:
            for p, theta0, ema in zip(
                self.thetat_params, self.theta0_params, self.ema_params
            ):
                data = p.data.detach().to(device)
                theta0.copy_(data)
                ema.copy_(data)
            self.initialized = True

        if not self.initialized or step % self.update_freq != 0:
            return

        # compute decay factors
        t = max(step - self.update_after + self.scaling_lag, 1)
        beta = self._ema_beta(t)
        alpha = self._bema_alpha(t)

        # gather θₜ
        new_thetas = [
            p.data.detach().to(device, non_blocking=True) for p in self.thetat_params
        ]

        # EMA update
        torch._foreach_mul_(self.ema_params, 1 - beta)
        torch._foreach_add_(self.ema_params, new_thetas, alpha=beta)

        # BEMA correction term
        deltas = [theta.clone() for theta in new_thetas]
        torch._foreach_sub_(deltas, self.theta0_params)  # θₜ − θ₀
        torch._foreach_mul_(deltas, alpha)  # α·(θₜ − θ₀)

        # curvature scaling (Adam second moment)
        if self.use_adam_curvature and len(trainer.optimizers) > 0:
            opt = trainer.optimizers[0]
            beta2 = opt.param_groups[0]["betas"][1]
            eps = opt.param_groups[0].get("eps", 1e-8)

            v_hats = []
            for p in self.thetat_params:
                st = opt.state[p]
                if "exp_avg_sq" in st:
                    v = st["exp_avg_sq"]
                    step_count = st.get("step", 1)
                    v_hat = v / (1 - beta2**step_count)
                    v_hats.append(v_hat.sqrt().to(device))
                else:
                    v_hats.append(torch.ones_like(p, device=device))

            for d, vhat in zip(deltas, v_hats):
                d.div_(vhat + eps)

        # add EMA + corrected deltas
        torch._foreach_add_(deltas, self.ema_params)

        # write back to running_model
        sd_run = self.running_model.state_dict()
        for name, corr in zip(self.param_names, deltas):
            sd_run[name].copy_(corr)
        self.running_model.load_state_dict(sd_run, strict=False)
