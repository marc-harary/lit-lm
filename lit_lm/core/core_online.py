import torch
from transformers import TrainerCallback, TrainerState, TrainingArguments, TrainerControl
import copy


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class BEMACallback(TrainerCallback):
    """
    Bias-Corrected Exponential Moving Average (BEMA) with optional
    curvature scaling from Adam's second moment (exp_avg_sq).
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
        device: str = "cpu",
        use_adam_curvature: bool = True,
    ):
        # hyperparameters
        self.update_freq = update_freq
        self.ema_power = ema_power
        self.eta_power = eta_power
        self.update_after = update_after or 0
        self.scaling_lag = scaling_lag
        self.ema_gamma = ema_gamma
        self.min_ema_multiplier = min_ema_multiplier
        self.device = device
        self.use_adam_curvature = use_adam_curvature

        # internal state
        self.initialized = False
        self.param_names = []
        self.thetat_params = []
        self.theta0_params = []
        self.ema_params = []
        self.running_model = None

    # ------------------------------------------------------------------
    # decay schedules
    # ------------------------------------------------------------------
    def _ema_beta(self, t: int) -> float:
        if self.ema_power < 0:
            return 1.0
        beta = (1 + self.ema_gamma * t) ** (-self.ema_power)
        return max(beta, self.min_ema_multiplier)

    def _bema_alpha(self, t: int) -> float:
        if self.eta_power < 0:
            return 0.0
        return (1 + self.ema_gamma * t) ** (-self.eta_power)

    # ------------------------------------------------------------------
    # Trainer hooks
    # ------------------------------------------------------------------
    @torch.no_grad()
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = _unwrap_model(kwargs["model"])
        self.running_model = copy.deepcopy(model).to(self.device)

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.param_names.append(name)
            self.thetat_params.append(p)

            theta0_buf = p.data.detach().to(self.device).clone()
            ema_buf = theta0_buf.clone()
            self.theta0_params.append(theta0_buf)
            self.ema_params.append(ema_buf)

        self.initialized = False

    @torch.no_grad()
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step
        if step is None:
            return

        # init snapshot θ₀ and EMA
        if not self.initialized and step >= self.update_after:
            for p, theta0, ema in zip(self.thetat_params, self.theta0_params, self.ema_params):
                data = p.data.detach().to(self.device)
                theta0.copy_(data)
                ema.copy_(data)
            self.initialized = True

        if not self.initialized:
            return
        if step % self.update_freq != 0:
            return

        # compute decay factors
        t = max(step - self.update_after + self.scaling_lag, 1)
        beta = self._ema_beta(t)
        alpha = self._bema_alpha(t)

        # gather θₜ
        new_thetas = [p.data.detach().to(self.device, non_blocking=True) for p in self.thetat_params]

        # EMA update
        torch._foreach_mul_(self.ema_params, 1 - beta)
        torch._foreach_add_(self.ema_params, new_thetas, alpha=beta)

        # BEMA correction term
        deltas = [theta.clone() for theta in new_thetas]
        torch._foreach_sub_(deltas, self.theta0_params)   # θₜ − θ₀
        torch._foreach_mul_(deltas, alpha)                # α·(θₜ − θ₀)

        # curvature scaling (Adam second moment)
        if self.use_adam_curvature and "optimizer" in kwargs:
            opt = kwargs["optimizer"]
            beta2 = opt.param_groups[0]["betas"][1]
            eps = opt.param_groups[0].get("eps", 1e-8)

            v_hats = []
            for p in self.thetat_params:
                st = opt.state[p]
                if "exp_avg_sq" in st:
                    v = st["exp_avg_sq"]
                    step_count = st.get("step", 1)
                    v_hat = v / (1 - beta2**step_count)
                    v_hats.append(v_hat.sqrt().to(self.device))
                else:
                    v_hats.append(torch.ones_like(p, device=self.device))

            # elementwise division
            for d, vhat in zip(deltas, v_hats):
                d.div_(vhat + eps)

        # add EMA + corrected deltas
        torch._foreach_add_(deltas, self.ema_params)

        # write back to running_model
        sd_run = self.running_model.state_dict()
        for name, corr in zip(self.param_names, deltas):
            sd_run[name].copy_(corr)
        self.running_model.load_state_dict(sd_run, strict=False)

    @torch.no_grad()
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            out_path = f"{args.output_dir}/bema.pt"
            torch.save(self.running_model.state_dict(), out_path)
            print(f"[BEMA] Saved stabilized weights to {out_path}")
