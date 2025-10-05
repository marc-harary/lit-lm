import torch
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM
from torchmetrics.classification import CalibrationError
from torchmetrics.text import Perplexity
from torchmetrics import MeanMetric


class LitLM(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        ignore_index: int = -100,
        n_bins: int = 15,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Hugging Face causal LM
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
        self.model.train()
        self.hparams.vocab_size = self.model.config.vocab_size

        # --- Metrics ---
        self.train_acc = MeanMetric()
        self.val_acc = MeanMetric()
        self.val_perplexity = Perplexity(ignore_index=self.hparams.ignore_index)
        self.val_ece = CalibrationError(
            n_bins=self.hparams.n_bins,
            task="multiclass",
            num_classes=self.hparams.vocab_size,
            ignore_index=self.hparams.ignore_index,
        )

    def log(self, name, value, *args, **kwargs):
        kwargs.setdefault("sync_dist", True)
        kwargs.setdefault("on_step", False)
        return super().log(name, value, *args, **kwargs)

    def _shared_step(self, batch, batch_idx, stage):
        outputs = self.model(**batch)
        step_dict = dict(
            loss=outputs.loss,
            labels=batch["labels"],
            logits=outputs.logits,
            preds=torch.argmax(outputs.logits, dim=-1),
            probs=torch.softmax(outputs.logits.float(), dim=-1),
        )

        self.log(
            f"{stage}/loss",
            step_dict["loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        acc = getattr(self, f"{stage}_acc")
        mask = step_dict["labels"] != self.hparams.ignore_index
        correct = (step_dict["preds"][mask] == step_dict["labels"][mask]).float()
        acc_val = (
            correct.mean() if mask.any() else torch.tensor(0.0, device=self.device)
        )
        acc.update(acc_val)
        self.log(f"{stage}_token_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return step_dict

    def training_step(self, batch, batch_idx):
        step_dict = self._shared_step(batch, batch_idx, stage="train")
        return step_dict["loss"]

    def validation_step(self, batch, batch_idx):
        step_dict = self._shared_step(batch, batch_idx, stage="val")

        self.val_perplexity.update(step_dict["logits"], step_dict["labels"])
        self.log("val/perp", self.val_perplexity)

        self.val_ece.update(
            step_dict["probs"].view(-1, step_dict["probs"].size(-1)),
            step_dict["labels"].view(-1),
        )
        self.log("val/ece", self.val_ece)

        return step_dict["loss"]

    def configure_optimizers(self):
        # Hydra/LightningCLI supplies optimizer
        return None
