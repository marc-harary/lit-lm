import torch
from torch import nn
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer


class LitLM(pl.LightningModule):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)

    def _shared_step(self, batch, stage: str):
        """
        Shared forward + loss computation used by both training and validation.
        """
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],  # HuggingFace does the shifting inside
        )
        loss = outputs.loss
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch["input_ids"].size(0),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")
