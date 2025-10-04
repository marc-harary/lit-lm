# scripts/train.py

from lit_lm.learner import LitLM
from lit_lm.data import TextDatasets
from pytorch_lightning.cli import LightningCLI, SaveConfigCallback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import *
from pytorch_lightning import Trainer, LightningModule
import wandb


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, WandbLogger):
            # Dump Hydra config to string for reproducibility
            config = self.parser.dump(self.config, skip_none=False)
            trainer.logger.log_hyperparams({"config": config})
            # Optionally log as W&B artifact
            artifact = wandb.Artifact(name="experiment-config", type="config")
            artifact.add_file("config.yaml")
            trainer.logger.experiment.log_artifact(artifact)


def main():
    # LightningCLI automatically instantiates LitLM + DataModule from config
    cli = LightningCLI(
        LitLM,                          # LightningModule (in lit_lm/learner.py)
        TextDatasets,       # reads from `data.*` in config (your style)
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=LoggerSaveConfigCallback,
    )


if __name__ == "__main__":
    main()
