from lit_lm.learner import LitLM
from lit_lm.data import TextDatasets
from lit_lm.callbacks import LoggerSaveConfigCallback
from pytorch_lightning.cli import LightningCLI


def main():
    cli = LightningCLI(
        LitLM,
        TextDatasets,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=LoggerSaveConfigCallback,
    )


if __name__ == "__main__":
    main()
