# lit_lm/data.py
import os
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl

def _maybe_load(path_or_name: str, dataset_config: str | None = None):
    if os.path.exists(path_or_name):
        return load_from_disk(path_or_name)
    return load_dataset(path_or_name, dataset_config) if dataset_config else load_dataset(path_or_name)

class TextDatasets(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_name: str,                    # e.g. "wikitext" or local path
        dataset_config: str | None = None,    # e.g. "wikitext-2-raw-v1"
        pretrained_model_name_or_path: str = "gpt2",
        text_key: str = "text",
        max_length: int = 1024,
        num_workers: int = 4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.text_key = text_key
        self.max_length = max_length
        self.num_workers = num_workers

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.train_dataset = None
        self.valid_dataset = None

    def prepare_data(self):
        # trigger HF cache download under rank-zero
        _maybe_load(self.dataset_name, self.dataset_config)

    def setup(self, stage: str | None = None):
        dataset = _maybe_load(self.dataset_name, self.dataset_config)

        def tokenize_fn(examples):
            enc = self.tokenizer(
                examples[self.text_key],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
            # causal LM loss needs labels
            enc["labels"] = enc["input_ids"].copy()
            return enc

        if isinstance(dataset, DatasetDict):
            remove_cols = dataset["train"].column_names
        else:
            remove_cols = dataset.column_names

        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols)
        dataset.set_format(type="torch")

        if isinstance(dataset, DatasetDict):
            self.train_dataset = dataset["train"]
            self.valid_dataset = dataset.get("validation", dataset.get("test"))
        elif isinstance(dataset, Dataset):
            split = dataset.train_test_split(0.1, seed=42)
            self.train_dataset, self.valid_dataset = split["train"], split["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
