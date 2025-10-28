import os
from typing import Dict, Iterator

import pandas as pd
from datasets import DatasetInfo, Features, Value, Split, SplitGenerator, GeneratorBasedBuilder
from datasets import Version

CSV_URL = "https://huggingface.co/datasets/JonusNattapong/xauusd-dataset/resolve/main/XAUUSD_enhanced_ml_dataset_clean.csv"


class XAUUSDDataset(GeneratorBasedBuilder):
    """XAUUSD dataset loader that reads the CSV and yields examples.

    This script dynamically infers feature types from the CSV header/sample so
    it avoids storing complex pickled metadata that can cause compatibility
    issues when loaded from different environments.
    """

    VERSION = Version("1.0.0")

    def _info(self) -> DatasetInfo:
        # Read a small sample to infer dtypes
        sample = pd.read_csv(CSV_URL, nrows=100)

        features_dict: Dict[str, Value] = {}
        for col in sample.columns:
            # Treat the Date column as string
            if col.lower() == "date":
                features_dict[col] = Value("string")
                continue

            # Infer numeric -> float32, else string
            if pd.api.types.is_numeric_dtype(sample[col]):
                features_dict[col] = Value("float32")
            else:
                features_dict[col] = Value("string")

        features = Features(features_dict)

        return DatasetInfo(
            description="XAUUSD enhanced ML dataset (generated loader)",
            features=features,
            homepage="https://huggingface.co/datasets/JonusNattapong/xauusd-dataset",
            license="MIT",
        )

    def _split_generators(self, dl_manager):
        # We directly use the CSV URL; dl_manager can download it if needed
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"filepath": CSV_URL},
            )
        ]

    def _generate_examples(self, filepath: str) -> Iterator:
        # Use pandas iterator to avoid loading whole file into memory
        for idx, row in pd.read_csv(filepath, chunksize=1000):
            # rows from chunksize iterator come as DataFrame; iterate rows
            for i, r in row.iterrows():
                example = {}
                for col in row.columns:
                    val = r[col]
                    if pd.isna(val):
                        example[col] = None
                    else:
                        # Cast numpy types to native Python types where possible
                        if hasattr(val, "item"):
                            try:
                                example[col] = val.item()
                            except Exception:
                                example[col] = val
                        else:
                            example[col] = val
                yield f"{idx}-{i}", example
