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
        # Use pandas chunked iterator to avoid loading the whole file into memory
        for chunk_idx, chunk in enumerate(pd.read_csv(filepath, chunksize=1000)):
            for row_idx, row in chunk.iterrows():
                example = {}
                for col in chunk.columns:
                    val = row[col]
                    if pd.isna(val):
                        example[col] = None
                    else:
                        # Convert numpy scalars to native Python scalars to avoid
                        # pickling / numpy dtype issues when the dataset is
                        # serialized by the HF datasets library.
                        try:
                            # pandas/numpy scalar -> python native
                            if hasattr(val, "item"):
                                cast_val = val.item()
                            else:
                                cast_val = val

                            # Ensure datetimes/strings are strings
                            if isinstance(cast_val, (pd.Timestamp,)):
                                example[col] = str(cast_val)
                            else:
                                example[col] = cast_val
                        except Exception:
                            example[col] = str(val)

                yield f"{chunk_idx}-{row_idx}", example
