#!/usr/bin/env python3
"""
Upload a file to the Hugging Face dataset repo using the HfApi.
"""
from huggingface_hub import HfApi

repo_id = "JonusNattapong/xauusd-dataset"
filepath = "xauusd_dataset.py"

api = HfApi()
api.upload_file(
    path_or_fileobj=filepath,
    path_in_repo=filepath,
    repo_id=repo_id,
    repo_type="dataset",
)
print(f"Uploaded {filepath} to {repo_id}")
