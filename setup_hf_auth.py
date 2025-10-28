#!/usr/bin/env python3
"""
Setup Hugging Face Authentication for Dataset Upload
"""

import os
from huggingface_hub import login, whoami
from pathlib import Path

def setup_hf_auth():
    """Setup Hugging Face authentication"""

    print("🔑 Setting up Hugging Face Authentication")
    print("=" * 50)

    # Check if already logged in
    try:
        user = whoami()
        print(f"✅ Already logged in as: {user['name']}")
        print(f"📧 Email: {user['email']}")
        return True
    except Exception as e:
        print(f"❌ Not logged in: {e}")
        print()

    print("To upload datasets to Hugging Face, you need to:")
    print("1. Create a Hugging Face account at: https://huggingface.co/join")
    print("2. Get your API token from: https://huggingface.co/settings/tokens")
    print("3. Run this script and enter your token when prompted")
    print()

    # Try to login
    try:
        print("Enter your Hugging Face API token:")
        print("(You can find it at: https://huggingface.co/settings/tokens)")
        print("Token: ", end="")

        # For security, we'll use environment variable or manual input
        token = os.getenv('HF_TOKEN')

        if not token:
            print("\nFor security, please set your token as an environment variable:")
            print("Run: set HF_TOKEN=your_token_here")
            print("Then run this script again.")
            return False

        login(token=token)
        print("✅ Login successful!")

        # Verify login
        user = whoami()
        print(f"Logged in as: {user['name']}")

        return True

    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

def create_dataset_repo():
    """Create the dataset repository on Hugging Face"""

    from huggingface_hub import HfApi, create_repo

    repo_name = "xauusd-dataset"
    repo_id = f"JonusNattapong/{repo_name}"

    print(f"\n📁 Creating dataset repository: {repo_id}")

    try:
        api = HfApi()
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print(f"✅ Repository created: https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return False

if __name__ == "__main__":
    print("🚀 XAUUSD Dataset Upload Setup")
    print("=" * 40)

    # Setup authentication
    if setup_hf_auth():
        # Create repository
        if create_dataset_repo():
            print("\n🎯 Ready to upload datasets!")
            print("Run 'python upload_datasets.py' to upload your datasets.")
        else:
            print("\n❌ Repository creation failed.")
    else:
        print("\n❌ Authentication failed.")
        print("Please set up your Hugging Face token and try again.")