"""
Utilities for fine-tuning workshop.

🫵 You shouldn't be here, no touching unless you want something to break!
"""

import os
from finetuning._defaults import MODEL_DEFAULTS, DATA_DEFAULTS

def setup_hf() -> None:
    """Configure HF caches to local folders."""
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("HF_TOKEN environment variable not set!")
    os.environ["HF_HOME"] = str(MODEL_DEFAULTS.model_dir)
    os.environ["HF_HUB_CACHE"] = str(MODEL_DEFAULTS.model_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DEFAULTS.model_dir)
    os.environ["DATASETS_CACHE"] = str(DATA_DEFAULTS.data_dir)