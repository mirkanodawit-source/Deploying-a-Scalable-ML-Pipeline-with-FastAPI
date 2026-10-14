"""Artifact locations are independent of the process working directory."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT / "model")))
