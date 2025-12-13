import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("GaussianSplatter")

def get_project_root() -> Path:
    return Path(__file__).parent

def ensure_directory(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
