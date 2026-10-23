import sys
from pathlib import Path

# allow `pytest` from a fresh clone without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
