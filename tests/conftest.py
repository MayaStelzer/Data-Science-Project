import sys
from pathlib import Path


def pytest_configure(config):
    # Ensure project root (two levels up from tests/) is on sys.path so local package imports work
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
