"""Entry point: `python -m src.main ["optional proposal"]` or `python src/main.py`."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.config import load_settings  # noqa: E402
from src.ui.app import MagiApp  # noqa: E402


def main() -> None:
    settings = load_settings()
    initial = " ".join(sys.argv[1:]).strip() or None
    MagiApp(settings=settings, initial_prompt=initial).run()


if __name__ == "__main__":
    main()
