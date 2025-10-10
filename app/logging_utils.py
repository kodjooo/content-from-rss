"""Настройка логирования приложения."""

from __future__ import annotations

import logging
from typing import Sequence


def setup_logging(level: str = "INFO") -> None:
    """Настраивает корневой логгер."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


__all__: Sequence[str] = ("setup_logging",)
