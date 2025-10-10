"""Проверка готовности приложения."""

from __future__ import annotations

import sys

from .config import load_settings


def main() -> None:
    """Запускает проверку конфигурации."""
    try:
        load_settings()
    except Exception as exc:  # noqa: BLE001
        print(f"healthcheck failed: {exc}", file=sys.stderr)
        raise
    print("ok")  # noqa: T201 - вывод для healthcheck


if __name__ == "__main__":
    main()
