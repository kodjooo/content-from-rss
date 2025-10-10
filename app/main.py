"""Точка входа приложения AI News Auto Writer."""

from __future__ import annotations

import argparse

from .config import load_settings
from .logging_utils import setup_logging
from .orchestrator import PipelineRunner
from .scheduler import PipelineScheduler


def main() -> None:
    """Запускает приложение в выбранном режиме."""
    parser = argparse.ArgumentParser(description="AI News Auto Writer")
    parser.add_argument(
        "--mode",
        choices=("scheduler", "run-once"),
        default="scheduler",
        help="Режим работы: по расписанию или единичный запуск",
    )
    args = parser.parse_args()

    config = load_settings()
    setup_logging(config.log_level)

    if args.mode == "run-once":
        stats = PipelineRunner(config).run()
        print(  # noqa: T201 - консольный вывод допущен для CLI
            f"Готово: обработано={stats.processed}, принято={stats.accepted}, опубликовано={stats.published}, ошибки={stats.failed}",
        )
        return

    scheduler = PipelineScheduler(config)
    scheduler.start(block=True)


if __name__ == "__main__":
    main()
