"""Планировщик регулярных запусков пайплайна."""

from __future__ import annotations

import logging
import time
from typing import Callable, Sequence

from apscheduler.schedulers.background import BackgroundScheduler

from .config import AppConfig
from .logging_utils import setup_logging
from .orchestrator import PipelineRunner

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """Регулярно запускает PipelineRunner по расписанию."""

    def __init__(
        self,
        config: AppConfig,
        runner_factory: Callable[[], PipelineRunner] | None = None,
    ) -> None:
        self._config = config
        self._runner_factory = runner_factory or (lambda: PipelineRunner(config))
        self._scheduler = BackgroundScheduler(timezone=config.scheduler.timezone)

    def schedule_jobs(self) -> None:
        """Добавляет задачи в планировщик."""
        for hour in self._config.scheduler.run_hours:
            self._scheduler.add_job(self._run_job, "cron", hour=hour, minute=0)
            logger.info("Запланирован запуск пайплайна на %02d:00", hour)

    def start(self, block: bool = True) -> None:
        """Запускает планировщик."""
        setup_logging(self._config.log_level)
        if self._config.scheduler.run_once_on_start:
            logger.info("Выполняется разовый запуск пайплайна при старте")
            self.run_once()
        self.schedule_jobs()
        self._scheduler.start()
        logger.info("Планировщик запущен")
        if not block:
            return
        try:
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            self.stop()

    def stop(self) -> None:
        """Останавливает планировщик."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Планировщик остановлен")

    def run_once(self) -> None:
        """Запускает задачу немедленно (для тестов и дебага)."""
        self._run_job()

    def _run_job(self) -> None:
        """Выполнение задачи пайплайна."""
        runner = self._runner_factory()
        stats = runner.run()
        logger.info(
            "Завершен цикл: обработано=%d, принято=%d, опубликовано=%d, ошибки=%d",
            stats.processed,
            stats.accepted,
            stats.published,
            stats.failed,
        )


__all__: Sequence[str] = ("PipelineScheduler",)
