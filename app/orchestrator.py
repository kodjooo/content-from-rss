"""Основной оркестратор пайплайна новостей."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import pytz

from .config import AppConfig, load_settings
from .image_pipeline import ImageSelector
from .logging_utils import setup_logging
from .models import GeneratedPost, ImageAsset, NewsItem, PublicationRecord
from .post_generator import PostComposer, PostGenerationError
from .rss import RSSCollector
from .scoring import RelevanceScorer
from .sheets import GoogleSheetsWriter

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Статистика выполнения пайплайна."""

    processed: int = 0
    accepted: int = 0
    published: int = 0
    failed: int = 0


class PipelineRunner:
    """Класс, orchestrating полный бизнес-процесс."""

    def __init__(
        self,
        config: AppConfig,
        rss_collector: RSSCollector | None = None,
        scorer: RelevanceScorer | None = None,
        composer: PostComposer | None = None,
        image_selector: ImageSelector | None = None,
        sheets_writer: GoogleSheetsWriter | None = None,
    ) -> None:
        self._config = config
        self._rss = rss_collector or RSSCollector(config.rss)
        self._scorer = scorer or RelevanceScorer(config.openai, config.cache_dir)
        self._composer = composer or PostComposer(config.openai)
        self._image_selector = image_selector or ImageSelector(
            config.pexels,
            config.freeimagehost,
            config.openai,
        )
        self._sheets = sheets_writer or GoogleSheetsWriter(config.sheets)
        self._timezone = pytz.timezone(config.scheduler.timezone)

    def run(self) -> PipelineStats:
        """Запускает полный цикл обработки."""
        stats = PipelineStats()
        news_items = self._rss.collect()
        stats.processed = len(news_items)

        ranked = self._scorer.evaluate_many(news_items)
        accepted = [item for item in ranked if item.score >= 7]
        stats.accepted = len(accepted)

        if not accepted:
            logger.info("Нет релевантных новостей для публикации")
            return stats

        records: list[PublicationRecord] = []
        for ranked_item in accepted:
            try:
                post = self._composer.generate(ranked_item.news)
                image = self._image_selector.select(ranked_item.news, post)
                record = self._build_record(ranked_item.score, post, image, ranked_item.news)
                records.append(record)
            except PostGenerationError:
                stats.failed += 1
                logger.exception("Не удалось сгенерировать пост для %s", ranked_item.news.link)
            except Exception:  # noqa: BLE001
                stats.failed += 1
                logger.exception("Ошибка обработки новости %s", ranked_item.news.link)

        if records:
            try:
                self._sheets.append_records(records)
                stats.published = len(records)
            except Exception:  # noqa: BLE001
                stats.failed += len(records)
                logger.exception("Не удалось записать данные в Google Sheets")

        return stats

    def _build_record(
        self,
        score: int,
        post: GeneratedPost,
        image: ImageAsset,
        news: NewsItem,
    ) -> PublicationRecord:
        """Формирует объект для сохранения."""
        now = datetime.now(self._timezone)
        return PublicationRecord(
            date=now,
            source=news.source,
            title=news.title,
            link=news.link,
            summary=news.summary,
            post=post,
            image=image,
            score=score,
        )


def main() -> PipelineStats:
    """Точка входа для запуска из CLI."""
    config = load_settings()
    setup_logging(config.log_level)
    runner = PipelineRunner(config)
    return runner.run()


__all__: Sequence[str] = ("PipelineRunner", "PipelineStats", "main")
