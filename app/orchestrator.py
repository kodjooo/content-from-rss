"""Основной оркестратор пайплайна новостей."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
        if self._should_reset_sheet():
            try:
                self._sheets.clear_records()
                logger.info("Очистка Google Sheets перед утренним запуском")
            except Exception:  # noqa: BLE001
                logger.exception("Не удалось очистить Google Sheets")

        try:
            existing_links = self._sheets.fetch_existing_links()
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось получить список ссылок из Google Sheets")
            existing_links = set()

        news_items = [item for item in self._rss.collect() if item.link not in existing_links]
        recent_items = self._filter_recent(news_items)
        stats.processed = len(recent_items)

        if not recent_items:
            logger.info("Нет новостей за последние 12 часов")
            return stats

        ranked = self._scorer.evaluate_many(recent_items)
        accepted = self._select_top_ranked(ranked)
        stats.accepted = len(accepted)

        if not accepted:
            logger.info("Нет релевантных новостей с оценкой >= 8")
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
        date_str = now.strftime("%Y-%m-%d %H:%M:%S")
        status = "Revised" if score >= 9 else "Written"
        return PublicationRecord(
            date=date_str,
            source=news.source,
            title=post.translated_title,
            link=news.link,
            summary=post.summary,
            post=post,
            image=image,
            score=score,
            image_source=self._image_source_label(image.source),
            status=status,
        )

    def _should_reset_sheet(self) -> bool:
        """Определяет, нужно ли очистить таблицу перед запуском."""
        run_hours = getattr(self._config.scheduler, "run_hours", (7, 19))
        if not run_hours:
            return False
        earliest_hour = min(run_hours)
        now_local = datetime.now(self._timezone)
        return now_local.hour == earliest_hour

    def _filter_recent(self, items: Iterable[NewsItem]) -> list[NewsItem]:
        """Фильтрует новости по давности публикации (12 часов)."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=12)
        recent: list[NewsItem] = []
        for item in items:
            published = item.published
            if published is None:
                continue
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            else:
                published = published.astimezone(timezone.utc)
            if published >= cutoff:
                recent.append(item)
        return recent

    def _image_source_label(self, source: str) -> str:
        """Читабельное название источника изображения."""
        mapping = {
            "rss": "RSS",
            "pexels": "Библиотека",
            "openai": "Генерация",
        }
        return mapping.get(source.lower(), source)

    def _select_top_ranked(self, ranked: Iterable[RankedNews]) -> list[RankedNews]:
        """Выбирает минимум три новости, постепенно снижая порог с 10 до 8."""
        ranked_by_score: dict[int, list[RankedNews]] = {10: [], 9: [], 8: []}
        for item in ranked:
            if item.score >= 10:
                ranked_by_score[10].append(item)
            elif item.score == 9:
                ranked_by_score[9].append(item)
            elif item.score == 8:
                ranked_by_score[8].append(item)

        result: list[RankedNews] = []
        for score in (10, 9, 8):
            pool = ranked_by_score[score]
            while pool and len(result) < 5:
                result.append(pool.pop(0))
            if len(result) >= 3:
                break
        return result[:5]


def main() -> PipelineStats:
    """Точка входа для запуска из CLI."""
    config = load_settings()
    setup_logging(config.log_level)
    runner = PipelineRunner(config)
    return runner.run()


__all__: Sequence[str] = ("PipelineRunner", "PipelineStats", "main")
