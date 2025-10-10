"""Модуль сбора и фильтрации RSS-данных."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Iterable, Sequence

import feedparser

from .config import RSSConfig
from .models import NewsItem, RawFeedEntry

logger = logging.getLogger(__name__)


class RSSCollector:
    """Сборщик RSS-новостей."""

    def __init__(self, config: RSSConfig) -> None:
        self._config = config

    def collect(self) -> list[NewsItem]:
        """Возвращает список отфильтрованных новостей."""
        seen_links: set[str] = set()
        seen_titles: list[str] = []
        result: list[NewsItem] = []

        for source in self._sources():
            try:
                feed = feedparser.parse(source)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Ошибка чтения RSS %s: %s", source, exc)
                continue

            for entry in feed.entries:
                raw = self._to_raw_entry(source, entry)
                if not raw.link or raw.link in seen_links:
                    continue
                if not self._match_keywords(raw):
                    continue
                if self._is_similar(raw.title, seen_titles):
                    continue
                news = self._normalize(raw)
                result.append(news)
                seen_links.add(raw.link)
                seen_titles.append(raw.title)
                if len(result) >= self._config.max_items:
                    return result
        return result

    def _sources(self) -> Iterable[str]:
        """Возвращает список источников с учетом значений по умолчанию."""
        if self._config.sources:
            return self._config.sources
        return (
            "https://techcrunch.com/category/artificial-intelligence/feed/",
            "https://venturebeat.com/category/ai/feed/",
            "https://www.technologyreview.com/feed/",
            "https://www.theverge.com/artificial-intelligence/rss/index.xml",
            "https://openai.com/blog/rss/",
            "https://ai.googleblog.com/feeds/posts/default",
            "https://www.anthropic.com/news/rss",
        )

    def _to_raw_entry(self, source: str, entry: feedparser.FeedParserDict) -> RawFeedEntry:
        """Преобразует элемент feedparser в RawFeedEntry."""
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        summary = entry.get("summary", "").strip()
        published = None
        if published_struct := entry.get("published_parsed"):
            published = datetime(*published_struct[:6], tzinfo=timezone.utc)

        media_url = self._extract_media(entry)

        return RawFeedEntry(
            source=source,
            title=title,
            link=link,
            summary=summary,
            published=published,
            media_url=media_url,
        )

    def _extract_media(self, entry: feedparser.FeedParserDict) -> str | None:
        """Пытается получить ссылку на медиафайл из RSS."""
        enclosure = entry.get("enclosures") or ()
        for item in enclosure:
            href = item.get("href")
            if href:
                return href
        media_content = entry.get("media_content") or ()
        for item in media_content:
            url = item.get("url")
            if url:
                return url
        return entry.get("image", {}).get("href")

    def _match_keywords(self, entry: RawFeedEntry) -> bool:
        """Проверяет совпадение по ключевым словам."""
        if not self._config.keywords:
            return True
        haystack = f"{entry.title} {entry.summary}".lower()
        return any(keyword.lower() in haystack for keyword in self._config.keywords)

    def _is_similar(self, title: str, seen_titles: Sequence[str]) -> bool:
        """Дедупликация по схожести заголовков."""
        for seen in seen_titles:
            if SequenceMatcher(None, title.lower(), seen.lower()).ratio() >= self._config.similarity_threshold:
                return True
        return False

    def _normalize(self, entry: RawFeedEntry) -> NewsItem:
        """Преобразует запись к бизнес-модели."""
        keywords = tuple(keyword for keyword in self._config.keywords if keyword.lower() in entry.summary.lower())
        return NewsItem(
            source=entry.source,
            title=entry.title,
            link=entry.link,
            summary=entry.summary,
            published=entry.published,
            keywords=keywords,
            media_url=entry.media_url,
        )


__all__: Sequence[str] = ("RSSCollector",)
