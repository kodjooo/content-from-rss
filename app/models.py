"""Общие модели данных."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence


@dataclass
class RawFeedEntry:
    """Сырой элемент RSS-ленты."""

    source: str
    title: str
    link: str
    summary: str
    published: datetime | None = None
    media_url: str | None = None


@dataclass
class NewsItem:
    """Нормализованная новость после фильтрации."""

    source: str
    title: str
    link: str
    summary: str
    published: datetime | None
    keywords: tuple[str, ...]
    media_url: str | None = None


@dataclass
class RankedNews:
    """Новость с оценкой релевантности."""

    news: NewsItem
    score: int
    evaluation_notes: str | None = None


@dataclass
class GeneratedPost:
    """Сгенерированный текстовый пост."""

    title: str
    translated_title: str
    body: str
    summary: str
    short_body: str
    average_body: str
    hashtags: tuple[str, ...]

    def formatted(self) -> str:
        """Возвращает основной текст поста (без хэштегов)."""
        return self.body


@dataclass
class ImageAsset:
    """Информация об изображении."""

    url: str
    source: str
    prompt: str | None = None


@dataclass
class PublicationRecord:
    """Запись для сохранения в Google Sheets."""

    date: datetime | str
    source: str
    title: str
    link: str
    summary: str
    post: GeneratedPost
    image: ImageAsset
    score: int
    image_source: str
    status: str = "Written"
    notes: str | None = None
    telegraph_link: str | None = None
    vk_post_link: str | None = None
    tg_post_link: str | None = None

    def as_row(self) -> list[str]:
        """Преобразует запись в строку для Google Sheets."""
        hashtags_line = " ".join(f"#{tag}" for tag in self.post.hashtags)
        date_value = self.date.isoformat() if isinstance(self.date, datetime) else str(self.date)
        return [
            date_value,
            self.source,
            self.title,
            self.link,
            self.summary,
            self.post.short_body,
            self.post.average_body,
            self.post.title,
            self.post.formatted(),
            self.image.url,
            self.image_source,
            str(self.score),
            self.status,
            hashtags_line,
            self.notes or "",
            self.telegraph_link or "",
            self.vk_post_link or "",
            self.tg_post_link or "",
        ]


__all__: Sequence[str] = (
    "RawFeedEntry",
    "NewsItem",
    "RankedNews",
    "GeneratedPost",
    "ImageAsset",
    "PublicationRecord",
)
