"""Загрузка конфигурации приложения."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv


@dataclass(frozen=True)
class RSSConfig:
    """Настройки RSS источников."""

    sources: tuple[str, ...]
    keywords: tuple[str, ...]
    similarity_threshold: float
    max_items: int


@dataclass(frozen=True)
class OpenAIConfig:
    """Настройки взаимодействия с OpenAI."""

    api_key: str
    model_rank: str
    model_post: str
    model_image: str


@dataclass(frozen=True)
class PexelsConfig:
    """Настройки Pexels API."""

    api_key: str
    timeout: int


@dataclass(frozen=True)
class FreeImageHostConfig:
    """Настройки FreeImageHost API."""

    api_key: str
    endpoint: str
    timeout: int


@dataclass(frozen=True)
class SheetsConfig:
    """Настройки Google Sheets."""

    sheet_id: str
    service_account_json: Path
    worksheet: str


@dataclass(frozen=True)
class SchedulerConfig:
    """Настройки планировщика."""

    timezone: str
    run_hours: tuple[int, ...] = field(default=(7, 19))
    run_once_on_start: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Сводный объект конфигурации."""

    rss: RSSConfig
    openai: OpenAIConfig
    pexels: PexelsConfig
    freeimagehost: FreeImageHostConfig
    sheets: SheetsConfig
    scheduler: SchedulerConfig
    cache_dir: Path
    log_level: str


def _parse_list(env_value: str | None) -> tuple[str, ...]:
    """Преобразует строку из env в кортеж."""
    if not env_value:
        return ()
    return tuple(item.strip() for item in env_value.split(",") if item.strip())


def _require(value: str | None, name: str) -> str:
    """Проверяет наличие обязательных переменных."""
    if not value:
        raise ValueError(f"Не указана обязательная переменная окружения: {name}")
    return value


def _as_bool(value: str | None, default: bool = False) -> bool:
    """Преобразует строку в булево значение."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_settings(dotenv_path: str | None = None) -> AppConfig:
    """Загружает конфигурацию из переменных окружения."""
    load_dotenv(dotenv_path)

    rss_sources = _parse_list(os.getenv("RSS_SOURCES"))
    keywords = _parse_list(os.getenv("KEYWORDS"))
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    max_items = int(os.getenv("PIPELINE_MAX_ITEMS", "25"))

    rss = RSSConfig(
        sources=rss_sources,
        keywords=keywords,
        similarity_threshold=similarity_threshold,
        max_items=max_items,
    )

    openai_cfg = OpenAIConfig(
        api_key=_require(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY"),
        model_rank=os.getenv("OPENAI_MODEL_RANK", "gpt-4o-mini"),
        model_post=os.getenv("OPENAI_MODEL_POST", "gpt-4o-mini"),
        model_image=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
    )

    pexels = PexelsConfig(
        api_key=_require(os.getenv("PEXELS_API_KEY"), "PEXELS_API_KEY"),
        timeout=int(os.getenv("PEXELS_API_TIMEOUT", "20")),
    )

    freeimagehost = FreeImageHostConfig(
        api_key=_require(os.getenv("FREEIMAGEHOST_API_KEY"), "FREEIMAGEHOST_API_KEY"),
        endpoint=os.getenv(
            "FREEIMAGEHOST_API_ENDPOINT",
            "https://freeimage.host/api/1/upload",
        ),
        timeout=int(os.getenv("FREEIMAGEHOST_API_TIMEOUT", "30")),
    )

    sheets = SheetsConfig(
        sheet_id=_require(os.getenv("SHEET_ID"), "SHEET_ID"),
        service_account_json=Path(
            _require(
                os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"),
                "GOOGLE_SERVICE_ACCOUNT_JSON",
            )
        ),
        worksheet=os.getenv("SHEET_WORKSHEET", "Sheet1"),
    )

    scheduler = SchedulerConfig(
        timezone=os.getenv("SCHEDULER_TIMEZONE", "Europe/Moscow"),
        run_hours=(7, 19),
        run_once_on_start=_as_bool(os.getenv("RUN_PIPELINE_ON_START"), default=True),
    )

    cache_dir = Path(os.getenv("CACHE_DIR", ".cache"))
    log_level = os.getenv("LOG_LEVEL", "INFO")

    return AppConfig(
        rss=rss,
        openai=openai_cfg,
        pexels=pexels,
        freeimagehost=freeimagehost,
        sheets=sheets,
        scheduler=scheduler,
        cache_dir=cache_dir,
        log_level=log_level,
    )


__all__: Sequence[str] = ("load_settings", "AppConfig")
