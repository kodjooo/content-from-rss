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
    api_key_image: str
    model_rank: str
    model_post: str
    model_image: str
    image_quality: str
    image_size: str
    image_generation_enabled: bool = False


@dataclass(frozen=True)
class PexelsConfig:
    """Настройки Pexels API."""

    api_key: str
    timeout: int
    enabled: bool = True


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
    run_times: tuple[tuple[int, int], ...] = field(default=((17, 50),))
    run_once_on_start: bool = True

    @property
    def lookback_hours(self) -> int:
        """Глубина окна отбора новостей — интервал между запусками плюс час запаса."""
        if len(self.run_times) < 2:
            return 25
        minutes = sorted(hour * 60 + minute for hour, minute in self.run_times)
        gaps = [second - first for first, second in zip(minutes, minutes[1:])]
        gaps.append(minutes[0] + 24 * 60 - minutes[-1])
        return max(2, round(max(gaps) / 60) + 1)


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
    pipeline_max_posts: int = 2


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


def _parse_run_times(env_value: str | None) -> tuple[tuple[int, int], ...]:
    """Разбирает расписание вида '17:50' или '7:00,19:30' в пары (час, минута)."""
    if not env_value or not env_value.strip():
        return ((17, 50),)
    times: list[tuple[int, int]] = []
    for chunk in env_value.split(","):
        raw = chunk.strip()
        if not raw:
            continue
        hour_part, _, minute_part = raw.partition(":")
        try:
            hour = int(hour_part)
            minute = int(minute_part) if minute_part else 0
        except ValueError as err:
            raise ValueError(f"Некорректное время запуска в SCHEDULER_RUN_TIMES: {raw}") from err
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError(f"Время запуска вне диапазона в SCHEDULER_RUN_TIMES: {raw}")
        times.append((hour, minute))
    if not times:
        return ((17, 50),)
    return tuple(sorted(set(times)))


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

    # Декоративные картинки к новостям не дают прироста охвата (проверено на архиве
    # постов), поэтому по умолчанию генерация и поиск иллюстраций отключены:
    # картинка берётся только из самой новости, если она там есть.
    images_enabled = _as_bool(os.getenv("ENABLE_IMAGE_GENERATION"), default=False)

    openai_cfg = OpenAIConfig(
        api_key=_require(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY"),
        api_key_image=os.getenv("OPENAI_IMAGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
        model_rank=os.getenv("OPENAI_MODEL_RANK", "gpt-4o-mini"),
        model_post=os.getenv("OPENAI_MODEL_POST", "gpt-4o-mini"),
        model_image=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
        image_quality=os.getenv("IMAGE_QUALITY", "medium"),
        image_size=os.getenv("IMAGE_SIZE", "1024x1024"),
        image_generation_enabled=images_enabled,
    )

    skip_pexels = _as_bool(os.getenv("SKIP_PEXELS_SEARCH"), default=False) or not images_enabled
    pexels_key = os.getenv("PEXELS_API_KEY")
    if skip_pexels:
        pexels_key = pexels_key or ""
    else:
        pexels_key = _require(pexels_key, "PEXELS_API_KEY")

    pexels = PexelsConfig(
        api_key=pexels_key,
        timeout=int(os.getenv("PEXELS_API_TIMEOUT", "20")),
        enabled=not skip_pexels,
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
        run_times=_parse_run_times(os.getenv("SCHEDULER_RUN_TIMES")),
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
        pipeline_max_posts=int(os.getenv("PIPELINE_MAX_POSTS", "2")),
    )


__all__: Sequence[str] = ("load_settings", "AppConfig")
