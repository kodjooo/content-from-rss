"""Оценка релевантности новостей через OpenAI."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import OpenAIConfig
from .models import NewsItem, RankedNews

logger = logging.getLogger(__name__)

SCORE_PATTERN = re.compile(r"(\d{1,2})")


class RelevanceScorer:
    """Клиент OpenAI для оценки релевантности."""

    def __init__(self, config: OpenAIConfig, cache_dir: Path, client: OpenAI | None = None) -> None:
        self._config = config
        self._cache_path = cache_dir / "relevance_cache.json"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()
        self._client = client or OpenAI(api_key=config.api_key)

    def evaluate_many(self, items: Iterable[NewsItem]) -> list[RankedNews]:
        """Оценивает список новостей."""
        results: list[RankedNews] = []
        for item in items:
            ranked = self.evaluate(item)
            if ranked:
                results.append(ranked)
        return results

    def evaluate(self, item: NewsItem) -> RankedNews | None:
        """Возвращает оценку для конкретной новости."""
        if item.link in self._cache:
            cached = self._cache[item.link]
            return RankedNews(news=item, score=cached["score"], evaluation_notes=cached.get("notes"))
        try:
            response_text = self._request_score(item)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка оценки новости %s: %s", item.link, exc)
            return None

        score, notes = self._parse_score(response_text)
        if score is None:
            logger.warning("Не удалось извлечь оценку для новости %s", item.link)
            return None

        ranked = RankedNews(news=item, score=score, evaluation_notes=notes)
        self._cache[item.link] = {"score": score, "notes": notes}
        self._save_cache()
        return ranked

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _request_score(self, item: NewsItem) -> str:
        """Отправляет запрос в OpenAI и возвращает текст ответа."""
        prompt = self._build_prompt(item)
        logger.debug("Промт для оценки новости %s: %s", item.link, prompt)
        response = self._client.responses.create(
            model=self._config.model_rank,
            input=prompt,
        )
        return getattr(response, "output_text", "").strip()

    def _build_prompt(self, item: NewsItem) -> str:
        """Формирует текст промта."""
        return (
            "Оцени насколько эта новость даёт повод для практического применения, спора или инсайта для AI-специалистов и владельцев бизнеса (по шкале от 1 до 10).\n"
            "Ответь в формате 'score: <число> — <краткий комментарий>'.\n\n"
            f"Заголовок: {item.title}\n"
            f"Описание: {item.summary}\n"
            f"Ссылка: {item.link}"
        )

    def _parse_score(self, text: str) -> tuple[int | None, str | None]:
        """Извлекает числовую оценку из ответа модели."""
        if not text:
            return None, None
        match = SCORE_PATTERN.search(text)
        if not match:
            return None, text
        score = int(match.group(1))
        score = max(1, min(score, 10))
        notes = text
        return score, notes

    def _load_cache(self) -> dict[str, dict[str, object]]:
        """Загружает кэш из файла."""
        if self._cache_path.exists():
            try:
                return json.loads(self._cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Поврежден кэш оценок, создается заново.")
        return {}

    def _save_cache(self) -> None:
        """Сохраняет кэш на диск."""
        self._cache_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")


__all__: Sequence[str] = ("RelevanceScorer",)
