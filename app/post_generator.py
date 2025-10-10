"""Генерация финального поста через OpenAI."""

from __future__ import annotations

import json
import logging
from typing import Iterable, Sequence

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import OpenAIConfig
from .models import GeneratedPost, NewsItem

logger = logging.getLogger(__name__)


class PostGenerationError(RuntimeError):
    """Ошибка генерации поста."""


class PostComposer:
    """Класс, генерирующий конечный текст поста."""

    def __init__(self, config: OpenAIConfig, client: OpenAI | None = None) -> None:
        self._config = config
        self._client = client or OpenAI(api_key=config.api_key)

    def generate(self, item: NewsItem) -> GeneratedPost:
        """Генерирует пост для новости."""
        last_error: PostGenerationError | None = None
        for attempt in range(2):
            raw_text = self._request_post(item)
            try:
                payload = self._parse_payload(raw_text)
                self._validate_payload(payload)
                hashtags = tuple(payload["hashtags"])
                return GeneratedPost(
                    title=payload["title"],
                    body=payload["body"],
                    summary=payload["summary"],
                    short_body=payload["short_body"],
                    hashtags=hashtags,
                )
            except PostGenerationError as exc:
                last_error = exc
                if ("Некорректный JSON" in str(exc) and attempt < 1) or ("Длина текста вне" in str(exc) and attempt < 2):
                    logger.warning("Повторная попытка генерации поста для %s: %s", item.link, str(exc))
                    continue
                raise
        if last_error:
            raise last_error
        raise PostGenerationError("Не удалось сформировать пост")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _request_post(self, item: NewsItem) -> str:
        """Отправляет запрос в OpenAI."""
        prompt = self._build_prompt(item)
        logger.debug("Промт генерации поста для %s: %s", item.link, prompt)
        response = self._client.responses.create(
            model=self._config.model_post,
            input=prompt,
        )
        text = getattr(response, "output_text", "")
        if not text:
            raise PostGenerationError("Пустой ответ модели")
        return text

    def _build_prompt(self, item: NewsItem) -> str:
        """Формирует промт для модели."""
        keywords = ", ".join(item.keywords) or "AI"
        return (
            "Ты — Марк Аборчи, AI-специалист и IT-автоматизатор с прошлым опытом проектного менеджмента.\n"
            "Пиши строго на русском языке.\n"
            "Сгенерируй две версии поста: длинную (1000–1500 символов) и короткую (до 600 символов).\n"
            "Обе версии должны быть аналитическими, с акцентом на практические выводы для IT-специалистов и владельцев бизнеса.\n\n"

            "Требования:\n"
            "- Делай выводы и разворачивай личную позицию от первого лица.\n"
            "- В длинной версии выделяй ключевые мысли при помощи **жирного** форматирования (там, где это уместно).\n"
            "- Не вставляй хэштеги в тексты длинной и короткой версий.\n"
            "- Короткая версия должна быть ёмким пересказом основных тезисов (до 600 символов).\n"
            "- Используй только русские слова в списке hashtags (кириллица), без символа #.\n"
            "- Хэштеги отражают ключевые темы новости.\n\n"

            "Формат ответа — строго JSON:\n"
            "title — заголовок до 100 символов.\n"
            "summary — краткое изложение (300–400 символов).\n"
            "short_body — короткая версия поста (до 600 символов, без хэштегов).\n"
            "body — длинная версия поста (1000–1500 символов, без хэштегов).\n"
            "hashtags — список из 3–4 русских слов без символа #.\n\n"

            "Пример ответа:\n"
            "{"
            "\"title\": \"ИИ перестал быть инструктором — он стал бизнес-партнёром\","
            "\"summary\": \"Краткое резюме на 300–400 символов...\","
            "\"short_body\": \"Сжатый текст до 600 символов...\","
            "\"body\": \"Развёрнутый текст 1000–1500 символов с выделениями **жирным**...\","
            "\"hashtags\": [\"инвестиции\", \"автоматизация\", \"управление\"]"
            "}\n\n"

            "Не добавляй никаких дополнительных полей и не используй markdown помимо **жирного**.\n"
            f"Заголовок новости: {item.title}\n"
            f"Описание новости: {item.summary}\n"
            f"Ключевые слова: {keywords}"
        )

    def _parse_payload(self, text: str) -> dict[str, object]:
        """Парсит JSON-ответ модели."""
        try:
            return json.loads(text)
        except json.JSONDecodeError as err:
            logger.error("Ответ модели не является корректным JSON: %s", text)
            raise PostGenerationError("Некорректный JSON в ответе модели") from err

    def _validate_payload(self, payload: dict[str, object]) -> None:
        """Проверяет структуру и ограничения результата."""
        for field in ("title", "summary", "short_body", "body", "hashtags"):
            if field not in payload:
                raise PostGenerationError(f"Отсутствует поле {field}")
        body = payload["body"]
        summary = payload["summary"]
        short_body = payload["short_body"].strip()
        hashtags = payload["hashtags"]
        if not isinstance(body, str):
            raise PostGenerationError("Поле body должно быть строкой")
        length = len(body)
        if length < 800 or length > 1700:
            raise PostGenerationError(f"Длина текста вне требуемого диапазона: {length}")
        if not isinstance(summary, str) or not summary.strip():
            raise PostGenerationError("Поле summary должно быть непустой строкой")
        if not isinstance(short_body, str) or not short_body.strip():
            raise PostGenerationError("Поле short_body должно быть непустой строкой")
        if len(short_body) > 600:
            raise PostGenerationError("Поле short_body превышает 600 символов")
        if not isinstance(hashtags, Iterable):
            raise PostGenerationError("Поле hashtags должно быть массивом")
        hashtags_list = [tag for tag in hashtags if isinstance(tag, str) and tag.strip()]
        if len(hashtags_list) < 3 or len(hashtags_list) > 4:
            raise PostGenerationError("Количество хэштегов должно быть от 3 до 4")
        payload["hashtags"] = hashtags_list


__all__: Sequence[str] = ("PostComposer", "PostGenerationError")
