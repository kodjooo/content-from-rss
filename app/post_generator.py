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
        raw_text = self._request_post(item)
        payload = self._parse_payload(raw_text)
        self._validate_payload(payload)
        hashtags = tuple(payload["hashtags"])
        return GeneratedPost(title=payload["title"], body=payload["body"], hashtags=hashtags)

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
            "Сгенерируй пост длиной 1500-3000 символов на основе новости.\n"
            "Используй выделенные ключевые факты, добавь контекст и вывод. Нужно сделать акцент на том, как это повлияет на будущее развития IT и AI.\n"
            "Стиль: профессиональный, но не сухой. Должно быть полезно AI-специалистам, IT-специалистам и предпринимателям. Избегай «воды», используй реальные кейсы и понятные примеры.\n"
            "Подготовь 3-4 хэштега без символа # (мы добавим его сами).\n"
            "Ответ предоставь в формате JSON с ключами title, body, hashtags.\n"
            "Пример: {\"title\": \"...\", \"body\": \"...\", \"hashtags\": [\"AI\", \"Automation\"]}.\n\n"
            f"Заголовок: {item.title}\n"
            f"Описание: {item.summary}\n"
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
        for field in ("title", "body", "hashtags"):
            if field not in payload:
                raise PostGenerationError(f"Отсутствует поле {field}")
        body = payload["body"]
        hashtags = payload["hashtags"]
        if not isinstance(body, str):
            raise PostGenerationError("Поле body должно быть строкой")
        length = len(body)
        if length < 1500 or length > 3000:
            raise PostGenerationError(f"Длина текста вне требуемого диапазона: {length}")
        if not isinstance(hashtags, Iterable):
            raise PostGenerationError("Поле hashtags должно быть массивом")
        hashtags_list = [tag for tag in hashtags if isinstance(tag, str) and tag.strip()]
        if len(hashtags_list) < 3 or len(hashtags_list) > 4:
            raise PostGenerationError("Количество хэштегов должно быть от 3 до 4")
        payload["hashtags"] = hashtags_list


__all__: Sequence[str] = ("PostComposer", "PostGenerationError")
