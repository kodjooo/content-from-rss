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
        return GeneratedPost(
            title=payload["title"],
            body=payload["body"],
            summary=payload["summary"],
            hashtags=hashtags,
        )

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
            "Пиши на русском языке.\n"
            "Сгенерируй полноценный пост длиной 1500–3000 символов на основе новости.\n"
            "Это не пересказ, а аналитический и вовлекающий пост для соцсетей.\n\n"

            "Задача — выделить главные факты, объяснить, почему это важно именно сейчас, и показать, как это повлияет на будущее IT, бизнеса и автоматизации.\n"
            "Сфокусируйся на практических выводах: как это можно использовать, какие возможности или риски открываются, какие профессии или процессы могут измениться.\n"
            "Добавь личную позицию от первого лица — что ты об этом думаешь, согласен ли с подходом, какие последствия видишь.\n"
            "Если новость вызывает противоречия или может разделить мнения, подчеркни этот аспект и закончи открытым вопросом для обсуждения.\n\n"

            "Стиль: уверенный, живой, профессиональный. Без клише, без общих фраз, без лишней воды.\n"
            "Пиши как эксперт, который делится личным опытом и размышлениями с коллегами и предпринимателями.\n"
            "Используй конкретные примеры, короткие абзацы и естественную структуру.\n\n"

            "Аудитория: владельцы бизнеса, IT- и AI-специалисты, автоматизаторы и менеджеры, которым важно понимать реальные тенденции, а не маркетинг.\n\n"

            "Формат ответа — строго JSON:\n"
            "title — короткий, цепляющий заголовок (до 100 символов).\n"
            "summary — краткое изложение сути новости (300–400 символов).\n"
            "body — сам текст поста длиной 1500–3000 символов.\n"
            "hashtags — список из 3–4 ключевых слов без символа #.\n\n"

            "Пример:\n"
            "{"
            "\"title\": \"ИИ перестал быть инструментом — он стал бизнес-партнёром\", "
            "\"summary\": \"OpenAI представил новую модель, способную не только генерировать код, но и самостоятельно выполнять задачи. "
            "Это открывает путь к автономным агентам в бизнесе.\", "
            "\"body\": \"<Текст поста длиной 1500–3000 символов>.\", "
            "\"hashtags\": [\"AI\", \"Automation\", \"Business\", \"Productivity\"]"
            "}\n\n"

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
        for field in ("title", "summary", "body", "hashtags"):
            if field not in payload:
                raise PostGenerationError(f"Отсутствует поле {field}")
        body = payload["body"]
        summary = payload["summary"]
        hashtags = payload["hashtags"]
        if not isinstance(body, str):
            raise PostGenerationError("Поле body должно быть строкой")
        length = len(body)
        if length < 1500 or length > 3000:
            raise PostGenerationError(f"Длина текста вне требуемого диапазона: {length}")
        if not isinstance(summary, str) or not summary.strip():
            raise PostGenerationError("Поле summary должно быть непустой строкой")
        if not isinstance(hashtags, Iterable):
            raise PostGenerationError("Поле hashtags должно быть массивом")
        hashtags_list = [tag for tag in hashtags if isinstance(tag, str) and tag.strip()]
        if len(hashtags_list) < 3 or len(hashtags_list) > 4:
            raise PostGenerationError("Количество хэштегов должно быть от 3 до 4")
        payload["hashtags"] = hashtags_list


__all__: Sequence[str] = ("PostComposer", "PostGenerationError")
