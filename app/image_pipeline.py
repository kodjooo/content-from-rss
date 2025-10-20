"""Подбор и генерация изображений."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Sequence

import requests
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import FreeImageHostConfig, OpenAIConfig, PexelsConfig
from .models import GeneratedPost, ImageAsset, NewsItem

logger = logging.getLogger(__name__)


class ImageGenerationError(RuntimeError):
    """Ошибка подбора или генерации изображения."""


@dataclass
class _ImageCandidate:
    data: bytes
    source: str
    prompt: str | None = None


class ImageSelector:
    """Определяет изображение для новости и загружает его на FreeImageHost."""

    def __init__(
        self,
        pexels_config: PexelsConfig,
        freeimage_config: FreeImageHostConfig,
        openai_config: OpenAIConfig,
        session: requests.Session | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._pexels = pexels_config
        self._freeimage = freeimage_config
        self._openai_cfg = openai_config
        self._session = session or requests.Session()
        self._client = client

    def select(self, news: NewsItem, post: GeneratedPost) -> ImageAsset:
        """Основной метод подбора."""
        candidate = (
            self._from_rss(news)
            or self._from_pexels(news, post)
            or self._generate_image(news, post)
        )
        if not candidate:
            raise ImageGenerationError("Не удалось получить изображение")
        url = self._upload(candidate.data)
        return ImageAsset(url=url, source=candidate.source, prompt=candidate.prompt)

    def _from_rss(self, news: NewsItem) -> _ImageCandidate | None:
        """Попытка скачать изображение из RSS."""
        if not news.media_url:
            return None
        try:
            response = self._session.get(news.media_url, timeout=self._freeimage.timeout)
            response.raise_for_status()
            if not self._is_image_response(response):
                return None
            return _ImageCandidate(data=response.content, source="rss")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось загрузить изображение из RSS %s: %s", news.media_url, exc)
            return None

    def _from_pexels(self, news: NewsItem, post: GeneratedPost) -> _ImageCandidate | None:
        """Поиск изображения в Pexels."""
        if not self._pexels.enabled:
            return None

        hashtags = [tag.replace("#", "").strip() for tag in post.hashtags if tag.strip()]
        query = " ".join(hashtags) if hashtags else (news.title or "artificial intelligence")
        headers = {"Authorization": self._pexels.api_key}
        params = {"query": query, "per_page": 1}
        try:
            response = self._session.get(
                "https://api.pexels.com/v1/search",
                headers=headers,
                params=params,
                timeout=self._pexels.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ошибка поиска в Pexels: %s", exc)
            return None

        photos = data.get("photos") or []
        if not photos:
            return None
        image_url = photos[0]["src"].get("large2x") or photos[0]["src"].get("large")
        if not image_url:
            return None

        try:
            image_resp = self._session.get(image_url, timeout=self._pexels.timeout)
            image_resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось скачать изображение с Pexels: %s", exc)
            return None
        return _ImageCandidate(data=image_resp.content, source="pexels")

    def _generate_image(self, news: NewsItem, post: GeneratedPost) -> _ImageCandidate | None:
        """Генерация изображения через OpenAI."""
        prompt = self._build_image_prompt(news, post)
        client = self._ensure_client()
        try:
            response = client.images.generate(
                model=self._openai_cfg.model_image,
                prompt=prompt,
                size=self._openai_cfg.image_size,
                quality=self._openai_cfg.image_quality,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка генерации изображения: %s", exc)
            return None

        data = response.data[0].get("b64_json")
        if not data:
            return None
        try:
            binary = base64.b64decode(data)
        except Exception as exc:  # noqa: BLE001
            logger.error("Не удалось декодировать изображение: %s", exc)
            return None
        return _ImageCandidate(data=binary, source="openai", prompt=prompt)

    def _build_image_prompt(self, news: NewsItem, post: GeneratedPost) -> str:
        """Генерирует промт для изображения."""
        return (
            "Photorealistic illustration for an article about artificial intelligence.\n"
            f"Headline: {news.title}\n"
            f"Summary: {news.summary[:200]}\n"
            f"Post gist: {post.body[:200]}"
        )

    def _is_image_response(self, response: requests.Response) -> bool:
        """Проверяет, что ответ является изображением."""
        content_type = response.headers.get("Content-Type", "")
        return content_type.startswith("image/")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _upload(self, data: bytes) -> str:
        """Загружает файл на FreeImageHost."""
        files = {"source": ("image.jpg", data)}
        payload = {"key": self._freeimage.api_key}
        response = self._session.post(
            self._freeimage.endpoint,
            data=payload,
            files=files,
            timeout=self._freeimage.timeout,
        )
        response.raise_for_status()
        json_payload = response.json()
        image = json_payload.get("image") or {}
        url = image.get("url") or image.get("display_url")
        if not url:
            raise ImageGenerationError("Сервис не вернул ссылку на изображение")
        return url

    def _ensure_client(self) -> OpenAI:
        """Ленивая инициализация клиента OpenAI."""
        if self._client is None:
            self._client = OpenAI(api_key=self._openai_cfg.api_key_image or self._openai_cfg.api_key)
        return self._client


__all__: Sequence[str] = ("ImageSelector", "ImageGenerationError")
