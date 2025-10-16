from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
import requests
import responses

from app.config import FreeImageHostConfig, OpenAIConfig, PexelsConfig
from app.image_pipeline import ImageGenerationError, ImageSelector
from app.models import GeneratedPost, NewsItem


@pytest.fixture()
def news_item() -> NewsItem:
    return NewsItem(
        source="https://example.com/feed",
        title="AI breakthrough",
        link="https://example.com/1",
        summary="New AI model released",
        published=None,
        keywords=("AI",),
        media_url="https://cdn.example.com/image.jpg",
    )


@pytest.fixture()
def generated_post() -> GeneratedPost:
    return GeneratedPost(
        title="Generated",
        translated_title="Переведённый заголовок",
        body="A" * 1500,
        summary="Краткое описание",
        short_body="Сжатая версия поста",
        average_body="Средний формат поста",
        hashtags=("AI", "Automation", "Innovation"),
    )


@pytest.fixture()
def config_bundle(tmp_path_factory) -> tuple[PexelsConfig, FreeImageHostConfig, OpenAIConfig, requests.Session]:
    session = requests.Session()
    return (
        PexelsConfig(api_key="pexels", timeout=5, enabled=True),
        FreeImageHostConfig(api_key="freeimage", endpoint="https://freeimage.host/api/1/upload", timeout=5),
        OpenAIConfig(api_key="openai", model_rank="gpt", model_post="gpt", model_image="img"),
        session,
    )


@responses.activate
def test_select_uses_rss_image(news_item: NewsItem, generated_post: GeneratedPost, config_bundle) -> None:
    pexels, freeimage, openai_config, session = config_bundle
    responses.add(responses.GET, news_item.media_url, body=b"imgdata", status=200, content_type="image/jpeg")
    responses.add(
        responses.POST,
        freeimage.endpoint,
        json={"image": {"url": "https://freeimage.host/img/123"}},
        status=200,
    )

    selector = ImageSelector(pexels, freeimage, openai_config, session=session, client=None)
    asset = selector.select(news_item, generated_post)

    assert asset.url == "https://freeimage.host/img/123"
    assert asset.source == "rss"


@responses.activate
def test_select_falls_back_to_pexels(generated_post: GeneratedPost, config_bundle) -> None:
    pexels, freeimage, openai_config, session = config_bundle
    news = NewsItem(
        source="https://example.com/feed",
        title="AI robotics",
        link="https://example.com/2",
        summary="Robotics update",
        published=None,
        keywords=("AI",),
        media_url=None,
    )
    responses.add(
        responses.GET,
        "https://api.pexels.com/v1/search",
        json={
            "photos": [
                {
                    "src": {
                        "large2x": "https://images.pexels.com/photos/1.jpeg",
                    }
                }
            ]
        },
        status=200,
    )
    responses.add(
        responses.GET,
        "https://images.pexels.com/photos/1.jpeg",
        body=b"pexelsdata",
        status=200,
        content_type="image/jpeg",
    )
    responses.add(
        responses.POST,
        freeimage.endpoint,
        json={"image": {"url": "https://freeimage.host/img/pexels"}},
        status=200,
    )

    selector = ImageSelector(pexels, freeimage, openai_config, session=session, client=None)
    asset = selector.select(news, generated_post)

    assert asset.source == "pexels"
    assert asset.url.endswith("pexels")


class DummyImageClient:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    class _ImageWrapper:
        def __init__(self, payload: str) -> None:
            self._payload = payload

        def generate(self, model: str, prompt: str, size: str) -> SimpleNamespace:  # noqa: ARG002
            return SimpleNamespace(data=[{"b64_json": self._payload}])

    @property
    def images(self) -> "DummyImageClient._ImageWrapper":  # type: ignore[override]
        return DummyImageClient._ImageWrapper(self._payload)


@responses.activate
def test_select_generates_image_when_sources_fail(generated_post: GeneratedPost, config_bundle) -> None:
    pexels, freeimage, openai_config, session = config_bundle
    news = NewsItem(
        source="https://example.com/feed",
        title="AI robotics",
        link="https://example.com/3",
        summary="Robotics summary",
        published=None,
        keywords=("AI",),
        media_url=None,
    )
    responses.add(
        responses.GET,
        "https://api.pexels.com/v1/search",
        json={"photos": []},
        status=200,
    )
    responses.add(
        responses.POST,
        freeimage.endpoint,
        json={"image": {"url": "https://freeimage.host/img/generated"}},
        status=200,
    )
    payload = base64.b64encode(b"image-bytes").decode()
    selector = ImageSelector(
        pexels,
        freeimage,
        openai_config,
        session=session,
        client=DummyImageClient(payload),
    )

    asset = selector.select(news, generated_post)

    assert asset.source == "openai"
    assert asset.url.endswith("generated")


@responses.activate
def test_select_skips_pexels_when_disabled(generated_post: GeneratedPost, config_bundle) -> None:
    _, freeimage, openai_config, session = config_bundle
    pexels_disabled = PexelsConfig(api_key="", timeout=5, enabled=False)
    news = NewsItem(
        source="https://example.com/feed",
        title="AI robotics",
        link="https://example.com/4",
        summary="Robotics summary",
        published=None,
        keywords=("AI",),
        media_url=None,
    )
    responses.add(
        responses.POST,
        freeimage.endpoint,
        json={"image": {"url": "https://freeimage.host/img/generated-disabled"}},
        status=200,
    )
    payload = base64.b64encode(b"image-disabled").decode()

    selector = ImageSelector(
        pexels_disabled,
        freeimage,
        openai_config,
        session=session,
        client=DummyImageClient(payload),
    )

    asset = selector.select(news, generated_post)

    assert asset.source == "openai"
    assert asset.url.endswith("generated-disabled")


@responses.activate
def test_select_raises_when_upload_fails(news_item: NewsItem, generated_post: GeneratedPost, config_bundle) -> None:
    pexels, freeimage, openai_config, session = config_bundle
    responses.add(responses.GET, news_item.media_url, body=b"imgdata", status=200, content_type="image/jpeg")
    responses.add(responses.POST, freeimage.endpoint, json={}, status=200)

    selector = ImageSelector(pexels, freeimage, openai_config, session=session, client=None)

    with pytest.raises(ImageGenerationError):
        selector.select(news_item, generated_post)
