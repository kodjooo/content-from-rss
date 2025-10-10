from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.config import OpenAIConfig
from app.models import NewsItem
from app.post_generator import PostComposer, PostGenerationError


class DummyClient:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text
        self.calls = 0

    class _Wrapper:
        def __init__(self, parent: "DummyClient") -> None:
            self._parent = parent

        def create(self, model: str, input: str) -> SimpleNamespace:  # noqa: ARG002
            self._parent.calls += 1
            return SimpleNamespace(output_text=self._parent._output_text)

    @property
    def responses(self) -> "DummyClient._Wrapper":  # type: ignore[override]
        return DummyClient._Wrapper(self)


@pytest.fixture()
def news_item() -> NewsItem:
    return NewsItem(
        source="https://example.com/feed",
        title="AI breakthrough",
        link="https://example.com/1",
        summary="New AI model released",
        published=None,
        keywords=("AI",),
        media_url=None,
    )


def make_payload(body_length: int, hashtags: list[str]) -> str:
    body = "A" * body_length
    data = {
        "title": "Generated",
        "summary": "Краткое описание новости.",
        "short_body": "Короткая версия поста до 600 символов.",
        "body": body,
        "hashtags": hashtags,
    }
    return json.dumps(data)


def test_generate_returns_valid_post(news_item: NewsItem) -> None:
    payload = make_payload(1500, ["AI", "Automation", "Innovation"])
    composer = PostComposer(
        OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="img"),
        client=DummyClient(payload),
    )

    post = composer.generate(news_item)

    assert post.title == "Generated"
    assert post.summary.startswith("Краткое")
    assert len(post.body) == 1500
    assert post.short_body.startswith("Короткая версия")
    assert post.hashtags == ("AI", "Automation", "Innovation")


def test_generate_raises_on_short_text(news_item: NewsItem) -> None:
    payload = make_payload(700, ["AI", "Automation", "Innovation"])
    composer = PostComposer(
        OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="img"),
        client=DummyClient(payload),
    )

    with pytest.raises(PostGenerationError):
        composer.generate(news_item)


def test_generate_raises_on_invalid_json(news_item: NewsItem) -> None:
    composer = PostComposer(
        OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="img"),
        client=DummyClient("invalid json"),
    )

    with pytest.raises(PostGenerationError):
        composer.generate(news_item)
