from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.config import OpenAIConfig
from app.models import NewsItem
from app.post_generator import PostComposer, PostGenerationError


class DummyResponsesClient:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text
        self.calls = 0

    class _Wrapper:
        def __init__(self, parent: "DummyResponsesClient") -> None:
            self._parent = parent

        def create(self, model: str, input: str) -> SimpleNamespace:  # noqa: ARG002
            self._parent.calls += 1
            return SimpleNamespace(output_text=self._parent._output_text)

    @property
    def responses(self) -> "DummyResponsesClient._Wrapper":  # type: ignore[override]
        return DummyResponsesClient._Wrapper(self)


class DummyChatClient:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text
        self.calls = 0
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, parent: "DummyChatClient") -> None:
            self.completions = DummyChatClient._Completions(parent)

    class _Message:
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = DummyChatClient._Message(content)

    class _Completion:
        def __init__(self, content: str) -> None:
            self.choices = [DummyChatClient._Choice(content)]

    class _Completions:
        def __init__(self, parent: "DummyChatClient") -> None:
            self._parent = parent

        def create(self, model: str, messages: list[dict[str, str]]) -> "DummyChatClient._Completion":  # noqa: ARG002
            self._parent.calls += 1
            return DummyChatClient._Completion(self._parent._output_text)


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
    data = {"title": "Generated", "body": body, "hashtags": hashtags}
    return json.dumps(data)


def test_generate_returns_valid_post(news_item: NewsItem) -> None:
    payload = make_payload(1500, ["AI", "Automation", "Innovation"])
    composer = PostComposer(
        OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="img"),
        client=DummyResponsesClient(payload),
    )

    post = composer.generate(news_item)

    assert post.title == "Generated"
    assert len(post.body) == 1500
    assert post.hashtags == ("AI", "Automation", "Innovation")


def test_generate_raises_on_short_text(news_item: NewsItem) -> None:
    payload = make_payload(1200, ["AI", "Automation", "Innovation"])
    composer = PostComposer(
        OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="img"),
        client=DummyResponsesClient(payload),
    )

    with pytest.raises(PostGenerationError):
        composer.generate(news_item)


def test_generate_raises_on_invalid_json(news_item: NewsItem) -> None:
    composer = PostComposer(
        OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="img"),
        client=DummyResponsesClient("invalid json"),
    )

    with pytest.raises(PostGenerationError):
        composer.generate(news_item)


def test_generate_uses_chat_completion(news_item: NewsItem) -> None:
    payload = make_payload(1800, ["AI", "Automation", "Future"])
    composer = PostComposer(
        OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="img"),
        client=DummyChatClient(payload),  # type: ignore[arg-type]
    )

    post = composer.generate(news_item)

    assert post.body.startswith("A")
    assert post.hashtags == ("AI", "Automation", "Future")
