from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import OpenAIConfig
from app.models import NewsItem
from app.scoring import RelevanceScorer


class DummyResponsesClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: int = 0

    def responses(self) -> None:  # pragma: no cover - заглушка
        raise NotImplementedError

    class _Wrapper:
        def __init__(self, parent: "DummyResponsesClient") -> None:
            self._parent = parent

        def create(self, model: str, input: str) -> SimpleNamespace:  # noqa: ARG002
            self._parent.calls += 1
            return SimpleNamespace(output_text=self._parent._responses[self._parent.calls - 1])

    @property
    def responses(self) -> "DummyResponsesClient._Wrapper":  # type: ignore[override]
        return DummyResponsesClient._Wrapper(self)


class DummyChatClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
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
            return DummyChatClient._Completion(self._parent._responses[self._parent.calls - 1])


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> Path:
    return tmp_path


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


def test_evaluate_parses_score(tmp_cache: Path, news_item: NewsItem) -> None:
    config = OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="gpt-image")
    client = DummyResponsesClient(["score: 9 — стоит опубликовать"])
    scorer = RelevanceScorer(config, cache_dir=tmp_cache, client=client)

    ranked = scorer.evaluate(news_item)

    assert ranked is not None
    assert ranked.score == 9
    assert client.calls == 1


def test_evaluate_uses_cache(tmp_cache: Path, news_item: NewsItem) -> None:
    config = OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="gpt-image")
    client = DummyResponsesClient(["score: 8"])
    scorer = RelevanceScorer(config, cache_dir=tmp_cache, client=client)

    first = scorer.evaluate(news_item)
    second = scorer.evaluate(news_item)

    assert first is not None and second is not None
    assert first.score == second.score == 8
    assert client.calls == 1


def test_evaluate_handles_invalid_response(tmp_cache: Path, news_item: NewsItem) -> None:
    config = OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="gpt-image")
    client = DummyResponsesClient(["непонятный ответ"])
    scorer = RelevanceScorer(config, cache_dir=tmp_cache, client=client)

    ranked = scorer.evaluate(news_item)

    assert ranked is None


def test_evaluate_falls_back_to_chat(tmp_cache: Path, news_item: NewsItem) -> None:
    config = OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="gpt-image")
    client = DummyChatClient(["score: 7 — приемлемо"])
    scorer = RelevanceScorer(config, cache_dir=tmp_cache, client=client)  # type: ignore[arg-type]

    ranked = scorer.evaluate(news_item)

    assert ranked is not None
    assert ranked.score == 7
    assert client.calls == 1
