from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import OpenAIConfig
from app.models import NewsItem
from app.scoring import RelevanceScorer


class DummyClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: int = 0

    def responses(self) -> None:  # pragma: no cover - заглушка
        raise NotImplementedError

    class _Wrapper:
        def __init__(self, parent: "DummyClient") -> None:
            self._parent = parent

        def create(self, model: str, input: str) -> SimpleNamespace:  # noqa: ARG002
            self._parent.calls += 1
            return SimpleNamespace(output_text=self._parent._responses[self._parent.calls - 1])

    @property
    def responses(self) -> "DummyClient._Wrapper":  # type: ignore[override]
        return DummyClient._Wrapper(self)


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
    config = OpenAIConfig(
        api_key="test",
        api_key_image="test-images",
        model_rank="gpt",
        model_post="gpt",
        model_image="gpt-image",
        image_quality="medium",
        image_size="1024x1024",
    )
    client = DummyClient(["score: 9 — стоит опубликовать"])
    scorer = RelevanceScorer(config, cache_dir=tmp_cache, client=client)

    ranked = scorer.evaluate(news_item)

    assert ranked is not None
    assert ranked.score == 9
    assert client.calls == 1


def test_evaluate_uses_cache(tmp_cache: Path, news_item: NewsItem) -> None:
    config = OpenAIConfig(
        api_key="test",
        api_key_image="test-images",
        model_rank="gpt",
        model_post="gpt",
        model_image="gpt-image",
        image_quality="medium",
        image_size="1024x1024",
    )
    client = DummyClient(["score: 8"])
    scorer = RelevanceScorer(config, cache_dir=tmp_cache, client=client)

    first = scorer.evaluate(news_item)
    second = scorer.evaluate(news_item)

    assert first is not None and second is not None
    assert first.score == second.score == 8
    assert client.calls == 1


def test_evaluate_handles_invalid_response(tmp_cache: Path, news_item: NewsItem) -> None:
    config = OpenAIConfig(
        api_key="test",
        api_key_image="test-images",
        model_rank="gpt",
        model_post="gpt",
        model_image="gpt-image",
        image_quality="medium",
        image_size="1024x1024",
    )
    client = DummyClient(["непонятный ответ"])
    scorer = RelevanceScorer(config, cache_dir=tmp_cache, client=client)

    ranked = scorer.evaluate(news_item)

    assert ranked is None


def test_score_prompt_contains_priorities(tmp_cache: Path, news_item: NewsItem) -> None:
    """В промте оценки есть приоритеты тем и фильтр на практическую применимость."""
    scorer = RelevanceScorer(
        OpenAIConfig(
            api_key="test",
            api_key_image="test-images",
            model_rank="gpt",
            model_post="gpt",
            model_image="img",
            image_quality="medium",
            image_size="1024x1024",
        ),
        tmp_cache,
        client=DummyClient(["score: 9 — ok"]),
    )

    prompt = scorer._build_prompt(news_item)

    assert "понедельник утром" in prompt
    assert "рынок труда" in prompt
    assert "инвестиционные раунды" in prompt
    assert "антимонопольные расследования" in prompt
    assert "вайб-кодингом" in prompt
