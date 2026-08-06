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
        "translated_title": "Сгенерированный заголовок",
        "summary": "Краткое описание новости.",
        "short_body": "Короткая версия поста до 600 символов.",
        "average_body": "Средняя версия поста " + "A" * (min(body_length, 300)),
        "body": body,
        "hashtags": hashtags,
    }
    return json.dumps(data)


def test_generate_returns_valid_post(news_item: NewsItem) -> None:
    payload = make_payload(1500, ["AI", "Automation", "Innovation"])
    composer = PostComposer(
        OpenAIConfig(
            api_key="test",
            api_key_image="test-images",
            model_rank="gpt",
            model_post="gpt",
            model_image="img",
            image_quality="medium",
            image_size="1024x1024",
        ),
        client=DummyClient(payload),
    )

    post = composer.generate(news_item)

    assert post.title == "Generated"
    assert post.summary.startswith("Краткое")
    assert len(post.body) == 1500
    assert post.short_body.startswith("Короткая версия")
    assert post.average_body.startswith("Средняя версия поста")
    assert post.hashtags == ("AI", "Automation", "Innovation")


def test_generate_raises_on_short_text(news_item: NewsItem) -> None:
    payload = make_payload(700, ["AI", "Automation", "Innovation"])
    composer = PostComposer(
        OpenAIConfig(
            api_key="test",
            api_key_image="test-images",
            model_rank="gpt",
            model_post="gpt",
            model_image="img",
            image_quality="medium",
            image_size="1024x1024",
        ),
        client=DummyClient(payload),
    )

    with pytest.raises(PostGenerationError):
        composer.generate(news_item)


def test_generate_raises_on_invalid_json(news_item: NewsItem) -> None:
    composer = PostComposer(
        OpenAIConfig(
            api_key="test",
            api_key_image="test-images",
            model_rank="gpt",
            model_post="gpt",
            model_image="img",
            image_quality="medium",
            image_size="1024x1024",
        ),
        client=DummyClient("invalid json"),
    )

    with pytest.raises(PostGenerationError):
        composer.generate(news_item)


def test_generate_raises_on_average_overflow(news_item: NewsItem) -> None:
    data = {
        "title": "Generated",
        "translated_title": "Сгенерированный заголовок",
        "summary": "Краткое описание новости.",
        "short_body": "Короткая версия поста до 600 символов.",
        "average_body": "A" * 1001,
        "body": "B" * 1200,
        "hashtags": ["AI", "Automation", "Innovation"],
    }
    composer = PostComposer(
        OpenAIConfig(
            api_key="test",
            api_key_image="test-images",
            model_rank="gpt",
            model_post="gpt",
            model_image="img",
            image_quality="medium",
            image_size="1024x1024",
        ),
        client=DummyClient(json.dumps(data)),
    )

    with pytest.raises(PostGenerationError):
        composer.generate(news_item)


def make_composer(payload: str) -> PostComposer:
    return PostComposer(
        OpenAIConfig(
            api_key="test",
            api_key_image="test-images",
            model_rank="gpt",
            model_post="gpt",
            model_image="img",
            image_quality="medium",
            image_size="1024x1024",
        ),
        client=DummyClient(payload),
    )


def test_generate_rejects_banned_phrase(news_item: NewsItem) -> None:
    """Маркеры нейрослопа не должны проходить в готовый пост."""
    data = json.loads(make_payload(1300, ["ии", "автоматизация", "промпты"]))
    data["body"] = "Б" * 1200 + " Владельцам бизнеса стоит следить за этим."
    composer = make_composer(json.dumps(data))

    with pytest.raises(PostGenerationError):
        composer.generate(news_item)


def test_generate_rejects_question_in_title(news_item: NewsItem) -> None:
    data = json.loads(make_payload(1300, ["ии", "автоматизация", "промпты"]))
    data["title"] = "Нужен ли тут программист?"
    composer = make_composer(json.dumps(data))

    with pytest.raises(PostGenerationError):
        composer.generate(news_item)


def test_generate_rejects_multiple_questions_in_body(news_item: NewsItem) -> None:
    data = json.loads(make_payload(1300, ["ии", "автоматизация", "промпты"]))
    data["body"] = "В" * 1200 + " А у вас как? С чем столкнулись? Что выбрали?"
    composer = make_composer(json.dumps(data))

    with pytest.raises(PostGenerationError):
        composer.generate(news_item)


def test_prompt_variants_differ_between_news() -> None:
    """Комбинация приёмов подачи зависит от новости — посты не должны быть однотипными."""
    composer = make_composer(make_payload(1300, ["ии", "автоматизация", "промпты"]))
    variants = set()
    for index in range(12):
        item = NewsItem(
            source="https://example.com/feed",
            title=f"News {index}",
            link=f"https://example.com/news/{index}",
            summary="Summary",
            published=None,
            keywords=("AI",),
            media_url=None,
        )
        variants.add(composer._variants(item))
    assert len(variants) > 1
