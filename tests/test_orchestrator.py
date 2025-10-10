from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import (
    AppConfig,
    FreeImageHostConfig,
    OpenAIConfig,
    PexelsConfig,
    RSSConfig,
    SchedulerConfig,
    SheetsConfig,
)
from app.models import GeneratedPost, ImageAsset, NewsItem, RankedNews
from app.orchestrator import PipelineRunner


@dataclass
class DummyCollector:
    items: list[NewsItem]

    def collect(self) -> list[NewsItem]:
        return self.items


@dataclass
class DummyScorer:
    def __init__(self, score: int) -> None:
        self.score = score

    def evaluate_many(self, items):  # noqa: ANN001
        return [RankedNews(news=item, score=self.score) for item in items]


class DummyComposer:
    def generate(self, item: NewsItem) -> GeneratedPost:
        return GeneratedPost(
            title=f"RU {item.title}",
            body="A" * 1500,
            summary="Краткое описание",
            hashtags=("AI", "Tech", "News"),
        )


class DummyImageSelector:
    def select(self, news: NewsItem, post: GeneratedPost) -> ImageAsset:  # noqa: ARG002
        return ImageAsset(url="https://images.example.com/test.jpg", source="rss")


class DummySheetsWriter:
    def __init__(self) -> None:
        self.records = []

    def append_records(self, records):  # noqa: ANN001
        self.records.extend(records)


@pytest.fixture()
def config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        rss=RSSConfig(sources=(), keywords=("AI",), similarity_threshold=0.8, max_items=10),
        openai=OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="img"),
        pexels=PexelsConfig(api_key="pexels", timeout=5),
        freeimagehost=FreeImageHostConfig(api_key="freeimage", endpoint="https://freeimage.host/api", timeout=5),
        sheets=SheetsConfig(sheet_id="sheet", service_account_json=tmp_path / "credentials.json", worksheet="Sheet1"),
        scheduler=SchedulerConfig(timezone="Europe/Moscow"),
        cache_dir=tmp_path,
        log_level="INFO",
    )


def test_pipeline_runner_success(config: AppConfig) -> None:
    news = NewsItem(
        source="Test",
        title="AI breakthrough",
        link="https://example.com/1",
        summary="Summary",
        published=datetime.now(timezone.utc),
        keywords=("AI",),
        media_url=None,
    )
    collector = DummyCollector([news])
    scorer = DummyScorer(10)
    sheets = DummySheetsWriter()
    runner = PipelineRunner(
        config,
        rss_collector=collector,  # type: ignore[arg-type]
        scorer=scorer,  # type: ignore[arg-type]
        composer=DummyComposer(),
        image_selector=DummyImageSelector(),
        sheets_writer=sheets,  # type: ignore[arg-type]
    )

    stats = runner.run()

    assert stats.processed == 1
    assert stats.accepted == 1
    assert stats.published == 1
    assert stats.failed == 0
    assert len(sheets.records) == 1


def test_pipeline_runner_handles_failures(config: AppConfig) -> None:
    news = NewsItem(
        source="Test",
        title="AI breakthrough",
        link="https://example.com/1",
        summary="Summary",
        published=datetime.now(timezone.utc),
        keywords=("AI",),
        media_url=None,
    )
    collector = DummyCollector([news])
    scorer = DummyScorer(6)
    sheets = DummySheetsWriter()
    runner = PipelineRunner(
        config,
        rss_collector=collector,  # type: ignore[arg-type]
        scorer=scorer,  # type: ignore[arg-type]
        composer=DummyComposer(),
        image_selector=DummyImageSelector(),
        sheets_writer=sheets,  # type: ignore[arg-type]
    )

    stats = runner.run()

    assert stats.processed == 1
    assert stats.accepted == 0
    assert stats.published == 0
    assert stats.failed == 0
    assert sheets.records == []
