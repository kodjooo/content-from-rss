from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import RSSConfig
from app.rss import RSSCollector


@pytest.fixture()
def rss_config() -> RSSConfig:
    return RSSConfig(
        sources=("https://example.com/feed",),
        keywords=("AI", "Automation"),
        similarity_threshold=0.8,
        max_items=10,
    )


def fake_entry(title: str, summary: str, link: str, media_url: str | None = None) -> dict[str, object]:
    return {
        "title": title,
        "summary": summary,
        "link": link,
        "media_content": [{"url": media_url}] if media_url else [],
    }


def test_collect_filters_by_keyword(monkeypatch: pytest.MonkeyPatch, rss_config: RSSConfig) -> None:
    entries = [
        fake_entry("AI breakthrough", "All about AI", "https://example.com/1"),
        fake_entry("Sports news", "Football only", "https://example.com/2"),
    ]
    monkeypatch.setattr("feedparser.parse", lambda _: SimpleNamespace(entries=entries))

    collector = RSSCollector(rss_config)
    result = collector.collect()

    assert len(result) == 1
    assert result[0].link == "https://example.com/1"


def test_collect_deduplicates_similar_titles(monkeypatch: pytest.MonkeyPatch, rss_config: RSSConfig) -> None:
    entries = [
        fake_entry("AI breakthrough", "AI rocks", "https://example.com/1"),
        fake_entry("AI breakthrough!", "AI rocks again", "https://example.com/2"),
    ]
    monkeypatch.setattr("feedparser.parse", lambda _: SimpleNamespace(entries=entries))

    collector = RSSCollector(rss_config)
    result = collector.collect()

    assert len(result) == 1
    assert result[0].link == "https://example.com/1"
