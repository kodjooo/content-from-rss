from __future__ import annotations

from dataclasses import dataclass

from app.config import AppConfig, FreeImageHostConfig, OpenAIConfig, PexelsConfig, RSSConfig, SchedulerConfig, SheetsConfig
from app.scheduler import PipelineScheduler


@dataclass
class DummyRunner:
    executed: bool = False

    def run(self):  # noqa: D401, ANN001
        self.executed = True
        return type("Stats", (), {"processed": 0, "accepted": 0, "published": 0, "failed": 0})()


def make_config(tmp_path) -> AppConfig:
    return AppConfig(
        rss=RSSConfig(sources=(), keywords=(), similarity_threshold=0.8, max_items=10),
        openai=OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="img"),
        pexels=PexelsConfig(api_key="pexels", timeout=5),
        freeimagehost=FreeImageHostConfig(api_key="freeimage", endpoint="https://freeimage.host/api", timeout=5),
        sheets=SheetsConfig(sheet_id="sheet", service_account_json=tmp_path / "credentials.json", worksheet="Sheet1"),
        scheduler=SchedulerConfig(timezone="Europe/Moscow"),
        cache_dir=tmp_path,
        log_level="INFO",
    )


def test_scheduler_run_once(tmp_path) -> None:
    config = make_config(tmp_path)
    runner = DummyRunner()
    scheduler = PipelineScheduler(config, runner_factory=lambda: runner)

    scheduler.run_once()

    assert runner.executed is True
