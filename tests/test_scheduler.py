from __future__ import annotations

from dataclasses import dataclass

from app.config import AppConfig, FreeImageHostConfig, OpenAIConfig, PexelsConfig, RSSConfig, SchedulerConfig, SheetsConfig
from app.scheduler import PipelineScheduler


@dataclass
class DummyRunner:
    calls: int = 0

    def run(self):  # noqa: D401, ANN001
        self.calls += 1
        return type("Stats", (), {"processed": 0, "accepted": 0, "published": 0, "failed": 0})()


def make_config(tmp_path, run_once_on_start: bool = True) -> AppConfig:
    return AppConfig(
        rss=RSSConfig(sources=(), keywords=(), similarity_threshold=0.8, max_items=10),
        openai=OpenAIConfig(api_key="test", model_rank="gpt", model_post="gpt", model_image="img"),
        pexels=PexelsConfig(api_key="pexels", timeout=5),
        freeimagehost=FreeImageHostConfig(api_key="freeimage", endpoint="https://freeimage.host/api", timeout=5),
        sheets=SheetsConfig(sheet_id="sheet", service_account_json=tmp_path / "credentials.json", worksheet="Sheet1"),
        scheduler=SchedulerConfig(timezone="Europe/Moscow", run_hours=(), run_once_on_start=run_once_on_start),
        cache_dir=tmp_path,
        log_level="INFO",
    )


def test_scheduler_run_once(tmp_path) -> None:
    config = make_config(tmp_path)
    runner = DummyRunner()
    scheduler = PipelineScheduler(config, runner_factory=lambda: runner)

    scheduler.run_once()

    assert runner.calls == 1


def test_scheduler_start_triggers_initial_run(tmp_path) -> None:
    config = make_config(tmp_path)
    runner = DummyRunner()
    scheduler = PipelineScheduler(config, runner_factory=lambda: runner)

    scheduler.start(block=False)
    scheduler.stop()

    assert runner.calls >= 1


def test_scheduler_start_respects_flag(tmp_path) -> None:
    config = make_config(tmp_path, run_once_on_start=False)
    runner = DummyRunner()
    scheduler = PipelineScheduler(config, runner_factory=lambda: runner)

    scheduler.start(block=False)
    scheduler.stop()

    assert runner.calls == 0
