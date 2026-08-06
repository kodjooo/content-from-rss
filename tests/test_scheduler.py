from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from app.config import _parse_run_times, AppConfig, FreeImageHostConfig, OpenAIConfig, PexelsConfig, RSSConfig, SchedulerConfig, SheetsConfig
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
        openai=OpenAIConfig(
            api_key="test",
            api_key_image="test-images",
            model_rank="gpt",
            model_post="gpt",
            model_image="img",
            image_quality="medium",
            image_size="1024x1024",
        ),
        pexels=PexelsConfig(api_key="pexels", timeout=5, enabled=True),
        freeimagehost=FreeImageHostConfig(api_key="freeimage", endpoint="https://freeimage.host/api", timeout=5),
        sheets=SheetsConfig(
            sheet_id="sheet",
            service_account_json=tmp_path / "credentials.json",
            worksheet="Sheet1",
        ),
        scheduler=SchedulerConfig(timezone="Europe/Moscow", run_times=(), run_once_on_start=run_once_on_start),
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


def test_parse_run_times_defaults_to_single_evening_run() -> None:
    assert _parse_run_times(None) == ((17, 50),)
    assert _parse_run_times("   ") == ((17, 50),)


def test_parse_run_times_supports_minutes_and_lists() -> None:
    assert _parse_run_times("17:50") == ((17, 50),)
    assert _parse_run_times("19:30, 7") == ((7, 0), (19, 30))


def test_parse_run_times_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        _parse_run_times("25:00")
    with pytest.raises(ValueError):
        _parse_run_times("вечером")


def test_lookback_window_matches_schedule() -> None:
    assert SchedulerConfig(timezone="Europe/Moscow", run_times=((17, 50),)).lookback_hours == 25
    assert SchedulerConfig(timezone="Europe/Moscow", run_times=((7, 0), (19, 0))).lookback_hours == 13


def test_scheduler_registers_jobs_with_minutes(tmp_path) -> None:
    config = make_config(tmp_path, run_once_on_start=False)
    config = replace(config, scheduler=replace(config.scheduler, run_times=((17, 50),)))
    scheduler = PipelineScheduler(config, runner_factory=lambda: DummyRunner())

    scheduler.start(block=False)
    jobs = scheduler._scheduler.get_jobs()
    scheduler.stop()

    assert len(jobs) == 1
    fields = {field.name: str(field) for field in jobs[0].trigger.fields}
    assert fields["hour"] == "17"
    assert fields["minute"] == "50"
