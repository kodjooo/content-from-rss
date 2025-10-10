from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest

from app.config import SheetsConfig
from app.models import GeneratedPost, ImageAsset, PublicationRecord
from app.sheets import GoogleSheetsWriter


class DummyWorksheet:
    def __init__(self) -> None:
        self.rows: List[list[str]] = []

    def append_row(self, row: list[str], value_input_option: str = "RAW") -> None:  # noqa: ARG002
        self.rows.append(row)

    def row_values(self, index: int) -> list[str]:
        if 1 <= index <= len(self.rows):
            return self.rows[index - 1]
        return []

    def update(self, range_name: str, values: list[list[str]]) -> None:  # noqa: ARG002
        if not values:
            return
        if self.rows:
            self.rows[0] = values[0]
        else:
            self.rows.append(values[0])


class DummySpreadsheet:
    def __init__(self) -> None:
        self.sheet1 = DummyWorksheet()
        self._worksheets = {"Sheet1": self.sheet1}

    def worksheet(self, title: str) -> DummyWorksheet:
        if title in self._worksheets:
            return self._worksheets[title]
        raise Exception("Worksheet not found")


class DummyClient:
    def __init__(self) -> None:
        self.open_calls = 0
        self.spreadsheet = DummySpreadsheet()

    def open_by_key(self, key: str) -> DummySpreadsheet:  # noqa: ARG002
        self.open_calls += 1
        return self.spreadsheet


@pytest.fixture()
def publication_record() -> PublicationRecord:
    return PublicationRecord(
        date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source="Test Source",
        title="AI News",
        link="https://example.com/news",
        summary="Краткое описание",
        post=GeneratedPost(
            title="Post title",
            summary="Краткое описание",
            body="Body" * 400,
            short_body="Короткая версия",
            hashtags=("AI", "Automation", "Innovation"),
        ),
        image=ImageAsset(url="https://images.example.com/img.jpg", source="rss"),
        score=9,
        image_source="RSS",
    )


def test_append_records_writes_rows(tmp_path, publication_record: PublicationRecord) -> None:
    config = SheetsConfig(
        sheet_id="sheet",
        service_account_json=tmp_path / "credentials.json",
        worksheet="Sheet1",
    )
    client = DummyClient()
    writer = GoogleSheetsWriter(config, client=client)  # type: ignore[arg-type]

    writer.append_records([publication_record])

    assert client.open_calls == 1
    assert len(client.spreadsheet.sheet1.rows) == 2
    header = client.spreadsheet.sheet1.rows[0]
    data_row = client.spreadsheet.sheet1.rows[1]
    assert header[0] == "Date"
    assert data_row[0].startswith("2024-01-01")
    assert data_row[5].startswith("#главная_новость")
    assert "[Post title]" in data_row[6]
    assert data_row[8] == publication_record.image_source
    assert data_row[11].startswith("#")
