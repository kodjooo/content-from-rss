from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List

import pytest
from gspread.exceptions import WorksheetNotFound

from app.config import SheetsConfig
from app.models import GeneratedPost, ImageAsset, PublicationRecord
from app.sheets import GoogleSheetsWriter


class DummyWorksheet:
    def __init__(self, title: str) -> None:
        self.title = title
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

    def col_values(self, index: int) -> list[str]:  # noqa: ARG002
        result: List[str] = []
        for row in self.rows:
            if len(row) >= index:
                result.append(row[index - 1])
            else:
                result.append("")
        return result

    def resize(self, rows: int) -> None:
        self.rows = self.rows[:rows]


class DummySpreadsheet:
    def __init__(self) -> None:
        self.id = "dummy-spreadsheet"
        self.sheet1 = DummyWorksheet("Sheet1")
        self._worksheets = {"Sheet1": self.sheet1}

    def worksheet(self, title: str) -> DummyWorksheet:
        if title in self._worksheets:
            return self._worksheets[title]
        raise WorksheetNotFound(f"{title} not found")  # type: ignore[name-defined]

    def add_worksheet(self, title: str, rows: int, cols: int) -> DummyWorksheet:  # noqa: ARG002
        worksheet = DummyWorksheet(title)
        self._worksheets[title] = worksheet
        return worksheet


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
        title="Новости об ИИ",
        link="https://example.com/news",
        summary="Краткое описание",
        post=GeneratedPost(
            title="Post title",
            translated_title="Новости об ИИ",
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
    assert data_row[2] == publication_record.title
    assert data_row[5] == f"{publication_record.post.short_body}\n\nЧитать подробнее >"
    telegraph_payload = json.loads(data_row[7])
    assert telegraph_payload[-1]["children"][0]["attrs"]["href"] == publication_record.link
    assert data_row[9] == publication_record.image_source
    assert data_row[12] == " ".join(f"#{tag}" for tag in publication_record.post.hashtags)


def test_fetch_links_and_clear(tmp_path, publication_record: PublicationRecord) -> None:
    config = SheetsConfig(
        sheet_id="sheet",
        service_account_json=tmp_path / "credentials.json",
        worksheet="Sheet1",
    )
    client = DummyClient()
    writer = GoogleSheetsWriter(config, client=client)  # type: ignore[arg-type]

    writer.append_records([publication_record])
    links = writer.fetch_existing_links()
    assert publication_record.link in links

    writer.clear_records()
    assert writer.fetch_existing_links() == set()
    assert len(client.spreadsheet.sheet1.rows) == 1
