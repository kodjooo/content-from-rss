"""Модуль интеграции с Google Sheets."""

from __future__ import annotations

import json
import logging
import re
from typing import Iterable, Sequence

import gspread
from gspread.exceptions import APIError, WorksheetNotFound

from .config import SheetsConfig
from .models import PublicationRecord

logger = logging.getLogger(__name__)

SHEET_HEADERS = [
    "Date",
    "Source",
    "Title",
    "Link",
    "Summary",
    "Short Post",
    "Average Post",
    "GPT Post Title",
    "GPT Post",
    "Image URL",
    "Image Source",
    "Score",
    "Status",
    "Hashtags",
    "Notes",
    "Telegraph Link",
    "VK Post Link",
    "TG Post Link",
]

_BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")


class GoogleSheetsWriter:
    """Обертка над gspread для записи строк."""

    def __init__(self, config: SheetsConfig, client: gspread.Client | None = None) -> None:
        self._config = config
        self._client = client or gspread.service_account(filename=str(config.service_account_json))
        self._spreadsheet: gspread.Spreadsheet | None = None

    def append_records(self, records: Iterable[PublicationRecord]) -> None:
        """Добавляет несколько записей в таблицу."""
        spreadsheet = self._open_spreadsheet()
        worksheet = self._resolve_worksheet(spreadsheet)

        for record in records:
            row = self._serialize(record)
            try:
                worksheet.append_row(row, value_input_option="RAW")
                logger.info("Добавлена запись в Google Sheets: %s", record.link)
            except APIError as exc:
                logger.exception("Ошибка записи строки в Google Sheets: %s", exc)
                raise

    def _serialize(self, record: PublicationRecord) -> list[str]:
        """Формирует строку согласно структуре таблицы."""
        telegraph_content = self._build_telegraph_content(record)
        short_body = record.post.short_body.strip()
        short_text = f"{short_body}\n\nЧитать подробнее >" if short_body else "Читать подробнее >"
        average_body = record.post.average_body.strip()
        average_text = f"{average_body}\n\nИсточник >" if average_body else "Источник >"
        hashtags_line = " ".join(f"#{tag}" for tag in record.post.hashtags)
        date_value = record.date if isinstance(record.date, str) else record.date.isoformat()
        return [
            date_value,
            record.source,
            record.title,
            record.link,
            record.summary,
            short_text,
            average_text,
            record.post.title,
            telegraph_content,
            record.image.url,
            record.image_source,
            str(record.score),
            record.status,
            hashtags_line,
            record.notes or "",
            record.telegraph_link or "",
            record.vk_post_link or "",
            record.tg_post_link or "",
        ]

    def _resolve_worksheet(self, spreadsheet: gspread.Spreadsheet) -> gspread.Worksheet:
        """Возвращает рабочий лист, создавая при необходимости и гарантируя заголовок."""
        title = self._config.worksheet
        try:
            worksheet = spreadsheet.worksheet(title)
        except WorksheetNotFound:
            logger.info("Создается вкладка %s в таблице %s", title, spreadsheet.id)
            worksheet = spreadsheet.add_worksheet(title=title, rows=1, cols=len(SHEET_HEADERS))
        self._ensure_header(worksheet)
        return worksheet

    def _ensure_header(self, worksheet: gspread.Worksheet) -> None:
        """Убеждается, что заголовок таблицы присутствует."""
        try:
            first_row = worksheet.row_values(1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось прочитать заголовок: %s", exc)
            first_row = []
        normalized = [cell.strip() for cell in first_row]
        if not normalized:
            worksheet.append_row(list(SHEET_HEADERS), value_input_option="RAW")
        elif list(normalized) != list(SHEET_HEADERS):
            worksheet.update("1:1", [list(SHEET_HEADERS)])

    def _open_spreadsheet(self) -> gspread.Spreadsheet:
        """Открывает таблицу, кешируя объект гугл-таблицы."""
        if self._spreadsheet is not None:
            return self._spreadsheet
        try:
            self._spreadsheet = self._client.open_by_key(self._config.sheet_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Не удалось открыть Google Sheet: %s", exc)
            raise
        return self._spreadsheet

    def fetch_existing_links(self) -> set[str]:
        """Возвращает множество ссылок, уже присутствующих в таблице."""
        spreadsheet = self._open_spreadsheet()
        worksheet = self._resolve_worksheet(spreadsheet)
        try:
            values = worksheet.col_values(4)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Не удалось получить список ссылок из Google Sheets: %s", exc)
            raise
        return {value.strip() for value in values[1:] if value.strip()}

    def clear_records(self) -> None:
        """Удаляет все строки кроме заголовка."""
        spreadsheet = self._open_spreadsheet()
        worksheet = self._resolve_worksheet(spreadsheet)
        try:
            worksheet.resize(rows=1)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Не удалось очистить Google Sheet: %s", exc)
            raise
        self._ensure_header(worksheet)

    def _build_telegraph_content(self, record: PublicationRecord) -> str:
        """Собирает JSON-структуру для createPage API Telegra.ph."""
        body = record.post.formatted().strip()
        paragraphs = [chunk.strip() for chunk in body.split("\n\n") if chunk.strip()]
        nodes: list[dict[str, object]] = []
        for paragraph in paragraphs:
            children = self._paragraph_to_children(paragraph)
            nodes.append({"tag": "p", "children": children})
        nodes.append(
            {
                "tag": "p",
                "children": [
                    {
                        "tag": "a",
                        "attrs": {"href": record.link},
                        "children": ["Источник >"],
                    }
                ],
            }
        )
        return json.dumps(nodes, ensure_ascii=False)

    def _paragraph_to_children(self, paragraph: str) -> list[object]:
        """Преобразует параграф в список узлов Telegra.ph c поддержкой **жирного**."""
        children: list[object] = []
        last_index = 0
        for match in _BOLD_PATTERN.finditer(paragraph):
            start, end = match.span()
            if start > last_index:
                children.append(paragraph[last_index:start])
            bold_text = match.group(1).strip()
            if bold_text:
                children.append({"tag": "strong", "children": [bold_text]})
            last_index = end
        if last_index < len(paragraph):
            children.append(paragraph[last_index:])
        if not children:
            children.append("")
        return children


__all__: Sequence[str] = ("GoogleSheetsWriter",)
