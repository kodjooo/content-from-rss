"""Модуль интеграции с Google Sheets."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import gspread
from gspread.exceptions import APIError

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
    "GPT Post",
    "Image URL",
    "Image Source",
    "Score",
    "Status",
    "Notes",
]


class GoogleSheetsWriter:
    """Обертка над gspread для записи строк."""

    def __init__(self, config: SheetsConfig, client: gspread.Client | None = None) -> None:
        self._config = config
        self._client = client or gspread.service_account(filename=str(config.service_account_json))

    def append_records(self, records: Iterable[PublicationRecord]) -> None:
        """Добавляет несколько записей в таблицу."""
        worksheet = self._open_worksheet()

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
        base_body = record.post.formatted().strip()
        full_body = f"{record.post.title}\n\n{base_body}".strip()
        full_body = f"{full_body}\n\nИсточник: [{record.post.title}]({record.link})".strip()
        short_text = f"#главная_новость\n\n{record.post.short_body}".strip()
        short_text = f"{short_text}\n\nИсточник: [{record.post.title}]({record.link})".strip()
        hashtags_line = " ".join(f"#{tag}" for tag in record.post.hashtags)
        date_value = record.date if isinstance(record.date, str) else record.date.isoformat()
        return [
            date_value,
            record.source,
            record.title,
            record.link,
            record.summary,
            short_text,
            full_body,
            record.image.url,
            record.image_source,
            str(record.score),
            record.status,
            record.notes or hashtags_line,
        ]

    def _resolve_worksheet(self, spreadsheet: gspread.Spreadsheet):
        """Возвращает рабочий лист, используя название из конфигурации."""
        try:
            return spreadsheet.worksheet(self._config.worksheet)
        except Exception:
            logger.warning("Лист %s не найден, используется первый лист", self._config.worksheet)
            return spreadsheet.sheet1

    def _ensure_header(self, worksheet: gspread.Worksheet) -> None:
        """Убеждается, что заголовок таблицы присутствует."""
        try:
            first_row = worksheet.row_values(1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось прочитать заголовок: %s", exc)
            first_row = []
        normalized = [cell.strip() for cell in first_row]
        if not normalized:
            worksheet.append_row(SHEET_HEADERS, value_input_option="RAW")
        elif normalized != SHEET_HEADERS:
            worksheet.update("1:1", [SHEET_HEADERS])

    def _open_worksheet(self) -> gspread.Worksheet:
        """Возвращает рабочий лист с гарантированным заголовком."""
        try:
            spreadsheet = self._client.open_by_key(self._config.sheet_id)
            worksheet = self._resolve_worksheet(spreadsheet)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Не удалось открыть Google Sheet: %s", exc)
            raise
        self._ensure_header(worksheet)
        return worksheet

    def fetch_existing_links(self) -> set[str]:
        """Возвращает множество ссылок, уже присутствующих в таблице."""
        worksheet = self._open_worksheet()
        try:
            values = worksheet.col_values(4)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Не удалось получить список ссылок из Google Sheets: %s", exc)
            raise
        return {value.strip() for value in values[1:] if value.strip()}

    def clear_records(self) -> None:
        """Удаляет все строки кроме заголовка."""
        worksheet = self._open_worksheet()
        try:
            worksheet.resize(rows=1)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Не удалось очистить Google Sheet: %s", exc)
            raise
        self._ensure_header(worksheet)


__all__: Sequence[str] = ("GoogleSheetsWriter",)
