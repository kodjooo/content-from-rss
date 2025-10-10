"""Модуль интеграции с Google Sheets."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import gspread
from gspread.exceptions import APIError

from .config import SheetsConfig
from .models import PublicationRecord

logger = logging.getLogger(__name__)


class GoogleSheetsWriter:
    """Обертка над gspread для записи строк."""

    def __init__(self, config: SheetsConfig, client: gspread.Client | None = None) -> None:
        self._config = config
        self._client = client or gspread.service_account(filename=str(config.service_account_json))

    def append_records(self, records: Iterable[PublicationRecord]) -> None:
        """Добавляет несколько записей в таблицу."""
        try:
            spreadsheet = self._client.open_by_key(self._config.sheet_id)
            worksheet = self._resolve_worksheet(spreadsheet)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Не удалось открыть Google Sheet: %s", exc)
            raise

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
        post_text = record.post.formatted()
        hashtags_line = " ".join(f"#{tag}" for tag in record.post.hashtags)
        return [
            record.date.isoformat(),
            record.source,
            record.title,
            record.link,
            record.summary,
            post_text,
            record.image.url,
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


__all__: Sequence[str] = ("GoogleSheetsWriter",)
