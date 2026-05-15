"""Генерация финального поста через OpenAI."""

from __future__ import annotations

import json
import logging
from typing import Iterable, Sequence

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import OpenAIConfig
from .models import GeneratedPost, NewsItem

logger = logging.getLogger(__name__)


class PostGenerationError(RuntimeError):
    """Ошибка генерации поста."""


class PostComposer:
    """Класс, генерирующий конечный текст поста."""

    def __init__(self, config: OpenAIConfig, client: OpenAI | None = None) -> None:
        self._config = config
        self._client = client or OpenAI(api_key=config.api_key)

    def generate(self, item: NewsItem) -> GeneratedPost:
        """Генерирует пост для новости."""
        last_error: PostGenerationError | None = None
        for attempt in range(2):
            raw_text = self._request_post(item)
            try:
                payload = self._parse_payload(raw_text)
                self._validate_payload(payload)
                hashtags = tuple(payload["hashtags"])
                return GeneratedPost(
                    title=payload["title"],
                    translated_title=payload["translated_title"],
                    body=payload["body"],
                    summary=payload["summary"],
                    short_body=payload["short_body"],
                    average_body=payload["average_body"],
                    hashtags=hashtags,
                )
            except PostGenerationError as exc:
                last_error = exc
                if ("Некорректный JSON" in str(exc) and attempt < 1) or ("Длина текста вне" in str(exc) and attempt < 2):
                    logger.warning("Повторная попытка генерации поста для %s: %s", item.link, str(exc))
                    continue
                raise
        if last_error:
            raise last_error
        raise PostGenerationError("Не удалось сформировать пост")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _request_post(self, item: NewsItem) -> str:
        """Отправляет запрос в OpenAI."""
        prompt = self._build_prompt(item)
        logger.debug("Промт генерации поста для %s: %s", item.link, prompt)
        response = self._client.responses.create(
            model=self._config.model_post,
            input=prompt,
        )
        text = getattr(response, "output_text", "")
        if not text:
            raise PostGenerationError("Пустой ответ модели")
        return text

    def _build_prompt(self, item: NewsItem) -> str:
        """Формирует промт для модели."""
        keywords = ", ".join(item.keywords) or "AI"
        return (
            "Ты — Марк Аборчи, AI-специалист и IT-автоматизатор с опытом проектного менеджмента.\n"
            "Пиши строго на русском языке.\n"
            "Сгенерируй три версии поста: длинную (1000–1500 символов), среднюю (до 1000 символов без пробелов) и короткую (до 600 символов).\n"
            "Все версии должны быть аналитическими, с акцентом на практические выводы для одной выбранной аудитории: начинающие AI-практики и автоматизаторы без глубокого кода (n8n/Make/Python + AI-агенты).\n\n"
            "Стиль и манера автора:\n"
            "Пиши в стиле живого технического мышления, а не как вычитанную статью.\n"
            "Автор — практик по AI и автоматизации, который объясняет процессы так, будто думает вслух и прямо сейчас проектирует систему.\n"
            "Текст должен создавать ощущение реального инженерного процесса, а не заранее подготовленного контента.\n\n"
            "Особенности стиля:\n"
            "- разговорная речь;\n"
            "- декомпозиция задач на этапы;\n"
            "- объяснение через реальные сценарии и примеры;\n"
            "- постоянные уточнения мысли;\n"
            "- ощущение потока мышления;\n"
            "- минимальная литературность;\n"
            "- минимальный маркетинговый пафос;\n"
            "- акцент на эффективности, автоматизации и логике процессов.\n\n"
            "Используй конструкции: вот представим, например, то есть, сначала, потом, таким образом, по сути, в итоге.\n"
            "Не делай текст идеально вылизанным.\n"
            "Допустима легкая неровность речи, длинные предложения и постепенное уточнение мысли по ходу объяснения.\n"
            "Не пиши как копирайтер, журналист или преподаватель.\n"
            "Пиши как человек, который реально ежедневно работает с AI-инструментами и объясняет свой подход вживую.\n\n"
            "Анти-шаблоны (обязательно):\n"
            "1. Не используй клише и псевдо-образные штампы: серебряная пуля, магия, пушка, революция, игра в долгую, взлетит/не взлетит и т.п.\n"
            "2. Не делай искусственные примеры в стиле условный Вася/компания N без конкретного практического смысла.\n"
            "3. Не используй риторические украшения ради красоты: длинные метафоры, пафосные сравнения, эмоциональные гиперболы.\n"
            "4. Не пиши канцеляритом: в рамках, на текущем этапе, имеет место быть, осуществлять.\n"
            "5. Не дублируй одну мысль разными формулировками; каждый абзац должен добавлять новую практическую ценность.\n"
            "6. Не используй кавычки для псевдо-цитат и условных формулировок; кавычки допустимы только для реальных названий, терминов или буквальных цитат.\n"
            "7. Не используй сниженные или грубые формулировки (например: тупые ошибки, по факту как паразит).\n\n"

            "Требования:\n"
            "- Делай выводы и разворачивай личную позицию от первого лица.\n"
            "- В длинной версии выделяй ключевые мысли при помощи **жирного** форматирования (там, где это уместно).\n"
            "- Не вставляй хэштеги в тексты длинной, средней и короткой версий.\n"
            "- Короткая версия должна быть ёмким пересказом основных тезисов (до 600 символов).\n"
            "- Средняя версия (average_body) должна быть до 1000 символов без учёта пробелов.\n"
            "- Используй только русские слова в списке hashtags (кириллица), без символа #.\n"
            "- Хэштеги отражают ключевые темы новости.\n\n"

            "Формат ответа — строго JSON:\n"
            "translated_title — дословный перевод оригинального заголовка новости на русский язык (до 120 символов).\n"
            "title — заголовок до 100 символов.\n"
            "summary — краткое изложение (300–400 символов).\n"
            "short_body — короткая версия поста (до 600 символов, без хэштегов).\n"
            "average_body — средняя версия поста (до 1000 символов без пробелов, без хэштегов).\n"
            "body — длинная версия поста (1000–1500 символов, без хэштегов).\n"
            "hashtags — список из 3–4 русских слов без символа #.\n\n"

            "Пример ответа:\n"
            "{"
            "\"translated_title\": \"Переведённый заголовок новости...\","
            "\"title\": \"ИИ перестал быть инструктором — он стал партнёром\","
            "\"summary\": \"Краткое резюме на 300–400 символов...\","
            "\"short_body\": \"Сжатый текст до 600 символов...\","
            "\"average_body\": \"Средний текст...\","
            "\"body\": \"Развёрнутый текст 1000–1500 символов с выделениями **жирным**...\","
            "\"hashtags\": [\"инвестиции\", \"автоматизация\", \"управление\"]"
            "}\n\n"

            "Не добавляй никаких дополнительных полей и не используй markdown помимо **жирного**.\n"
            f"Заголовок новости: {item.title}\n"
            f"Описание новости: {item.summary}\n"
            f"Ключевые слова: {keywords}"
        )

    def _parse_payload(self, text: str) -> dict[str, object]:
        """Парсит JSON-ответ модели."""
        try:
            return json.loads(text)
        except json.JSONDecodeError as err:
            logger.error("Ответ модели не является корректным JSON: %s", text)
            raise PostGenerationError("Некорректный JSON в ответе модели") from err

    def _validate_payload(self, payload: dict[str, object]) -> None:
        """Проверяет структуру и ограничения результата."""
        for field in ("translated_title", "title", "summary", "short_body", "average_body", "body", "hashtags"):
            if field not in payload:
                raise PostGenerationError(f"Отсутствует поле {field}")
        translated_title = payload["translated_title"]
        if not isinstance(translated_title, str) or not translated_title.strip():
            raise PostGenerationError("Поле translated_title должно быть непустой строкой")
        if len(translated_title) > 200:
            raise PostGenerationError("Поле translated_title должно быть не длиннее 200 символов")
        body = payload["body"]
        summary = payload["summary"]
        short_body = payload["short_body"].strip()
        average_body = payload["average_body"]
        hashtags = payload["hashtags"]
        if not isinstance(body, str):
            raise PostGenerationError("Поле body должно быть строкой")
        length = len(body)
        if length < 800 or length > 1700:
            raise PostGenerationError(f"Длина текста вне требуемого диапазона: {length}")
        if not isinstance(summary, str) or not summary.strip():
            raise PostGenerationError("Поле summary должно быть непустой строкой")
        if not isinstance(short_body, str) or not short_body.strip():
            raise PostGenerationError("Поле short_body должно быть непустой строкой")
        if len(short_body) > 600:
            raise PostGenerationError("Поле short_body превышает 600 символов")
        if not isinstance(average_body, str) or not average_body.strip():
            raise PostGenerationError("Поле average_body должно быть непустой строкой")
        avg_compact = "".join(average_body.split())
        if len(avg_compact) > 1000:
            raise PostGenerationError("Поле average_body превышает 1000 символов без пробелов")
        if not isinstance(hashtags, Iterable):
            raise PostGenerationError("Поле hashtags должно быть массивом")
        hashtags_list = [tag for tag in hashtags if isinstance(tag, str) and tag.strip()]
        if len(hashtags_list) < 3 or len(hashtags_list) > 4:
            raise PostGenerationError("Количество хэштегов должно быть от 3 до 4")
        payload["hashtags"] = hashtags_list


__all__: Sequence[str] = ("PostComposer", "PostGenerationError")
