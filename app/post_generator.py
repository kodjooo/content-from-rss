"""Генерация финального поста через OpenAI."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Iterable, Sequence

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import OpenAIConfig
from .models import GeneratedPost, NewsItem

logger = logging.getLogger(__name__)

# Формулировки-маркеры «нейрослопа». Собраны по разбору архива постов:
# именно они чаще всего встречались в постах с нулевой вовлечённостью.
BANNED_PHRASES: tuple[str, ...] = (
    "владельцам бизнеса стоит",
    "для владельцев бизнеса",
    "что это значит для бизнеса",
    "новая эра",
    "новый этап",
    "новый стандарт",
    "не просто",
    "важный сигнал",
    "тревожный звоночек",
    "важно понимать",
    "по сути",
    "критически важн",
    "неизбежно повлия",
    "меняет правила игры",
    "мой вывод: бизнес",
    "остаться в стороне",
    "поделитесь кейсами",
    "серебряная пуля",
    "имеет место быть",
    "в рамках",
)

# Варианты подачи. Для каждой новости выбирается своя комбинация, чтобы посты
# не превращались в один шаблон, отличающийся только фактами.
_LEAD_VARIANTS: tuple[str, ...] = (
    "Начни с конкретной цифры из новости прямо в первом предложении и сразу скажи, что она значит для твоей работы.",
    "Начни со сценария: «вот представим…» — короткая узнаваемая ситуация из работы вайб-кодера, и только потом новость.",
    "Начни с личного признания или наблюдения из практики, а новость подтяни во втором абзаце как повод.",
    "Начни с короткого утверждения-тезиса, с которым многие не согласятся, и дальше объясняй, почему ты так решил.",
    "Начни с того, что ты в этой новости сначала понял неправильно, и как переобулся, когда дочитал детали.",
)

_SHAPE_VARIANTS: tuple[str, ...] = (
    "Структура свободная, без списков — сплошной рассуждающий текст.",
    "В середине допустим короткий список из 2–3 пунктов, всё остальное — обычный текст.",
    "Построй текст как «было / стало»: как я делал это раньше и что меняю теперь.",
    "Построй текст как разбор одного решения: что выбираю, что отбрасываю и почему.",
    "Построй текст как ход мысли по шагам: сначала проверил бы это, потом это, в итоге такой вывод.",
)

_ENDING_VARIANTS: tuple[str, ...] = (
    "Заканчивай одним конкретным вопросом к читателю про его собственный опыт.",
    "Заканчивай одним вопросом-провокацией: предположи, как у большинства читателей это устроено, и попроси проверить.",
    "Заканчивай своей позицией без вопроса, но так, чтобы с ней явно хотелось поспорить.",
    "Заканчивай приглашением: попроси рассказать, у кого получилось иначе, чем у тебя.",
)

_VOICE_VARIANTS: tuple[str, ...] = (
    "Тон спокойный, слегка скептичный.",
    "Тон живой, местами с иронией над хайпом.",
    "Тон рабочий и суховатый, как заметка для себя.",
    "Тон азартный: тебе правда интересно, что из этого получится.",
)


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
        attempts = 3
        for attempt in range(attempts):
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
                if attempt < attempts - 1:
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

    def _variants(self, item: NewsItem) -> tuple[str, str, str, str]:
        """Подбирает комбинацию приёмов подачи для конкретной новости."""
        digest = hashlib.sha256(item.link.encode("utf-8")).digest()
        return (
            _LEAD_VARIANTS[digest[0] % len(_LEAD_VARIANTS)],
            _SHAPE_VARIANTS[digest[1] % len(_SHAPE_VARIANTS)],
            _ENDING_VARIANTS[digest[2] % len(_ENDING_VARIANTS)],
            _VOICE_VARIANTS[digest[3] % len(_VOICE_VARIANTS)],
        )

    def _build_prompt(self, item: NewsItem) -> str:
        """Формирует промт для модели."""
        keywords = ", ".join(item.keywords) or "AI"
        lead, shape, ending, voice = self._variants(item)
        return (
            "Ты — Марк Аборчи, AI-специалист и IT-автоматизатор с опытом проектного менеджмента.\n"
            "Пиши строго на русском языке.\n"
            "Сгенерируй три версии поста: длинную (1200–1500 символов), среднюю (до 1000 символов без пробелов) и короткую (до 600 символов).\n"
            "Аудитория — такие же практики, как автор: люди, которые собирают автоматизации вайб-кодингом (ChatGPT, Claude Code, "
            "иногда локальные модели вроде Qwen), пишут скрипты на Python с помощью AI и не считают себя инженерами.\n"
            "Автор автоматизирует именно так: через AI-ассистентов и код, а не через готовые no-code платформы. "
            "Не приписывай автору опыт, которого нет, и не выдумывай ему клиентов, цифры и проекты.\n\n"
            "Стиль и манера автора:\n"
            "Пиши в стиле живого технического мышления, а не как вычитанную статью.\n"
            "Автор — практик по AI и автоматизации, который объясняет процессы так, будто думает вслух и прямо сейчас проектирует систему.\n"
            "Текст должен создавать ощущение реального инженерного процесса, а не заранее подготовленного контента.\n\n"
            "Особенности стиля:\n"
            "- разговорная речь, живой человек, а не редакция;\n"
            "- рабочий сленг и жаргон уместны там, где он естественен (прод, деплой, костыль, задеплоить, промпт, токены, контекст);\n"
            "- объяснение через реальные сценарии и примеры;\n"
            "- постоянные уточнения мысли, ощущение потока мышления;\n"
            "- минимальная литературность и минимальный маркетинговый пафос;\n"
            "- акцент на эффективности, автоматизации и логике процессов.\n\n"
            "Используй конструкции: вот представим, например, то есть, сначала, потом, в итоге.\n"
            "Не делай текст идеально вылизанным.\n"
            "Допустима легкая неровность речи, длинные предложения и постепенное уточнение мысли по ходу объяснения.\n"
            "Не пиши как копирайтер, журналист или преподаватель.\n"
            "Пиши как человек, который реально ежедневно работает с AI-инструментами и объясняет свой подход вживую.\n"
            "Не старайся попасть в шаблон: два поста подряд не должны читаться как один и тот же текст с другими фактами.\n\n"
            "Приёмы именно для этого поста (следуй им, но без насилия над текстом):\n"
            f"- {lead}\n"
            f"- {shape}\n"
            f"- {ending}\n"
            f"- {voice}\n\n"
            "Анти-шаблоны (обязательно):\n"
            "1. Не используй клише и псевдо-образные штампы: серебряная пуля, магия, пушка, революция, игра в долгую, взлетит/не взлетит и т.п.\n"
            "2. Не делай искусственные примеры в стиле условный Вася/компания N без конкретного практического смысла.\n"
            "3. Не используй риторические украшения ради красоты: длинные метафоры, пафосные сравнения, эмоциональные гиперболы.\n"
            "4. Не пиши канцеляритом: в рамках, на текущем этапе, имеет место быть, осуществлять.\n"
            "5. Не дублируй одну мысль разными формулировками; каждый абзац должен добавлять новую практическую ценность.\n"
            "6. Не используй кавычки для псевдо-цитат и условных формулировок; кавычки допустимы только для реальных названий, терминов или буквальных цитат.\n"
            "7. Не используй сниженные или грубые формулировки (например: тупые ошибки, по факту как паразит).\n\n"

            "Запрещённые формулировки (ни в одной из версий):\n"
            "«владельцам бизнеса стоит / для владельцев бизнеса», «что это значит для бизнеса», «новая эра / новый этап / новый стандарт», "
            "«не просто X, а Y», «важный сигнал», «тревожный звоночек», «важно понимать», «по сути», «критически важно», "
            "«неизбежно повлияет», «меняет правила игры», «рискует остаться в стороне», «поделитесь кейсами и инсайтами».\n"
            "Не заканчивай пост выводом вида «бизнесу надо внедрять ИИ» — это подходит к любой новости и поэтому ничего не значит.\n\n"

            "Заголовок:\n"
            "- 40–70 символов, лучше ближе к 50;\n"
            "- либо форма «Как <сделать X>», либо утверждение с конкретной цифрой или следствием;\n"
            "- главную цифру новости выноси в заголовок и в первую строку текста;\n"
            "- не начинай заголовок с названия продукта или компании, если это не общеизвестное имя (OpenAI, Google, Anthropic, Nvidia);\n"
            "- не делай заголовок вопросом и не начинай его с «Почему»;\n"
            "- не используй шаблон «… что это значит для …».\n\n"

            "Требования:\n"
            "- Делай выводы и разворачивай личную позицию от первого лица.\n"
            "- В посте должен быть один спорный тезис — такой, с которым читатель может не согласиться и захотеть возразить.\n"
            "- Заканчивай длинную версию максимум одним вопросом. Блок из трёх-четырёх вопросов подряд запрещён.\n"
            "- Оговорки и ограничения — не больше двух предложений, без отдельной лекции про риски.\n"
            "- Вывод из новости должен быть конкретным для работы через AI-ассистентов и код: что поменять в промпте, "
            "в скрипте, в выборе модели, в проверке результата. Общие рассуждения про рынок не нужны.\n"
            "- Проверь себя перед ответом: если подставить в текст другую компанию и другую цифру и он останется верным, "
            "значит в нём нет содержания — перепиши.\n"
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
            "body — длинная версия поста (1200–1500 символов, без хэштегов).\n"
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
        if length < 1100 or length > 1700:
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
        self._validate_style(payload["title"], body, short_body, average_body)

    def _validate_style(self, title: object, body: str, short_body: str, average_body: str) -> None:
        """Проверяет стилевые требования: стоп-лист, заголовок, число вопросов."""
        for name, text in (("body", body), ("average_body", average_body), ("short_body", short_body)):
            found = self._find_banned(text)
            if found:
                raise PostGenerationError(f"Поле {name} содержит запрещённую формулировку: {found}")

        if body.count("?") > 1:
            raise PostGenerationError("В длинной версии больше одного вопроса")

        if not isinstance(title, str) or not title.strip():
            raise PostGenerationError("Поле title должно быть непустой строкой")
        if "?" in title:
            raise PostGenerationError("Заголовок не должен быть вопросом")
        if len(title) > 100:
            raise PostGenerationError("Заголовок длиннее 100 символов")
        found_title = self._find_banned(title)
        if found_title:
            raise PostGenerationError(f"Заголовок содержит запрещённую формулировку: {found_title}")

        # Мягкие проверки: не заворачиваем генерацию, но фиксируем в логах,
        # чтобы было видно, если модель систематически съезжает со стиля.
        if not re.search(r"\d", body[:400]):
            logger.warning("В первых 400 символах поста нет конкретной цифры: %s", title)
        if not 40 <= len(title) <= 70:
            logger.warning("Длина заголовка вне диапазона 40–70 символов (%s): %s", len(title), title)

    def _find_banned(self, text: str) -> str | None:
        """Возвращает первую найденную запрещённую формулировку."""
        lowered = text.lower().replace("ё", "е")
        for phrase in BANNED_PHRASES:
            if phrase.replace("ё", "е") in lowered:
                return phrase
        return None


__all__: Sequence[str] = ("PostComposer", "PostGenerationError")
