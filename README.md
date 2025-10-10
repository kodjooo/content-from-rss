# AI News Auto Writer MVP

Сервис автоматически собирает англоязычные новости об искусственном интеллекте, фильтрует их по релевантности, генерирует готовые посты и подбирает изображения. Результат записывается в Google Sheets для ручной публикации.

## Основные возможности
- Сбор RSS-лент из TechCrunch, VentureBeat, MIT Technology Review, The Verge, OpenAI, Google AI и Anthropic.
- Дедупликация и фильтрация по ключевым словам.
- Оценка релевантности новостей и генерация постов через OpenAI API.
- Подбор изображений из RSS, Pexels или генерация через OpenAI Images с загрузкой на FreeImageHost.
- Сохранение результатов в Google Sheets с указанием статуса Written.
- Планировщик запусков в 07:00 и 19:00 по Европе/Москве.

## Требования
- Python 3.11+ (для локального запуска вне Docker).
- Docker / Docker Compose для контейнерного деплоя.
- Активные ключи OpenAI, Pexels, FreeImageHost и сервисный аккаунт Google.

## Настройка окружения
1. Установите зависимости:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. Сгенерируйте `.env` из шаблона (при необходимости будут созданы случайные секреты):
   ```bash
   python scripts/setup_env.py --force
   ```
3. Проверьте и дополните `.env` нужными значениями:
   - `OPENAI_API_KEY`
   - `PEXELS_API_KEY`
   - `FREEIMAGEHOST_API_KEY`
   - `SHEET_ID`
   - `SHEET_WORKSHEET` (название вкладки в таблице)
   - `GOOGLE_SERVICE_ACCOUNT_JSON` (путь до JSON сервисного аккаунта)

## Локальный запуск
- Единичный прогон пайплайна:
  ```bash
  python -m app.main --mode run-once
  ```
- Запуск по расписанию (блокирующий режим):
  ```bash
  python -m app.main --mode scheduler
  ```

## Docker
- Соберите и запустите сервис:
  ```bash
  docker compose up --build
  ```
- Планировщик поднимется внутри контейнера, healthcheck использует `python -m app.healthcheck`.

## Тесты и качество
- Запуск модульных тестов:
  ```bash
  python -m pytest
  ```
- CI-конвейер описан в `.github/workflows/tests.yml` и прогоняет тесты при push/PR.

## Развертывание на удалённом сервере
1. Подготовьте сервер с Docker и Docker Compose (Linux x86_64 или ARM64).
2. Склонируйте репозиторий:
   ```bash
   git clone git@github.com:kodjooo/content-from-rss.git
   cd content-from-rss
   ```
3. Скопируйте `.env.example` в `.env` и заполните переменные (`OPENAI_API_KEY`, `PEXELS_API_KEY`, `FREEIMAGEHOST_API_KEY`, `SHEET_ID`, `SHEET_WORKSHEET`, `GOOGLE_SERVICE_ACCOUNT_JSON`).
4. Загрузите файл сервисного аккаунта Google в каталог `secrets/` (путь должен совпадать с `GOOGLE_SERVICE_ACCOUNT_JSON`).
5. Запустите сервис в фоне:
   ```bash
   docker compose pull
   docker compose up -d --build
   ```
6. Проверьте статус контейнера и healthcheck:
   ```bash
   docker compose ps
   docker compose logs -f
   ```
7. Для обновления версии выполните `git pull`, затем `docker compose up -d --build`.

## Структура проекта
- `app/` — реализации модулей (RSS, OpenAI, генерация изображений, Sheets, оркестратор, планировщик).
- `tests/` — pytest-тесты с моками внешних сервисов.
- `docs/` — требования, архитектура и детальный план внедрения (этапы отмечаются в `docs/plan.md`).
- `scripts/setup_env.py` — утилита генерации `.env`.
- `docker-compose.yml` и `Dockerfile` — контейнеризация сервиса.

## Полезные команды
- Очистить и пересоздать `.env`:
  ```bash
  python scripts/setup_env.py --force
  ```
- Быстрый smoke-тест пайплайна (без расписания):
  ```bash
  python -m app.main --mode run-once
  ```

`.env` не коммитится в репозиторий; перед публикацией проверяйте, что файл отсутствует в индексе Git.
