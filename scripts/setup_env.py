"""Скрипт для генерации файла .env на основе .env.example."""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path


GENERATABLE_SUFFIXES = ("_SECRET", "_TOKEN")


def parse_env_file(path: Path) -> dict[str, str]:
    """Считывает пары ключ-значение из env-файла."""
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def generate_secret() -> str:
    """Генерирует случайный секрет."""
    return secrets.token_hex(32)


def build_env_content(values: dict[str, str]) -> str:
    """Формирует текст env-файла из словаря."""
    return "\n".join(f"{key}={values.get(key, '')}" for key in sorted(values)) + "\n"


def main() -> None:
    """Точка входа CLI."""
    parser = argparse.ArgumentParser(description="Генерация .env на основе .env.example")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующий .env",
    )
    parser.add_argument(
        "--target",
        default=".env",
        help="Путь до создаваемого .env файла",
    )
    args = parser.parse_args()

    example_path = Path(".env.example")
    if not example_path.exists():
        raise FileNotFoundError("Файл .env.example не найден")

    target_path = Path(args.target)
    if target_path.exists() and not args.force:
        print("Файл .env уже существует. Используйте --force для перезаписи.")
        return

    values = parse_env_file(example_path)
    for key, value in values.items():
        if value:
            continue
        if key.endswith(GENERATABLE_SUFFIXES):
            values[key] = generate_secret()
    target_path.write_text(build_env_content(values), encoding="utf-8")
    print(f".env успешно создан: {target_path}")


if __name__ == "__main__":
    main()
