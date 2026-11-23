from __future__ import annotations

import asyncio
import hashlib
import logging
import re

import httpx

from config import settings
import db
from loader import registry
from rules import build_personal_plan

logger = logging.getLogger("atlas.monitor")


def _normalize_html(text: str) -> str:
    # Убираем скрипты, стили и пробелы, чтобы не реагировать на динамические счётчики/токены
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:50_000]


async def check_nla_sources() -> list[dict]:
    """
    Автоматический мониторинг изменений на официальных страницах НПА
    (pravo.by, e-pasluga.by, lex.uz, emehmon.uz).
    """
    results: list[dict] = []
    sources = registry.all_nla_sources()
    async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
        for country_code, src in sources:
            try:
                resp = await client.get(
                    src.url,
                    headers={"User-Agent": "AtlasGovMonitor/1.0 (+VSU Belarus)"},
                )
                content_hash = (
                    hashlib.sha256(_normalize_html(resp.text).encode("utf-8")).hexdigest()[:16]
                    if resp.status_code < 400
                    else None
                )
                changed = await db.save_nla_snapshot(
                    url=src.url,
                    country_code=country_code,
                    name=src.name,
                    content_hash=content_hash,
                    http_status=resp.status_code,
                )
                results.append(
                    {
                        "country": country_code,
                        "name": src.name,
                        "url": src.url,
                        "status": resp.status_code,
                        "changed": changed,
                    }
                )
                if changed:
                    logger.warning("ОБНАРУЖЕНО ИЗМЕНЕНИЕ НПА: %s (%s)", src.name, src.url)
            except Exception as exc:
                await db.save_nla_snapshot(
                    url=src.url,
                    country_code=country_code,
                    name=src.name,
                    content_hash=None,
                    http_status=0,
                )
                results.append(
                    {
                        "country": country_code,
                        "name": src.name,
                        "url": src.url,
                        "status": 0,
                        "changed": False,
                        "error": str(exc),
                    }
                )
    return results


async def check_deadline_reminders(bot) -> None:
    """Отправляет Telegram-напоминания за 2 дня до дедлайна регистрации/шага вуза."""
    if bot is None:
        return
    users = await db.get_users_with_arrival()
    for user in users:
        completed = await db.get_checklist_progress(user["tg_id"])
        plan = build_personal_plan(
            citizenship=user["citizenship"],
            destination=user["destination"],
            university_id=user["university_id"],
            purpose=user["purpose"],
            housing_type=user["housing_type"],
            entry_via_rf=bool(user["entry_via_rf"]),
            arrival_date=user["arrival_date"],
            completed_map=completed,
        )
        for step in plan.steps:
            if step.completed or step.days_left is None:
                continue
            if 0 <= step.days_left <= 2 and not await db.was_reminded(user["tg_id"], step.id):
                try:
                    await bot.send_message(
                        user["tg_id"],
                        (
                            f"⏰ <b>Напоминание «Атлас»</b>\n\n"
                            f"<b>{step.title}</b>\n"
                            f"Осталось дней: {step.days_left}\n"
                            f"{step.description}"
                        ),
                        parse_mode="HTML",
                    )
                    await db.mark_step_reminded(user["tg_id"], step.id)
                except Exception as exc:
                    logger.debug("Не удалось отправить напоминание %s: %s", user["tg_id"], exc)


async def background_loop(bot) -> None:
    """Единый лёгкий фоновый цикл без сторонних планировщиков."""
    nla_every = max(10, settings.nla_check_interval_min) * 60
    rem_every = max(5, settings.reminder_check_interval_min) * 60
    last_nla = 0.0
    last_rem = 0.0

    while True:
        now = asyncio.get_running_loop().time()
        if now - last_nla >= nla_every:
            await check_nla_sources()
            last_nla = now
        if now - last_rem >= rem_every:
            await check_deadline_reminders(bot)
            last_rem = now
        await asyncio.sleep(30)
