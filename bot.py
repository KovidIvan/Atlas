from __future__ import annotations

import html
import io
import logging
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    WebAppInfo,
)

import ai
from config import settings
import db
from loader import registry
from rules import build_personal_plan

logger = logging.getLogger("atlas.bot")
router = Router()

# Временный кэш последнего вопроса в памяти (только обезличенный текст, для кнопки «Передать в вуз»)
_last_scrubbed_question: dict[int, str] = {}

CONSENT_TEXT = (
    "👋 <b>Добро пожаловать в «Атлас»!</b>\n"
    "Цифровая платформа адаптации и навигации для иностранных студентов и гостей Беларуси.\n\n"
    "🔒 <b>Конфиденциальность (Закон РБ № 99-З):</b>\n"
    "• Мы <b>не запрашиваем и не храним</b> ФИО, номера/фото паспортов, телефоны и адреса.\n"
    "• Перед обращением к ИИ срабатывает локальный <b>Privacy Firewall</b>, скрывающий случайные личные данные.\n"
    "• Сервис носит <b>справочный характер</b> (ответы со ссылками на НПА и датой проверки).\n"
    "• Сообщения проходят через инфраструктуру Telegram; вы можете удалить все свои настройки в 1 клик командой /delete_data.\n\n"
    "Нажмите кнопку ниже, чтобы подтвердить согласие и начать."
)


def main_keyboard() -> ReplyKeyboardMarkup:
    rows: list[list[KeyboardButton]] = []
    app_url = settings.app_base_url.rstrip("/") + "/app"
    if app_url.startswith("https://"):
        rows.append([KeyboardButton(text="🗺 Открыть «Атлас» (Карта и План)", web_app=WebAppInfo(url=app_url))])
    rows.append([KeyboardButton(text="📋 Мой план"), KeyboardButton(text="🆘 SOS и посольства")])
    rows.append([KeyboardButton(text="🏛 Вопрос в вуз"), KeyboardButton(text="🗑 Удалить мои данные")])
    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True)


def consent_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Понятно, принимаю и начать", callback_data="consent_ok")]
        ]
    )


async def _ensure_consent(message: Message) -> dict | None:
    if not message.from_user:
        return None
    user = await db.get_user(message.from_user.id)
    if not user or not user.get("consent_given"):
        await message.answer(CONSENT_TEXT, parse_mode="HTML", reply_markup=consent_keyboard())
        return None
    return user


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if not message.from_user:
        return
    user = await db.get_user(message.from_user.id)
    if user and user.get("consent_given"):
        await message.answer(
            "🕊 <b>AIst на связи!</b>\n\n"
            "• Напишите любой вопрос о регистрации, учёбе, быте или культуре.\n"
            "• Отправьте <b>фото вывески или объявления</b> (не паспорта!) — я переведу и объясню смысл.\n"
            "• Или откройте «Мой план» и карту кнопками ниже.",
            parse_mode="HTML",
            reply_markup=main_keyboard(),
        )
        return
    await message.answer(CONSENT_TEXT, parse_mode="HTML", reply_markup=consent_keyboard())


@router.callback_query(F.data == "consent_ok")
async def cb_consent(callback: CallbackQuery) -> None:
    if not callback.from_user:
        return
    await db.upsert_user(callback.from_user.id, consent_given=1)
    await callback.answer("Согласие сохранено")
    if callback.message:
        await callback.message.answer(
            "✅ Готово! Добро пожаловать в «Атлас».\n\n"
            "По умолчанию выбран профиль: <b>Узбекистан → Беларусь (ВГУ им. П. М. Машерова)</b>.\n"
            "Настроить свой план, карту и страну можно в Mini App или задав вопрос прямо здесь.",
            parse_mode="HTML",
            reply_markup=main_keyboard(),
        )


@router.message(Command("delete_data"))
@router.message(F.text == "🗑 Удалить мои данные")
async def cmd_delete_data(message: Message) -> None:
    if not message.from_user:
        return
    uid = message.from_user.id
    _last_scrubbed_question.pop(uid, None)
    await db.delete_user_data(uid)
    await message.answer(
        "🗑 <b>Все ваши данные удалены</b> из базы «Атлас» (профиль, чек-лист, напоминания и тикеты).\n"
        "Чтобы начать заново, отправьте /start.",
        parse_mode="HTML",
    )


@router.message(Command("plan"))
@router.message(F.text == "📋 Мой план")
async def cmd_plan(message: Message) -> None:
    user = await _ensure_consent(message)
    if not user or not message.from_user:
        return
    completed = await db.get_checklist_progress(message.from_user.id)
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
    lines = [
        f"📋 <b>Личный план: {html.escape(plan.destination_name)}</b>",
        f"Профиль: {plan.citizenship} → {plan.destination}"
        + (f" · {html.escape(plan.university_name)}" if plan.university_name else ""),
        "",
    ]
    for idx, step in enumerate(plan.steps, 1):
        icon = "✅" if step.completed else ("🔴" if step.urgency == "urgent" else "🔹")
        lines.append(f"{icon} <b>{idx}. {html.escape(step.title)}</b>")
        lines.append(html.escape(step.description))
        if step.location:
            lines.append(f"📍 {html.escape(step.location)}")
        if step.source_url:
            lines.append(f"🔗 Источник: {html.escape(step.source_title)} ({step.source_url})")
        lines.append("")
    lines.append(f"<i>{html.escape(plan.disclaimer)}</i>")
    await message.answer("\n".join(lines), parse_mode="HTML", disable_web_page_preview=True)


@router.message(Command("sos"))
@router.message(F.text == "🆘 SOS и посольства")
async def cmd_sos(message: Message) -> None:
    user = await _ensure_consent(message)
    if not user:
        return
    dest = user["destination"].upper()
    country = registry.countries.get(dest)
    uni = registry.universities.get(user["university_id"]) if user.get("university_id") else None

    lines = [f"🆘 <b>Экстренные контакты ({html.escape(country.name if country else dest)})</b>\n"]
    if country:
        for c in country.sos_contacts:
            note = f" — {html.escape(c.note)}" if c.note else ""
            lines.append(f"• <b>{html.escape(c.title)}:</b> <code>{html.escape(c.phone)}</code>{note}")
    if uni:
        lines.append(
            f"\n🏛 <b>{html.escape(uni.international_office_name)} ({html.escape(uni.short_name)}):</b>\n"
            f"📍 {html.escape(uni.international_office_address)}\n"
            f"📞 <code>{html.escape(uni.international_office_phone)}</code>"
        )
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(F.text == "🏛 Вопрос в вуз")
async def cmd_escalate_hint(message: Message) -> None:
    user = await _ensure_consent(message)
    if not user:
        return
    uni = registry.universities.get(user["university_id"] or "")
    uni_name = uni.short_name if uni else "выбранного вуза"
    await message.answer(
        f"🏛 Чтобы отправить обращение в международный отдел <b>{html.escape(uni_name)}</b>, "
        "напишите сообщение с префиксом <code>/ticket ваш вопрос</code>.\n"
        "<i>Личные номера паспортов и телефоны автоматически скрываются перед сохранением.</i>",
        parse_mode="HTML",
    )


@router.message(Command("ticket"))
async def cmd_ticket(message: Message) -> None:
    user = await _ensure_consent(message)
    if not user or not message.from_user or not message.text:
        return
    raw_q = message.text.removeprefix("/ticket").strip()
    if not raw_q:
        await message.answer("Напишите вопрос после команды, например: <code>/ticket Нужна справка для общежития</code>", parse_mode="HTML")
        return
    clean_q, pii_hits = ai.scrub_pii(raw_q)
    uni_id = user.get("university_id") or "vsu_vitebsk"
    ticket_id = await db.create_ticket(message.from_user.id, uni_id, clean_q)
    await db.log_event("ticket", university_id=uni_id, country_code=user["destination"])
    pii_note = "\n🔒 Личные данные в тексте автоматически скрыты." if pii_hits else ""
    await message.answer(
        f"✅ Обращение <b>#{ticket_id}</b> передано в международный отдел вуза.{pii_note}\n"
        "Ответ специалиста придёт прямо сюда в чат.",
        parse_mode="HTML",
    )


@router.callback_query(F.data == "escalate_last")
async def cb_escalate_last(callback: CallbackQuery) -> None:
    if not callback.from_user:
        return
    uid = callback.from_user.id
    user = await db.get_user(uid)
    uni_id = (user or {}).get("university_id") or "vsu_vitebsk"
    q = _last_scrubbed_question.get(uid, "Запрос консультации из чата «Атлас»")
    ticket_id = await db.create_ticket(uid, uni_id, q)
    await db.log_event("ticket", university_id=uni_id, country_code=(user or {}).get("destination"))
    await callback.answer(f"Обращение #{ticket_id} отправлено в вуз")
    if callback.message:
        await callback.message.answer(
            f"🏛 Вопрос передан в международный отдел вуза (обращение <b>#{ticket_id}</b>). "
            "Ответ придёт в этот чат.",
            parse_mode="HTML",
        )


@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot) -> None:
    user = await _ensure_consent(message)
    if not user or not message.photo:
        return
    status = await message.answer("📷 Распознаю и перевожу вывеску в оперативной памяти (без сохранения фото)…")
    buf = io.BytesIO()
    await bot.download(message.photo[-1], destination=buf)
    result = await ai.translate_photo_bytes(buf.getvalue(), target_lang=user.get("lang", "ru"))
    buf.close()
    await db.log_event("photo_translate", university_id=user.get("university_id"), country_code=user["destination"])
    await status.edit_text(
        f"📷 <b>Перевод вывески (AIst):</b>\n\n{html.escape(result)}\n\n"
        "<i>🔒 Фото обработано в RAM и не сохранено на диск.</i>",
        parse_mode="HTML",
    )


@router.message(F.text)
async def handle_question(message: Message) -> None:
    user = await _ensure_consent(message)
    if not user or not message.from_user or not message.text:
        return
    clean_q, _ = ai.scrub_pii(message.text)
    _last_scrubbed_question[message.from_user.id] = clean_q

    res = await ai.ask_aist(
        question=message.text,
        destination=user["destination"],
        university_id=user.get("university_id"),
        lang=user.get("lang", "ru"),
    )
    await db.log_event(
        "ask",
        university_id=user.get("university_id"),
        country_code=user["destination"],
        topic=res.topic,
    )

    parts = [f"🕊 <b>AIst:</b>\n{html.escape(res.answer)}"]
    if res.pii_redacted_count:
        parts.append(
            f"\n🔒 <i>Privacy Firewall скрыл персональные данные ({res.pii_redacted_count}) перед обработкой.</i>"
        )
    if res.sources:
        parts.append("\n<b>Проверенные источники:</b>")
        for s in res.sources:
            parts.append(
                f"• <a href=\"{html.escape(s.source_url)}\">{html.escape(s.source_title)}</a> "
                f"(проверено {html.escape(s.verified_at)})"
            )
    parts.append(f"\n<i>{html.escape(res.disclaimer)}</i>")

    kb = None
    if res.escalate_to_university:
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🏛 Передать в международный отдел вуза", callback_data="escalate_last")]
            ]
        )
    await message.answer(
        "\n".join(parts),
        parse_mode="HTML",
        disable_web_page_preview=True,
        reply_markup=kb,
    )


def create_bot_and_dispatcher() -> tuple[Bot | None, Dispatcher]:
    dp = Dispatcher()
    dp.include_router(router)
    if not settings.bot_token:
        logger.info("BOT_TOKEN не задан — Telegram-бот отключён, работает Web / Mini App режим.")
        return None, dp
    return Bot(token=settings.bot_token), dp
