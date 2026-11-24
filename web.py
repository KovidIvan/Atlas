from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

import ai
from config import persist_llm_config, settings
import db
from loader import registry
import monitor
from rules import build_personal_plan

logger = logging.getLogger("atlas.web")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await db.init_db()
    registry.reload()
    ai.rag_index.rebuild()
    yield


app = FastAPI(
    title="Атлас — модульная платформа адаптации и навигации",
    version="1.1.0",
    lifespan=lifespan,
)

# Ссылка на запущенный Telegram-бот (подставляется из main.py для отправки ответов на тикеты)
_active_bot = None


def set_active_bot(bot) -> None:
    global _active_bot
    _active_bot = bot


def _require_admin(x_admin_token: str | None) -> None:
    if not x_admin_token or x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="Неверный токен администратора")


# --- Страницы Mini App и Мини-кабинета вуза ---

@app.get("/")
@app.get("/app")
async def serve_mini_app() -> FileResponse:
    return FileResponse(settings.static_dir / "app.html")


@app.get("/manifest.json")
async def serve_manifest() -> FileResponse:
    return FileResponse(settings.static_dir / "manifest.json", media_type="application/manifest+json")


@app.get("/sw.js")
async def serve_service_worker() -> FileResponse:
    return FileResponse(
        settings.static_dir / "sw.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )


@app.get("/admin")
async def serve_admin_panel() -> FileResponse:
    return FileResponse(settings.static_dir / "admin.html")


# --- Публичный API Mini App ---

@app.get("/api/modules")
async def get_modules() -> dict:
    """Возвращает загруженные модули стран и вузов (демонстрация модульной архитектуры)."""
    return {
        "countries": [
            {
                "code": c.code,
                "name": c.name,
                "flag": c.flag,
                "languages": c.languages,
                "cards_count": len(c.cards),
            }
            for c in registry.countries.values()
        ],
        "universities": [
            {
                "id": u.id,
                "name": u.name,
                "short_name": u.short_name,
                "country_code": u.country_code,
                "city": u.city,
                "map_center": u.map_center,
                "international_office_name": u.international_office_name,
                "international_office_address": u.international_office_address,
                "international_office_phone": u.international_office_phone,
                "points_count": len(u.map_points),
                "cards_count": len(u.cards),
            }
            for u in registry.universities.values()
        ],
        "privacy": {
            "zero_pii": True,
            "pii_firewall": True,
            "sovereign_llm": settings.is_sovereign_llm,
            "law_ref": "Закон РБ № 99-З · Указ № 60",
        },
    }


@app.get("/api/profile")
async def get_profile_and_plan(tg_id: int = Query(10001)) -> dict:
    """Читает сохранённый профиль и план без перезаписи параметров по умолчанию."""
    user = await db.get_user(tg_id)
    if user:
        completed = await db.get_checklist_progress(tg_id)
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
        return {
            "exists": True,
            "lang": user.get("lang") or "ru",
            "plan": plan.model_dump(),
        }
    default_plan = build_personal_plan(
        citizenship="UZ",
        destination="BY",
        university_id="vsu_vitebsk",
        purpose="study",
        housing_type="dorm",
        entry_via_rf=False,
        arrival_date=None,
        completed_map={},
    )
    return {"exists": False, "lang": "ru", "plan": default_plan.model_dump()}


class PlanRequest(BaseModel):
    tg_id: int = 10001
    lang: str = "ru"
    citizenship: str = "UZ"
    destination: str = "BY"
    university_id: str | None = "vsu_vitebsk"
    purpose: str = "study"
    housing_type: str = "dorm"
    entry_via_rf: bool = False
    arrival_date: str | None = None


@app.post("/api/plan")
async def save_and_build_plan(req: PlanRequest) -> dict:
    await db.upsert_user(
        req.tg_id,
        consent_given=1,
        lang=req.lang,
        citizenship=req.citizenship.upper(),
        destination=req.destination.upper(),
        university_id=req.university_id,
        purpose=req.purpose,
        housing_type=req.housing_type,
        entry_via_rf=req.entry_via_rf,
        arrival_date=req.arrival_date,
    )
    completed = await db.get_checklist_progress(req.tg_id)
    plan = build_personal_plan(
        citizenship=req.citizenship,
        destination=req.destination,
        university_id=req.university_id,
        purpose=req.purpose,
        housing_type=req.housing_type,
        entry_via_rf=req.entry_via_rf,
        arrival_date=req.arrival_date,
        completed_map=completed,
    )
    await db.log_event("plan", university_id=req.university_id, country_code=req.destination.upper())
    return plan.model_dump()


class ToggleStepRequest(BaseModel):
    tg_id: int = 10001
    step_id: str
    completed: bool


@app.post("/api/plan/toggle")
async def api_toggle_step(req: ToggleStepRequest) -> dict:
    await db.toggle_step(req.tg_id, req.step_id, req.completed)
    return {"ok": True}


class AskRequest(BaseModel):
    question: str
    destination: str = "BY"
    university_id: str | None = "vsu_vitebsk"
    lang: str = "ru"
    history: list[dict[str, str]] = []


@app.post("/api/ask")
async def api_ask(req: AskRequest) -> dict:
    res = await ai.ask_aist(
        question=req.question,
        destination=req.destination,
        university_id=req.university_id,
        lang=req.lang,
        history=req.history,
    )
    await db.log_event(
        "ask",
        university_id=req.university_id,
        country_code=req.destination.upper(),
        topic=res.topic,
    )
    return res.model_dump()


@app.get("/api/map")
async def api_map(
    university_id: str = Query("vsu_vitebsk"),
    category: str | None = Query(None),
    halal: bool = Query(False),
    accessible: bool = Query(False),
) -> dict:
    uni = registry.universities.get(university_id)
    if not uni:
        return {"center": [55.1904, 30.2049], "city": "Витебск", "points": []}
    pts = uni.map_points
    if category:
        pts = [p for p in pts if p.category == category]
    if halal:
        pts = [p for p in pts if p.halal]
    if accessible:
        pts = [p for p in pts if p.accessible]
    return {
        "university": uni.short_name,
        "city": uni.city,
        "center": uni.map_center,
        "points": [p.model_dump() for p in pts],
    }


@app.get("/api/toolkit")
async def api_toolkit(
    destination: str = Query("BY"),
    university_id: str | None = Query("vsu_vitebsk"),
) -> dict:
    """Единый пакет для вкладки «Сервисы» и офлайн-кэша: карточки показа, ЕРИП, фразы, SOS."""
    dest = destination.upper()
    country = registry.countries.get(dest)
    uni = registry.universities.get(university_id) if university_id else None
    if uni and uni.country_code.upper() != dest:
        uni = None

    erip = []
    show_cards = []
    if uni:
        erip.extend([e.model_dump() for e in uni.erip_guides])
        show_cards.extend([c.model_dump() for c in uni.show_cards])
    if country:
        erip.extend([e.model_dump() for e in country.erip_guides])
        show_cards.extend([c.model_dump() for c in country.show_cards])

    return {
        "country": country.name if country else dest,
        "university": uni.short_name if uni else None,
        "erip_guides": erip,
        "show_cards": show_cards,
        "phrases": [p.model_dump() for p in country.phrasebook] if country else [],
        "contacts": [c.model_dump() for c in country.sos_contacts] if country else [],
        "university_office": (
            {
                "name": uni.international_office_name,
                "university": uni.short_name,
                "address": uni.international_office_address,
                "phone": uni.international_office_phone,
                "email": uni.international_office_email,
            }
            if uni
            else None
        ),
    }


@app.get("/api/phrasebook")
async def api_phrasebook(destination: str = Query("BY")) -> dict:
    country = registry.countries.get(destination.upper())
    if not country:
        return {"phrases": []}
    return {"country": country.name, "phrases": [p.model_dump() for p in country.phrasebook]}


@app.get("/api/sos")
async def api_sos(
    destination: str = Query("BY"),
    university_id: str | None = Query("vsu_vitebsk"),
) -> dict:
    country = registry.countries.get(destination.upper())
    uni = registry.universities.get(university_id) if university_id else None
    return {
        "country": country.name if country else destination,
        "contacts": [c.model_dump() for c in country.sos_contacts] if country else [],
        "university_office": (
            {
                "name": uni.international_office_name,
                "university": uni.short_name,
                "address": uni.international_office_address,
                "phone": uni.international_office_phone,
                "email": uni.international_office_email,
            }
            if uni
            else None
        ),
    }


class PhotoTranslateRequest(BaseModel):
    image_base64: str
    lang: str = "ru"
    destination: str = "BY"
    university_id: str | None = "vsu_vitebsk"


@app.post("/api/translate-photo")
async def api_translate_photo(req: PhotoTranslateRequest) -> dict:
    raw = req.image_base64.strip()
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    if len(raw) > 8_500_000:
        raise HTTPException(status_code=413, detail="Файл слишком большой (макс. 6 МБ)")
    try:
        img_bytes = base64.b64decode(raw, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Некорректный формат изображения") from exc

    translation = await ai.translate_photo_bytes(img_bytes, target_lang=req.lang)
    await db.log_event(
        "photo_translate",
        university_id=req.university_id,
        country_code=req.destination.upper(),
    )
    return {"translation": translation, "ram_only": True}


class TicketRequest(BaseModel):
    tg_id: int = 10001
    university_id: str = "vsu_vitebsk"
    question: str


@app.post("/api/tickets")
async def api_create_ticket(req: TicketRequest) -> dict:
    clean_q, pii_hits = ai.scrub_pii(req.question)
    ticket_id = await db.create_ticket(req.tg_id, req.university_id, clean_q)
    uni = registry.universities.get(req.university_id)
    await db.log_event(
        "ticket",
        university_id=req.university_id,
        country_code=uni.country_code if uni else None,
    )
    return {"ticket_id": ticket_id, "pii_redacted_count": pii_hits, "question_scrubbed": clean_q}


@app.get("/api/tickets/my")
async def api_my_tickets(tg_id: int = Query(10001)) -> dict:
    items = await db.list_user_tickets(tg_id, limit=20)
    return {"tickets": items}


@app.delete("/api/user/{tg_id}")
async def api_delete_user(tg_id: int) -> dict:
    await db.delete_user_data(tg_id)
    return {"deleted": True, "tg_id": tg_id}


# --- API Мини-кабинета международного отдела вуза (/admin) ---

@app.get("/api/admin/overview")
async def admin_overview(x_admin_token: str | None = Header(None)) -> dict:
    _require_admin(x_admin_token)
    ai.rag_index.ensure_built()
    stats = await db.get_admin_stats()
    tickets = await db.list_tickets(limit=30)
    raw_key = settings.llm_api_key
    masked_key = (
        f"{raw_key[:4]}…{raw_key[-4:]}" if len(raw_key) > 8 else ("задан" if raw_key else "")
    )
    return {
        "stats": stats,
        "tickets": tickets,
        "modules": {
            "countries": [
                {"code": c.code, "name": c.name, "cards": len(c.cards), "nla_sources": len(c.nla_sources)}
                for c in registry.countries.values()
            ],
            "universities": [
                {"id": u.id, "short_name": u.short_name, "city": u.city, "points": len(u.map_points), "cards": len(u.cards)}
                for u in registry.universities.values()
            ],
        },
        "security": {
            "zero_pii": True,
            "pii_firewall": True,
            "rag_chunks": len(ai.rag_index.chunks),
            "llm_mode": (
                "Локальный суверенный ИИ"
                if settings.is_sovereign_llm and settings.llm_api_key
                else ("Подключён внешний LLM + Strict RAG" if settings.llm_api_key else "Автономный BM25 RAG (без API-ключа)")
            ),
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
            "llm_vision_model": settings.llm_vision_model,
            "llm_has_key": bool(raw_key),
            "llm_key_masked": masked_key,
        },
    }


class LlmConfigRequest(BaseModel):
    base_url: str
    api_key: str = ""
    clear_key: bool = False
    model: str
    vision_model: str = ""


@app.post("/api/admin/llm")
async def admin_update_llm(
    req: LlmConfigRequest,
    x_admin_token: str | None = Header(None),
) -> dict:
    _require_admin(x_admin_token)
    new_key: str | None
    if req.clear_key:
        new_key = ""
    elif req.api_key.strip():
        new_key = req.api_key.strip()
    else:
        new_key = None  # сохраняем ранее введённый ключ, если поле оставили пустым

    persist_llm_config(
        base_url=req.base_url,
        api_key=new_key,
        model=req.model,
        vision_model=req.vision_model or req.model,
    )
    ok, msg = await ai.test_llm_connection()
    return {"ok": ok, "message": msg}


class TicketReplyRequest(BaseModel):
    reply_text: str


@app.post("/api/admin/tickets/{ticket_id}/reply")
async def admin_reply_ticket(
    ticket_id: int,
    req: TicketReplyRequest,
    x_admin_token: str | None = Header(None),
) -> dict:
    _require_admin(x_admin_token)
    ticket = await db.reply_ticket(ticket_id, req.reply_text)
    if not ticket:
        raise HTTPException(status_code=404, detail="Тикет не найден")

    delivered = False
    if _active_bot is not None and ticket["tg_id"] > 100000:
        try:
            await _active_bot.send_message(
                ticket["tg_id"],
                f"🏛 <b>Ответ международного отдела вуза (обращение #{ticket_id}):</b>\n\n{req.reply_text}",
                parse_mode="HTML",
            )
            delivered = True
        except Exception as exc:
            logger.warning("Не удалось доставить ответ в Telegram: %s", exc)
    return {"ok": True, "ticket": ticket, "delivered_to_telegram": delivered}


@app.post("/api/admin/reload")
async def admin_reload_modules(x_admin_token: str | None = Header(None)) -> dict:
    _require_admin(x_admin_token)
    counts = registry.reload()
    ai.rag_index.rebuild()
    return {"ok": True, "loaded": counts, "rag_chunks": len(ai.rag_index.chunks)}


@app.post("/api/admin/check-nla")
async def admin_check_nla(x_admin_token: str | None = Header(None)) -> dict:
    _require_admin(x_admin_token)
    results = await monitor.check_nla_sources()
    return {"ok": True, "checked": results}
