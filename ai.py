from __future__ import annotations

import base64
import json
import logging
import re
from pydantic import BaseModel, Field

from openai import AsyncOpenAI

from config import settings
from loader import KnowledgeCard, registry

logger = logging.getLogger("atlas.ai")


# --- Privacy Firewall (локальное обезличивание ПД до отправки в любой ИИ) ---
# ВАЖНО: порядок от более точных форматов (карта, 14-значный ID, паспорт) к общим (телефон)
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Email
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[СКРЫТО_EMAIL]"),
    # Номера банковских карт (16 цифр с пробелами/дефисами или слитно)
    (re.compile(r"(?<!\d)(?:\d{4}[\s\-]?){3}\d{4}(?!\d)"), "[СКРЫТО_КАРТА]"),
    # Личный номер РБ (14 символов: 7 цифр + буква + 3 цифры + 2 буквы + цифра)
    (re.compile(r"\b\d{7}[A-ZА-Я]\d{3}[A-ZА-Я]{2}\d\b", re.IGNORECASE), "[СКРЫТО_ИДЕНТИФИКАТОР]"),
    # ПИНФЛ / 14-значный национальный идентификатор
    (re.compile(r"(?<!\d)\d{14}(?!\d)"), "[СКРЫТО_ИДЕНТИФИКАТОР]"),
    # Номера паспортов (РБ/РУз: 2 буквы + 7 цифр; КНР: E/G + 8 цифр; РФ: 4 цифры + 6 цифр)
    (
        re.compile(r"\b(?:[A-ZА-Я]{2}\s?\d{7}|[EG]\d{8}|\d{2}\s?\d{2}\s?\d{6})\b", re.IGNORECASE),
        "[СКРЫТО_ПАСПОРТ]",
    ),
    # Телефоны (РБ, СНГ, международные: 10–15 цифр)
    (re.compile(r"(?<!\d)(?:\+?\d[\s\-()]*){9,14}\d(?!\d)"), "[СКРЫТО_ТЕЛЕФОН]"),
]


def scrub_pii(text: str) -> tuple[str, int]:
    """Удаляет персональные данные из текста до отправки во внешние сервисы (Закон РБ № 99-З)."""
    scrubbed = text
    total_hits = 0
    for pattern, placeholder in _PII_PATTERNS:
        scrubbed, count = pattern.subn(placeholder, scrubbed)
        total_hits += count
    return scrubbed, total_hits


class VerifiedSource(BaseModel):
    card_id: str
    title: str
    source_title: str
    source_url: str
    verified_at: str


class AiAnswer(BaseModel):
    answer: str
    sources: list[VerifiedSource] = Field(default_factory=list)
    confident: bool = True
    escalate_to_university: bool = False
    pii_redacted_count: int = 0
    scrubbed_query: str = ""
    topic: str = "general"
    mode: str = "verified_kb"  # sovereign_llm | cloud_llm_scrubbed | verified_kb
    disclaimer: str = "Справочный ответ на основе проверенной базы «Атлас»."


_MULTILINGUAL_BRIDGES: list[tuple[tuple[str, ...], str]] = [
    (
        ("fine", "fines", "penalty", "overstay", "jarima", "штраф", "просроч", "24.35", "коап", "депортац", "罚款", "逾期"),
        "штраф просрочка наказание депортация 24.35 коап базовая",
    ),
    (
        ("registration", "register", "ogim", "police", "ro'yxat", "royxat", "рэгістрац", "注册", "登记"),
        "регистрация огим епэу срок",
    ),
    (
        ("russia", "russian", "moscow", "smolensk", "rossiya", "расі", "через рф", "через россию", "俄罗斯", "转机"),
        "рф россия граница",
    ),
    (
        ("dorm", "dormitory", "hostel", "checkin", "yotoqxona", "інтэрнат", "заселен", "пропуск", "宿舍", "入住"),
        "общежитие заселение международный отдел",
    ),
    (
        ("sim", "phone", "bank", "card", "money", "pay", "payment", "erip", "aloqa", "to'lov", "tolov", "картк", "аплат", "ерип", "пошлин", "银行", "电话卡", "支付"),
        "сим связь банк карта оплата ерип деньги",
    ),
    (
        ("doctor", "hospital", "clinic", "medical", "insurance", "pharmacy", "shifokor", "dorixona", "sug'urta", "лекарств", "страховк", "медосмотр", "урач", "аптэк", "医院", "保险", "药店"),
        "медицина страховка врач поликлиника аптека медосмотр 103",
    ),
    (
        ("halal", "food", "eat", "cafe", "shop", "halol", "ovqat", "еда", "кафе", "халяль", "харчаван", "清真", "餐厅"),
        "халяль еда кафе магазин продукты",
    ),
    (
        ("visa", "visafree", "tourist", "tourism", "безвиз", "туризм", "турист", "bepul viza", "бязвіз", "免签", "旅游"),
        "безвиз виза туризм въезд",
    ),
    (
        ("lost", "stolen", "embassy", "consulate", "потерял", "украли", "утеря", "паспорт", "посольств", "yo'qot", "элчихона", "丢失", "护照", "大使馆"),
        "утеря паспорт милиция 102 посольство справка",
    ),
]


def _expand_query(query: str) -> str:
    low = query.lower()
    extras: list[str] = []
    for triggers, bridge in _MULTILINGUAL_BRIDGES:
        if any(t in low for t in triggers):
            extras.append(bridge)
    return f"{query} {' '.join(extras)}" if extras else query


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁўЎқҚғҒҳҲіІ0-9']{2,}", text.lower())
    stems = {w[:5] if len(w) > 5 else w for w in words}
    # Поддержка иероглифов (биграммы для китайского языка)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_tokens = {cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)}
    return set(words) | stems | cjk_tokens


def retrieve_cards(
    query: str,
    destination: str | None = None,
    university_id: str | None = None,
    top_k: int = 4,
) -> list[KnowledgeCard]:
    """Быстрый локальный поиск по проверенным карточкам без векторной БД."""
    cards = registry.get_cards(destination=destination, university_id=university_id)
    if not cards:
        cards = registry.get_cards()

    expanded = _expand_query(query)
    q_tokens = _tokenize(expanded)
    if not q_tokens:
        return cards[:top_k]

    scored: list[tuple[int, KnowledgeCard]] = []
    for card in cards:
        haystack = f"{card.title} {' '.join(card.tags)} {card.content}"
        c_tokens = _tokenize(haystack)
        tag_tokens = _tokenize(" ".join(card.tags) + " " + card.title)
        score = len(q_tokens & c_tokens) + 2 * len(q_tokens & tag_tokens)
        if score > 0:
            scored.append((score, card))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def _llm_client() -> AsyncOpenAI | None:
    if not settings.llm_api_key:
        return None
    return AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


async def ask_aist(
    question: str,
    destination: str | None = None,
    university_id: str | None = None,
    lang: str = "ru",
) -> AiAnswer:
    """
    Strict RAG ассистент AIst:
    1. Обезличивает вопрос (Privacy Firewall).
    2. Ищет релевантные проверенные карточки в локальных модулях.
    3. Если подключён LLM — генерирует ответ строго по карточкам,
       а ссылки и даты проверки подставляет бэкенд по card_ids (нельзя выдумать ссылку).
    4. Если уверенности нет — предлагает передать тикет в международный отдел вуза.
    """
    clean_q, pii_hits = scrub_pii(question)
    matched = retrieve_cards(clean_q, destination=destination, university_id=university_id, top_k=4)

    if not matched:
        return AiAnswer(
            answer=(
                "В проверенной базе «Атласа» пока нет точного ответа на этот вопрос. "
                "Чтобы не ошибиться в правилах, я не буду додумывать — лучше передать вопрос "
                "специалисту международного отдела вуза."
            ),
            sources=[],
            confident=False,
            escalate_to_university=True,
            pii_redacted_count=pii_hits,
            scrubbed_query=clean_q,
            topic="unanswered",
            mode="verified_kb",
        )

    cards_by_id = {c.id: c for c in matched}
    client = _llm_client()

    # Fallback без внешнего API-ключа (отлично работает на демо и в офлайне)
    if client is None:
        top = matched[0]
        extra = ""
        if len(matched) > 1:
            extra = f"\n\nТакже по теме: «{matched[1].title}»."
        return AiAnswer(
            answer=f"{top.title}\n\n{top.content}{extra}",
            sources=[
                VerifiedSource(
                    card_id=top.id,
                    title=top.title,
                    source_title=top.source_title,
                    source_url=top.source_url,
                    verified_at=top.verified_at,
                )
            ],
            confident=True,
            escalate_to_university=False,
            pii_redacted_count=pii_hits,
            scrubbed_query=clean_q,
            topic=top.topic,
            mode="verified_kb",
        )

    context_blocks = [
        f"[{c.id}] {c.title}\n{c.content}"
        for c in matched
    ]
    system_prompt = (
        "Ты — AIst, интеллектуальный проводник платформы «Атлас» для иностранных студентов и гостей. "
        "Отвечай КРАТКО, по делу и СТРОГО на основе переданных карточек.\n"
        "ЗАПРЕЩЕНО выдумывать законы, сроки, адреса или ссылки, которых нет в карточках.\n"
        "Если в карточках нет ответа на вопрос, установи confident=false.\n"
        f"Язык ответа: {lang}.\n"
        "Верни строго JSON без пояснений: "
        '{"answer": "текст ответа", "card_ids": ["id1"], "confident": true}'
    )
    user_prompt = (
        "Проверенные карточки базы знаний:\n"
        + "\n\n".join(context_blocks)
        + f"\n\nВопрос гостя (обезличен): {clean_q}"
    )

    try:
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_content = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw_content)
        answer_text = str(parsed.get("answer", "")).strip()
        used_ids = [cid for cid in parsed.get("card_ids", []) if cid in cards_by_id]
        confident = bool(parsed.get("confident", False)) and bool(used_ids)
    except Exception as exc:
        logger.warning("Ошибка LLM (%s), переключение на базу знаний", exc)
        top = matched[0]
        return AiAnswer(
            answer=f"{top.title}\n\n{top.content}",
            sources=[
                VerifiedSource(
                    card_id=top.id,
                    title=top.title,
                    source_title=top.source_title,
                    source_url=top.source_url,
                    verified_at=top.verified_at,
                )
            ],
            confident=True,
            escalate_to_university=False,
            pii_redacted_count=pii_hits,
            scrubbed_query=clean_q,
            topic=top.topic,
            mode="verified_kb",
        )

    # Ссылки и даты подставляются строго бэкендом по проверенным ID
    verified_sources = [
        VerifiedSource(
            card_id=cards_by_id[cid].id,
            title=cards_by_id[cid].title,
            source_title=cards_by_id[cid].source_title,
            source_url=cards_by_id[cid].source_url,
            verified_at=cards_by_id[cid].verified_at,
        )
        for cid in used_ids
    ]
    topic = cards_by_id[used_ids[0]].topic if used_ids else "unanswered"

    if not confident or not answer_text:
        return AiAnswer(
            answer=(
                "У меня нет проверенных данных по этой детали, поэтому я не стану угадывать. "
                "Нажмите кнопку ниже, чтобы передать вопрос специалисту международного отдела вуза."
            ),
            sources=[],
            confident=False,
            escalate_to_university=True,
            pii_redacted_count=pii_hits,
            scrubbed_query=clean_q,
            topic="unanswered",
            mode="sovereign_llm" if settings.is_sovereign_llm else "cloud_llm_scrubbed",
        )

    return AiAnswer(
        answer=answer_text,
        sources=verified_sources,
        confident=True,
        escalate_to_university=False,
        pii_redacted_count=pii_hits,
        scrubbed_query=clean_q,
        topic=topic,
        mode="sovereign_llm" if settings.is_sovereign_llm else "cloud_llm_scrubbed",
    )


async def translate_photo_bytes(image_bytes: bytes, target_lang: str = "ru") -> str:
    """
    Перевод вывесок и объявлений строго в оперативной памяти (без записи фото на диск).
    """
    client = _llm_client()
    if client is None:
        return (
            "📷 Фото-переводчик готов к работе, но в .env пока не задан LLM_API_KEY "
            "(поддерживается OpenAI / Gemini / OpenRouter или локальная Vision-модель Ollama).\n"
            "Фото не сохранялось на диск."
        )

    b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        f"На фото вывеска, указатель или бытовое объявление. Переведи текст на язык '{target_lang}' "
        "и в 1–2 предложениях поясни, что это значит для гостя города. "
        "Если на фото похож паспорт или личный документ — НЕ переписывай личные номера, "
        "а вежливо напомни не отправлять фото документов."
    )
    try:
        resp = await client.chat.completions.create(
            model=settings.llm_vision_model,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        return (resp.choices[0].message.content or "Не удалось распознать текст на фото.").strip()
    except Exception as exc:
        logger.warning("Ошибка Vision API: %s", exc)
        return "Не удалось обработать фото через Vision API. Проверьте настройки модели в .env."
