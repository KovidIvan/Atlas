from __future__ import annotations

import base64
from collections import Counter
import json
import logging
import math
import re
from pydantic import BaseModel, Field

from openai import AsyncOpenAI

from config import persist_llm_config, settings
from loader import registry

logger = logging.getLogger("atlas.ai")


# --- Privacy Firewall (локальное обезличивание ПД до отправки в любой ИИ) ---
# Порядок от точных форматов (карта, 14-значный ID, паспорт) к общим (телефон)
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
    retrieved_chunks: int = 0
    topic: str = "general"
    mode: str = "verified_kb"  # sovereign_llm | cloud_llm_scrubbed | verified_kb
    disclaimer: str = "Справочный ответ на основе верифицированной базы «Атлас»."


class RagChunk(BaseModel):
    """Единица индекса гибридного RAG (охватывает НПА, правила, точки карты, ЕРИП, шаги и SOS)."""
    id: str
    kind: str  # card | rule | step | place | erip | sos
    country_code: str
    university_id: str | None = None
    topic: str
    title: str
    content: str
    keywords: str = ""
    source_title: str
    source_url: str
    verified_at: str


_STOPWORDS = {
    "как", "где", "что", "когда", "какой", "какая", "какие", "для", "при", "или", "это", "мне",
    "нужно", "можно", "есть", "надо", "его", "она", "они", "мы", "вы", "вас", "нас", "тут", "там",
    "про", "без", "через", "после", "до", "из", "по", "на", "за", "от", "под", "над",
    "the", "and", "for", "with", "how", "where", "what", "when", "can", "should", "from", "into",
}

_MULTILINGUAL_BRIDGES: list[tuple[tuple[str, ...], str]] = [
    (
        ("fine", "fines", "penalty", "overstay", "jarima", "штраф", "просроч", "24.35", "коап", "депортац", "罚款", "逾期"),
        "штраф просрочка наказание депортация 24.35 коап базовая",
    ),
    (
        ("registration", "register", "ogim", "migration", "ro'yxat", "royxat", "рэгістрац", "агім", "огим", "епэу", "pasluga", "注册", "登记", "移民局"),
        "регистрация огим епэу e-pasluga временная срок",
    ),
    (
        ("russia", "russian", "moscow", "smolensk", "rossiya", "расі", "через рф", "через россию", "москв", "смоленск", "俄罗斯", "转机"),
        "рф россия граница билет огим лично",
    ),
    (
        ("dorm", "dormitory", "hostel", "checkin", "room", "yotoqxona", "інтэрнат", "общежит", "заселен", "пропуск", "фрунзе", "терешков", "дзержинск", "宿舍", "入住"),
        "общежитие заселение пропуск студгородок",
    ),
    (
        ("sim", "mobile", "telecom", "mts", "bank", "card", "exchange", "currency", "pay", "payment", "erip", "aloqa", "to'lov", "tolov", "картк", "аплат", "ерип", "арип", "пошлин", "валют", "обмен", "беларусбанк", "银行", "电话卡", "支付", "缴费"),
        "сим связь мтс а1 банк беларусбанк карта обмен оплата ерип госпошлина",
    ),
    (
        ("doctor", "hospital", "clinic", "medical", "checkup", "insurance", "pharmacy", "medicine", "shifokor", "dorixona", "sug'urta", "лекарств", "страховк", "медосмотр", "здравпункт", "поликлиник", "аптек", "белгосстрах", "урач", "аптэк", "医院", "保险", "药店", "体检"),
        "медицина страховка белгосстрах врач поликлиника здравпункт аптека круглосуточно медосмотр 103",
    ),
    (
        ("halal", "food", "eat", "cafe", "shop", "supermarket", "halol", "ovqat", "еда", "кафе", "поесть", "магазин", "продукт", "халяль", "харчаван", "清真", "餐厅", "超市"),
        "халяль еда питание кафе магазин супермаркет восточная",
    ),
    (
        ("visa", "visafree", "tourist", "tourism", "airport", "безвиз", "виза", "туризм", "турист", "аэропорт", "bepul viza", "бязвіз", "免签", "旅游", "机场"),
        "безвиз виза туризм аэропорт въезд",
    ),
    (
        ("lost", "stolen", "embassy", "consulate", "police", "потерял", "украли", "утеря", "паспорт", "посольств", "консульств", "милици", "yo'qot", "elchixona", "丢失", "护照", "大使馆"),
        "утеря паспорт милиция 102 посольство справка",
    ),
    (
        ("notary", "translate", "translation", "нотариус", "перевод паспорта", "белтпп", "tarjima", "notarius", "公证", "翻译"),
        "перевод нотариус белтпп нотариальная паспорт",
    ),
    (
        ("train", "bus", "station", "вокзал", "автовокзал", "поезд", "автобус", "метро", "vokzal", "火车站", "车站"),
        "вокзал автовокзал транспорт поезд автобус",
    ),
    (
        ("international office", "dean", "международный отдел", "деканат", "фоиг", "умс", "201б", "ленинградская", "xalqaro bo'lim", "国际处"),
        "международный отдел связей корпус 201б фоиг умс",
    ),
]


def _expand_query_terms(query: str) -> list[str]:
    low = query.lower()
    extras: list[str] = []
    for triggers, bridge in _MULTILINGUAL_BRIDGES:
        if any(t in low for t in triggers):
            extras.append(bridge)
    return _tokenize_list(" ".join(extras)) if extras else []


def _stem_token(w: str) -> str:
    if len(w) <= 4:
        return w
    # Мягкий стемминг русских/белорусских/узбекских/английских окончаний
    for suf in (
        "ами", "ями", "ого", "его", "ому", "ему", "иях", "иям", "ией", "ться", "ость",
        "lar", "ning", "dan", "ga", "da", "ing", "tion", "ments", "ment",
        "ов", "ев", "ей", "ам", "ям", "ах", "ях", "ой", "ий", "ый", "ая", "яя", "ое", "ее", "ые", "ие", "ую", "юю",
    ):
        if len(w) - len(suf) >= 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w[:6] if len(w) > 6 else w


def _tokenize_list(text: str) -> list[str]:
    raw_words = re.findall(r"[a-zA-Zа-яА-ЯёЁўЎқҚғҒҳҲіІ0-9.\-]{2,}", text.lower())
    tokens: list[str] = []
    for w in raw_words:
        w_clean = w.strip(".-")
        if len(w_clean) < 2 or w_clean in _STOPWORDS:
            continue
        tokens.append(w_clean)
        stem = _stem_token(w_clean)
        if stem != w_clean:
            tokens.append(stem)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(cjk_chars) - 1):
        tokens.append(cjk_chars[i] + cjk_chars[i + 1])
    return tokens


def _char_trigrams(text: str) -> set[str]:
    clean = re.sub(r"\s+", " ", text.lower()).strip()
    if len(clean) < 3:
        return {clean} if clean else set()
    return {clean[i : i + 3] for i in range(min(len(clean) - 2, 260))}


class HybridRagIndex:
    """
    Полноценный локальный гибридный RAG-индекс (Okapi BM25 + символьные 3-граммы):
    индексирует карточки НПА, миграционную матрицу, шаги вуза, все точки карты с адресами/телефонами,
    деревья ЕРИП и контакты посольств. Автоматически пересобирается при обновлении модулей.
    """

    def __init__(self) -> None:
        self._signature: tuple = ()
        self.chunks: list[RagChunk] = []
        self._doc_tf: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._avgdl: float = 1.0
        self._idf: dict[str, float] = {}
        self._trigrams: list[set[str]] = []

    def _current_signature(self) -> tuple:
        return (
            tuple(sorted(registry.countries.keys())),
            tuple(sorted(registry.universities.keys())),
            sum(len(c.cards) + len(c.registration_rules) + len(c.sos_contacts) for c in registry.countries.values()),
            sum(len(u.cards) + len(u.map_points) + len(u.erip_guides) + len(u.onboarding_steps) for u in registry.universities.values()),
        )

    def ensure_built(self) -> None:
        sig = self._current_signature()
        if sig == self._signature and self.chunks:
            return
        self.rebuild()

    def rebuild(self) -> None:
        chunks: list[RagChunk] = []

        # 1. Модули стран (карточки НПА, матрица сроков регистрации, ЕРИП, SOS/посольства)
        for code, country in registry.countries.items():
            cc = code.upper()
            for card in country.cards:
                chunks.append(
                    RagChunk(
                        id=card.id,
                        kind="card",
                        country_code=cc,
                        topic=card.topic,
                        title=card.title,
                        content=card.content,
                        keywords=" ".join(card.tags),
                        source_title=card.source_title,
                        source_url=card.source_url,
                        verified_at=card.verified_at,
                    )
                )

            # Сводный чанк по миграционной матрице страны
            if country.registration_rules:
                rules_lines = []
                for r in country.registration_rules:
                    cit_label = "Все остальные страны" if r.citizenship == "*" else f"Гражданство {r.citizenship}"
                    rules_lines.append(f"• {cit_label} (срок {r.grace_days} сут.): {r.summary} {r.protocol_note}".strip())
                r0 = country.registration_rules[0]
                chunks.append(
                    RagChunk(
                        id=f"{cc.lower()}_rules_matrix",
                        kind="rule",
                        country_code=cc,
                        topic="migration",
                        title=f"Сроки регистрации по гражданству ({country.name})",
                        content="\n".join(rules_lines),
                        keywords="гражданство россия рф казахстан китай узбекистан армения кыргызстан туркменистан индия 90 30 10 дней суток регистрация",
                        source_title=r0.source_title,
                        source_url=r0.source_url,
                        verified_at=r0.verified_at,
                    )
                )

            for erip in country.erip_guides:
                path_str = " → ".join(erip.steps)
                chunks.append(
                    RagChunk(
                        id=erip.id,
                        kind="erip",
                        country_code=cc,
                        topic="finance",
                        title=erip.title,
                        content=f"Путь оплаты: {path_str}.\nЧто вводить: {erip.input_hint}\n{erip.fee_note}".strip(),
                        keywords="оплата сбор ерип пошлина",
                        source_title=erip.source_title or country.name,
                        source_url="https://raschet.by" if cc == "BY" else "https://emehmon.uz",
                        verified_at=erip.verified_at or "2026-11-24",
                    )
                )

            if country.sos_contacts:
                sos_lines = [
                    f"• {s.title}: {s.phone}" + (f" ({s.note})" if s.note else "")
                    for s in country.sos_contacts
                ]
                chunks.append(
                    RagChunk(
                        id=f"{cc.lower()}_sos_embassies",
                        kind="sos",
                        country_code=cc,
                        topic="migration",
                        title=f"Экстренные телефоны и посольства ({country.name})",
                        content="\n".join(sos_lines),
                        keywords="sos экстренный телефон милиция скорая мчс 112 102 103 посольство консульство кнр китай рф россия узбекистан казахстан туркменистан дгим мвд",
                        source_title=f"Справочник экстренных служб и диппредставительств ({country.name})",
                        source_url="https://mfa.gov.by" if cc == "BY" else "https://uzbekistan.travel",
                        verified_at="2026-11-24",
                    )
                )

        # 2. Модули университетов и городов (карточки, шаги, ЕРИП, все точки карты с адресами и часами!)
        for uid, uni in registry.universities.items():
            cc = uni.country_code.upper()
            uni_url = "https://bsu.by" if "bsu" in uid else "https://vsu.by"

            # Карточка контактов международного отдела и ОГиМ вуза
            chunks.append(
                RagChunk(
                    id=f"{uid}_office_ogim",
                    kind="step",
                    country_code=cc,
                    university_id=uid,
                    topic="university",
                    title=f"{uni.international_office_name} и ОГиМ ({uni.short_name})",
                    content=(
                        f"Международный отдел ({uni.short_name}, г. {uni.city}): {uni.international_office_address}. "
                        f"Телефон: {uni.international_office_phone}, email: {uni.international_office_email}.\n"
                        f"Закреплённый ОГиМ: {uni.ogim_name}, адрес: {uni.ogim_address}. "
                        f"Режим работы ОГиМ: {uni.ogim_hours}."
                    ),
                    keywords=f"{uni.short_name} {uni.city} международный отдел телефон адрес кабинет огим график часы работы",
                    source_title=uni.short_name,
                    source_url=uni_url,
                    verified_at="2026-11-24",
                )
            )

            for card in uni.cards:
                chunks.append(
                    RagChunk(
                        id=card.id,
                        kind="card",
                        country_code=cc,
                        university_id=uid,
                        topic=card.topic,
                        title=card.title,
                        content=card.content,
                        keywords=" ".join(card.tags) + f" {uni.city} {uni.short_name}",
                        source_title=card.source_title,
                        source_url=card.source_url,
                        verified_at=card.verified_at,
                    )
                )

            for step in uni.onboarding_steps:
                chunks.append(
                    RagChunk(
                        id=f"{uid}_{step.id}",
                        kind="step",
                        country_code=cc,
                        university_id=uid,
                        topic="university",
                        title=f"{step.title} ({uni.short_name})",
                        content=f"{step.description} (Рекомендуемый срок: первые {step.deadline_days} дн. после приезда. Адрес: {step.location}).",
                        keywords=f"{uni.short_name} {uni.city} заселение медосмотр поликлиника общежитие шаг",
                        source_title=step.source_title or uni.short_name,
                        source_url=step.source_url or uni_url,
                        verified_at="2026-11-24",
                    )
                )

            for erip in uni.erip_guides:
                path_str = " → ".join(erip.steps)
                chunks.append(
                    RagChunk(
                        id=erip.id,
                        kind="erip",
                        country_code=cc,
                        university_id=uid,
                        topic="finance",
                        title=f"{erip.title} ({uni.city})",
                        content=f"Дерево ЕРИП: {path_str}.\nЧто ввести: {erip.input_hint}\n{erip.fee_note}".strip(),
                        keywords=f"ерип оплата реквизиты госпошлина общежитие обучение {uni.short_name} {uni.city}",
                        source_title=erip.source_title or uni.short_name,
                        source_url="https://raschet.by",
                        verified_at=erip.verified_at or "2026-11-24",
                    )
                )

            # Индексируем каждую проверенную точку города/вуза (адреса, телефоны, часы работы)
            cat_topic_map = {
                "ogim": "migration",
                "university": "university",
                "dorm": "university",
                "pharmacy": "medicine",
                "bank": "finance",
                "telecom": "finance",
                "halal": "culture",
                "services": "general",
            }
            for pt in uni.map_points:
                details = [f"Адрес: г. {uni.city}, {pt.address}."]
                if pt.hours:
                    details.append(f"Часы работы: {pt.hours}.")
                if pt.phone:
                    details.append(f"Телефон: {pt.phone}.")
                if pt.note:
                    details.append(pt.note)
                if pt.halal:
                    details.append("Доступна продукция / блюда Халяль.")
                if pt.accessible:
                    details.append("Объект оборудован элементами безбарьерной среды.")
                chunks.append(
                    RagChunk(
                        id=f"place_{pt.id}",
                        kind="place",
                        country_code=cc,
                        university_id=uid,
                        topic=cat_topic_map.get(pt.category, "general"),
                        title=f"{pt.name} (г. {uni.city})",
                        content=" ".join(details),
                        keywords=f"{pt.category} {uni.city} {uni.short_name} адрес телефон график где находится",
                        source_title=f"Верифицированная карта «Атлас» · г. {uni.city}",
                        source_url=uni_url,
                        verified_at=pt.verified_at or "2026-11-24",
                    )
                )

        # Строим BM25 индекс (TF, DF, IDF) + символьные 3-граммы
        doc_tf: list[Counter[str]] = []
        doc_len: list[int] = []
        df: Counter[str] = Counter()
        trigrams: list[set[str]] = []

        for ch in chunks:
            # Заголовок и ключевые слова получают тройной вес в BM25
            weighted_text = f"{ch.title} {ch.title} {ch.keywords} {ch.keywords} {ch.content}"
            toks = _tokenize_list(weighted_text)
            tf = Counter(toks)
            doc_tf.append(tf)
            doc_len.append(len(toks))
            for term in tf:
                df[term] += 1
            trigrams.append(_char_trigrams(f"{ch.title} {ch.keywords}"))

        n_docs = max(len(chunks), 1)
        avgdl = sum(doc_len) / n_docs if doc_len else 1.0
        idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

        self.chunks = chunks
        self._doc_tf = doc_tf
        self._doc_len = doc_len
        self._avgdl = max(avgdl, 1.0)
        self._idf = idf
        self._trigrams = trigrams
        self._signature = self._current_signature()
        logger.info("Гибридный RAG-индекс построен: %d верифицированных фрагментов", len(chunks))

    def search(
        self,
        query: str,
        destination: str | None = None,
        university_id: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[float, RagChunk]]:
        self.ensure_built()
        dest_code = destination.upper() if destination else None

        orig_tokens = _tokenize_list(query)
        orig_set = set(orig_tokens)
        bridge_tokens = [t for t in _expand_query_terms(query) if t not in orig_set]
        q_tris = _char_trigrams(query)

        k1 = 1.5
        b = 0.75
        scored: list[tuple[float, RagChunk]] = []

        for idx, ch in enumerate(self.chunks):
            # Строгая фильтрация по стране и выбранному вузу/городу (чтобы не путать Витебск и Минск)
            if dest_code and ch.country_code != dest_code:
                continue
            if university_id and ch.university_id and ch.university_id != university_id:
                continue

            tf = self._doc_tf[idx]
            dl = self._doc_len[idx]
            bm25 = 0.0
            # Основные слова запроса — полный вес 1.0
            for term in orig_tokens:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf.get(term, 0.5)
                bm25 += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / self._avgdl))

            # Мультиязычные синонимы-мосты — мягкий вес 0.22 (не перебивают точные слова запроса)
            for term in bridge_tokens:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf.get(term, 0.5)
                bm25 += 0.22 * idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / self._avgdl))

            # Нечёткое совпадение по символьным триграммам (устойчивость к опечаткам и падежам)
            tri_overlap = 0.0
            if q_tris and self._trigrams[idx]:
                inter = len(q_tris & self._trigrams[idx])
                if inter >= 2:
                    tri_overlap = inter / max(len(q_tris), 1)

            total_score = bm25 + (2.2 * tri_overlap if bm25 > 0 or tri_overlap >= 0.35 else 0.0)

            if total_score >= 0.65:
                scored.append((total_score, ch))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]


rag_index = HybridRagIndex()


def _llm_client() -> AsyncOpenAI | None:
    if not settings.llm_api_key:
        return None
    return AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=25.0,
        max_retries=1,
    )


def _parse_llm_output(raw: str, valid_ids: set[str]) -> tuple[str, list[str], bool]:
    """
    Надёжный парсер ответов любых моделей (Groq, Gemini, OpenRouter Free, DeepSeek-R1, Qwen, Ollama):
    очищает <think>...</think>, markdown-блоки ```json и умеет работать даже если бесплатная модель
    вернула чистый текст вместо JSON.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S | re.I)
    json_candidate = fenced.group(1) if fenced else None
    if not json_candidate:
        brace_match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if brace_match:
            json_candidate = brace_match.group(0)

    if json_candidate:
        try:
            parsed = json.loads(json_candidate)
            ans = str(parsed.get("answer", "")).strip()
            used = [cid for cid in parsed.get("card_ids", []) if cid in valid_ids]
            conf = bool(parsed.get("confident", True))
            if ans:
                return ans, used, conf
        except Exception:
            pass

    # Если бесплатная модель ответила обычным текстом без обёртки JSON — извлекаем упомянутые ID
    mentioned = [cid for cid in valid_ids if cid in cleaned]
    plain = re.sub(r"\[[a-zA-Z0-9_]+\]", "", cleaned).strip()
    if plain and len(plain) > 12:
        low = plain.lower()
        refusal = any(p in low for p in ("нет данных в карточках", "нет информации в базе", "confident=false", "no information in the provided"))
        return plain, mentioned, not refusal
    return "", [], False


def _synthesize_local_rag_answer(matched: list[tuple[float, RagChunk]], lang: str = "ru") -> tuple[str, list[RagChunk]]:
    """
    Структурированный локальный синтез по найденным RAG-фрагментам (когда внешний LLM не подключён
    или недоступен): объединяет правило/карточку с точными адресами, ЕРИП или телефонами.
    """
    primary = matched[0][1]
    used: list[RagChunk] = [primary]
    parts: list[str] = [f"{primary.title}\n{primary.content}"]

    # Если второй фрагмент дополняет ответ конкретным адресом, ЕРИП или шагом вуза — аккуратно добавляем
    for score, cand in matched[1:3]:
        if score >= matched[0][0] * 0.52 and cand.id != primary.id:
            if cand.kind in ("place", "erip", "step", "sos") or cand.topic == primary.topic:
                parts.append(f"Дополнительно — {cand.title}:\n{cand.content}")
                used.append(cand)
                if len(used) >= 3:
                    break

    return "\n\n".join(parts), used


_LANG_NAMES = {
    "ru": "русский (Russian)",
    "by": "белорусский (Беларуская мова)",
    "en": "английский (English)",
    "uz": "узбекский латиницей (O‘zbek tili)",
    "zh": "китайский упрощённый (简体中文)",
}


async def ask_aist(
    question: str,
    destination: str | None = None,
    university_id: str | None = None,
    lang: str = "ru",
    history: list[dict[str, str]] | None = None,
) -> AiAnswer:
    """
    Гибридный Strict-RAG ассистент AIst:
    1. Обезличивает вопрос и историю диалога через локальный Privacy Firewall (Закон РБ № 99-З).
    2. Ищет топ-5 фрагментов через гибридный индекс Okapi BM25 + символьные триграммы
       по всем 6 слоям базы (НПА, правила гражданств, шаги вуза, адреса/часы карты, ЕРИП, посольства).
    3. Если подключён LLM (Groq / Gemini / OpenRouter / Ollama) — генерирует точный ответ на выбранном
       языке СТРОГО по найденным фрагментам, а верифицированные ссылки подставляет бэкенд.
    4. Если уверенности нет — предлагает в 1 клик передать вопрос в международный отдел вуза.
    """
    clean_q, pii_hits = scrub_pii(question)

    # Учитываем контекст предыдущего сообщения при поиске (для уточняющих вопросов «А какой телефон?»)
    search_query = clean_q
    clean_history: list[dict[str, str]] = []
    if history:
        for turn in history[-4:]:
            role = turn.get("role", "")
            content = str(turn.get("content", "")).strip()
            if role in ("user", "assistant") and content:
                scrubbed_turn, _ = scrub_pii(content[:600])
                clean_history.append({"role": role, "content": scrubbed_turn})
        if len(clean_q.split()) <= 6:
            prev_users = [t["content"] for t in clean_history if t["role"] == "user"]
            if prev_users:
                search_query = f"{prev_users[-1]} {clean_q}"

    matched = rag_index.search(
        search_query,
        destination=destination,
        university_id=university_id,
        top_k=5,
    )

    if not matched:
        return AiAnswer(
            answer=(
                "В проверенной базе «Атласа» пока нет точных сведений по этому вопросу. "
                "Чтобы исключить ошибки в сроках и документах, ассистент не додумывает факты — "
                "передайте вопрос специалисту международного отдела кнопкой ниже."
            ),
            sources=[],
            confident=False,
            escalate_to_university=True,
            pii_redacted_count=pii_hits,
            scrubbed_query=clean_q,
            retrieved_chunks=0,
            topic="unanswered",
            mode="verified_kb",
        )

    chunks_by_id = {ch.id: ch for _, ch in matched}
    client = _llm_client()

    # Надёжный локальный синтез, если API-ключ ещё не указан
    if client is None:
        synth_text, used_chunks = _synthesize_local_rag_answer(matched, lang=lang)
        seen_urls: set[tuple[str, str]] = set()
        sources: list[VerifiedSource] = []
        for ch in used_chunks:
            key = (ch.source_title, ch.source_url)
            if key not in seen_urls:
                seen_urls.add(key)
                sources.append(
                    VerifiedSource(
                        card_id=ch.id,
                        title=ch.title,
                        source_title=ch.source_title,
                        source_url=ch.source_url,
                        verified_at=ch.verified_at,
                    )
                )
        return AiAnswer(
            answer=synth_text,
            sources=sources,
            confident=True,
            escalate_to_university=False,
            pii_redacted_count=pii_hits,
            scrubbed_query=clean_q,
            retrieved_chunks=len(matched),
            topic=used_chunks[0].topic,
            mode="verified_kb",
        )

    context_blocks = [
        f"[{ch.id}] ({ch.kind}) {ch.title}\n{ch.content}"
        for _, ch in matched
    ]
    target_lang_desc = _LANG_NAMES.get(lang, lang)
    system_prompt = (
        "Ты — AIst, интеллектуальный ассистент платформы «Атлас» для иностранных студентов и гостей Беларуси.\n"
        "ПРАВИЛА STRICT RAG:\n"
        "1. Отвечай ЧЁТКО, структурированно (2–6 предложений или короткий список) ИСКЛЮЧИТЕЛЬНО на основе переданных фрагментов базы.\n"
        "2. Обязательно указывай точные сроки, статьи законов, адреса, кабинеты, телефоны и шаги ЕРИП из фрагментов.\n"
        "3. СТРОГО ЗАПРЕЩЕНО выдумывать факты, цены, законы или адреса, которых нет во фрагментах. Если ответа во фрагментах нет — верни confident=false.\n"
        f"4. Язык ответа: {target_lang_desc} (переведи смысл точно на этот язык, сохраняя названия улиц и порталов понятно).\n"
        "5. Верни ТОЛЬКО валидный JSON без markdown-обёртки:\n"
        '{"answer": "текст ответа", "card_ids": ["id1", "id2"], "confident": true}'
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(clean_history[-4:])
    messages.append(
        {
            "role": "user",
            "content": (
                "Верифицированные фрагменты из RAG-индекса «Атлас»:\n\n"
                + "\n\n".join(context_blocks)
                + f"\n\nВопрос пользователя (обезличен): {clean_q}"
            ),
        }
    )

    try:
        try:
            resp = await client.chat.completions.create(
                model=settings.llm_model,
                temperature=0.1,
                max_tokens=1400,
                messages=messages,
            )
        except Exception as first_err:
            if "404" in str(first_err) or "model_not_found" in str(first_err):
                alt_model = await _discover_active_chat_model(client)
                if alt_model:
                    persist_llm_config(settings.llm_base_url, None, alt_model, settings.llm_vision_model)
                    resp = await client.chat.completions.create(
                        model=settings.llm_model,
                        temperature=0.1,
                        max_tokens=1400,
                        messages=messages,
                    )
                else:
                    raise
            else:
                raise
        raw_content = resp.choices[0].message.content or ""
        answer_text, used_ids, confident = _parse_llm_output(raw_content, set(chunks_by_id.keys()))
        if confident and answer_text and not used_ids:
            # Если бесплатная модель дала корректный ответ по первому фрагменту, но не перечислила ID
            used_ids = [matched[0][1].id]
    except Exception as exc:
        logger.warning("Ошибка вызова LLM (%s), переключение на локальный синтез RAG", exc)
        synth_text, used_chunks = _synthesize_local_rag_answer(matched, lang=lang)
        return AiAnswer(
            answer=synth_text,
            sources=[
                VerifiedSource(
                    card_id=used_chunks[0].id,
                    title=used_chunks[0].title,
                    source_title=used_chunks[0].source_title,
                    source_url=used_chunks[0].source_url,
                    verified_at=used_chunks[0].verified_at,
                )
            ],
            confident=True,
            escalate_to_university=False,
            pii_redacted_count=pii_hits,
            scrubbed_query=clean_q,
            retrieved_chunks=len(matched),
            topic=used_chunks[0].topic,
            mode="verified_kb",
        )

    if not confident or not answer_text:
        return AiAnswer(
            answer=(
                "По этой детали нет подтверждённых данных в верифицированной базе, поэтому я не стану угадывать. "
                "Нажмите кнопку ниже, чтобы передать вопрос специалисту международного отдела вуза."
            ),
            sources=[],
            confident=False,
            escalate_to_university=True,
            pii_redacted_count=pii_hits,
            scrubbed_query=clean_q,
            retrieved_chunks=len(matched),
            topic="unanswered",
            mode="sovereign_llm" if settings.is_sovereign_llm else "cloud_llm_scrubbed",
        )

    seen_src: set[tuple[str, str]] = set()
    verified_sources: list[VerifiedSource] = []
    for cid in used_ids:
        ch = chunks_by_id[cid]
        key = (ch.source_title, ch.source_url)
        if key not in seen_src:
            seen_src.add(key)
            verified_sources.append(
                VerifiedSource(
                    card_id=ch.id,
                    title=ch.title,
                    source_title=ch.source_title,
                    source_url=ch.source_url,
                    verified_at=ch.verified_at,
                )
            )

    topic = chunks_by_id[used_ids[0]].topic if used_ids else matched[0][1].topic

    return AiAnswer(
        answer=answer_text,
        sources=verified_sources,
        confident=True,
        escalate_to_university=False,
        pii_redacted_count=pii_hits,
        scrubbed_query=clean_q,
        retrieved_chunks=len(matched),
        topic=topic,
        mode="sovereign_llm" if settings.is_sovereign_llm else "cloud_llm_scrubbed",
    )


async def translate_photo_bytes(image_bytes: bytes, target_lang: str = "ru") -> str:
    """
    Перевод вывесок, указателей и объявлений строго в оперативной памяти (0 байт на диске).
    Поддерживает бесплатные мультимодальные модели (Groq Llama 4 Scout, Google Gemini Flash, OpenRouter, Ollama).
    """
    client = _llm_client()
    if client is None:
        return (
            "Модуль распознавания вывесок работает в режиме RAM-only, но сейчас не подключён API-ключ ИИ. "
            "Откройте кабинет /admin (или файл .env) и укажите бесплатный ключ Groq / Google AI Studio."
        )

    b64 = base64.b64encode(image_bytes).decode("ascii")
    target_lang_desc = _LANG_NAMES.get(target_lang, target_lang)
    prompt = (
        f"На фото городская вывеска, дорожный указатель, расписание или объявление. "
        f"Переведи текст на {target_lang_desc} и в 1–2 коротких предложениях поясни практический смысл для иностранного студента или туриста. "
        "Если на фото личный документ (паспорт, банковская карта) — НЕ переписывай номера и вежливо напомни не фотографировать личные документы."
    )
    vision_model = settings.llm_vision_model or settings.llm_model
    try:
        resp = await client.chat.completions.create(
            model=vision_model,
            temperature=0.1,
            max_tokens=900,
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
        raw = resp.choices[0].message.content or "Не удалось распознать текст на изображении."
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I).strip()
    except Exception as exc:
        logger.warning("Ошибка Vision API (%s): %s", vision_model, exc)
        return (
            f"Текущая Vision-модель ({vision_model}) не смогла обработать изображение. "
            "Для бесплатного перевода фото укажите в /admin модель qwen/qwen3.8-27b (Groq) или gemini-2.5-flash (Gemini)."
        )


async def _discover_active_chat_model(client: AsyncOpenAI) -> str | None:
    """Если указанная модель устарела (404), запрашивает список доступных моделей у провайдера и выбирает лучшую."""
    try:
        models_page = await client.models.list()
        ids = [m.id for m in models_page.data if getattr(m, "id", "")]
        skip_words = ("whisper", "tts", "guard", "safeguard", "embed", "moderation", "orpheus")
        chat_ids = [m for m in ids if not any(sw in m.lower() for sw in skip_words)]
        for preferred in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "gemini-2.5-flash", "openrouter/free"):
            if preferred in chat_ids:
                return preferred
        for kw in ("gpt-oss", "qwen3", "gemini", "llama", "deepseek"):
            for m in chat_ids:
                if kw in m.lower():
                    return m
        return chat_ids[0] if chat_ids else None
    except Exception:
        return None


async def test_llm_connection() -> tuple[bool, str]:
    """Проверяет работоспособность настроенного API-ключа и при 404 автоматически подбирает актуальную модель."""
    client = _llm_client()
    if client is None:
        return True, "Внешний API-ключ не задан — активен автономный гибридный BM25 RAG по локальной базе"
    try:
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.0,
            max_tokens=256,
            messages=[{"role": "user", "content": "Ответь одним словом: ОК"}],
        )
        reply = re.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=re.S).strip()
        return True, f"Соединение успешно ({settings.llm_model}): {reply[:40] or 'ОК'}"
    except Exception as exc:
        if "404" in str(exc) or "model_not_found" in str(exc):
            alt = await _discover_active_chat_model(client)
            if alt:
                persist_llm_config(settings.llm_base_url, None, alt, settings.llm_vision_model)
                try:
                    resp2 = await client.chat.completions.create(
                        model=alt,
                        temperature=0.0,
                        max_tokens=256,
                        messages=[{"role": "user", "content": "Ответь одним словом: ОК"}],
                    )
                    reply2 = re.sub(r"<think>.*?</think>", "", resp2.choices[0].message.content or "", flags=re.S).strip()
                    return True, f"Автоматически выбрана актуальная модель {alt} — соединение успешно ({reply2[:30] or 'ОК'})"
                except Exception as exc2:
                    return False, f"Ошибка подключения ({alt}): {exc2}"
        return False, f"Ошибка подключения к {settings.llm_base_url} ({settings.llm_model}): {exc}"
