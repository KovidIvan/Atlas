from __future__ import annotations

import json
import logging
from pathlib import Path
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger("atlas.loader")


class KnowledgeCard(BaseModel):
    id: str
    topic: str  # migration | housing | medicine | finance | culture | university
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    source_title: str
    source_url: str
    verified_at: str  # YYYY-MM-DD


class NlaSource(BaseModel):
    name: str
    url: str


class PhraseItem(BaseModel):
    situation: str
    ru: str
    by: str = ""
    uz: str = ""
    en: str = ""


class SosContact(BaseModel):
    title: str
    phone: str
    note: str = ""


class EripGuide(BaseModel):
    id: str
    title: str
    category: str  # university | migration | other
    steps: list[str]
    input_hint: str
    fee_note: str = ""
    source_title: str = ""
    verified_at: str = ""


class ShowCard(BaseModel):
    id: str
    target_role: str  # Кому показывать: "Водителю такси", "Инспектору ОГиМ" и т.д.
    local_text: str   # Крупный текст на языке страны
    sub_note: str = ""
    native_hint: str = ""  # Пояснение/перевод для гостя


class RegistrationRule(BaseModel):
    citizenship: str  # код страны гражданства (например, "UZ", "BY", "*" для всех)
    grace_days: int
    online_portal_name: str = ""
    online_portal_url: str = ""
    online_blocked_if_via_rf: bool = False
    hotel_auto_registers: bool = True
    summary: str
    protocol_note: str = ""
    source_title: str
    source_url: str
    verified_at: str


class CountryModule(BaseModel):
    code: str  # "BY", "UZ", ...
    name: str
    flag: str = ""
    languages: list[str] = Field(default_factory=list)
    registration_rules: list[RegistrationRule] = Field(default_factory=list)
    nla_sources: list[NlaSource] = Field(default_factory=list)
    cards: list[KnowledgeCard] = Field(default_factory=list)
    phrasebook: list[PhraseItem] = Field(default_factory=list)
    sos_contacts: list[SosContact] = Field(default_factory=list)
    erip_guides: list[EripGuide] = Field(default_factory=list)
    show_cards: list[ShowCard] = Field(default_factory=list)


class MapPoint(BaseModel):
    id: str
    name: str
    category: str  # ogim | university | dorm | services | pharmacy | bank | telecom | halal
    address: str
    lat: float | None = None
    lon: float | None = None
    hours: str = ""
    phone: str = ""
    note: str = ""
    halal: bool = False
    accessible: bool = False
    verified_at: str = ""


class UniversityStep(BaseModel):
    id: str
    title: str
    description: str
    deadline_days: int  # через сколько дней после въезда
    location: str = ""
    source_title: str = ""
    source_url: str = ""


class UniversityModule(BaseModel):
    id: str
    name: str
    short_name: str
    country_code: str
    city: str
    map_center: list[float]  # [lat, lon]
    international_office_name: str
    international_office_address: str
    international_office_phone: str = ""
    international_office_email: str = ""
    admin_tg_chat_id: int | None = None
    ogim_name: str
    ogim_address: str
    ogim_hours: str = ""
    onboarding_steps: list[UniversityStep] = Field(default_factory=list)
    map_points: list[MapPoint] = Field(default_factory=list)
    cards: list[KnowledgeCard] = Field(default_factory=list)
    erip_guides: list[EripGuide] = Field(default_factory=list)
    show_cards: list[ShowCard] = Field(default_factory=list)


class ModuleRegistry:
    """
    Двухслойный реестр сменных модулей данных:
    - data/countries/*.json — правила, НПА, культура, SOS по странам
    - data/universities/*.json — инфраструктура, карта и памятки конкретных вузов
    Добавление новой страны или вуза = 1 JSON-файл без изменения кода.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or settings.data_dir
        self.countries: dict[str, CountryModule] = {}
        self.universities: dict[str, UniversityModule] = {}
        self.reload()

    def reload(self) -> dict[str, int]:
        countries: dict[str, CountryModule] = {}
        universities: dict[str, UniversityModule] = {}

        countries_dir = self.data_dir / "countries"
        if countries_dir.exists():
            for path in sorted(countries_dir.glob("*.json")):
                raw = json.loads(path.read_text(encoding="utf-8"))
                mod = CountryModule.model_validate(raw)
                countries[mod.code.upper()] = mod

        unis_dir = self.data_dir / "universities"
        if unis_dir.exists():
            uni_files = sorted(
                unis_dir.glob("*.json"),
                key=lambda p: (0 if p.stem == "vsu_vitebsk" else 1, p.name),
            )
            for path in uni_files:
                raw = json.loads(path.read_text(encoding="utf-8"))
                mod = UniversityModule.model_validate(raw)
                universities[mod.id] = mod

        self.countries = countries
        self.universities = universities
        logger.info(
            "Загружены модули: %d стран (%s), %d вузов (%s)",
            len(countries),
            ", ".join(countries.keys()),
            len(universities),
            ", ".join(universities.keys()),
        )
        return {"countries": len(countries), "universities": len(universities)}

    def get_cards(
        self,
        destination: str | None = None,
        university_id: str | None = None,
    ) -> list[KnowledgeCard]:
        cards: list[KnowledgeCard] = []
        if destination and destination.upper() in self.countries:
            cards.extend(self.countries[destination.upper()].cards)
        else:
            for c in self.countries.values():
                cards.extend(c.cards)

        if university_id and university_id in self.universities:
            cards.extend(self.universities[university_id].cards)
        elif not destination:
            for u in self.universities.values():
                cards.extend(u.cards)
        return cards

    def all_nla_sources(self) -> list[tuple[str, NlaSource]]:
        result: list[tuple[str, NlaSource]] = []
        for code, country in self.countries.items():
            for src in country.nla_sources:
                result.append((code, src))
        return result


registry = ModuleRegistry()
