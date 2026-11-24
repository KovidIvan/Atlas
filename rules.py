from __future__ import annotations

from datetime import date, timedelta
from pydantic import BaseModel

from loader import CountryModule, RegistrationRule, UniversityModule, registry


class PlanStep(BaseModel):
    id: str
    title: str
    description: str
    deadline_date: str | None = None
    days_left: int | None = None
    urgency: str = "normal"  # done | urgent | soon | normal
    location: str = ""
    action_url: str = ""
    action_label: str = ""
    source_title: str = ""
    source_url: str = ""
    verified_at: str = ""
    completed: bool = False


class PersonalPlan(BaseModel):
    citizenship: str
    destination: str
    destination_name: str
    university_id: str | None
    university_name: str | None
    purpose: str
    housing_type: str
    entry_via_rf: bool
    arrival_date: str
    steps: list[PlanStep]
    disclaimer: str = (
        "Справочный план рассчитан детерминированным алгоритмом по проверенной базе правил "
        "(без генерации ИИ). Окончательное решение принимают органы по гражданству и миграции."
    )


def _find_registration_rule(country: CountryModule, citizenship: str) -> RegistrationRule | None:
    cit = citizenship.upper()
    for rule in country.registration_rules:
        if rule.citizenship.upper() == cit:
            return rule
    for rule in country.registration_rules:
        if rule.citizenship == "*":
            return rule
    return None


def _parse_arrival(raw: str | None) -> date:
    if raw:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            pass
    return date.today()


def _calc_urgency(days_left: int | None, completed: bool) -> str:
    if completed:
        return "done"
    if days_left is None:
        return "normal"
    if days_left <= 2:
        return "urgent"
    if days_left <= 5:
        return "soon"
    return "normal"


def build_personal_plan(
    citizenship: str,
    destination: str,
    university_id: str | None,
    purpose: str,
    housing_type: str,
    entry_via_rf: bool,
    arrival_date: str | None,
    completed_map: dict[str, bool] | None = None,
) -> PersonalPlan:
    """
    Детерминированный движок расчёта формальностей (0% галлюцинаций ИИ).
    Учитывает страну въезда, гражданство, способ въезда (через РФ или напрямую),
    тип жилья и локальные шаги конкретного университета.
    """
    completed_map = completed_map or {}
    dest_code = destination.upper()
    country = registry.countries.get(dest_code)
    uni: UniversityModule | None = (
        registry.universities.get(university_id) if university_id else None
    )
    if uni and uni.country_code.upper() != dest_code:
        uni = None

    arrival = _parse_arrival(arrival_date)
    today = date.today()
    steps: list[PlanStep] = []

    # Шаг 1: Въезд и базовая фиксация
    steps.append(
        PlanStep(
            id="entry_docs",
            title="Въезд и сохранение подтверждения пересечения границы",
            description=(
                "Сохраните посадочный талон / билет и проверьте штамп о въезде в паспорте. "
                "При въезде в РБ через границу с РФ штамп в паспорт не ставится — обязательно сохраните билет."
                if dest_code == "BY"
                else "Сохраните посадочный талон и проверьте штамп пограничного контроля в паспорте."
            ),
            deadline_date=arrival.isoformat(),
            days_left=(arrival - today).days,
            urgency=_calc_urgency(None, completed_map.get("entry_docs", True)),
            completed=completed_map.get("entry_docs", True),
        )
    )

    # Шаг 2: Регистрация (детерминированный расчёт срока и канала подачи!)
    if country:
        rule = _find_registration_rule(country, citizenship)
        if rule:
            deadline = arrival + timedelta(days=rule.grace_days)
            days_left = (deadline - today).days
            step_done = completed_map.get("registration", False)

            if housing_type == "hotel" and rule.hotel_auto_registers:
                reg_desc = (
                    f"В гостинице/хостеле регистрацию оформляет администрация в день заселения. "
                    f"Обязательно возьмите и сохраните отрывной талон/подтверждение регистрации до выезда. "
                    f"{rule.protocol_note}"
                ).strip()
                action_url = rule.online_portal_url
                action_label = f"О системе {rule.online_portal_name}" if rule.online_portal_name else ""
                location = "Стойка регистрации отеля / хостела"
            elif dest_code == "BY" and entry_via_rf and rule.online_blocked_if_via_rf:
                ogim_info = f"{uni.ogim_name} ({uni.ogim_address})" if uni else "местный ОГиМ по месту проживания"
                reg_desc = (
                    f"Вы указали въезд в Беларусь через территорию РФ: бесплатная онлайн-регистрация "
                    f"на {rule.online_portal_name} в этом случае недоступна. "
                    f"Необходимо лично обратиться в {ogim_info} (или через международный отдел вуза). "
                    f"{rule.protocol_note}"
                ).strip()
                action_url = rule.source_url
                action_label = "Памятка по ОГиМ"
                location = ogim_info
            elif housing_type == "dorm" and uni:
                reg_desc = (
                    f"Студентам в общежитии {uni.short_name} регистрацию оформляют централизованно через "
                    f"{uni.international_office_name}. Передайте паспорт и медстраховку специалисту вуза в первые 2 дня. "
                    f"{rule.protocol_note}"
                ).strip()
                action_url = rule.online_portal_url
                action_label = rule.online_portal_name
                location = f"{uni.international_office_name}, {uni.international_office_address}"
            else:
                reg_desc = f"{rule.summary} {rule.protocol_note}".strip()
                action_url = rule.online_portal_url
                action_label = (
                    f"Перейти в {rule.online_portal_name}" if rule.online_portal_name else ""
                )
                location = (
                    f"Онлайн ({rule.online_portal_name}) или {uni.ogim_name}"
                    if uni
                    else f"Онлайн ({rule.online_portal_name})"
                )

            if rule.grace_days >= 30:
                reg_title = f"Льготный срок без регистрации — {rule.grace_days} сут. (до {deadline.strftime('%d.%m.%Y')})"
            else:
                reg_title = f"Временная регистрация (срок до {deadline.strftime('%d.%m.%Y')})"

            steps.append(
                PlanStep(
                    id="registration",
                    title=reg_title,
                    description=reg_desc,
                    deadline_date=deadline.isoformat(),
                    days_left=days_left,
                    urgency=_calc_urgency(days_left, step_done),
                    location=location,
                    action_url=action_url,
                    action_label=action_label,
                    source_title=rule.source_title,
                    source_url=rule.source_url,
                    verified_at=rule.verified_at,
                    completed=step_done,
                )
            )

    # Шаги конкретного университета (Слой 2: модуль «Вуз и город»)
    if uni and purpose == "study":
        for ustep in uni.onboarding_steps:
            dl = arrival + timedelta(days=ustep.deadline_days)
            d_left = (dl - today).days
            done = completed_map.get(ustep.id, False)
            steps.append(
                PlanStep(
                    id=ustep.id,
                    title=ustep.title,
                    description=ustep.description,
                    deadline_date=dl.isoformat(),
                    days_left=d_left,
                    urgency=_calc_urgency(d_left, done),
                    location=ustep.location,
                    source_title=ustep.source_title or uni.short_name,
                    source_url=ustep.source_url,
                    completed=done,
                )
            )
    elif purpose == "tourism":
        ins_done = completed_map.get("tourism_insurance", False)
        steps.append(
            PlanStep(
                id="tourism_insurance",
                title="Медицинская страховка и правила безвизового режима",
                description=(
                    "Проверьте наличие действующего полиса медицинского страхования на весь срок поездки "
                    "(для РБ подходит «Белгосстрах» или «Белэксимгарант») и соблюдайте установленный коридор пребывания."
                    if dest_code == "BY"
                    else "Проверьте медицинскую страховку и сохраняйте подтверждения регистрации в отелях до выезда."
                ),
                deadline_date=(arrival + timedelta(days=1)).isoformat(),
                days_left=((arrival + timedelta(days=1)) - today).days,
                urgency=_calc_urgency(((arrival + timedelta(days=1)) - today).days, ins_done),
                location=f"Город прибытия: {uni.city}" if uni else "",
                source_title="МИД РБ / Госпрограмма «Туризм»" if dest_code == "BY" else "",
                source_url="https://mfa.gov.by" if dest_code == "BY" else "",
                completed=ins_done,
            )
        )

    # Бытовой шаг (связь + оплата)
    daily_done = completed_map.get("daily_basics", False)
    steps.append(
        PlanStep(
            id="daily_basics",
            title="Связь, банковская карта и нужные места рядом",
            description=(
                "Оформите местную SIM-карту (нужен паспорт) и проверьте работу платёжных карт. "
                "Откройте вкладку «Карта», чтобы увидеть ближайшие салоны связи, банки, аптеки и халяль-кафе."
            ),
            deadline_date=(arrival + timedelta(days=2)).isoformat(),
            days_left=((arrival + timedelta(days=2)) - today).days,
            urgency=_calc_urgency(((arrival + timedelta(days=2)) - today).days, daily_done),
            completed=daily_done,
        )
    )

    return PersonalPlan(
        citizenship=citizenship.upper(),
        destination=dest_code,
        destination_name=country.name if country else dest_code,
        university_id=uni.id if uni else None,
        university_name=uni.short_name if uni else None,
        purpose=purpose,
        housing_type=housing_type,
        entry_via_rf=entry_via_rf,
        arrival_date=arrival.isoformat(),
        steps=steps,
    )
