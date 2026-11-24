from __future__ import annotations

from datetime import datetime, timezone
import aiosqlite

from config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    tg_id INTEGER PRIMARY KEY,
    consent_given INTEGER NOT NULL DEFAULT 0,
    lang TEXT NOT NULL DEFAULT 'ru',
    citizenship TEXT NOT NULL DEFAULT 'UZ',
    destination TEXT NOT NULL DEFAULT 'BY',
    university_id TEXT DEFAULT 'vsu_vitebsk',
    purpose TEXT NOT NULL DEFAULT 'study',
    housing_type TEXT NOT NULL DEFAULT 'dorm',
    entry_via_rf INTEGER NOT NULL DEFAULT 0,
    arrival_date TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checklist_progress (
    tg_id INTEGER NOT NULL,
    step_id TEXT NOT NULL,
    completed INTEGER NOT NULL DEFAULT 0,
    reminded_at TEXT,
    PRIMARY KEY (tg_id, step_id)
);

CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tg_id INTEGER NOT NULL,
    university_id TEXT NOT NULL,
    question_scrubbed TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    reply_text TEXT,
    created_at TEXT NOT NULL,
    replied_at TEXT
);

CREATE TABLE IF NOT EXISTS analytics_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    university_id TEXT,
    country_code TEXT,
    topic TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nla_snapshots (
    url TEXT PRIMARY KEY,
    country_code TEXT NOT NULL,
    name TEXT NOT NULL,
    content_hash TEXT,
    http_status INTEGER,
    changed INTEGER NOT NULL DEFAULT 0,
    last_checked TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def init_db() -> None:
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.executescript(SCHEMA_SQL)
        await conn.commit()


async def get_user(tg_id: int) -> dict | None:
    async with aiosqlite.connect(settings.db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute("SELECT * FROM users WHERE tg_id = ?", (tg_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def upsert_user(tg_id: int, **fields) -> dict:
    existing = await get_user(tg_id)
    now = _now_iso()
    data = {
        "consent_given": 0,
        "lang": "ru",
        "citizenship": "UZ",
        "destination": "BY",
        "university_id": "vsu_vitebsk",
        "purpose": "study",
        "housing_type": "dorm",
        "entry_via_rf": 0,
        "arrival_date": None,
    }
    if existing:
        for k in data:
            data[k] = existing[k]
    for k, v in fields.items():
        if k in data:
            data[k] = int(v) if isinstance(v, bool) else v

    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute(
            """
            INSERT INTO users (
                tg_id, consent_given, lang, citizenship, destination,
                university_id, purpose, housing_type, entry_via_rf, arrival_date, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tg_id) DO UPDATE SET
                consent_given = excluded.consent_given,
                lang = excluded.lang,
                citizenship = excluded.citizenship,
                destination = excluded.destination,
                university_id = excluded.university_id,
                purpose = excluded.purpose,
                housing_type = excluded.housing_type,
                entry_via_rf = excluded.entry_via_rf,
                arrival_date = excluded.arrival_date,
                updated_at = excluded.updated_at
            """,
            (
                tg_id,
                data["consent_given"],
                data["lang"],
                data["citizenship"],
                data["destination"],
                data["university_id"],
                data["purpose"],
                data["housing_type"],
                data["entry_via_rf"],
                data["arrival_date"],
                now,
            ),
        )
        await conn.commit()
    return await get_user(tg_id) or {"tg_id": tg_id, **data}


async def delete_user_data(tg_id: int) -> None:
    """Право на забвение (Закон РБ № 99-З): безвозвратное удаление всех данных пользователя."""
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("DELETE FROM checklist_progress WHERE tg_id = ?", (tg_id,))
        await conn.execute("DELETE FROM tickets WHERE tg_id = ?", (tg_id,))
        await conn.execute("DELETE FROM users WHERE tg_id = ?", (tg_id,))
        await conn.commit()


async def get_checklist_progress(tg_id: int) -> dict[str, bool]:
    async with aiosqlite.connect(settings.db_path) as conn:
        cur = await conn.execute(
            "SELECT step_id, completed FROM checklist_progress WHERE tg_id = ?", (tg_id,)
        )
        rows = await cur.fetchall()
        return {r[0]: bool(r[1]) for r in rows}


async def toggle_step(tg_id: int, step_id: str, completed: bool) -> None:
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute(
            """
            INSERT INTO checklist_progress (tg_id, step_id, completed)
            VALUES (?, ?, ?)
            ON CONFLICT(tg_id, step_id) DO UPDATE SET completed = excluded.completed
            """,
            (tg_id, step_id, int(completed)),
        )
        await conn.commit()


async def mark_step_reminded(tg_id: int, step_id: str) -> None:
    now = _now_iso()
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute(
            """
            INSERT INTO checklist_progress (tg_id, step_id, completed, reminded_at)
            VALUES (?, ?, 0, ?)
            ON CONFLICT(tg_id, step_id) DO UPDATE SET reminded_at = excluded.reminded_at
            """,
            (tg_id, step_id, now),
        )
        await conn.commit()


async def get_users_with_arrival() -> list[dict]:
    async with aiosqlite.connect(settings.db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT * FROM users WHERE consent_given = 1 AND arrival_date IS NOT NULL AND arrival_date != ''"
        )
        return [dict(r) for r in await cur.fetchall()]


async def was_reminded(tg_id: int, step_id: str) -> bool:
    async with aiosqlite.connect(settings.db_path) as conn:
        cur = await conn.execute(
            "SELECT completed, reminded_at FROM checklist_progress WHERE tg_id = ? AND step_id = ?",
            (tg_id, step_id),
        )
        row = await cur.fetchone()
        if not row:
            return False
        return bool(row[0]) or bool(row[1])


async def create_ticket(tg_id: int, university_id: str, question_scrubbed: str) -> int:
    now = _now_iso()
    async with aiosqlite.connect(settings.db_path) as conn:
        cur = await conn.execute(
            """
            INSERT INTO tickets (tg_id, university_id, question_scrubbed, status, created_at)
            VALUES (?, ?, ?, 'open', ?)
            """,
            (tg_id, university_id, question_scrubbed, now),
        )
        await conn.commit()
        return cur.lastrowid or 0


async def list_tickets(limit: int = 50) -> list[dict]:
    async with aiosqlite.connect(settings.db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT * FROM tickets ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in await cur.fetchall()]


async def list_user_tickets(tg_id: int, limit: int = 20) -> list[dict]:
    async with aiosqlite.connect(settings.db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            """
            SELECT id, university_id, question_scrubbed, status, reply_text, created_at, replied_at
            FROM tickets
            WHERE tg_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (tg_id, limit),
        )
        return [dict(r) for r in await cur.fetchall()]


async def reply_ticket(ticket_id: int, reply_text: str) -> dict | None:
    now = _now_iso()
    async with aiosqlite.connect(settings.db_path) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute(
            """
            UPDATE tickets
            SET status = 'answered', reply_text = ?, replied_at = ?
            WHERE id = ?
            """,
            (reply_text, now, ticket_id),
        )
        await conn.commit()
        cur = await conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def log_event(
    event_type: str,
    university_id: str | None = None,
    country_code: str | None = None,
    topic: str | None = None,
) -> None:
    """Полностью обезличенная аналитика (без tg_id и ПД) для панели вуза."""
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute(
            """
            INSERT INTO analytics_events (event_type, university_id, country_code, topic, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_type, university_id, country_code, topic, _now_iso()),
        )
        await conn.commit()


async def save_nla_snapshot(
    url: str, country_code: str, name: str, content_hash: str | None, http_status: int
) -> bool:
    """Сохраняет хэш страницы НПА. Возвращает True, если обнаружено изменение."""
    now = _now_iso()
    async with aiosqlite.connect(settings.db_path) as conn:
        cur = await conn.execute(
            "SELECT content_hash FROM nla_snapshots WHERE url = ?", (url,)
        )
        row = await cur.fetchone()
        changed = bool(row and row[0] and content_hash and row[0] != content_hash)
        await conn.execute(
            """
            INSERT INTO nla_snapshots (url, country_code, name, content_hash, http_status, changed, last_checked)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                content_hash = COALESCE(excluded.content_hash, nla_snapshots.content_hash),
                http_status = excluded.http_status,
                changed = excluded.changed,
                last_checked = excluded.last_checked
            """,
            (url, country_code, name, content_hash, http_status, int(changed), now),
        )
        await conn.commit()
        return changed


async def get_admin_stats() -> dict:
    async with aiosqlite.connect(settings.db_path) as conn:
        conn.row_factory = aiosqlite.Row
        users_count = (await (await conn.execute("SELECT COUNT(*) FROM users")).fetchone())[0]
        open_tickets = (
            await (await conn.execute("SELECT COUNT(*) FROM tickets WHERE status = 'open'")).fetchone()
        )[0]
        total_questions = (
            await (
                await conn.execute(
                    "SELECT COUNT(*) FROM analytics_events WHERE event_type = 'ask'"
                )
            ).fetchone()
        )[0]

        cur = await conn.execute(
            """
            SELECT COALESCE(topic, 'general') AS topic, COUNT(*) AS cnt
            FROM analytics_events
            WHERE event_type = 'ask'
            GROUP BY topic
            ORDER BY cnt DESC
            LIMIT 8
            """
        )
        top_topics = [dict(r) for r in await cur.fetchall()]

        cur = await conn.execute("SELECT * FROM nla_snapshots ORDER BY country_code, name")
        nla_snapshots = [dict(r) for r in await cur.fetchall()]

    return {
        "users_count": users_count,
        "open_tickets": open_tickets,
        "total_questions": total_questions,
        "top_topics": top_topics,
        "nla_snapshots": nla_snapshots,
    }
