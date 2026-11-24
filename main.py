from __future__ import annotations

import asyncio
import logging
import sys

import uvicorn

from bot import create_bot_and_dispatcher
from config import settings
import db
from loader import registry
import monitor
from web import app, set_active_bot

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(line_buffering=True)
        except Exception:
            pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("atlas.main")


async def run_all() -> None:
    await db.init_db()
    registry.reload()

    bot, dp = create_bot_and_dispatcher()
    set_active_bot(bot)

    uvicorn_config = uvicorn.Config(
        app=app,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
    server = uvicorn.Server(uvicorn_config)

    tasks: list[asyncio.Task] = [
        asyncio.create_task(server.serve(), name="fastapi"),
        asyncio.create_task(monitor.background_loop(bot), name="monitor"),
    ]
    if bot is not None:
        tasks.append(
            asyncio.create_task(
                dp.start_polling(bot, handle_signals=False),
                name="telegram_bot",
            )
        )

    logger.info("Атлас запущен: Mini App -> http://%s:%s/app | Админка вуза -> http://%s:%s/admin",
                settings.host, settings.port, settings.host, settings.port)

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        if bot is not None:
            await bot.session.close()


if __name__ == "__main__":
    asyncio.run(run_all())
