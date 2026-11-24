from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str = ""
    app_base_url: str = "http://127.0.0.1:8000"
    host: str = "127.0.0.1"
    port: int = 8000
    admin_token: str = "atlas-vsu-2026"

    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    nla_check_interval_min: int = 360
    reminder_check_interval_min: int = 60
    db_file: str = ""

    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def static_dir(self) -> Path:
        return BASE_DIR / "static"

    @property
    def db_path(self) -> Path:
        if self.db_file:
            p = Path(self.db_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        return BASE_DIR / "atlas.db"

    @property
    def is_sovereign_llm(self) -> bool:
        """True, если LLM развернута локально / внутри контура РБ (localhost / 127.0.0.1 / .by)."""
        url = self.llm_base_url.lower()
        return any(host in url for host in ("127.0.0.1", "localhost", "0.0.0.0", ".by/"))


settings = Settings()


def persist_llm_config(
    base_url: str,
    api_key: str | None,
    model: str,
    vision_model: str,
) -> None:
    """Обновляет параметры LLM в памяти и сохраняет их в .env без затирания остальных настроек."""
    settings.llm_base_url = base_url.strip()
    if api_key is not None:
        settings.llm_api_key = api_key.strip()
    settings.llm_model = model.strip()
    settings.llm_vision_model = vision_model.strip() or model.strip()

    env_path = BASE_DIR / ".env"
    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    updates = {
        "LLM_BASE_URL": settings.llm_base_url,
        "LLM_API_KEY": settings.llm_api_key,
        "LLM_MODEL": settings.llm_model,
        "LLM_VISION_MODEL": settings.llm_vision_model,
    }
    seen: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}")
                seen.add(k)
                continue
        new_lines.append(line)

    for k, v in updates.items():
        if k not in seen:
            new_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
