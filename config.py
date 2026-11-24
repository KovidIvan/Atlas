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

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_vision_model: str = "gpt-4o-mini"

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
