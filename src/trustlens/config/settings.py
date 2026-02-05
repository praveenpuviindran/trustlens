from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized config.

    Key idea:
    - read from env first (so tests/CI can override),
    - otherwise default to local dev values.
    """

    model_config = SettingsConfigDict(env_prefix="TRUSTLENS_", extra="ignore")

    # DuckDB file by default (portable, zero-setup)
    db_url: str = "duckdb:///data/trustlens.duckdb"

    # GDELT DOC 2.1 API base URL
    gdelt_doc_base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"

    # safety defaults
    gdelt_timeout_s: float = 20.0
    gdelt_max_records_default: int = 50

    # Deployment safety defaults
    evidence_timeout_s: float = 10.0
    max_records_cap: int = 50
    rate_limit_per_min: int = 30

    # LLM explanation settings (default stub for deterministic tests)
    llm_provider: str = "stub"
    llm_model_name: str | None = None
    openai_api_key: str | None = None


settings = Settings()
