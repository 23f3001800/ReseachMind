import os
from typing import List, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str
    tavily_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_iterations: int = 10
    confidence_threshold: float = 0.7
    search_provider: Literal["auto", "tavily", "duckduckgo"] = "auto"
    # Storage root. Everything that must outlive a container lives here:
    # conversation history, the vector index, the document catalog. Point it at
    # a mounted volume in production or it is lost on every restart.
    data_dir: str = "data"
    db_path: str = ""  # defaults to {data_dir}/memory.db — see resolved_db_path

    @property
    def resolved_db_path(self) -> str:
        return self.db_path or os.path.join(self.data_dir, "memory.db")
    langsmith_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "researchmind"

    # Pipeline tuning
    request_timeout: int = 180        # seconds; must be <= the platform's request timeout
    max_tool_rounds: int = 5          # researcher ReAct loop cap
    max_research_retries: int = 1     # self-reflection passes back to the researcher

    # API protection. Empty api_key = open access (fine locally, not in public deploys).
    api_key: str = ""
    rate_limit_per_minute: int = 20
    max_upload_mb: int = 10

    # Deliberately a plain string, not List[str]. pydantic-settings JSON-decodes
    # complex field types straight out of the environment source — before any
    # field_validator runs — so a List[str] here makes ALLOWED_ORIGINS=https://x
    # raise SettingsError and kill the process at import. Parse it ourselves.
    allowed_origins: str = "*"

    @property
    def allowed_origins_list(self) -> List[str]:
        """CORS origins, from a comma-separated env var."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()] or ["*"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
