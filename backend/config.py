from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str
    tavily_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_iterations: int = 10
    confidence_threshold: float = 0.7

    class Config:
        env_file = ".env"


settings = Settings()