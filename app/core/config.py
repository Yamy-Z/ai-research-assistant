from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    anthropic_api_key: str
    openai_api_key: str
    tavily_api_key: str
    
    # Database
    database_url: str
    
    # Redis
    redis_url: str
    
    # Vector DB
    qdrant_url: str = "http://localhost:6333"
    
    # Application
    app_name: str = "AI Research Assistant"
    app_version: str = "0.1.0"
    debug: bool = True
    log_level: str = "INFO"
    
    # Embedding
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072
    
    # LLM
    llm_model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.7
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()  # type: ignore[call-arg]
