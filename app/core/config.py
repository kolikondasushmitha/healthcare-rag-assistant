from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    database_url: str = ""
    redis_url: str = "redis://localhost:6379"
    secret_key: str = "mysupersecretkey123"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    chroma_persist_dir: str = "./chroma_store"

    class Config:
        env_file = ".env"

settings = Settings()