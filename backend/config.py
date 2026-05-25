from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    whisper_model: str = "large-v3"

    speech_model: str = "qwen-audio-turbo-latest"
    speech_api_key: str = ""
    speech_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    speech_enabled: bool = True

    vision_enabled: bool = False
    vision_model: str = "qwen-vl-max"
    vision_api_key: str = ""
    vision_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "knowledge"
    postgres_password: str = "knowledge123"
    postgres_db: str = "knowledge_db"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "knowledge123"

    redis_url: str = "redis://localhost:6379/0"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "knowledge-files"

    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 1024

    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    vector_dim: int = 1024

    secret_key: str = "dev-secret-key-change-in-production"
    access_token_expire_minutes: int = 60

    low_confidence_threshold: float = 0.4
    auto_review_interval_days: int = 7
    enable_external_search: bool = False

    dedup_file_hash_enabled: bool = True
    dedup_content_similarity_enabled: bool = True
    dedup_strict_threshold: float = 0.95
    dedup_warn_threshold: float = 0.85
    dedup_mode: str = "manual"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()