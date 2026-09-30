import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    neo4j_uri: str = Field(default="neo4j+s://8d2d5bc7.databases.neo4j.io", validation_alias="NEO4J_URI")
    neo4j_user: str = Field(default="8d2d5bc7", validation_alias="NEO4J_USER")
    neo4j_password: str = Field(default="bw8o9R3NXXApN2DXy72Gv7iXNI2KcVDGp_zZ7h0D5BU", validation_alias="NEO4J_PASSWORD")
    ollama_base_url: str = Field(default="https://corporate-economic-automatic.ngrok-free.dev", validation_alias="OLLAMA_BASE_URL")
    llm_model: str = Field(default="qwen2.5:7b-instruct", validation_alias="LLM_MODEL")
    embedding_model: str = Field(default="all-minilm", validation_alias="EMBEDDING_MODEL")
    similarity_threshold: float = Field(default=0.65, validation_alias="SIMILARITY_THRESHOLD")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

def setup_logger(name: str = "m4") -> logging.Logger:
    config = AppConfig()
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(config.log_level.upper())
    return logger
