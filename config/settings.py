from pydantic_settings import BaseSettings
from pydantic import AliasChoices, Field


class Settings(BaseSettings):
    # ── AWS ──────────────────────────────────────────────────────────────────
    aws_access_key_id: str = Field(..., alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field(
        "us-east-1", validation_alias=AliasChoices("AWS_DEFAULT_REGION", "AWS_REGION")
    )

    # ── Bedrock Models ────────────────────────────────────────────────────────
    embedding_model_id: str = Field(
        "amazon.titan-embed-text-v2:0", alias="EMBEDDING_MODEL_ID"
    )
    llm_model_id: str = Field(
        "anthropic.claude-3-5-sonnet-20240620-v1:0", alias="LLM_MODEL_ID"
    )
    llm_temperature: float = Field(0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(2048, alias="LLM_MAX_TOKENS")

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field("./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("legalmail_rag", alias="CHROMA_COLLECTION_NAME")

    # ── Text Splitting ────────────────────────────────────────────────────────
    chunk_size: int = Field(1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(200, alias="CHUNK_OVERLAP")

    # ── Ingestion ─────────────────────────────────────────────────────────────
    enable_curation: bool = Field(False, alias="ENABLE_CURATION")
    email_data_dir: str = Field("./data/clean", alias="EMAIL_DATA_DIR")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(15, alias="TOP_K")

    model_config = {"env_file": ".env", "populate_by_name": True, "extra": "ignore"}
