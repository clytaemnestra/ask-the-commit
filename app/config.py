"""Typed application settings, loaded from the environment and ``.env``.

One settings object is the single source of truth for every tunable in the
system; nothing else reads ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LlmProvider = Literal["groq", "ollama", "openai", "echo"]
EmbeddingProvider = Literal["onnx", "local", "jina", "openai", "google"]
VectorStoreKind = Literal["numpy", "chroma"]


class Settings(BaseSettings):
    """All configuration for ingestion, retrieval, generation and the API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Paths -------------------------------------------------------------
    episodes_dir: Path = Field(Path("episodes"), description="Folder holding podcast audio files.")
    data_dir: Path = Field(Path("data"), description="Root for all generated state.")
    collection_name: str = "podcast_chunks"

    # --- Transcription (faster-whisper) ------------------------------------
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    whisper_language: str | None = Field(None, description="Force a language, e.g. 'en'. None = autodetect.")
    whisper_vad_filter: bool = True
    whisper_initial_prompt: str | None = Field(
        None,
        description="Comma-separated domain vocabulary that biases transcription, e.g. "
        "'Qubes OS, Tor, CPython, PyPI'. Homophones otherwise resolve to the commoner "
        "general-English spelling. Changing this invalidates cached transcripts.",
    )

    # --- Embeddings --------------------------------------------------------
    # "onnx" runs the committed MiniLM export through ONNX Runtime: local, no
    # API key, no torch, ~90MB resident. This is the default and what the
    # deployed service uses.
    # "local" runs sentence-transformers instead (identical vectors, but ~830MB
    # resident) and is for ingest/development only.
    # The hosted providers exist for when you want a stronger embedding model
    # than fits in a free tier.
    embedding_provider: EmbeddingProvider = "onnx"
    onnx_model_dir: Path = Field(
        Path("models/minilm-onnx"), description="Committed ONNX model + tokenizer."
    )
    onnx_threads: int = 1
    embedding_model: str | None = Field(None, description="Overrides the provider's default model.")
    embedding_api_key: str | None = None
    embedding_base_url: str | None = Field(None, description="Overrides the provider's default base URL.")
    embedding_dimension: int | None = Field(None, description="Expected vector width; checked per response.")
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    embedding_timeout_s: float = 30.0
    embedding_max_retries: int = 3

    # --- Vector store ------------------------------------------------------
    vector_store: VectorStoreKind = "numpy"

    # --- Chunking ----------------------------------------------------------
    chunk_tokens: int = Field(500, ge=32, description="Target chunk size, in embedding-model tokens.")
    chunk_overlap_tokens: int = Field(50, ge=0, description="Tokens of overlap carried between chunks.")

    # --- Retrieval ---------------------------------------------------------
    top_k: int = Field(5, ge=1, le=50)
    min_score: float = Field(
        0.20,
        ge=-1.0,
        le=1.0,
        description="Cosine similarity floor. If the best hit scores below this, "
        "the pipeline refuses without spending a generation call. Set to -1 to disable.",
    )

    # --- Generation --------------------------------------------------------
    llm_provider: LlmProvider = "groq"
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(700, ge=64)
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434/v1"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # --- Answer cache ------------------------------------------------------
    # Repeat questions are common on a demo link. In-memory only: the host has
    # no persistent disk and idles the process out anyway.
    answer_cache_size: int = Field(256, ge=0, description="Entries to keep. 0 disables caching.")
    answer_cache_ttl_s: float | None = Field(
        None, description="Optional entry lifetime. None = keep until evicted."
    )

    # --- Observability -----------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # --- API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    # Render and most PaaS hosts inject the port as $PORT. The start command
    # passes it to uvicorn directly; this alias keeps `python main.py` in step.
    api_port: int = Field(8000, validation_alias=AliasChoices("API_PORT", "PORT"))

    @property
    def index_dir(self) -> Path:
        """Committed NumPy index: embeddings.npy, chunks.json, manifest.json."""
        return self.data_dir / "index"

    @property
    def chroma_dir(self) -> Path:
        """Where the persistent Chroma database lives."""
        return self.data_dir / "chroma"

    @property
    def transcripts_dir(self) -> Path:
        """Cache of transcribed episodes, so re-chunking never re-transcribes."""
        return self.data_dir / "transcripts"

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        """Reject an overlap that would prevent the chunker from advancing."""
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_tokens")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
