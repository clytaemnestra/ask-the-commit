"""Typed application settings, loaded from the environment and ``.env``.

One settings object is the single source of truth for every tunable in the
system; nothing else reads ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LlmProvider = Literal["groq", "ollama", "openai", "echo"]
EmbeddingProvider = Literal["onnx", "local", "jina", "openai", "google"]
VectorStoreKind = Literal["numpy", "chroma"]

#: Settings that may legitimately be left blank in a ``.env`` file. A blank line
#: like ``ANSWER_CACHE_TTL_S=`` reaches pydantic as ``""``, which is not a float
#: and not None, so without this the documented ``cp .env.example .env`` would
#: fail validation at boot. Blank means "unset" for all of them.
_OPTIONAL_FIELDS = (
    "whisper_language",
    "whisper_initial_prompt",
    "embedding_model",
    "embedding_api_key",
    "embedding_base_url",
    "embedding_dimension",
    "groq_api_key",
    "openai_api_key",
    "answer_cache_ttl_s",
)


def secret_value(secret: SecretStr | None) -> str | None:
    """Unwrap a :class:`~pydantic.SecretStr`, preserving ``None``.

    Credentials are held as ``SecretStr`` so that a stray ``repr(settings)``, a
    pydantic validation error or a logged settings dump prints ``**********``
    instead of the key. They are unwrapped only at the point a provider client
    is constructed.
    """
    return secret.get_secret_value() if secret is not None else None


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
    embedding_api_key: SecretStr | None = None
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

    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434/v1"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # --- Answer cache ------------------------------------------------------
    # Repeat questions are common on a demo link. In-memory only: the host has
    # no persistent disk and idles the process out anyway.
    answer_cache_size: int = Field(256, ge=0, description="Entries to keep. 0 disables caching.")
    answer_cache_ttl_s: float | None = Field(
        None, description="Optional entry lifetime. None = keep until evicted."
    )

    # --- Rate limiting -----------------------------------------------------
    # /ask is the only expensive endpoint and the only one that spends quota.
    # In-memory and per-process, which is exactly right for a single free-tier
    # instance and not enough for a horizontally scaled one.
    rate_limit_requests: int = Field(
        20, ge=0, description="Requests allowed per client per window. 0 disables the limit."
    )
    rate_limit_window_s: float = Field(60.0, gt=0, description="Length of the sliding window, in seconds.")
    rate_limit_max_clients: int = Field(
        4096, ge=1, description="Distinct clients tracked before the least recent is forgotten."
    )
    trust_proxy_headers: bool = Field(
        True,
        description="Read the client IP from X-Forwarded-For. Correct behind a proxy that sets it "
        "(Render, Fly, any load balancer); turn it off when the service is directly exposed, "
        "because a client can otherwise spoof the header and bypass the rate limit.",
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

    @field_validator(*_OPTIONAL_FIELDS, mode="before")
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        """Treat a blank env var as absent rather than as an empty value."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

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
