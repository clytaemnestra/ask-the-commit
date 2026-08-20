"""Composition root: the only module that knows which concrete provider is in use.

Everything else depends on the protocols in :mod:`app.interfaces`. To add a
provider, write an adapter and register it here.
"""

from __future__ import annotations

from typing import Callable

from app.config import Settings, secret_value
from app.interfaces import ChatModel, Embedder, Transcriber, VectorStore
from app.logging_config import get_logger
from app.providers.embeddings_remote import PROVIDER_PRESETS, RemoteEmbedder
from app.providers.llm import EchoChatModel, OpenAICompatibleChatModel
from app.providers.store_numpy import NumpyVectorStore

log = get_logger(__name__)


class ConfigurationError(RuntimeError):
    """Raised when settings ask for a provider that cannot be constructed."""


DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_embedder(settings: Settings) -> Embedder:
    """Construct the configured embedding backend.

    ``local`` runs sentence-transformers in-process and is for ingestion and
    development only — it needs torch, which the deployed runtime does not ship.
    Every other provider is an HTTP call.

    Raises:
        ConfigurationError: If a hosted provider is selected with no API key, or
            the provider name is unknown.
    """
    if settings.embedding_provider == "onnx":
        from app.providers.embeddings_onnx import OnnxEmbedder

        return OnnxEmbedder(settings.onnx_model_dir, threads=settings.onnx_threads)

    if settings.embedding_provider == "local":
        # Imported lazily: sentence-transformers is not installed in the
        # runtime image, and importing it there would break startup.
        try:
            from app.providers.embeddings import SentenceTransformerEmbedder
        except ImportError as exc:
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=local needs sentence-transformers, which is deliberately "
                "absent from requirements.txt because torch does not fit a free tier.\n"
                "  - to ingest locally:  pip install -r requirements-ingest.txt\n"
                "  - to serve:           set EMBEDDING_PROVIDER to a hosted provider "
                f"({', '.join(sorted(PROVIDER_PRESETS))}) and rebuild the index with it."
            ) from exc

        return SentenceTransformerEmbedder(
            settings.embedding_model or DEFAULT_LOCAL_EMBEDDING_MODEL,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )

    preset = PROVIDER_PRESETS.get(settings.embedding_provider)
    if preset is None:
        raise ConfigurationError(
            f"unknown EMBEDDING_PROVIDER {settings.embedding_provider!r}; expected 'onnx', "
            f"'local', or one of {sorted(PROVIDER_PRESETS)}"
        )
    if not settings.embedding_api_key:
        raise ConfigurationError(
            f"EMBEDDING_API_KEY is not set but EMBEDDING_PROVIDER={settings.embedding_provider}. "
            f"Get a key from {preset['signup']}, or set EMBEDDING_PROVIDER=local for a fully "
            "offline setup (requires requirements-ingest.txt)."
        )
    return RemoteEmbedder(
        provider=settings.embedding_provider,
        model=settings.embedding_model or preset["model"],
        base_url=settings.embedding_base_url or preset["base_url"],
        api_key=secret_value(settings.embedding_api_key) or "",
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        timeout_s=settings.embedding_timeout_s,
        max_retries=settings.embedding_max_retries,
    )


def build_transcriber(settings: Settings) -> Transcriber:
    """Construct the configured speech-to-text backend (ingest only)."""
    from app.providers.transcription import FasterWhisperTranscriber  # needs faster-whisper

    return FasterWhisperTranscriber(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        beam_size=settings.whisper_beam_size,
        language=settings.whisper_language,
        vad_filter=settings.whisper_vad_filter,
        initial_prompt=settings.whisper_initial_prompt,
    )


def build_vector_store(settings: Settings, *, read_only: bool = False) -> VectorStore:
    """Construct the configured vector store.

    Args:
        settings: Application settings.
        read_only: Set by the serving path. The deployed host has no persistent
            disk, so writes must be impossible rather than merely unused.
    """
    if settings.vector_store == "numpy":
        return NumpyVectorStore(settings.index_dir, read_only=read_only)

    from app.providers.store import ChromaVectorStore  # needs chromadb

    return ChromaVectorStore(settings.chroma_dir, settings.collection_name)


def validate_serving_config(settings: Settings) -> None:
    """Check everything the query path needs, before the server accepts traffic.

    Collects *all* problems rather than surfacing them one redeploy at a time.

    Raises:
        ConfigurationError: If any required credential or file is missing.
    """
    problems: list[str] = []

    if settings.llm_provider == "groq" and not settings.groq_api_key:
        problems.append(
            "  - GROQ_API_KEY is not set (needed for LLM_PROVIDER=groq).\n"
            "    Free key: https://console.groq.com/keys"
        )
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        problems.append("  - OPENAI_API_KEY is not set (needed for LLM_PROVIDER=openai).")

    if settings.embedding_provider not in ("onnx", "local") and not settings.embedding_api_key:
        preset = PROVIDER_PRESETS.get(settings.embedding_provider, {})
        problems.append(
            f"  - EMBEDDING_API_KEY is not set (needed for "
            f"EMBEDDING_PROVIDER={settings.embedding_provider}).\n"
            f"    Free key: {preset.get('signup', 'see the provider docs')}"
        )

    if settings.embedding_provider == "onnx" and not (settings.onnx_model_dir / "model.onnx").exists():
        problems.append(
            f"  - No ONNX model at {settings.onnx_model_dir}/model.onnx\n"
            "    It is committed to the repo; see models/minilm-onnx/SOURCE.md."
        )

    if settings.vector_store == "numpy" and not (settings.index_dir / "embeddings.npy").exists():
        problems.append(
            f"  - No index found at {settings.index_dir}/embeddings.npy\n"
            "    Build it locally with `python ingest.py` and commit data/index/."
        )

    if problems:
        raise ConfigurationError(
            "the service cannot start because it is misconfigured:\n\n"
            + "\n".join(problems)
            + "\n\nSet these as environment variables (see .env.example). "
            "On Render: Dashboard -> your service -> Environment."
        )


def _build_groq(settings: Settings) -> ChatModel:
    """Groq's free OpenAI-compatible endpoint."""
    if not settings.groq_api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a key from "
            "https://console.groq.com/keys, or set LLM_PROVIDER=ollama / echo."
        )
    return _openai_compatible(
        settings, "groq", settings.groq_model, settings.groq_base_url, secret_value(settings.groq_api_key)
    )


def _build_ollama(settings: Settings) -> ChatModel:
    """A local Ollama server (no credentials required)."""
    return _openai_compatible(settings, "ollama", settings.ollama_model, settings.ollama_base_url, "ollama")


def _build_openai(settings: Settings) -> ChatModel:
    """OpenAI proper, for when the free tier stops being enough."""
    if not settings.openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai")
    return _openai_compatible(
        settings, "openai", settings.openai_model, settings.openai_base_url, secret_value(settings.openai_api_key)
    )


def _build_echo(settings: Settings) -> ChatModel:
    """Offline stub backend used by smoke tests and CI."""
    return EchoChatModel()


def _openai_compatible(
    settings: Settings, provider: str, model: str, base_url: str, api_key: str | None
) -> ChatModel:
    """Shared constructor for the OpenAI-dialect providers."""
    return OpenAICompatibleChatModel(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


#: Registry of generation backends. Add new providers here, nowhere else.
CHAT_BUILDERS: dict[str, Callable[[Settings], ChatModel]] = {
    "groq": _build_groq,
    "ollama": _build_ollama,
    "openai": _build_openai,
    "echo": _build_echo,
}


def build_chat_model(settings: Settings) -> ChatModel:
    """Construct the generation backend named by ``settings.llm_provider``.

    Raises:
        ConfigurationError: If the provider is unknown or misconfigured.
    """
    try:
        builder = CHAT_BUILDERS[settings.llm_provider]
    except KeyError:
        raise ConfigurationError(
            f"unknown LLM_PROVIDER {settings.llm_provider!r}; expected one of {sorted(CHAT_BUILDERS)}"
        ) from None
    model = builder(settings)
    log.info("chat_model.selected", extra={"event": "chat_model.selected", "model": model.name})
    return model
