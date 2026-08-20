"""Concrete adapters implementing the protocols in :mod:`app.interfaces`.

Only the dependency-light adapters are re-exported here. The ones needing torch,
faster-whisper or chromadb are imported directly, at call sites inside
:mod:`app.factory`, because the deployed runtime does not install those packages
and an eager import would break startup.

Light (always importable)::

    app.providers.embeddings_remote.RemoteEmbedder      httpx
    app.providers.store_numpy.NumpyVectorStore          numpy
    app.providers.llm.OpenAICompatibleChatModel         openai
    app.providers.llm.EchoChatModel                     -

Heavy (ingest and development only, see requirements-ingest.txt)::

    app.providers.embeddings.SentenceTransformerEmbedder    torch
    app.providers.transcription.FasterWhisperTranscriber    ctranslate2
    app.providers.store.ChromaVectorStore                   chromadb
"""

from app.providers.embeddings_remote import RemoteEmbedder
from app.providers.llm import EchoChatModel, OpenAICompatibleChatModel
from app.providers.store_numpy import NumpyVectorStore

__all__ = [
    "EchoChatModel",
    "NumpyVectorStore",
    "OpenAICompatibleChatModel",
    "RemoteEmbedder",
]
