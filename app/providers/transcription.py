"""Speech-to-text adapter backed by faster-whisper (local CPU inference)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from app.logging_config import get_logger
from app.models import Transcript, TranscriptSegment

log = get_logger(__name__)


def _vocab_tag(prompt: str) -> str:
    """Short stable digest of a vocabulary prompt, for cache-key purposes."""
    return hashlib.sha1(prompt.strip().encode("utf-8")).hexdigest()[:8]


class FasterWhisperTranscriber:
    """Local :class:`~app.interfaces.Transcriber` using CTranslate2 Whisper weights.

    Defaults target CPU-only machines: the ``base`` model with ``int8``
    quantisation transcribes roughly 5-10x faster than real time on a modern
    laptop core, which is the sweet spot for hour-long podcast episodes.
    """

    def __init__(
        self,
        model_size: str = "base",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        language: str | None = None,
        vad_filter: bool = True,
        initial_prompt: str | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._language = language
        self._vad_filter = vad_filter
        self._initial_prompt = initial_prompt
        self._model: Any | None = None

    @property
    def name(self) -> str:
        """Identifier recorded in the transcript cache and in logs.

        The vocabulary hint is part of the identity: changing it changes the
        output, so it must invalidate cached transcripts the way a model change
        does. A short hash keeps the name readable.
        """
        suffix = f"+v{_vocab_tag(self._initial_prompt)}" if self._initial_prompt else ""
        return f"faster-whisper:{self._model_size}{suffix}"

    @property
    def model(self) -> Any:
        """The underlying ``WhisperModel``, loaded on first access."""
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy import

            started = time.perf_counter()
            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
            log.info(
                "transcriber.loaded",
                extra={
                    "event": "transcriber.loaded",
                    "model": self._model_size,
                    "device": self._device,
                    "compute_type": self._compute_type,
                    "load_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        return self._model

    def transcribe(self, audio_path: Path, *, episode: str) -> Transcript:
        """Transcribe one audio file into timestamped segments.

        Args:
            audio_path: Audio file to transcribe (any format ffmpeg can decode).
            episode: Episode name to record on the transcript.

        Returns:
            A :class:`~app.models.Transcript` with per-segment start/end offsets.
        """
        started = time.perf_counter()
        segment_iter, info = self.model.transcribe(
            str(audio_path),
            beam_size=self._beam_size,
            language=self._language,
            vad_filter=self._vad_filter,
            # Biases decoding toward domain vocabulary. Whisper picks the most
            # probable transcription, so homophones resolve to whichever spelling
            # is commoner in general English ("cubes" over "Qubes") unless the
            # decoder is told this domain exists.
            initial_prompt=self._initial_prompt,
        )

        segments: list[TranscriptSegment] = []
        for segment in segment_iter:  # lazily evaluated: work happens here
            text = segment.text.strip()
            if text:
                segments.append(TranscriptSegment(text=text, start=segment.start, end=segment.end))
            if len(segments) % 100 == 0 and segments:
                log.info(
                    "transcribe.progress",
                    extra={
                        "event": "transcribe.progress",
                        "episode": episode,
                        "segments": len(segments),
                        "audio_position_s": round(segments[-1].end, 1),
                    },
                )

        elapsed = time.perf_counter() - started
        duration = float(getattr(info, "duration", 0.0) or (segments[-1].end if segments else 0.0))
        log.info(
            "transcribe.completed",
            extra={
                "event": "transcribe.completed",
                "episode": episode,
                "segments": len(segments),
                "audio_duration_s": round(duration, 1),
                "elapsed_s": round(elapsed, 1),
                "realtime_factor": round(duration / elapsed, 2) if elapsed else None,
                "language": getattr(info, "language", None),
            },
        )
        return Transcript(
            episode=episode,
            segments=segments,
            language=getattr(info, "language", None),
            duration=duration,
            source_path=str(audio_path),
        )
