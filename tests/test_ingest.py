"""Tests for episode discovery and the transcript cache."""

from __future__ import annotations

import json
from pathlib import Path

from ingest import discover_episodes, episode_name, load_or_transcribe, transcript_cache_path
from tests.conftest import FakeTranscriber


def test_discover_ignores_non_audio_and_hidden_files(tmp_path: Path) -> None:
    for name in ["ep-01.mp3", "ep-02.m4a", "cover.jpg", ".DS_Store", "notes.txt"]:
        (tmp_path / name).touch()
    (tmp_path / "subdir").mkdir()

    found = discover_episodes(tmp_path)

    assert [path.name for path in found] == ["ep-01.mp3", "ep-02.m4a"]


def test_discover_missing_folder_returns_empty(tmp_path: Path) -> None:
    assert discover_episodes(tmp_path / "nope") == []


def test_episode_name_is_the_file_stem(tmp_path: Path) -> None:
    assert episode_name(tmp_path / "Ep 12 - Burnout.mp3") == "Ep 12 - Burnout"


def test_transcription_is_cached_and_reused(tmp_path: Path, transcript) -> None:
    audio = tmp_path / "ep-01.mp3"
    audio.touch()
    cache_dir = tmp_path / "transcripts"
    transcriber = FakeTranscriber(transcript)

    first, transcribed_first = load_or_transcribe(audio, transcriber=transcriber, cache_dir=cache_dir)
    second, transcribed_second = load_or_transcribe(audio, transcriber=transcriber, cache_dir=cache_dir)

    assert transcribed_first is True
    assert transcribed_second is False  # served from cache
    assert [s.text for s in second.segments] == [s.text for s in first.segments]
    assert transcript_cache_path(cache_dir, "ep-01").exists()


def test_retranscribe_bypasses_the_cache(tmp_path: Path, transcript) -> None:
    audio = tmp_path / "ep-01.mp3"
    audio.touch()
    cache_dir = tmp_path / "transcripts"
    transcriber = FakeTranscriber(transcript)

    load_or_transcribe(audio, transcriber=transcriber, cache_dir=cache_dir)
    _, transcribed = load_or_transcribe(audio, transcriber=transcriber, cache_dir=cache_dir, retranscribe=True)

    assert transcribed is True


def test_cache_is_invalidated_when_the_transcriber_changes(tmp_path: Path, transcript) -> None:
    audio = tmp_path / "ep-01.mp3"
    audio.touch()
    cache_dir = tmp_path / "transcripts"
    load_or_transcribe(audio, transcriber=FakeTranscriber(transcript), cache_dir=cache_dir)

    cached = json.loads(transcript_cache_path(cache_dir, "ep-01").read_text())
    cached["transcriber"] = "faster-whisper:tiny"
    transcript_cache_path(cache_dir, "ep-01").write_text(json.dumps(cached))

    _, transcribed = load_or_transcribe(audio, transcriber=FakeTranscriber(transcript), cache_dir=cache_dir)

    assert transcribed is True


def test_vocabulary_prompt_is_part_of_the_transcriber_identity() -> None:
    """A changed vocabulary changes the output, so it must invalidate the cache."""
    from app.providers.transcription import FasterWhisperTranscriber

    plain = FasterWhisperTranscriber("small")
    hinted = FasterWhisperTranscriber("small", initial_prompt="Qubes OS, Tor")
    same = FasterWhisperTranscriber("small", initial_prompt="Qubes OS, Tor")
    other = FasterWhisperTranscriber("small", initial_prompt="librosa, ISMIR")

    assert plain.name == "faster-whisper:small"
    assert hinted.name != plain.name
    assert hinted.name == same.name          # stable across processes
    assert hinted.name != other.name         # different vocabulary, different cache
    assert hinted.name.startswith("faster-whisper:small+v")


def test_vocabulary_prompt_whitespace_does_not_change_identity() -> None:
    from app.providers.transcription import FasterWhisperTranscriber

    a = FasterWhisperTranscriber("small", initial_prompt="Qubes OS, Tor")
    b = FasterWhisperTranscriber("small", initial_prompt="  Qubes OS, Tor  ")

    assert a.name == b.name
