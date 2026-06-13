"""Event handler for clients of the server."""

import argparse
import asyncio
import io
import logging
import math
import re
import tempfile
import wave
from typing import Any, Dict, List, NamedTuple, Optional, Union

from piper import PiperVoice, SynthesisConfig
from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .download import ensure_voice_exists, find_voice

_LOGGER = logging.getLogger(__name__)

# Keep the most recently used voice loaded
_VOICE: Optional[PiperVoice] = None
_VOICE_NAME: Optional[str] = None
_VOICE_LOCK = asyncio.Lock()

# Matches [[speaker:#5]], [[speaker:name]], or [[gap:500]]
_TOKEN_RE = re.compile(r"\[\[(speaker|gap):([^\]]+)\]\]")


class TextSegment(NamedTuple):
    text: str
    speaker: Optional[str]  # raw token value ('#5', 'angry') or None for default


class GapSegment(NamedTuple):
    ms: int  # silence duration in milliseconds


Segment = Union[TextSegment, GapSegment]


def _parse_segments(
    text: str, current_speaker: Optional[str]
) -> List[Segment]:
    """Split text on [[speaker:...]] and [[gap:...]] tokens.

    Returns a list of TextSegment and GapSegment objects. Empty text runs are
    omitted. Invalid gap values are logged and skipped.
    """
    segments: List[Segment] = []
    pos = 0
    active_speaker = current_speaker

    for match in _TOKEN_RE.finditer(text):
        kind = match.group(1)
        value = match.group(2)

        # Emit text before this token
        seg = text[pos : match.start()]
        if seg.strip():
            segments.append(TextSegment(text=seg, speaker=active_speaker))

        if kind == "speaker":
            active_speaker = value
        elif kind == "gap":
            try:
                ms = int(value)
                if ms < 0:
                    raise ValueError("negative")
                segments.append(GapSegment(ms=ms))
            except ValueError:
                _LOGGER.warning("Invalid gap value '%s', skipping token", value)

        pos = match.end()

    # Remaining text after last token
    tail = text[pos:]
    if tail.strip():
        segments.append(TextSegment(text=tail, speaker=active_speaker))

    return segments


def _resolve_speaker_id(
    raw: Optional[str],
    voice: "PiperVoice",
    default_speaker_id: Optional[int],
) -> Optional[int]:
    """Resolve a raw speaker token value to a speaker_id integer.

    Syntax:
      '#5'   -> direct id 5 (bypasses name map)
      'name' -> look up in voice.config.speaker_id_map

    Returns default_speaker_id and logs a warning if the value cannot be
    resolved.
    """
    if raw is None:
        return default_speaker_id

    if raw.startswith("#"):
        try:
            return int(raw[1:])
        except ValueError:
            _LOGGER.warning("Invalid direct speaker id '%s', reverting to default", raw)
            return default_speaker_id

    # Name lookup
    speaker_id_map = getattr(getattr(voice, "config", None), "speaker_id_map", {}) or {}
    if raw in speaker_id_map:
        return speaker_id_map[raw]

    _LOGGER.warning(
        "Speaker '%s' not found in voice speaker map, reverting to default", raw
    )
    return default_speaker_id


def _silence_bytes(ms: int, rate: int, width: int, channels: int) -> bytes:
    """Return ms milliseconds of silence as PCM zero-bytes."""
    num_samples = int(rate * ms / 1000)
    return b"\x00" * (num_samples * width * channels)


class PiperEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        voices_info: Dict[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.voices_info = voices_info
        self.is_streaming: Optional[bool] = None
        self.sbd = SentenceBoundaryDetector()
        self._synthesize: Optional[Synthesize] = None
        self._current_speaker: Optional[str] = None  # tracks active speaker across sentences

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    # Ignore since this is only sent for compatibility reasons.
                    # For streaming, we expect:
                    # [synthesize-start] -> [synthesize-chunk]+ -> [synthesize]? -> [synthesize-stop]
                    return True

                # Sent outside a stream, so we must process it
                synthesize = Synthesize.from_event(event)
                self._synthesize = Synthesize(text="", voice=synthesize.voice)
                self.sbd = SentenceBoundaryDetector()
                self._current_speaker = None
                start_sent = False
                for i, sentence in enumerate(self.sbd.add_chunk(synthesize.text)):
                    self._synthesize.text = sentence
                    await self._handle_synthesize(
                        self._synthesize, send_start=(i == 0), send_stop=False
                    )
                    start_sent = True

                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    # Last sentence
                    await self._handle_synthesize(
                        self._synthesize, send_start=(not start_sent), send_stop=True
                    )
                else:
                    # No final sentence
                    await self.write_event(AudioStop().event())

                return True

            if self.cli_args.no_streaming:
                # Streaming is not enabled
                return True

            if SynthesizeStart.is_type(event.type):
                # Start of a stream
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                self._current_speaker = None
                _LOGGER.debug("Text stream started: voice=%s", stream_start.voice)
                return True

            if SynthesizeChunk.is_type(event.type):
                assert self._synthesize is not None
                stream_chunk = SynthesizeChunk.from_event(event)
                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    _LOGGER.debug("Synthesizing stream sentence: %s", sentence)
                    self._synthesize.text = sentence
                    await self._handle_synthesize(self._synthesize)

                return True

            if SynthesizeStop.is_type(event.type):
                assert self._synthesize is not None
                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    # Final audio chunk(s)
                    await self._handle_synthesize(self._synthesize)

                # End of audio
                await self.write_event(SynthesizeStopped().event())

                _LOGGER.debug("Text stream stopped")
                return True

            if not Synthesize.is_type(event.type):
                return True

            synthesize = Synthesize.from_event(event)
            return await self._handle_synthesize(synthesize)
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_synthesize(
        self, synthesize: Synthesize, send_start: bool = True, send_stop: bool = True
    ) -> bool:
        global _VOICE, _VOICE_NAME

        _LOGGER.debug(synthesize)

        raw_text = synthesize.text

        # Join multiple lines
        text = " ".join(raw_text.strip().splitlines())

        if self.cli_args.auto_punctuation and text:
            # Add automatic punctuation (important for some voices)
            has_punctuation = False
            for punc_char in self.cli_args.auto_punctuation:
                if text[-1] == punc_char:
                    has_punctuation = True
                    break

            if not has_punctuation:
                text = text + self.cli_args.auto_punctuation[0]

        # Resolve voice
        _LOGGER.debug("synthesize: raw_text=%s, text='%s'", raw_text, text)
        voice_name: Optional[str] = None
        voice_speaker: Optional[str] = None
        if synthesize.voice is not None:
            voice_name = synthesize.voice.name
            voice_speaker = synthesize.voice.speaker

        if voice_name is None:
            # Default voice
            voice_name = self.cli_args.voice

        if voice_name == self.cli_args.voice:
            # Default speaker from CLI (used as the revert-to-default value)
            voice_speaker = voice_speaker or self.cli_args.speaker

        assert voice_name is not None

        # Resolve alias
        voice_info = self.voices_info.get(voice_name, {})
        voice_name = voice_info.get("key", voice_name)
        assert voice_name is not None

        # Parse [[speaker:...]] and [[gap:...]] tokens, splitting text into segments.
        segments = _parse_segments(text, self._current_speaker)
        if not segments:
            # Nothing to synthesize (e.g. text was only whitespace/tokens)
            if send_stop:
                await self.write_event(AudioStop().event())
            return True

        with tempfile.NamedTemporaryFile(mode="wb+", suffix=".wav") as output_file:
            async with _VOICE_LOCK:
                if voice_name != _VOICE_NAME:
                    # Load new voice
                    _LOGGER.debug("Loading voice: %s", voice_name)
                    ensure_voice_exists(
                        voice_name,
                        self.cli_args.data_dir,
                        self.cli_args.download_dir,
                        self.voices_info,
                    )
                    model_path, config_path = find_voice(
                        voice_name, self.cli_args.data_dir
                    )
                    _VOICE = PiperVoice.load(
                        model_path, config_path, use_cuda=self.cli_args.use_cuda
                    )
                    _VOICE_NAME = voice_name

                assert _VOICE is not None

                # Resolve the default speaker id once so _resolve_speaker_id
                # can revert to it on unknown tokens.
                default_speaker_id: Optional[int] = None
                if voice_speaker is not None:
                    default_speaker_id = _resolve_speaker_id(
                        voice_speaker, _VOICE, default_speaker_id=None
                    )

                # Accumulate all PCM frames here; WAV params from first TextSegment.
                all_pcm: bytes = b""
                rate: Optional[int] = None
                width: Optional[int] = None
                channels: Optional[int] = None

                # GapSegments that arrive before the first TextSegment (WAV params
                # not yet known) are stored as millisecond values and flushed once
                # we have audio format information.
                pending_gap_ms: int = 0

                # Track whether the previous segment was a TextSegment so we know
                # when to insert auto-padding vs. when a [[gap:n]] replaces it.
                prev_was_text = False

                sentence_silence_ms = getattr(
                    self.cli_args, "sentence_silence_ms", 80
                )

                for seg in segments:
                    if isinstance(seg, GapSegment):
                        if rate is None:
                            # WAV params not yet known — accumulate ms for later.
                            pending_gap_ms += seg.ms
                        else:
                            # Explicit gap replaces auto-padding at this boundary.
                            prev_was_text = False
                            all_pcm += _silence_bytes(seg.ms, rate, width, channels)  # type: ignore[arg-type]
                        continue

                    # TextSegment — resolve speaker id.
                    assert isinstance(seg, TextSegment)

                    _FALLBACK = object()
                    if seg.speaker is None:
                        syn_config_speaker_id = default_speaker_id
                    else:
                        resolved = _resolve_speaker_id(
                            seg.speaker, _VOICE, _FALLBACK  # type: ignore[arg-type]
                        )
                        if resolved is _FALLBACK:
                            self._current_speaker = None
                            syn_config_speaker_id = default_speaker_id
                        else:
                            self._current_speaker = seg.speaker
                            syn_config_speaker_id = resolved  # type: ignore[assignment]

                    syn_config = SynthesisConfig()
                    syn_config.speaker_id = syn_config_speaker_id
                    if self.cli_args.length_scale is not None:
                        syn_config.length_scale = self.cli_args.length_scale
                    if self.cli_args.noise_scale is not None:
                        syn_config.noise_scale = self.cli_args.noise_scale
                    if self.cli_args.noise_w_scale is not None:
                        syn_config.noise_w_scale = self.cli_args.noise_w_scale

                    _LOGGER.debug(
                        "Synthesizing segment: speaker=%s, text='%s'",
                        seg.speaker,
                        seg.text,
                    )

                    # Synthesize into a separate buffer per segment to avoid the
                    # "cannot change parameters after starting to write" WAV error
                    # that occurs when reusing a single wav_writer across segments.
                    seg_buf = io.BytesIO()
                    seg_wav: wave.Wave_write = wave.open(seg_buf, "wb")
                    with seg_wav:
                        _VOICE.synthesize_wav(seg.text, seg_wav, syn_config)
                    seg_buf.seek(0)
                    seg_read: wave.Wave_read = wave.open(seg_buf, "rb")
                    with seg_read:
                        seg_rate = seg_read.getframerate()
                        seg_width = seg_read.getsampwidth()
                        seg_channels = seg_read.getnchannels()
                        seg_pcm = seg_read.readframes(seg_read.getnframes())

                    if rate is None:
                        # First TextSegment — establish WAV params.
                        rate, width, channels = seg_rate, seg_width, seg_channels
                        # Flush any gap tokens that preceded the first text.
                        if pending_gap_ms > 0:
                            all_pcm += _silence_bytes(pending_gap_ms, rate, width, channels)
                    else:
                        # Between TextSegments: use auto-padding unless a [[gap:n]]
                        # already separated them (prev_was_text would be False then).
                        if prev_was_text and sentence_silence_ms > 0:
                            all_pcm += _silence_bytes(
                                sentence_silence_ms, rate, width, channels  # type: ignore[arg-type]
                            )

                    all_pcm += seg_pcm
                    prev_was_text = True

                if not all_pcm:
                    # Only gap tokens, no text — nothing to write
                    if send_stop:
                        await self.write_event(AudioStop().event())
                    return True

                wav_writer: wave.Wave_write = wave.open(output_file, "wb")
                with wav_writer:
                    wav_writer.setnchannels(channels)  # type: ignore[arg-type]
                    wav_writer.setsampwidth(width)  # type: ignore[arg-type]
                    wav_writer.setframerate(rate)  # type: ignore[arg-type]
                    wav_writer.writeframes(all_pcm)

            output_file.seek(0)

            wav_file: wave.Wave_read = wave.open(output_file, "rb")
            with wav_file:
                r = wav_file.getframerate()
                w = wav_file.getsampwidth()
                c = wav_file.getnchannels()

                if send_start:
                    await self.write_event(
                        AudioStart(
                            rate=r,
                            width=w,
                            channels=c,
                        ).event(),
                    )

                # Audio
                audio_bytes = wav_file.readframes(wav_file.getnframes())
                bytes_per_sample = w * c
                bytes_per_chunk = bytes_per_sample * self.cli_args.samples_per_chunk
                num_chunks = int(math.ceil(len(audio_bytes) / bytes_per_chunk))

                # Split into chunks
                for i in range(num_chunks):
                    offset = i * bytes_per_chunk
                    chunk = audio_bytes[offset : offset + bytes_per_chunk]

                    await self.write_event(
                        AudioChunk(
                            audio=chunk,
                            rate=r,
                            width=w,
                            channels=c,
                        ).event(),
                    )

            if send_stop:
                await self.write_event(AudioStop().event())

        return True
