"""Demo script: synthesize German text with speaker/gap tokens directly via piper-tts."""

import argparse
import io
import wave
from piper import PiperVoice, SynthesisConfig

from wyoming_piper.handler import (
    GapSegment,
    TextSegment,
    _parse_segments,
    _resolve_speaker_id,
    _silence_bytes,
)

VOICE_NAME = "de_DE-thorsten_emotional-medium"
MODEL_PATH = f"local/{VOICE_NAME}.onnx"
CONFIG_PATH = f"local/{VOICE_NAME}.onnx.json"
OUTPUT = "demo_output.wav"
SENTENCE_SILENCE_MS = 150

# Demo text: cycles through emotional speakers with an explicit gap
TEXT = (
    "Guten Morgen! "
    "[[speaker:amused]] Ich freue mich wirklich sehr, Sie zu treffen! "
    "[[speaker:angry]] Das ist absolut inakzeptabel! "
    "[[gap:400]]"
    "[[speaker:whisper]] Und jetzt noch ein kleines Geheimnis unter uns... "
    "[[speaker:neutral]] Vielen Dank und auf Wiedersehen."
)

print(f"Loading voice: {VOICE_NAME}")
voice = PiperVoice.load(MODEL_PATH, CONFIG_PATH, use_cuda=False)
print(f"Speaker map: {voice.config.speaker_id_map}")

segments = _parse_segments(TEXT, current_speaker=None)
print(f"\nSegments ({len(segments)}):")
for seg in segments:
    if isinstance(seg, TextSegment):
        sid = _resolve_speaker_id(seg.speaker, voice, default_speaker_id=None)
        print(f"  TEXT  speaker={seg.speaker!r} (id={sid}): {seg.text!r}")
    else:
        print(f"  GAP   {seg.ms}ms")

print(f"\nSynthesizing to {OUTPUT} (sentence_silence={SENTENCE_SILENCE_MS}ms) ...")

all_pcm = b""
rate = width = channels = None
prev_was_text = False

for seg in segments:
    if isinstance(seg, GapSegment):
        if rate is not None:
            prev_was_text = False
            all_pcm += _silence_bytes(seg.ms, rate, width, channels)
        continue

    syn_config = SynthesisConfig()
    syn_config.speaker_id = _resolve_speaker_id(seg.speaker, voice, default_speaker_id=None)

    buf = io.BytesIO()
    wav_buf = wave.open(buf, "wb")
    with wav_buf:
        voice.synthesize_wav(seg.text, wav_buf, syn_config)
    buf.seek(0)
    wav_read = wave.open(buf, "rb")
    with wav_read:
        seg_rate = wav_read.getframerate()
        seg_width = wav_read.getsampwidth()
        seg_channels = wav_read.getnchannels()
        seg_pcm = wav_read.readframes(wav_read.getnframes())

    if rate is None:
        rate, width, channels = seg_rate, seg_width, seg_channels
    elif prev_was_text and SENTENCE_SILENCE_MS > 0:
        all_pcm += _silence_bytes(SENTENCE_SILENCE_MS, rate, width, channels)

    all_pcm += seg_pcm
    prev_was_text = True

with wave.open(OUTPUT, "wb") as wav_out:
    wav_out.setnchannels(channels)
    wav_out.setsampwidth(width)
    wav_out.setframerate(rate)
    wav_out.writeframes(all_pcm)

duration = len(all_pcm) / (rate * width * channels)
print(f"Done: {OUTPUT} ({duration:.1f}s, {rate}Hz, {channels}ch)")
