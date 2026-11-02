"""Amplitude-based pause detection for already isolated vocal WAV files."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import soundfile as sf

from audio_core import AudioInfo


@dataclass(frozen=True)
class PauseAnalysis:
    pauses: list[tuple[float, float]]
    clips: list[tuple[float, float]]


def detect_pauses(
    info: AudioInfo,
    *,
    threshold_db: float = -38.0,
    min_silence: float = 0.4,
    padding: float = 0.10,
    min_clip: float = 0.20,
    frame_ms: float = 20.0,
) -> PauseAnalysis:
    """Split a WAV at long low-RMS sections and return audio-global ranges.

    Pause ranges include long quiet intervals at the file boundaries. Each output
    clip includes a short margin next to speech. This is an amplitude detector;
    very quiet syllables may be classified as silence at aggressive thresholds.
    """
    if not (-100.0 <= threshold_db <= 0.0):
        raise ValueError("静音阈值应位于 -100 到 0 dBFS 之间")
    if not (math.isfinite(min_silence) and min_silence > 0):
        raise ValueError("最短停顿必须大于零")
    if not (math.isfinite(padding) and padding >= 0 and math.isfinite(min_clip) and min_clip > 0):
        raise ValueError("片段时长或留白参数无效")
    window = max(1, round(info.samplerate * frame_ms / 1000))
    threshold = 10.0 ** (threshold_db / 20.0)
    active_parts: list[np.ndarray] = []
    with sf.SoundFile(info.path) as stream:
        while True:
            data = stream.read(window * 2048, dtype="float32", always_2d=True)
            if not len(data):
                break
            whole = len(data) // window
            if whole:
                block = data[:whole * window].reshape(whole, window, info.channels)
                # float64 avoids excess rounding at low amplitudes.
                rms = np.sqrt(np.mean(np.square(block, dtype=np.float64), axis=(1, 2)))
                active_parts.append(rms >= threshold)
            if len(data) % window:
                tail = data[whole * window:]
                rms_tail = np.sqrt(np.mean(np.square(tail, dtype=np.float64)))
                active_parts.append(np.array([rms_tail >= threshold], dtype=bool))
    active = np.concatenate(active_parts) if active_parts else np.zeros(0, dtype=bool)
    if active.size == 0 or not np.any(active):
        return PauseAnalysis([(0.0, info.duration)], [])

    quiet = ~active
    edges = np.flatnonzero(np.diff(np.r_[False, quiet, False].astype(np.int8)))
    min_frames = max(1, math.ceil(min_silence * info.samplerate / window))
    pauses: list[tuple[float, float]] = []
    for start, stop in edges.reshape(-1, 2):
        if stop - start >= min_frames:
            a = min(info.duration, start * window / info.samplerate)
            b = min(info.duration, stop * window / info.samplerate)
            pauses.append((a, b))

    clips: list[tuple[float, float]] = []
    for i in range(len(pauses) + 1):
        left = pauses[i - 1][1] if i else 0.0
        right = pauses[i][0] if i < len(pauses) else info.duration
        earliest = sum(pauses[i - 1]) / 2 if i else 0.0
        latest = sum(pauses[i]) / 2 if i < len(pauses) else info.duration
        start_frame = min(active.size, max(0, math.floor(left * info.samplerate / window)))
        stop_frame = min(active.size, max(start_frame, math.ceil(right * info.samplerate / window)))
        voice = np.flatnonzero(active[start_frame:stop_frame])
        if not voice.size:
            continue
        first_voice = (start_frame + int(voice[0])) * window / info.samplerate
        last_voice = min(info.duration, (start_frame + int(voice[-1]) + 1) * window / info.samplerate)
        a = max(earliest, first_voice - padding)
        b = min(latest, last_voice + padding)
        if b - a >= min_clip:
            clips.append((float(round(a, 6)), float(round(b, 6))))
    return PauseAnalysis(pauses, clips)
