"""WAV metadata, waveform envelope and sample-accurate cutting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioInfo:
    path: str
    samplerate: int
    frames: int
    channels: int
    subtype: str

    @property
    def duration(self) -> float:
        return self.frames / self.samplerate


def probe(path: str) -> AudioInfo:
    info = sf.info(path)
    if info.frames < 1 or info.samplerate < 1:
        raise ValueError("空音频或采样率无效")
    return AudioInfo(str(Path(path).resolve()), info.samplerate, info.frames, info.channels, info.subtype)


def clamp_range(info: AudioInfo, start: float, end: float) -> tuple[int, int]:
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("截取时间必须为有效数字")
    left = max(0, min(info.frames, round(start * info.samplerate)))
    right = max(0, min(info.frames, round(end * info.samplerate)))
    if right <= left:
        raise ValueError("结束时间必须晚于开始时间")
    return left, right


def waveform_envelope(path: str, max_bins: int = 14000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read in bounded chunks; keep positive/negative peaks for long recordings."""
    info = probe(path)
    step = max(1, math.ceil(info.frames / max_bins))
    bin_count = math.ceil(info.frames / step)
    mins = np.empty(bin_count, dtype=np.float32)
    maxs = np.empty(bin_count, dtype=np.float32)
    with sf.SoundFile(path) as stream:
        written = 0
        while written < bin_count:
            batch_bins = min(bin_count - written, max(1, 1_000_000 // step))
            data = stream.read(batch_bins * step, dtype="float32", always_2d=True)
            if len(data) == 0:
                break
            full_bins = len(data) // step
            if full_bins:
                block = data[: full_bins * step].reshape(full_bins, step, info.channels)
                mins[written : written + full_bins] = block.min(axis=(1, 2))
                maxs[written : written + full_bins] = block.max(axis=(1, 2))
                written += full_bins
            remainder = len(data) % step
            if remainder:
                tail = data[-remainder:]
                mins[written] = tail.min()
                maxs[written] = tail.max()
                written += 1
    indices = np.arange(written, dtype=np.float64)
    seconds = (indices + 0.5) * step / info.samplerate
    seconds = np.minimum(seconds, info.duration)
    return seconds, mins[:written], maxs[:written]


def export_clip(info: AudioInfo, start: float, end: float, destination: str, *, asr_temp: bool = False) -> None:
    first, last = clamp_range(info, start, end)
    subtype = "PCM_16" if asr_temp else info.subtype
    if not sf.check_format("WAV", subtype):
        subtype = "PCM_16"
    with sf.SoundFile(info.path, mode="r") as source, sf.SoundFile(
        destination, mode="w", samplerate=info.samplerate,
        channels=info.channels, format="WAV", subtype=subtype,
    ) as target:
        source.seek(first)
        remaining = last - first
        while remaining:
            data = source.read(min(262144, remaining), dtype="float32", always_2d=True)
            if len(data) == 0:
                raise IOError("音频在导出过程中提前结束")
            target.write(data)
            remaining -= len(data)
