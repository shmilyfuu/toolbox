"""Playback of an arbitrary WAV range without loading the entire file."""
from __future__ import annotations

import threading
import soundfile as sf
import sounddevice as sd

from audio_core import AudioInfo, clamp_range


class ClipPlayer:
    def __init__(self):
        self._stream = None
        self._file = None
        self._position = 0
        self._end = 0
        self._samplerate = 1
        self._lock = threading.Lock()

    def start(self, info: AudioInfo, start: float, end: float) -> None:
        self.stop()
        first, last = clamp_range(info, start, end)
        source = sf.SoundFile(info.path)
        source.seek(first)
        self._file = source
        self._position = first
        self._end = last
        self._samplerate = info.samplerate
        output_channels = info.channels if info.channels <= 2 else 1

        def audio_callback(outdata, frames, time, status):
            outdata.fill(0)
            with self._lock:
                remaining = max(0, self._end - self._position)
                wanted = min(frames, remaining)
                if wanted:
                    chunk = source.read(wanted, dtype="float32", always_2d=True)
                    n = len(chunk)
                    if source.channels > 2:
                        outdata[:n, 0] = chunk.mean(axis=1)
                    else:
                        outdata[:n] = chunk
                    self._position += n
                else:
                    n = 0
            if n < frames or self._position >= self._end:
                raise sd.CallbackStop

        try:
            self._stream = sd.OutputStream(
                samplerate=info.samplerate, channels=output_channels,
                dtype="float32", blocksize=1024, callback=audio_callback,
            )
            self._stream.start()
        except Exception:
            self.stop()
            raise

    @property
    def active(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def position_seconds(self) -> float:
        with self._lock:
            return self._position / self._samplerate

    def stop(self) -> None:
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                stream.stop()
            finally:
                stream.close()
        if self._file is not None:
            self._file.close()
            self._file = None
