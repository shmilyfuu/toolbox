import numpy as np
import soundfile as sf
import pytest

from audio_core import probe, waveform_envelope, export_clip, clamp_range
from formatting import timestamp, as_srt_time


def test_wav_cut_and_duration(tmp_path):
    rate = 48000
    frames = rate * 2
    t = np.arange(frames) / rate
    stereo = np.column_stack((0.4 * np.sin(2 * np.pi * 330 * t), 0.25 * np.sin(2 * np.pi * 660 * t)))
    inp, out = tmp_path / "人声.wav", tmp_path / "片段.wav"
    sf.write(inp, stereo, rate, subtype="PCM_24")
    info = probe(str(inp))
    assert info.samplerate == rate
    assert info.channels == 2
    assert info.subtype == "PCM_24"
    sec, mn, mx = waveform_envelope(str(inp), max_bins=1000)
    assert len(sec) == len(mn) == len(mx)
    assert np.max(mx) > 0.35
    assert np.min(mn) < -0.35
    export_clip(info, .425, 1.275, str(out))
    clipped = probe(str(out))
    assert clipped.frames == round(1.275 * rate) - round(.425 * rate)
    assert clipped.subtype == "PCM_24"
    assert clipped.channels == 2
    audio, sr = sf.read(out)
    assert sr == rate
    assert np.max(np.abs(audio - stereo[round(.425*rate):round(1.275*rate)])) < 1e-6


def test_float_wav_and_invalid_range(tmp_path):
    inp = tmp_path / "float.wav"
    sf.write(inp, np.linspace(-0.1, 0.1, 333, dtype=np.float32), 16000, subtype="FLOAT")
    info = probe(str(inp))
    assert info.subtype == "FLOAT"
    assert clamp_range(info, 0, 1)[1] == info.frames
    with pytest.raises(ValueError):
        clamp_range(info, 2, 1)
    sec, mn, mx = waveform_envelope(str(inp), max_bins=100)
    assert len(sec) <= 100
    assert np.all(mn <= mx)


def test_timestamps():
    assert timestamp(62.321) == "00:01:02.321"
    assert as_srt_time(62.321) == "00:01:02,321"
