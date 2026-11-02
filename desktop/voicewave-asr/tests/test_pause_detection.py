import numpy as np
import pytest
import soundfile as sf

from audio_core import probe, export_clip
from pause_detection import detect_pauses


def make_wav(path, sample_rate=16000, channels=1, ranges=(), duration=3.0, amplitude=0.25):
    data = np.zeros((int(duration * sample_rate), channels), dtype=np.float32)
    for a, b in ranges:
        data[round(a * sample_rate):round(b * sample_rate), :] = amplitude
    sf.write(path, data, sample_rate, subtype='FLOAT')
    return probe(str(path))


def test_multiple_pauses_and_clips(tmp_path):
    info = make_wav(tmp_path / 'speech.wav', ranges=((0.2, 0.8), (1.35, 1.85), (2.35, 2.8)))
    found = detect_pauses(info, threshold_db=-38, min_silence=.4)
    assert len(found.pauses) == 2
    assert found.pauses[0] == pytest.approx((.8, 1.34), abs=.025)
    assert found.pauses[1] == pytest.approx((1.86, 2.34), abs=.025)
    assert len(found.clips) == 3
    assert found.clips[0] == pytest.approx((.1, .9), abs=.025)
    assert found.clips[1] == pytest.approx((1.25, 1.96), abs=.025)
    assert found.clips[2] == pytest.approx((2.24, 2.9), abs=.025)


def test_short_pause_retained_inside_clip(tmp_path):
    info = make_wav(tmp_path / 'quick.wav', ranges=((0.05, .7), (.85, 1.5)), duration=1.6)
    result = detect_pauses(info, min_silence=.3)
    assert len(result.clips) == 1
    assert result.clips[0][0] == pytest.approx(0, abs=.02)
    assert result.clips[0][1] == pytest.approx(1.6, abs=.02)


def test_boundary_silences_and_stereo_non_multiple_window(tmp_path):
    info = make_wav(tmp_path / 'stereo.wav', sample_rate=44100, channels=2,
                    ranges=((.51, .99),), duration=1.93)
    result = detect_pauses(info, min_silence=.3)
    assert len(result.pauses) == 2
    assert len(result.clips) == 1
    assert result.clips[0] == pytest.approx((.41, 1.09), abs=.022)


def test_silent_file_yields_no_clips(tmp_path):
    info = make_wav(tmp_path / 'silent.wav')
    result = detect_pauses(info)
    assert result.pauses == [(0.0, 3.0)]
    assert result.clips == []


def test_volume_threshold_affects_pause(tmp_path):
    info = make_wav(tmp_path / 'quiet.wav', ranges=((0, .7), (1.2, 1.9)),
                    duration=2.0, amplitude=.007)
    assert len(detect_pauses(info, threshold_db=-50).clips) == 2
    assert len(detect_pauses(info, threshold_db=-35).clips) == 0


def test_ranges_export_separately_keep_sample_rate_channels(tmp_path):
    info = make_wav(tmp_path / 'voice.wav', sample_rate=44100, channels=2,
                    ranges=((0.2, .8), (1.35, 1.85), (2.35, 2.8)))
    result = detect_pauses(info)
    for i, (a, b) in enumerate(result.clips):
        output = tmp_path / f'clip_{i}.wav'
        export_clip(info, a, b, str(output))
        actual = probe(str(output))
        assert actual.samplerate == 44100
        assert actual.channels == 2
        assert actual.frames == round(b * 44100) - round(a * 44100)


def test_invalid_pause_parameters(tmp_path):
    info = make_wav(tmp_path / 'voice.wav')
    with pytest.raises(ValueError):
        detect_pauses(info, min_silence=0)
    with pytest.raises(ValueError):
        detect_pauses(info, threshold_db=2)
