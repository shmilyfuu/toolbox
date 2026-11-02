import sys
import types

import numpy as np
import soundfile as sf

import asr_engine
from audio_core import probe


def test_chinese_english_transcription_with_relative_time_offsets(tmp_path, monkeypatch):
    wav = tmp_path / 'speech.wav'
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    info = probe(str(wav))
    captured = {}

    class FakeWhisperModel:
        def __init__(self, model_id, **kwargs):
            captured['model_id'] = model_id
            captured['kwargs'] = kwargs

        def transcribe(self, path, **kwargs):
            captured['settings'] = kwargs
            captured['clip_frames'] = sf.info(path).frames
            segments = [types.SimpleNamespace(start=0.1, end=0.3, text='你好 Hello world')]
            return iter(segments), types.SimpleNamespace(language='zh')

    monkeypatch.setitem(sys.modules, 'faster_whisper', types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setitem(sys.modules, 'ctranslate2', types.SimpleNamespace(get_cuda_device_count=lambda: 0))
    monkeypatch.setattr(asr_engine, '_cached_model', None)
    monkeypatch.setattr(asr_engine, '_cached_settings', None)
    messages = []
    result = asr_engine.transcribe(info, 0.2, 0.8, 'small', 'zh', 'auto', messages.append)
    assert captured['settings']['language'] == 'zh'
    assert captured['settings']['task'] == 'transcribe'
    assert captured['settings']['multilingual'] is True
    assert captured['clip_frames'] == round(0.8 * 16000) - round(0.2 * 16000)
    assert len(result) == 1
    assert result[0].text == '你好 Hello world'
    assert abs(result[0].start - 0.3) < 1e-6
    assert abs(result[0].end - 0.5) < 1e-6
    assert captured['kwargs']['device'] == 'cpu'


def test_full_file_and_english_mode(tmp_path, monkeypatch):
    wav = tmp_path / 'en.wav'
    sf.write(wav, np.zeros(24000, dtype=np.float32), 48000)
    info = probe(str(wav))
    seen = {}

    class FakeWhisperModel:
        def __init__(self, model_id, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            seen['path'] = path
            seen['kwargs'] = kwargs
            return iter([types.SimpleNamespace(start=0, end=0.4, text='Hello')]), types.SimpleNamespace(language='en')

    monkeypatch.setitem(sys.modules, 'faster_whisper', types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setitem(sys.modules, 'ctranslate2', types.SimpleNamespace(get_cuda_device_count=lambda: 1))
    monkeypatch.setattr(asr_engine, '_cached_model', None)
    monkeypatch.setattr(asr_engine, '_cached_settings', None)
    result = asr_engine.transcribe(info, 0, info.duration, 'large-v3-turbo', 'en', 'auto', lambda _: None)
    assert seen['path'] == info.path
    assert seen['kwargs']['multilingual'] is False
    assert result[0].start == 0
    assert result[0].end == 0.4
