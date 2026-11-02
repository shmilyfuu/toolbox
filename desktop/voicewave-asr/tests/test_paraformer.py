import sys
import types

import numpy as np
import soundfile as sf
import pytest

import asr_engine
from audio_core import probe


@pytest.fixture(autouse=True)
def reset_cache(monkeypatch):
    monkeypatch.setattr(asr_engine, '_cached_model', None)
    monkeypatch.setattr(asr_engine, '_cached_settings', None)


def make_model_root(tmp_path):
    root = tmp_path / 'models'
    for folder in asr_engine.PARAFORMER_DIRS:
        (root / folder).mkdir(parents=True)
    return root


def test_find_model_root(tmp_path):
    root = make_model_root(tmp_path)
    assert asr_engine.find_paraformer_root(root) == root.resolve()
    assert asr_engine.find_paraformer_root(root / asr_engine.PARAFORMER_DIRS[0]) == root.resolve()
    assert asr_engine.find_paraformer_root(tmp_path) is None


def test_paraformer_uses_three_local_models_and_sentence_timestamps(tmp_path, monkeypatch):
    root = make_model_root(tmp_path)
    wav = tmp_path / 'test.wav'
    sf.write(wav, np.zeros(16000 * 2, dtype=np.float32), 16000)
    seen = {}

    class FakeAutoModel:
        def __init__(self, **kwargs):
            seen['model_init'] = kwargs

        def generate(self, **kwargs):
            seen['inference'] = kwargs
            seen['frames'] = sf.info(kwargs['input']).frames
            return [{'text': '你好，这是测试。', 'sentence_info': [
                {'start': 100, 'end': 600, 'text': '你好，'},
                {'start': 700, 'end': 1100, 'text': '这是测试。'},
            ]}]

    monkeypatch.setitem(sys.modules, 'funasr', types.SimpleNamespace(AutoModel=FakeAutoModel))
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)))
    info = probe(str(wav))
    result = asr_engine.transcribe(info, .2, 1.5, f'paraformer:{root}', 'zh', 'auto', lambda _: None)
    assert seen['model_init']['model'] == str(root / asr_engine.PARAFORMER_DIRS[0])
    assert seen['model_init']['vad_model'] == str(root / asr_engine.PARAFORMER_DIRS[1])
    assert seen['model_init']['punc_model'] == str(root / asr_engine.PARAFORMER_DIRS[2])
    assert seen['model_init']['device'] == 'cuda:0'
    assert seen['inference']['sentence_timestamp'] is True
    assert seen['frames'] == (round(1.5 * 16000) - round(.2 * 16000))
    assert [s.start for s in result] == pytest.approx([.3, .9])
    assert [s.end for s in result] == pytest.approx([.8, 1.3])
    assert [s.text for s in result] == ['你好，', '这是测试。']


def test_no_sentence_info_keeps_complete_transcript(tmp_path, monkeypatch):
    root = make_model_root(tmp_path)
    wav = tmp_path / 'test.wav'
    sf.write(wav, np.zeros(16000), 16000)

    class FakeAutoModel:
        def __init__(self, **kwargs): pass
        def generate(self, **kwargs): return [{'text': '中文 Hello'}]

    monkeypatch.setitem(sys.modules, 'funasr', types.SimpleNamespace(AutoModel=FakeAutoModel))
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    info = probe(str(wav))
    assert asr_engine.transcribe(info, 0, 1, f'paraformer:{root}', 'zh', 'auto', lambda _: None) == [
        asr_engine.Segment(0, 1, '中文 Hello')]


def test_missing_model_files_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='三套本地模型'):
        asr_engine.transcribe(None, 0, 1, f'paraformer:{tmp_path}', 'zh', 'cpu', lambda _: None)


def test_english_only_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match='英文音频'):
        asr_engine.transcribe(None, 0, 1, f'paraformer:{tmp_path}', 'en', 'cpu', lambda _: None)
