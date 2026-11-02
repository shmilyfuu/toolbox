"""Exercise the GUI recognition workflow with a source-extracted Qt-free method."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from audio_core import probe
from pause_detection import detect_pauses


def make_wav(path, voices, duration=3):
    sr = 16000
    audio = np.zeros(int(duration * sr), np.float32)
    for start, end in voices:
        audio[round(start * sr):round(end * sr)] = .3
    sf.write(path, audio, sr)
    return probe(str(path))


class FakeApplication:
    @staticmethod
    def setOverrideCursor(value):
        pass

    @staticmethod
    def restoreOverrideCursor():
        pass


class FakeMessageBox:
    @staticmethod
    def critical(*args):
        raise AssertionError(f"Unexpected failure: {args}")

    @staticmethod
    def information(*args):
        raise AssertionError(f"Unexpected information dialog: {args}")

    @staticmethod
    def warning(*args):
        raise AssertionError(f"Unexpected selection warning: {args}")


def make_gui_method():
    source = Path(__file__).resolve().parents[1] / 'main.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MainWindow')
    method = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == 'run_asr')
    ns = {
        'QMessageBox': FakeMessageBox,
        'QApplication': FakeApplication,
        'Qt': SimpleNamespace(WaitCursor=0),
        'detect_pauses': detect_pauses,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)
    return ns['run_asr']


def make_ui(info, selection=(.4, 2.0)):
    obj = SimpleNamespace(
        info=info,
        worker=None,
        clip_ranges=[],
        pause_ranges=[],
        pause_db_spin=SimpleNamespace(value=lambda: -38),
        pause_duration_spin=SimpleNamespace(value=lambda: .4),
        selection=lambda: selection,
        note=lambda msg: None,
        _refresh_overlays=lambda: None,
        _clear_transcript=lambda: None,
    )
    recordings = []
    obj._start_batch_asr = lambda ranges, group_range=None: recordings.append((list(ranges), group_range))
    return obj, recordings


def test_recognize_full_uses_all_detected_pause_clips(tmp_path):
    info = make_wav(tmp_path / 'test.wav', [(0.2, .8), (1.35, 1.85), (2.35, 2.8)])
    obj, jobs = make_ui(info)
    make_gui_method()(obj, False)
    assert len(jobs) == 1
    assert len(jobs[0][0]) == 3
    assert jobs[0][1] == (0.0, info.duration)
    assert obj.clip_ranges == []
    assert len(obj.pause_ranges) == 2


def test_recognize_selection_splits_inside_selected_time_scope(tmp_path):
    info = make_wav(tmp_path / 'test.wav', [(0.2, .8), (1.35, 1.85), (2.35, 2.8)])
    obj, jobs = make_ui(info, (.5, 1.7))
    make_gui_method()(obj, True)
    assert len(jobs[0][0]) == 2
    assert jobs[0][0][0][0] == .5
    assert jobs[0][0][-1][1] == 1.7
    assert jobs[0][1] == (.5, 1.7)
    assert all(.5 <= start < end <= 1.7 for start, end in jobs[0][0])


def test_recognize_full_ignores_previous_clip_list_and_scans_entire_audio(tmp_path):
    info = make_wav(tmp_path / 'test.wav', [(0.2, .8), (1.35, 1.85), (2.35, 2.8)])
    obj, jobs = make_ui(info)
    obj.clip_ranges = [(0.0, .2)]
    make_gui_method()(obj, False)
    assert len(jobs[0][0]) == 3
    assert obj.clip_ranges == [(0.0, .2)]


def test_silence_skips_asr(tmp_path):
    info = make_wav(tmp_path / 'empty.wav', [])
    obj, jobs = make_ui(info)
    make_gui_method()(obj, False)
    assert obj.clip_ranges == []
    assert jobs == []
