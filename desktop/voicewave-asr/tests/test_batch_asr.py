"""Batch worker regression tests without importing a Qt GUI in CI."""
import ast
from pathlib import Path
from types import SimpleNamespace
from asr_engine import Segment


def test_batch_worker_processes_all_clips_with_global_timing():
    source = Path(__file__).resolve().parents[1] / 'main.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BatchASRWorker')

    class ThreadStub:
        def __init__(self):
            pass
        def isInterruptionRequested(self):
            return False

    class SignalStub:
        def __init__(self, *_):
            self.values = []
        def emit(self, value):
            self.values.append(value)

    def fake_transcribe(info, begin, end, model, language, device, *args):
        assert model == 'local-model'
        assert end > begin
        return [SimpleNamespace(start=begin + .1, end=end - .1, text='测试')]

    ns = {'QThread': ThreadStub, 'Signal': SignalStub, 'AudioInfo': object,
          'Segment': Segment, 'transcribe': fake_transcribe,
          'join_clip_transcript': lambda parts: ''.join(parts),
          'timestamp': lambda value: f'{value:.2f}', 'traceback': __import__('traceback')}
    exec(compile(ast.Module(body=[worker], type_ignores=[]), str(source), 'exec'), ns)
    obj = ns['BatchASRWorker']('voice', [(.2, .8), (1.4, 2.1)], 'local-model', 'zh', 'auto')
    obj.run()
    assert not obj.failed.values
    grouped = obj.succeeded.values[0]
    assert [(i, round(x.start, 2), round(x.end, 2)) for i, x in grouped] == [
        (1, .2, .8), (2, 1.4, 2.1)]


def test_batch_worker_does_not_mask_thread_start():
    source = Path(__file__).resolve().parents[1] / 'main.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BatchASRWorker')

    class ThreadStub:
        def __init__(self): pass
        def start(self): return 'running'

    class SignalStub:
        def __init__(self, *_): pass

    ns = {'QThread': ThreadStub, 'Signal': SignalStub, 'AudioInfo': object, 'Segment': object}
    exec(compile(ast.Module(body=[worker], type_ignores=[]), str(source), 'exec'), ns)
    obj = ns['BatchASRWorker']('voice', [(0, 1)], 'model', 'zh', 'auto')
    assert obj.start() == 'running'


def test_batch_worker_keeps_successes_when_one_clip_fails():
    source = Path(__file__).resolve().parents[1] / 'main.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BatchASRWorker')

    class ThreadStub:
        def __init__(self): pass
        def isInterruptionRequested(self): return False

    class SignalStub:
        def __init__(self, *_): self.values = []
        def emit(self, value): self.values.append(value)

    def fake_transcribe(info, begin, end, model, lang, device, *args):
        if begin == 0:
            raise RuntimeError('bad clip')
        return [SimpleNamespace(start=begin, end=end, text='ok')]

    ns = {'QThread': ThreadStub, 'Signal': SignalStub, 'AudioInfo': object,
          'Segment': Segment, 'transcribe': fake_transcribe,
          'join_clip_transcript': lambda parts: ''.join(parts),
          'timestamp': str, 'traceback': __import__('traceback')}
    exec(compile(ast.Module(body=[worker], type_ignores=[]), str(source), 'exec'), ns)
    obj = ns['BatchASRWorker']('voice', [(0, 1), (2, 3)], 'local-model', 'zh', 'auto')
    obj.run()
    assert len(obj.succeeded.values[0]) == 1
    assert obj.succeeded.values[0][0][0] == 2
    assert '片段 1' in obj.failed.values[0]


def test_batch_worker_combines_multiple_internal_sentences_per_pause_clip():
    source = Path(__file__).resolve().parents[1] / 'main.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BatchASRWorker')
    join = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'join_clip_transcript')

    class ThreadStub:
        def __init__(self): pass
        def isInterruptionRequested(self): return False

    class SignalStub:
        def __init__(self, *_): self.values = []
        def emit(self, value): self.values.append(value)

    ns = {'QThread': ThreadStub, 'Signal': SignalStub, 'AudioInfo': object,
          'Segment': Segment, 'timestamp': str, 'traceback': __import__('traceback'),
          'transcribe': lambda *args: [SimpleNamespace(text='你好'), SimpleNamespace(text='world'), SimpleNamespace(text='again')]}
    exec(compile(ast.Module(body=[join, worker], type_ignores=[]), str(source), 'exec'), ns)
    obj = ns['BatchASRWorker']('voice', [(1, 3)], 'local-model', 'zh', 'auto')
    obj.run()
    assert not obj.failed.values
    assert len(obj.succeeded.values[0]) == 1
    index, segment = obj.succeeded.values[0][0]
    assert index == 1
    assert (segment.start, segment.end) == (1, 3)
    assert segment.text == '你好world again'
