"""One selected audio range produces a single transcript with acoustic-pause punctuation."""
import ast
from pathlib import Path
from types import SimpleNamespace
from asr_engine import Segment


def _source_functions():
    src = Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    joiner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "join_paused_transcript")
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BatchASRWorker")
    return src, joiner, worker


def test_grouped_recognition_keeps_one_range():
    src, joiner, worker = _source_functions()
    class ThreadStub:
        def __init__(self): pass
        def isInterruptionRequested(self): return False
    class SignalStub:
        def __init__(self, *_): self.values = []
        def emit(self, value): self.values.append(value)
    texts = iter(["百丽集团品牌当中", "思加图是我们高端产品的代表", "也是我们集团"])
    def fake_transcribe(*args):
        return [SimpleNamespace(text=next(texts))]
    ns = {"QThread": ThreadStub, "Signal": SignalStub, "AudioInfo": object,
          "Segment": Segment, "transcribe": fake_transcribe,
          "join_clip_transcript": lambda parts: "".join(parts),
          "timestamp": str, "traceback": __import__("traceback")}
    exec(compile(ast.Module(body=[joiner, worker], type_ignores=[]), str(src), "exec"), ns)
    task = ns["BatchASRWorker"]("voice", [(0.0, 3.2), (3.8, 4.8), (5.1, 9.9)],
                                "model", "zh", "cpu", group_range=(0.0, 9.9))
    task.run()
    assert not task.failed.values
    result = task.succeeded.values[0]
    assert len(result) == 1
    assert result[0][0] is None
    assert result[0][1] == Segment(0.0, 9.9,
        "百丽集团品牌当中，思加图是我们高端产品的代表，也是我们集团")


def test_pause_join_keeps_existing_punctuation():
    src, joiner, _ = _source_functions()
    ns = {}
    exec(compile(ast.Module(body=[joiner], type_ignores=[]), str(src), "exec"), ns)
    assert ns["join_paused_transcript"](["你好，", "世界。"] ) == "你好，世界。"
    assert ns["join_paused_transcript"](["hello", "world"]) == "hello, world"
