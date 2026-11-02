"""Local Paraformer/FunASR and Faster-Whisper transcription backends."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from audio_core import AudioInfo, export_clip
from windows_runtime import configure_cuda_dlls


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


PARAFORMER_DIRS = (
    "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
)
PARAFORMER_PREFIX = "paraformer:"

_cached_model = None
_cached_settings = None


def find_paraformer_root(directory: str | Path) -> Path | None:
    """Accept either the models folder or one of the three model subfolders."""
    path = Path(directory).expanduser()
    for parent in (path, path.parent):
        if all((parent / name).is_dir() for name in PARAFORMER_DIRS):
            return parent.resolve()
    return None


def _load_whisper(model_id: str, device_choice: str, notify: Callable[[str], None]):
    global _cached_model, _cached_settings
    configure_cuda_dlls()
    from faster_whisper import WhisperModel
    import ctranslate2

    model_id = str(Path(model_id).resolve()) if Path(model_id).is_dir() else model_id.strip()
    if not model_id:
        raise ValueError("请选择识别模型或填写本地模型文件夹路径")
    if device_choice == "auto":
        try:
            device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = device_choice
    precision = "float16" if device == "cuda" else "int8"
    key = ("faster-whisper", model_id, device, precision)
    if key != _cached_settings:
        notify(f"正在加载 Faster-Whisper：{model_id}（{device}，{precision}）")
        _cached_model = WhisperModel(model_id, device=device, compute_type=precision)
        _cached_settings = key
    return _cached_model


def _load_paraformer(root: str, device_choice: str, notify: Callable[[str], None]):
    global _cached_model, _cached_settings
    local_root = find_paraformer_root(root)
    if local_root is None:
        raise FileNotFoundError(
            "达摩 ASR 需要完整的三套本地模型（Paraformer / FSMN VAD / 标点）。\n"
            f"当前模型目录：{root}\n请选择 GPT-SoVITS\\tools\\asr\\models。"
        )

    try:
        import torch
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError("本地达摩 ASR 依赖 FunASR 和 PyTorch。请运行 install_paraformer.cmd 安装。") from exc

    gpu_available = torch.cuda.is_available()
    if device_choice == "cuda" and not gpu_available:
        raise RuntimeError("当前 PyTorch 环境无法使用 CUDA。请安装兼容 NVIDIA GPU 的 PyTorch，或选择 CPU。")
    device = "cuda:0" if gpu_available and device_choice != "cpu" else "cpu"
    key = ("paraformer", str(local_root), device)
    if key != _cached_settings:
        notify(f"正在加载本地达摩 ASR（{device}）；模型文件直接读取：{local_root}")
        _cached_model = AutoModel(
            model=str(local_root / PARAFORMER_DIRS[0]),
            vad_model=str(local_root / PARAFORMER_DIRS[1]),
            punc_model=str(local_root / PARAFORMER_DIRS[2]),
            device=device,
            disable_update=True,
            trust_remote_code=False,
            vad_kwargs={"max_single_segment_time": 30000},
        )
        _cached_settings = key
    return _cached_model


def _parse_paraformer_result(
    raw: list[dict], offset: float, duration: float
) -> list[Segment]:
    """FunASR timestamps are in milliseconds; return audio-global seconds."""
    segments: list[Segment] = []
    for item in raw:
        prior_count = len(segments)
        sentence_info = item.get("sentence_info") or []
        for sentence in sentence_info:
            text = str(sentence.get("text") or sentence.get("sentence") or "").strip()
            if not text:
                continue
            start_ms, end_ms = sentence.get("start"), sentence.get("end")
            if start_ms is None or end_ms is None:
                continue
            start = max(0.0, min(duration, float(start_ms) / 1000))
            end = max(start, min(duration, float(end_ms) / 1000))
            segments.append(Segment(offset + start, offset + end, text))
        if len(segments) == prior_count and (text := str(item.get("text") or "").strip()):
            # The installed FunASR/model version may omit sentence timestamps.
            # Keep exact complete text; avoid fabricating sentence timing boundaries.
            segments.append(Segment(offset, offset + duration, text))
    return segments


def transcribe(
    info: AudioInfo,
    start: float,
    end: float,
    model_id: str,
    language: str,
    device: str,
    notify: Callable[[str], None],
    interrupted: Callable[[], bool] = lambda: False,
) -> list[Segment]:
    if model_id.startswith(PARAFORMER_PREFIX):
        if language == "en":
            raise ValueError("达摩 Paraformer 以中文识别为主。英文音频请选择 Faster-Whisper 模型。")
        model = _load_paraformer(model_id[len(PARAFORMER_PREFIX):], device, notify)
        backend = "paraformer"
    else:
        model = _load_whisper(model_id, device, notify)
        backend = "whisper"

    full_file = start <= 0 and end >= info.duration - 1 / info.samplerate
    with TemporaryDirectory(prefix="voicewave_asr_") as tmp:
        if full_file:
            path = info.path
            offset = 0.0
        else:
            path = str(Path(tmp) / "selection.wav")
            notify("正在提取选区用于识别…")
            export_clip(info, start, end, path, asr_temp=True)
            offset = max(0, round(start * info.samplerate)) / info.samplerate
        if interrupted():
            return []
        notify("正在识别语音…")
        if backend == "paraformer":
            raw = model.generate(
                input=path,
                batch_size_s=60,
                sentence_timestamp=True,
            )
            if interrupted():
                return []
            result = _parse_paraformer_result(raw, offset, end - offset)
            notify(f"本地达摩 ASR 完成：{len(result)} 段")
            return result

        segments, detected = model.transcribe(
            path,
            language=None if language == "auto" else language,
            task="transcribe",
            multilingual=(language != "en"),
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=True,
        )
        notify(f"语言：{detected.language}；正在整理带时间戳的识别结果…")
        result: list[Segment] = []
        for item in segments:
            if interrupted():
                return []
            phrase = item.text.strip()
            if phrase:
                result.append(Segment(offset + item.start, offset + item.end, phrase))
        return result
