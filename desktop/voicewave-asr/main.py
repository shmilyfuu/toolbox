"""VoiceWave ASR — local WAV editor and Chinese/English transcriber."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import os
import sys
import tempfile
import traceback

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton, QProgressDialog,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from audio_core import AudioInfo, export_clip, probe, waveform_envelope
from asr_engine import Segment, transcribe, find_paraformer_root, PARAFORMER_PREFIX
from player import ClipPlayer
from formatting import timestamp, as_srt_time
from pause_detection import detect_pauses


def join_clip_transcript(texts: list[str]) -> str:
    """Keep Chinese text continuous and separate adjacent English words."""
    parts = [str(text).strip() for text in texts if str(text).strip()]
    if not parts:
        return ""
    output = parts[0]
    for part in parts[1:]:
        if (output[-1].isascii() and output[-1].isalnum()
                and part[0].isascii() and part[0].isalnum()):
            output += " "
        output += part
    return output


def join_paused_transcript(texts: list[str]) -> str:
    """Join speech separated by an acoustic pause in one editable transcript.

    A pause is text formatting, never an instruction to create another audio clip.
    Preserve punctuation from ASR; only add a delimiter when it is absent.
    """
    parts = [str(text).strip() for text in texts if str(text).strip()]
    if not parts:
        return ""
    result = parts[0]
    sentence_ends = "，。、；：！？,.!?;:"
    for part in parts[1:]:
        if result.endswith(tuple(sentence_ends)) or part.startswith(tuple(sentence_ends)):
            glue = " " if result[-1].isascii() and part[0].isascii() and part[0].isalnum() else ""
        else:
            glue = ", " if result[-1].isascii() and part[0].isascii() and part[0].isalnum() else "，"
        result += glue + part
    return result


class BatchASRWorker(QThread):
    message = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, info: AudioInfo, ranges: list[tuple[float, float]], model: str, lang: str, device: str,
                 group_range: tuple[float, float] | None = None):
        super().__init__()
        self.info = info
        self.ranges = list(ranges)
        self.group_range = group_range
        self.model, self.lang, self.device = model, lang, device

    def run(self):
        try:
            completed: list[tuple[int, Segment]] = []
            errors: list[str] = []
            for index, (begin, finish) in enumerate(self.ranges, start=1):
                if self.isInterruptionRequested():
                    return
                self.message.emit(f"批量 ASR：第 {index}/{len(self.ranges)} 段 · {timestamp(begin)} - {timestamp(finish)}")
                try:
                    results = transcribe(
                        self.info, begin, finish, self.model, self.lang, self.device,
                        self.message.emit, self.isInterruptionRequested,
                    )
                    if self.isInterruptionRequested():
                        return
                    # The pause-defined clip is the unit of the transcript. ASR may
                    # return multiple internal sentences; show one editable row per clip.
                    text = join_clip_transcript([result.text for result in results])
                    completed.append((index, Segment(begin, finish, text)))
                except Exception:
                    errors.append(f"片段 {index}:\n{traceback.format_exc()}")
            if self.group_range is not None:
                # One user selection -> one transcript row, with pause punctuation
                # between recognition subranges; the source WAV range is unchanged.
                combined = join_paused_transcript([seg.text for _, seg in completed])
                a, b = self.group_range
                self.succeeded.emit([(None, Segment(a, b, combined))] if combined else [])
            else:
                self.succeeded.emit(completed)
            if errors:
                self.failed.emit(f"{len(errors)} 个片段识别失败；成功的结果已经保留。\n\n" + "\n\n".join(errors))
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoiceWave ASR · 音频裁剪与识别")
        self.resize(1320, 960)
        self.info: AudioInfo | None = None
        self.original_path: str | None = None
        self.temporary_conversion = None
        self.player = ClipPlayer()
        self.worker: BatchASRWorker | None = None
        self.segments: list[Segment] = []
        self.segment_clip_numbers: list[int | None] = []
        self.clip_ranges: list[tuple[float, float]] = []
        self.pause_ranges: list[tuple[float, float]] = []
        self.clip_buttons: list[QPushButton] = []
        self.paraformer_dir = os.environ.get(
            "VOICEWAVE_PARAFORMER_MODELS", r"N:\SoVITS\GPT-SoVITS\tools\asr\models"
        )
        self._changing_selection = False

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(9)

        file_row = QHBoxLayout()
        self.open_btn = QPushButton("打开 WAV…")
        self.open_btn.clicked.connect(self.open_file)
        self.file_label = QLabel("请选择本地 WAV 人声文件")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        file_row.addWidget(self.open_btn)
        file_row.addWidget(self.file_label, 1)
        layout.addLayout(file_row)

        pg.setConfigOptions(antialias=False)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#151923")
        self.plot.setMinimumHeight(260)
        self.plot.setLabel("bottom", "时间", units="s")
        self.plot.setYRange(-1.05, 1.05, padding=0)
        self.plot.hideAxis("left")
        self.plot.showGrid(x=True, y=False, alpha=0.25)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.getViewBox().setMenuEnabled(False)
        self.wave_item = pg.PlotDataItem(pen=pg.mkPen("#76baff", width=1), connect="pairs")
        self.plot.addItem(self.wave_item)
        self.pause_overlay = pg.BarGraphItem(x=[], y=-1.05, width=[], height=2.1,
            brush=pg.mkBrush(150, 150, 150, 45), pen=None)
        self.pause_overlay.setZValue(-10)
        self.plot.addItem(self.pause_overlay)
        self.clip_overlay = pg.BarGraphItem(x=[], y=-1.05, width=[], height=2.1,
            brush=pg.mkBrush(70, 210, 145, 32), pen=None)
        self.clip_overlay.setZValue(-8)
        self.plot.addItem(self.clip_overlay)
        self.region = pg.LinearRegionItem(values=(0, 1),
            brush=pg.mkBrush(54, 157, 255, 55), pen=pg.mkPen("#ffbb57", width=2))
        self.plot.addItem(self.region)
        self.region.sigRegionChanged.connect(self.on_region_changed)
        self.playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#f87171", width=1.5))
        self.playhead.hide()
        self.plot.addItem(self.playhead)
        layout.addWidget(self.plot, 4)

        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("选区开始 (秒)"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setSingleStep(0.1)
        self.start_spin.setKeyboardTracking(False)
        select_row.addWidget(self.start_spin)
        select_row.addWidget(QLabel("结束 (秒)"))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setDecimals(3)
        self.end_spin.setSingleStep(0.1)
        self.end_spin.setKeyboardTracking(False)
        select_row.addWidget(self.end_spin)
        self.start_spin.valueChanged.connect(self.on_spin_changed)
        self.end_spin.valueChanged.connect(self.on_spin_changed)
        select_row.addSpacing(20)
        for label, handler in [
            ("播放选区", self.play_selection),
            ("停止", self.stop_audio),
            ("放大选区", self.zoom_to_selection),
            ("显示全部", self.zoom_all),
            ("导出选区 WAV…", self.save_audio),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(handler)
            select_row.addWidget(btn)
        select_row.addStretch(1)
        layout.addLayout(select_row)

        # Each yellow selection can be saved as a clip; the list supports batch operations.
        pause_row = QHBoxLayout()
        pause_row.addWidget(QLabel("停顿阈值"))
        self.pause_db_spin = QDoubleSpinBox()
        self.pause_db_spin.setRange(-80.0, -10.0)
        self.pause_db_spin.setDecimals(0)
        self.pause_db_spin.setValue(-38)
        self.pause_db_spin.setSuffix(" dBFS")
        pause_row.addWidget(self.pause_db_spin)
        pause_row.addWidget(QLabel("最短停顿"))
        self.pause_duration_spin = QDoubleSpinBox()
        self.pause_duration_spin.setRange(0.10, 5.0)
        self.pause_duration_spin.setSingleStep(0.05)
        self.pause_duration_spin.setDecimals(2)
        self.pause_duration_spin.setSuffix(" 秒")
        self.pause_duration_spin.setValue(0.40)
        pause_row.addWidget(self.pause_duration_spin)
        self.pause_btn = QPushButton("检测停顿并生成片段")
        self.pause_btn.clicked.connect(self.detect_and_split)
        pause_row.addWidget(self.pause_btn)
        pause_row.addWidget(QLabel("停顿区域显示为灰色；可以手动调整自动生成的片段"))
        pause_row.addStretch(1)
        layout.addLayout(pause_row)

        clip_row = QHBoxLayout()
        for caption, callback in (
            ("添加当前选区", self.add_clip),
            ("更新所选片段", self.update_clip),
            ("删除所选片段", self.remove_clip),
            ("清空片段", self.clear_clips),
            ("批量导出 WAV…", self.export_all_clips),
        ):
            button = QPushButton(caption)
            button.clicked.connect(callback)
            self.clip_buttons.append(button)
            clip_row.addWidget(button)
        clip_row.addStretch(1)
        layout.addLayout(clip_row)

        self.clip_table = QTableWidget(0, 4)
        self.clip_table.setHorizontalHeaderLabels(["片段", "开始", "结束", "时长"])
        for column in range(4):
            self.clip_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        self.clip_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.clip_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.clip_table.setSelectionMode(QTableWidget.SingleSelection)
        self.clip_table.setAlternatingRowColors(True)
        self.clip_table.setMinimumHeight(115)
        self.clip_table.cellClicked.connect(self.select_clip_row)
        layout.addWidget(self.clip_table, 2)

        cfg = QHBoxLayout()
        cfg.addWidget(QLabel("语言"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("中文为主（混合英语）", "zh")
        self.lang_combo.addItem("自动检测", "auto")
        self.lang_combo.addItem("英文", "en")
        cfg.addWidget(self.lang_combo)
        cfg.addWidget(QLabel("模型"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(["达摩 ASR（本地 Paraformer）", "large-v3-turbo", "large-v3", "medium", "small"])
        if not find_paraformer_root(self.paraformer_dir):
            self.model_combo.setCurrentIndex(1)
        self.model_combo.setMinimumWidth(180)
        self.model_combo.setToolTip("达摩 ASR 读取本地三套 FunASR 模型；也可填写 Faster-Whisper 模型文件夹路径")
        cfg.addWidget(self.model_combo)
        model_dir_btn = QPushButton("本地模型…")
        model_dir_btn.clicked.connect(self.choose_model_dir)
        cfg.addWidget(model_dir_btn)
        cfg.addWidget(QLabel("计算设备"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("自动（优先 NVIDIA GPU）", "auto")
        self.device_combo.addItem("CUDA GPU", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        cfg.addWidget(self.device_combo)
        cfg.addStretch(1)
        layout.addLayout(cfg)

        asr_row = QHBoxLayout()
        self.recognize_full_btn = QPushButton("识别整段")
        self.recognize_sel_btn = QPushButton("识别选区")
        self.recognize_batch_btn = QPushButton("批量识别片段")
        self.export_text_btn = QPushButton("导出识别结果…")
        self.recognize_full_btn.clicked.connect(lambda: self.run_asr(False))
        self.recognize_sel_btn.clicked.connect(lambda: self.run_asr(True))
        self.recognize_batch_btn.clicked.connect(self.run_batch_asr)
        self.export_text_btn.clicked.connect(self.save_transcript)
        asr_row.addWidget(self.recognize_full_btn)
        asr_row.addWidget(self.recognize_sel_btn)
        asr_row.addWidget(self.recognize_batch_btn)
        asr_row.addWidget(self.export_text_btn)
        asr_row.addWidget(QLabel("点击识别结果行可定位波形；文字单元格可直接修改"))
        asr_row.addStretch(1)
        layout.addLayout(asr_row)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["片段", "起始时间", "结束时间", "识别文本（双击编辑）"])
        for column in range(3):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.cellClicked.connect(self.jump_to_segment)
        layout.addWidget(self.table, 5)

        self.timer = QTimer(self)
        self.timer.setInterval(70)
        self.timer.timeout.connect(self.update_playhead)
        self.statusBar().showMessage("就绪；达摩 ASR 直接使用本地模型，Faster-Whisper 首次使用可能下载权重")

    def note(self, msg: str):
        self.statusBar().showMessage(msg)

    def choose_model_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择本地 GPT-SoVITS ASR models 或 Faster-Whisper 模型目录")
        if path:
            paraformer_root = find_paraformer_root(path)
            if paraformer_root:
                self.paraformer_dir = str(paraformer_root)
                self.model_combo.setCurrentText("达摩 ASR（本地 Paraformer）")
                self.note(f"本地达摩 ASR 模型：{paraformer_root}")
            else:
                self.model_combo.setEditText(path)
                self.note(f"Faster-Whisper 模型目录：{path}")

    def _prepare_wav(self, path: str) -> AudioInfo:
        try:
            return probe(path)
        except Exception as original_error:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                answer = QMessageBox.question(self, "WAV 需要转码",
                    f"当前 WAV 编码无法直接读取：\n{original_error}\n\n可选择已有的 ffmpeg.exe 转换成 PCM WAV 后打开。是否选择？")
                if answer != QMessageBox.Yes:
                    raise RuntimeError("已取消导入。") from original_error
                ffmpeg, _ = QFileDialog.getOpenFileName(self, "选择 ffmpeg.exe", "", "FFmpeg (ffmpeg.exe);;所有文件 (*)")
                if not ffmpeg:
                    raise RuntimeError("已取消 FFmpeg 选择。") from original_error
            else:
                answer = QMessageBox.question(self, "WAV 需要转码",
                    f"当前 WAV 编码无法直接读取：\n{original_error}\n\n使用已找到的 FFmpeg 将其转换成 PCM WAV 后打开？")
                if answer != QMessageBox.Yes:
                    raise RuntimeError("已取消导入。") from original_error
            self.temporary_conversion = tempfile.TemporaryDirectory(prefix="voicewave_wav_")
            converted = str(Path(self.temporary_conversion.name) / "decoded.wav")
            self.note("正在使用 FFmpeg 读取 WAV 编码…")
            subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", path, "-vn", "-c:a", "pcm_f32le", converted],
                           check=True, capture_output=True, text=True)
            return probe(converted)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开人声 WAV", "", "WAV 文件 (*.wav *.wave)")
        if not path:
            return
        self.stop_audio()
        previous_conversion = self.temporary_conversion
        self.temporary_conversion = None
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            info = self._prepare_wav(path)
            x, minimum, maximum = waveform_envelope(info.path)
        except Exception as exc:
            if self.temporary_conversion:
                self.temporary_conversion.cleanup()
            self.temporary_conversion = previous_conversion
            QMessageBox.critical(self, "无法打开音频", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        if previous_conversion:
            previous_conversion.cleanup()
        self.info = info
        self.original_path = path
        self.file_label.setText(f"{path}    ({info.duration:.2f} 秒 / {info.samplerate} Hz / {info.channels} 声道 / {info.subtype})")
        self.file_label.setToolTip(path)
        y = np.empty(minimum.size * 2, dtype=np.float32)
        xs = np.repeat(x, 2)
        y[::2] = minimum
        y[1::2] = maximum
        self.wave_item.setData(x=xs, y=np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), connect="pairs", skipFiniteCheck=True)
        self.region.setBounds((0, info.duration))
        self.start_spin.setRange(0, info.duration)
        self.end_spin.setRange(0, info.duration)
        self.region.setRegion((0, min(10.0, info.duration)))
        self.zoom_all()
        self._clear_transcript()
        self.clip_ranges.clear()
        self.pause_ranges.clear()
        self._refresh_clip_table()
        self._refresh_overlays()
        self.playhead.hide()
        self.note("WAV 已打开；拖动波形上的黄色边界可调整裁剪范围")

    def _clear_transcript(self):
        self.segments.clear()
        self.segment_clip_numbers.clear()
        self.table.setRowCount(0)

    def _refresh_overlays(self):
        for item, ranges in ((self.pause_overlay, self.pause_ranges),
                             (self.clip_overlay, self.clip_ranges)):
            item.setOpts(
                x=[(a + b) / 2 for a, b in ranges],
                width=[b - a for a, b in ranges],
                y=-1.05, height=2.1,
            )

    def _refresh_clip_table(self, selected_range: tuple[float, float] | None = None):
        self.clip_table.blockSignals(True)
        self.clip_table.setRowCount(len(self.clip_ranges))
        for i, (a, b) in enumerate(self.clip_ranges):
            for col, value in enumerate((str(i + 1), timestamp(a), timestamp(b), f"{b-a:.3f} 秒")):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.clip_table.setItem(i, col, item)
        if selected_range and selected_range in self.clip_ranges:
            self.clip_table.selectRow(self.clip_ranges.index(selected_range))
        else:
            self.clip_table.clearSelection()
        self.clip_table.blockSignals(False)

    def _clips_changed(self, selected_range: tuple[float, float] | None = None):
        self.clip_ranges.sort(key=lambda item: (item[0], item[1]))
        self._refresh_clip_table(selected_range)
        self._refresh_overlays()
        # Clip indices in previously generated batch transcripts have changed.
        if any(number is not None for number in self.segment_clip_numbers):
            self._clear_transcript()

    def _clip_row_index(self) -> int:
        row = self.clip_table.currentRow()
        return row if self.clip_table.selectionModel().hasSelection() and 0 <= row < len(self.clip_ranges) else -1

    def select_clip_row(self, row: int, col: int):
        if 0 <= row < len(self.clip_ranges):
            a, b = self.clip_ranges[row]
            self.region.setRegion((a, b))
            self.zoom_to_selection()

    def add_clip(self):
        try:
            clip = self.selection()
            if clip in self.clip_ranges:
                self.note("该片段已在列表中")
                return
            self.clip_ranges.append(clip)
            self._clips_changed(clip)
            self.note(f"已添加片段，共 {len(self.clip_ranges)} 段")
        except ValueError as exc:
            QMessageBox.warning(self, "添加片段失败", str(exc))

    def update_clip(self):
        index = self._clip_row_index()
        if index < 0:
            QMessageBox.information(self, "选择片段", "请先在片段列表中选择一行")
            return
        try:
            clip = self.selection()
            if clip in self.clip_ranges and self.clip_ranges[index] != clip:
                QMessageBox.information(self, "重复片段", "列表中已有相同时间范围")
                return
            self.clip_ranges[index] = clip
            self._clips_changed(clip)
            self.note("所选片段时间已更新")
        except ValueError as exc:
            QMessageBox.warning(self, "更新片段失败", str(exc))

    def remove_clip(self):
        index = self._clip_row_index()
        if index < 0:
            QMessageBox.information(self, "选择片段", "请先在片段列表中选择一行")
            return
        del self.clip_ranges[index]
        self._clips_changed()
        self.note(f"已删除片段，剩余 {len(self.clip_ranges)} 段")

    def clear_clips(self):
        self.clip_ranges.clear()
        self._clips_changed()
        self.note("片段列表已清空")

    def detect_and_split(self):
        if self.info is None:
            QMessageBox.information(self, "缺少音频", "请先打开 WAV 文件")
            return
        try:
            self.note("正在分析音频的停顿位置…")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            analysis = detect_pauses(self.info,
                threshold_db=self.pause_db_spin.value(),
                min_silence=self.pause_duration_spin.value())
        except Exception as exc:
            QMessageBox.critical(self, "停顿检测失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        # Detection replaces the current clip plan so repeated analysis is predictable.
        self.pause_ranges = analysis.pauses
        self.clip_ranges = list(analysis.clips)
        self._clips_changed()
        if self.clip_ranges:
            self.region.setRegion(self.clip_ranges[0])
            self.clip_table.selectRow(0)
        self.note(f"检测到 {len(self.pause_ranges)} 处停顿，生成 {len(self.clip_ranges)} 个片段；可拖动黄色选区微调")

    def export_all_clips(self):
        if self.info is None or not self.clip_ranges:
            QMessageBox.information(self, "没有片段", "请先添加片段或检测停顿")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择批量 WAV 输出文件夹", str(Path(self.original_path).parent))
        if not folder:
            return
        stem = Path(self.original_path).stem
        destinations = [Path(folder) / f"{stem}_{i:03d}_{round(a*1000):09d}-{round(b*1000):09d}.wav"
                        for i, (a, b) in enumerate(self.clip_ranges, 1)]
        if any(path.exists() for path in destinations):
            answer = QMessageBox.question(self, "文件已存在", "输出目录已有同名片段文件，是否覆盖？")
            if answer != QMessageBox.Yes:
                return
        progress = QProgressDialog("正在导出音频片段…", "取消", 0, len(self.clip_ranges), self)
        progress.setWindowTitle("批量导出 WAV")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        completed = 0
        try:
            for i, ((a, b), dest) in enumerate(zip(self.clip_ranges, destinations), 1):
                if progress.wasCanceled():
                    break
                export_clip(self.info, a, b, str(dest))
                completed += 1
                progress.setValue(i)
        except Exception as exc:
            QMessageBox.critical(self, "批量导出失败", f"已完成 {completed} 段。\n{exc}")
            return
        finally:
            progress.close()
        self.note(f"批量导出完成：{completed}/{len(self.clip_ranges)} 段 → {folder}")

    def on_region_changed(self, *_):
        if self._changing_selection or not self.info:
            return
        self._changing_selection = True
        try:
            a, b = self.region.getRegion()
            self.start_spin.setValue(max(0, a))
            self.end_spin.setValue(min(self.info.duration, b))
        finally:
            self._changing_selection = False

    def on_spin_changed(self, *_):
        if self._changing_selection or not self.info:
            return
        a, b = self.start_spin.value(), self.end_spin.value()
        if b <= a:
            return
        self._changing_selection = True
        try:
            self.region.setRegion((a, b))
        finally:
            self._changing_selection = False

    def selection(self) -> tuple[float, float]:
        if not self.info:
            raise ValueError("请先打开 WAV 文件")
        a, b = self.region.getRegion()
        if b - a < 1 / self.info.samplerate:
            raise ValueError("选区过短，请调整开始和结束时间")
        return a, b

    def zoom_all(self):
        if self.info:
            self.plot.setXRange(0, self.info.duration, padding=0.015)

    def zoom_to_selection(self):
        if self.info:
            a, b = self.selection()
            pad = max(.15, (b - a) * .08)
            self.plot.setXRange(max(0, a - pad), min(self.info.duration, b + pad), padding=0)

    def play_selection(self):
        try:
            a, b = self.selection()
            self.player.start(self.info, a, b)
            self.playhead.setValue(a)
            self.playhead.show()
            self.timer.start()
            self.note(f"正在播放：{timestamp(a)} ~ {timestamp(b)}")
        except Exception as exc:
            QMessageBox.warning(self, "播放失败", str(exc))

    def stop_audio(self):
        self.timer.stop()
        self.player.stop()
        self.playhead.hide()

    def update_playhead(self):
        self.playhead.setValue(self.player.position_seconds)
        if not self.player.active:
            self.stop_audio()

    def save_audio(self):
        try:
            a, b = self.selection()
        except Exception as exc:
            QMessageBox.warning(self, "无法导出", str(exc))
            return
        base = Path(self.original_path).stem
        suggested = str(Path(self.original_path).with_name(f"{base}_{a:.2f}-{b:.2f}.wav"))
        dest, _ = QFileDialog.getSaveFileName(self, "导出选区 WAV", suggested, "WAV 文件 (*.wav)")
        if dest:
            try:
                export_clip(self.info, a, b, dest)
                self.note(f"已导出音频：{dest}")
            except Exception as exc:
                QMessageBox.critical(self, "音频导出失败", str(exc))

    def run_asr(self, selection_only: bool):
        if self.info is None:
            QMessageBox.information(self, "缺少音频", "请先打开 WAV 文件")
            return
        if self.worker and self.worker.isRunning():
            return
        try:
            a, b = self.selection() if selection_only else (0.0, self.info.duration)
        except ValueError as exc:
            QMessageBox.warning(self, "无效选区", str(exc))
            return

        try:
            self.note("正在检测停顿以划分 ASR 识别片段…")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            analysis = detect_pauses(
                self.info,
                threshold_db=self.pause_db_spin.value(),
                min_silence=self.pause_duration_spin.value(),
            )
            ranges = [
                (max(a, clip_start), min(b, clip_end))
                for clip_start, clip_end in analysis.clips
                if min(b, clip_end) - max(a, clip_start) >= 1 / self.info.samplerate
            ]
        except Exception as exc:
            QMessageBox.critical(self, "停顿检测失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        # Recognition respects the user's complete time range. Acoustic pauses
        # format its text; the explicit clip list and selected WAV stay intact.
        self.pause_ranges = [
            (max(a, pause_start), min(b, pause_end))
            for pause_start, pause_end in analysis.pauses
            if min(b, pause_end) > max(a, pause_start)
        ]
        self._refresh_overlays()
        if not ranges:
            self._clear_transcript()
            self.note("当前范围未检测到可识别的语音片段；可以调整停顿阈值")
            return
        self._start_batch_asr(ranges, group_range=(a, b))

    def _selected_model(self) -> str:
        model_name = self.model_combo.currentText().strip()
        return PARAFORMER_PREFIX + self.paraformer_dir if model_name == "达摩 ASR（本地 Paraformer）" else model_name

    def _set_asr_busy(self, busy: bool):
        self.open_btn.setEnabled(not busy)
        self.recognize_full_btn.setEnabled(not busy)
        self.recognize_sel_btn.setEnabled(not busy)
        self.recognize_batch_btn.setEnabled(not busy)
        self.pause_btn.setEnabled(not busy)
        for button in self.clip_buttons:
            button.setEnabled(not busy)

    def run_batch_asr(self):
        if self.info is None or not self.clip_ranges:
            QMessageBox.information(self, "没有片段", "请先添加片段或检测停顿")
            return
        if self.worker and self.worker.isRunning():
            return
        self._start_batch_asr(self.clip_ranges)

    def _start_batch_asr(self, ranges: list[tuple[float, float]],
                         group_range: tuple[float, float] | None = None):
        self.stop_audio()
        self._clear_transcript()
        self._set_asr_busy(True)
        self.worker = BatchASRWorker(self.info, ranges, self._selected_model(),
            self.lang_combo.currentData(), self.device_combo.currentData(), group_range=group_range)
        self.worker.message.connect(self.note)
        self.worker.succeeded.connect(self.receive_batch_segments)
        self.worker.failed.connect(self.asr_error)
        self.worker.finished.connect(self.asr_finished)
        self.worker.start()
        self.note(f"准备识别 {len(ranges)} 个语音区间；" +
                  ("将停顿整合为一条文本…" if group_range else "每个片段输出一行…"))

    def asr_finished(self):
        self._set_asr_busy(False)

    def _show_segments(self, segments: list[Segment], clip_numbers: list[int | None]):
        self.segments = segments
        self.segment_clip_numbers = clip_numbers
        self.table.setRowCount(len(segments))
        for row, (seg, clip) in enumerate(zip(segments, clip_numbers)):
            for col, value in enumerate((str(clip) if clip is not None else "整段/选区",
                                         timestamp(seg.start), timestamp(seg.end), seg.text)):
                item = QTableWidgetItem(value)
                if col < 3:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.note(f"识别完成：{len(segments)} 条结果，可编辑并导出 TXT / SRT / JSON。")

    def receive_batch_segments(self, results: list[tuple[int | None, Segment]]):
        self._show_segments([seg for _, seg in results], [index for index, _ in results])

    def asr_error(self, description: str):
        self.note("识别失败；请查看错误信息")
        QMessageBox.critical(self, "ASR 出错", description[-7000:])

    def jump_to_segment(self, row: int, col: int):
        if row >= len(self.segments) or not self.info:
            return
        seg = self.segments[row]
        self.region.setRegion((seg.start, min(self.info.duration, seg.end)))
        self.zoom_to_selection()

    def edited_segments(self) -> list[Segment]:
        result = []
        for row, seg in enumerate(self.segments):
            item = self.table.item(row, 3)
            text = item.text().strip() if item else seg.text
            if text:
                result.append(Segment(seg.start, seg.end, text))
        return result

    def save_transcript(self):
        rows = self.edited_segments()
        if not rows:
            QMessageBox.information(self, "没有识别结果", "请先运行 ASR 识别")
            return
        base = Path(self.original_path).stem if self.original_path else "transcript"
        dest, selected = QFileDialog.getSaveFileName(
            self, "导出识别结果", str(Path(self.original_path).with_name(base + "_transcript.txt")),
            "纯文本 (*.txt);;SRT 字幕 (*.srt);;JSON 时间戳 (*.json)", "纯文本 (*.txt)")
        if not dest:
            return
        suffix = Path(dest).suffix.lower()
        if suffix not in {".txt", ".srt", ".json"}:
            suffix = {"纯文本 (*.txt)": ".txt", "SRT 字幕 (*.srt)": ".srt",
                      "JSON 时间戳 (*.json)": ".json"}.get(selected, ".txt")
            dest += suffix
        try:
            if suffix == ".json":
                source_rows = [(seg, number) for row, (seg, number) in enumerate(zip(self.segments, self.segment_clip_numbers))
                               if (item := self.table.item(row, 3)) and item.text().strip()]
                data = json.dumps([dict(start=seg.start, end=seg.end, text=edited.text, clip=number)
                                   for (seg, number), edited in zip(source_rows, rows)], ensure_ascii=False, indent=2)
            elif suffix == ".srt":
                data = "\n\n".join(f"{i}\n{as_srt_time(s.start)} --> {as_srt_time(s.end)}\n{s.text}"
                    for i, s in enumerate(rows, 1)) + "\n"
            else:
                if any(number is not None for number in self.segment_clip_numbers):
                    lines = []
                    previous = None
                    for row, (seg, number) in enumerate(zip(self.segments, self.segment_clip_numbers)):
                        item = self.table.item(row, 3)
                        phrase = item.text().strip() if item else seg.text
                        if not phrase:
                            continue
                        if number != previous:
                            if lines:
                                lines.append("")
                            lines.append(f"片段 {number:03d}")
                            previous = number
                        lines.append(phrase)
                    data = "\n".join(lines) + "\n"
                else:
                    data = "\n".join(s.text for s in rows) + "\n"
            Path(dest).write_text(data, encoding="utf-8-sig" if suffix == ".txt" else "utf-8")
            self.note(f"已导出识别结果：{dest}")
        except Exception as exc:
            QMessageBox.critical(self, "文本导出失败", str(exc))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "识别正在运行", "请等待当前识别结束后关闭程序")
            event.ignore()
            return
        self.stop_audio()
        if self.temporary_conversion:
            self.temporary_conversion.cleanup()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VoiceWave ASR")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
