# VoiceWave ASR

Windows 本地 WAV 波形编辑、停顿检测与语音识别工具，支持中文及中英混合音频，提供 Faster-Whisper 和本地 FunASR Paraformer 识别后端。

## 运行环境

Windows 10/11、Python 3.11、Miniconda。

```cmd
install.cmd
install_paraformer.cmd
start.cmd
```

`install.cmd` 创建位于程序目录**上一级**的 `VoiceWaveASREnv` Conda 环境并安装基础依赖。使用本地 Paraformer 时执行 `install_paraformer.cmd` 安装 PyTorch（CUDA 12.8）、TorchAudio、FunASR 和 ModelScope。仅使用 Faster-Whisper 时可省略第二步。安装需要网络，模型权重按所选后端自行管理。程序可以在已经配置完成的 Python 环境中运行 `python main.py`。

## 本地 ASR 模型

使用 GPT-SoVITS 已下载的达摩模型时，在界面点击「本地模型…」选择其 `tools\asr\models` 文件夹。该目录需要包含 Paraformer、FSMN VAD 和标点模型三个子目录。程序只引用模型目录，不复制权重。

也可以设置环境变量 `VOICEWAVE_PARAFORMER_MODELS` 指向模型目录。用户已有的 `N:\SoVITS\GPT-SoVITS\tools\asr\models` 仍作为默认候选路径；其他机器可直接通过界面选择模型目录。

使用 Faster-Whisper 时选择识别模型名称或兼容的本地模型目录。中文为主选 `zh`，英文选 `en`，也提供自动语言检测。混合语音需人工校对。

## 音频裁剪与识别

1. 「打开 WAV」读取本地文件。可拖动波形选区或输入开始与结束秒数；可试听与导出当前选区 WAV。
2. 「检测停顿并生成片段」根据静音阈值和最短停顿长度创建片段列表。片段可单独调整、添加、删除与批量导出 WAV。
3. 「识别选区」将选中音频作为一条识别记录。内部利用停顿区间进行逐段 ASR，在合并文字时插入逗号，识别结果使用所选音频原本的起止时间。
4. 「识别整段」处理整个文件，并生成一条识别记录。
5. 「批量识别片段」根据片段列表分别识别，每个片段一条记录。可双击编辑文字，并导出 TXT、SRT 或 JSON。

停顿检测根据音量阈值判断静音，默认 `-38 dBFS` 与 `0.40 秒`。处理含底噪的音频时可能需要手动调节阈值。文字中的停顿标点按照声学停顿添加，不能代替语义标点校对。

音频处理使用原始采样率、声道以及可兼容的 WAV subtype。遇到不兼容的 WAV 格式，可通过本机 FFmpeg 转换后导入。识别在本地进行。

## 测试

```cmd
python -m pip install pytest
python -m pytest -q
```

## 参考项目

ASR 后端通过公开 Python API 调用 [FunASR](https://github.com/modelscope/FunASR) 和 [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)。模型目录约定参考 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)。模型权重和第三方依赖遵循各自许可。
