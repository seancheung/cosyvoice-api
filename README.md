# CosyVoice OpenAI-Compatible TTS API

一个兼容 OpenAI TTS API 的 CosyVoice 语音合成服务，提供简单易用的 HTTP REST API 接口。

## 功能特点

- ✅ **OpenAI API 兼容**: 完全兼容 OpenAI 的 `/v1/audio/speech` 接口
- 🎙️ **多种合成模式**: 支持零样本复刻、跨语种复刻和自然语言控制
- 🌍 **多语言支持**: 支持中文、英文、日语、粤语等多种语言
- 🔐 **可选的 API Key 认证**: 支持通过环境变量设置 API Key
- 🎨 **自定义声音**: 通过上传音频样本创建自定义声音
- ⚡ **流式传输**: 支持音频流式传输，降低延迟
- 🔧 **灵活配置**: 支持语速控制、多种音频格式

## 系统要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速）
- 预训练模型：CosyVoice3-0.5B（或其他 CosyVoice 模型）

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/your-repo/cosyvoice-api.git
cd cosyvoice-api
```

### 2. 安装依赖

```bash
pip install -r CosyVoice/requirements.txt
pip install fastapi uvicorn soundfile librosa pydantic
```

### 3. 下载模型

确保 `CosyVoice/pretrained_models/CosyVoice3-0.5B` 目录下有完整的模型文件。

### 4. 准备声音文件

在 `voices` 目录下创建声音文件：

```
voices/
  ├── voice1.wav     # 音频样本（至少 3 秒，采样率 >= 16kHz。针对 CosyVoice 3，使用24KHz效果最佳）
  ├── voice1.txt     # 对应的文本内容
  ├── voice2.wav
  └── voice2.txt
```

## 快速开始

### 启动服务

```bash
# 基本启动
python api.py

# 指定端口和主机
python api.py --host 0.0.0.0 --port 8000

# 启用 CORS
python api.py --allow-cors

# 指定模型目录
python api.py --model-dir /path/to/your/model

# 使用批处理脚本（Windows）
run_api.cmd
```

### 环境变量

```bash
# 设置 API Key（可选）
export OPENAI_API_KEY=your-secret-key

# 设置主机和端口
export HOST=0.0.0.0
export PORT=8000
```

## API 使用示例

### 1. 基本调用

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "你好，欢迎使用 CosyVoice 语音合成服务！",
    "voice": "voice1",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### 2. 使用 API Key 认证

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "这是一段需要认证的语音合成请求。",
    "voice": "voice1"
  }' \
  --output speech.wav
```

### 3. 跨语种复刻（中文声音说英文）

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "X-Mode: cross_lingual" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is cross-lingual voice cloning.",
    "voice": "voice1"
  }' \
  --output speech.wav 
```

### 4. 自然语言控制模式

当请求中包含 `instructions` 参数时，API 会自动使用 instruct 模式（除非显式设置了 `X-Mode` 请求头）：

```bash
# 方式一：自动检测（推荐）- 提供 instructions 时会自动使用 instruct 模式
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "今天天气真不错。",
    "voice": "voice1",
    "instructions": "请用温柔甜美的声音朗读"
  }' \
  --output speech.wav

# 方式二：显式指定模式
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "X-Mode: instruct" \
  -d '{
    "model": "tts-1",
    "input": "今天天气真不错。",
    "voice": "voice1",
    "instructions": "请用温柔甜美的声音朗读"
  }' \
  --output speech.wav
```

### 5. 调整语速

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "这是一段快速播放的语音。",
    "voice": "voice1",
    "speed": 1.5
  }' \
  --output speech.wav
```

### 6. 列出可用声音

```bash
curl http://localhost:8000/v1/voices
```

响应示例：
```json
{
  "voices": [
    {
      "id": "voice1",
      "name": "voice1",
      "preview_url": "http://localhost:8000/v1/voices/voice1/preview"
    },
    {
      "id": "voice2",
      "name": "voice2",
      "preview_url": "http://localhost:8000/v1/voices/voice2/preview"
    }
  ]
}
```

### 7. 预览声音

```bash
curl http://localhost:8000/v1/voices/voice1/preview --output preview.wav
```

### 8. 使用ffplay实时播放

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "你好，欢迎使用 CosyVoice 语音合成服务！",
    "voice": "voice1",
    "response_format": "wav"
  }' | ffplay -autoexit -nodisp -i - 
```

## Python 客户端示例

```python
import requests

# 基本调用
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    headers={
        "Content-Type": "application/json",
        # "Authorization": "Bearer your-secret-key"  # 如果需要认证
    },
    json={
        "model": "tts-1",
        "input": "你好，这是一段测试语音。",
        "voice": "voice1",
        "response_format": "wav",
        "speed": 1.0
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)

# 跨语种复刻
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    headers={
        "Content-Type": "application/json",
        "X-Mode": "cross_lingual"
    },
    json={
        "model": "tts-1",
        "input": "Hello, world!",
        "voice": "voice1"
    }
)

# 自然语言控制（方式一：自动检测模式）
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    headers={
        "Content-Type": "application/json",
        # 不需要设置 X-Mode，提供 instructions 会自动使用 instruct 模式
    },
    json={
        "model": "tts-1",
        "input": "今天天气真不错。",
        "voice": "voice1",
        "instructions": "请用欢快的语调朗读"
    }
)

# 自然语言控制（方式二：显式指定模式）
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    headers={
        "Content-Type": "application/json",
        "X-Mode": "instruct"
    },
    json={
        "model": "tts-1",
        "input": "今天天气真不错。",
        "voice": "voice1",
        "instructions": "请用欢快的语调朗读"
    }
)
```

## API 参数说明

### 请求体参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | string | 否 | - | 模型名称（为了兼容性，会被忽略） |
| `input` | string | 是 | - | 要合成的文本（最大 4096 字符） |
| `voice` | string | 是 | - | 声音名称（来自 voices 目录） |
| `instructions` | string | 否 | null | 指令文本（用于自然语言控制，提供此参数时会自动使用 instruct 模式，除非显式设置了 `X-Mode`） |
| `response_format` | string | 否 | "wav" | 音频格式（wav, flac, pcm, mp3, opus, aac） |
| `speed` | float | 否 | 1.0 | 语速（0.5-2.0） |
| `stream_format` | string | 否 | "audio" | 流格式（audio 或 sse，sse 暂不支持） |

### 自定义请求头

| 请求头 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `X-Mode` | string | "zero_shot"<br>（如果提供了 `instructions` 则自动为 "instruct"） | 推理模式：<br>• `zero_shot`: 零样本复刻（3秒极速复刻）<br>• `cross_lingual`: 跨语种复刻<br>• `instruct`: 自然语言控制<br><br>**注意**：如果请求中包含 `instructions` 参数且未设置 `X-Mode`，系统会自动使用 `instruct` 模式 |
| `X-Stream-Inference` | string | "False" | 是否启用流式推理（True/False） |

### 推理模式说明

1. **Zero-Shot（零样本复刻）**：基于提供的音频样本和对应文本，克隆声音特征
2. **Cross-Lingual（跨语种复刻）**：保持音色特征，支持跨语言合成
3. **Instruct（自然语言控制）**：通过自然语言指令控制语音的情感、语调等特征

## 音频格式支持

- ✅ **WAV**: 完全支持
- ✅ **FLAC**: 完全支持
- ✅ **PCM**: 完全支持
- ⚠️ **MP3/OPUS/AAC**: 基础支持（需要 ffmpeg 以获得更好的支持）

## 故障排除

### 1. 模型加载失败

```
RuntimeError: Model directory not found
```

**解决方案**：确保模型目录存在且包含所有必要文件
```bash
ls CosyVoice/pretrained_models/CosyVoice3-0.5B/
# 应该包含: cosyvoice3.yaml, llm.pt, flow.pt, hift.pt 等文件
```

### 2. 声音文件未找到

```
Voice 'xxx' not found
```

**解决方案**：确保 voices 目录包含对应的 .wav 和 .txt 文件
```bash
ls voices/
# voice1.wav voice1.txt voice2.wav voice2.txt
```

### 3. 采样率错误

```
wav sample rate must be greater than 16000
```

**解决方案**：使用 ffmpeg 转换音频采样率
```bash
ffmpeg -i input.wav -ar 16000 output.wav
```

### 4. CUDA 内存不足

**解决方案**：
- 减少批处理大小
- 使用 CPU 模式（模型会自动检测）
- 使用更小的模型

## 性能优化建议

1. **GPU 加速**: 使用 CUDA GPU 可以显著提升合成速度
2. **流式推理**: 启用 `X-Stream-Inference: True` 可以降低首字节时间
3. **音频质量**: 提供高质量的音频样本（清晰、无噪音）可以获得更好的合成效果
4. **文本长度**: 较长的文本会自动分段处理

## 与 OpenAI API 的兼容性

本 API 设计为与 OpenAI TTS API 兼容，可以作为替代品使用。主要区别：

| 特性 | OpenAI API | CosyVoice API |
|------|------------|---------------|
| 端点 | ✅ `/v1/audio/speech` | ✅ `/v1/audio/speech` |
| 认证 | ✅ Bearer Token | ✅ Bearer Token（可选） |
| 基本参数 | ✅ model, input, voice | ✅ 完全兼容 |
| 预设声音 | ✅ 11 种预设 | ⚠️ 自定义声音 |
| 自定义声音 | ❌ 不支持 | ✅ 支持 |
| 推理模式 | ❌ 无 | ✅ 3 种模式 |
| 指令控制 | ✅ instructions | ✅ instructions |

## 命令行参数

```bash
python api.py [OPTIONS]

Options:
  --host TEXT              主机地址（默认: 0.0.0.0）
  --port INTEGER           端口号（默认: 8000）
  --model-dir TEXT         模型目录路径
  --allow-cors            启用 CORS 支持
  --cors-origins TEXT     允许的 CORS 来源（默认: *）
  --help                  显示帮助信息
```

## 项目结构

```
cosyvoice-api/
├── api.py                 # FastAPI 主应用
├── app.py                 # Gradio WebUI（可选）
├── README.md              # 本文件
├── API.md                 # API 接口文档
├── run_api.cmd            # Windows 启动脚本
├── run_app.cmd            # WebUI 启动脚本
├── voices/                # 声音样本目录
│   ├── voice1.wav
│   ├── voice1.txt
│   └── ...
├── CosyVoice/             # CosyVoice 核心库
│   ├── cosyvoice/
│   └── pretrained_models/
│       └── CosyVoice3-0.5B/
└── outputs/               # 生成的音频文件（可选）
```

## 许可证

本项目基于 Apache License 2.0 开源。CosyVoice 模型遵循其自身的许可证。

## 相关链接

- [CosyVoice 官方仓库](https://github.com/FunAudioLLM/CosyVoice)
- [OpenAI TTS API 文档](https://platform.openai.com/docs/guides/text-to-speech)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0.0
- ✅ 实现 OpenAI 兼容的 TTS API
- ✅ 支持 CosyVoice、CosyVoice2 和 CosyVoice3 模型（自动检测）
- ✅ 默认使用 CosyVoice3-0.5B 模型
- ✅ 支持零样本、跨语种和指令控制模式
- ✅ 支持自定义声音
- ✅ 支持多种音频格式
- ✅ 支持 API Key 认证
- ✅ 支持 CORS

