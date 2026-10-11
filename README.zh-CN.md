# MSST API

[English](README.md) | **简体中文**

面向 [Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training)
的 GPU 推理 HTTP 服务。提交一段音频和一个模型，返回分离后的各个音轨。

- `POST /api/msst/inference`：表单推理，支持指定 MSST 模型及其推理参数。
- `POST /api/rvc/inference`：**RVC 声音转换**——把一段音频转换成已安装 RVC 音色模型的音色。
- `POST /api/asr/transcribe`：**ASR 单句识别**（Qwen3-ASR / PinyinASR / RomajiASR）。
- `WS /api/asr/streaming`：**流式 ASR**——通过 WebSocket 实时识别（仅 Qwen3-ASR）。
- `POST /api/v2m/inference`：**Vocal2Midi**——人声转“带歌词对齐的 MIDI”（及 USTX/VSQX/TextGrid），以 zip 返回。
- 缺失的 MSST 模型在首次推理时**自动下载**，支持 **ModelScope（默认）** 与 **Hugging Face**；
  RVC 音色模型从 `MSST_RVC_MODEL_DIR`（默认 `models/rvc_models`）读取。
- 模型存放路径通过环境变量 `MSST_MODEL_DIR` 指定。
- 使用 [uv](https://docs.astral.sh/uv/) 管理依赖，默认**仅支持 GPU 推理**。
- 内置 51 个来自 MSST 生态的预训练模型（人声/伴奏、单音轨、多音轨）。

> 参考实现：[MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI)、
> [RVCSVC-API-MSST](https://github.com/sdfsfsk/RVCSVC-API-MSST)、
> [applio-api-plugin](https://github.com/Deliay/applio-api-plugin)、
> [Applio](https://github.com/IAHispano/Applio)。

---

## 目录结构

```
.
├── Dockerfile                     # CUDA 12.8 runtime + uv
├── pyproject.toml / uv.lock       # uv 依赖与锁文件
├── scripts/
│   ├── build_registry.py          # 由上游目录生成 registry.json
│   └── models_info.json           # 上游模型目录（MSST-WebUI）
└── src/msst_api/
    ├── main.py                    # FastAPI 应用与路由
    ├── separator.py               # MSST 推理封装 + 模型 LRU 缓存
    ├── rvc_engine.py              # RVC 音色发现 + 转换器 LRU 缓存
    ├── rvc/                       # 内置 RVC 推理引擎（移植自 Applio）
    │   ├── infer/                 # VoiceConverter 与转换流程
    │   ├── lib/                   # 网络结构、F0 预测器（rmvpe/crepe/fcpe）、工具
    │   └── configs/               # 采样率预设
    ├── asr/                       # ASR 能力（asr + streaming_asr）
    │   ├── catalog.py             # Qwen3-ASR / PinyinASR / RomajiASR 目录
    │   ├── engines.py             # Qwen3-ASR + ONNX CTC 引擎与流式会话
    │   └── manager.py             # 共享缓存模型管理器
    ├── v2m/                       # Vocal2Midi 能力
    │   ├── catalog.py             # GAME / TiFA / HubertFA / RMVPE / kashi-g2p
    │   ├── manager.py             # 目录 + 存储 + 共享缓存
    │   ├── pipeline.py            # 端到端流程（zip 输出）
    │   ├── capabilities.py        # 分能力接口（切片/音高/音符/对齐/G2P）
    │   ├── runtime.py             # 内置 loader 与共享缓存的桥接
    │   └── _vendor/               # 内置 Vocal2Midi 推理运行时（Apache-2.0）
    ├── model_store.py             # 多文件模型包 + 快照下载
    ├── download.py                # ModelScope / HuggingFace / URL 下载
    ├── registry.py                # 模型目录读取
    ├── audio.py                   # 音频解码/编码
    ├── config.py                  # 环境变量配置
    └── data/
        ├── registry.json          # 生成的模型目录
        └── configs/               # 每个 checkpoint 对应的 YAML 配置
```

---

## 快速开始（Docker，推荐）

前置条件：已安装 NVIDIA 驱动与 `nvidia-container-toolkit`。

```bash
# 构建镜像
docker build -t msst-api:latest .

# 运行（挂载模型目录，使用 GPU）
docker run --gpus all -p 8000:8000 \
  -v /data/msst-models:/models \
  -e MSST_DOWNLOAD_BACKEND=modelscope \
  msst-api:latest
```

打开 <http://localhost:8000/docs> 查看交互式 API 文档。

> 镜像内的 Ubuntu apt、PyPI 与 PyTorch 均已指向国内镜像
> （清华 apt/PyPI、SJTU CUDA 12.8）。

---

## API 使用

### 1. 列出可用模型

```bash
curl http://localhost:8000/api/msst/models
curl "http://localhost:8000/api/msst/models?downloaded=false"
```

### 2. 推理（默认返回 zip）

```bash
curl -X POST http://localhost:8000/api/msst/inference \
  -F "audio=@song.wav" \
  -F "model=model_bs_roformer_ep_317_sdr_12.9755.ckpt" \
  -F "output_format=wav" \
  -o stems.zip
```

返回的 `stems.zip` 内含 `vocals.wav`、`instrumental.wav` 等音轨与 `manifest.json`。

### 3. 推理（返回 JSON + base64）

```bash
curl -X POST http://localhost:8000/api/msst/inference \
  -F "audio=@song.wav" \
  -F "model=model_bs_roformer_ep_317_sdr_12.9755.ckpt" \
  -F "response_format=json" \
  -F "output_format=flac"
```

### 4. Python 客户端

```python
import io, zipfile, requests

with open("song.wav", "rb") as fh:
    resp = requests.post(
        "http://localhost:8000/api/msst/inference",
        files={"audio": ("song.wav", fh, "audio/wav")},
        data={
            "model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "num_overlap": 4,
            "batch_size": 2,
            "use_tta": "false",
            "output_format": "wav",
        },
        timeout=3600,
    )
resp.raise_for_status()

with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
    for name in zf.namelist():
        print(name)
        zf.extract(name, "stems")
```

### `POST /api/msst/inference` 表单字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `audio` | file（必填） | 输入音频，支持 wav/flac/mp3/m4a 等 |
| `model` | str | 目录中的模型 id（`GET /api/msst/models` 获取） |
| `model_type` | str | 自定义模型的架构 key（如 `bs_roformer`） |
| `config` / `config_url` | file / url | 自定义模型的 YAML 配置 |
| `checkpoint` / `checkpoint_url` | file / url | 自定义模型的权重 |
| `device` | str | 默认取 `MSST_DEVICE`（`cuda:0`） |
| `batch_size` | int | 覆盖配置中的 `inference.batch_size` |
| `num_overlap` | int | 覆盖 `inference.num_overlap`（越大越平滑、越慢） |
| `chunk_size` | int | 覆盖 `audio.chunk_size` |
| `normalize` | bool | 覆盖 `inference.normalize` |
| `use_tta` | bool | 测试时增强（更慢，通常更稳） |
| `bigshifts` | int | BigShifts 位移平均次数（默认 1） |
| `extract_instrumental` | bool | 额外输出 `instrumental`（mix − vocals） |
| `stems` | str | 以逗号分隔的输出音轨子集，如 `vocals,instrumental` |
| `output_format` | str | `wav`（默认）/ `flac` / `mp3` |
| `bit_depth` | str | 如 `PCM_16`、`PCM_24`、`FLOAT`、`320k` |
| `response_format` | str | `zip`（默认）/ `json` |
| `source` | str | 本次请求的下载后端：`modelscope` / `huggingface` / `auto` |

### 其它端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查、CUDA 状态、已加载模型 |
| `GET` | `/api/msst/models` | 模型列表 |
| `GET` | `/api/msst/models/{id}` | 模型详情 |
| `POST` | `/api/msst/models/{id}/download` | 预下载某个模型 |

---

## RVC 声音转换

`POST /api/rvc/inference` 会把输入音频转换成已安装 RVC 音色模型的音色，并直接在
响应体中返回转换后的音频（全程内存处理，不落盘）。

RVC 音色模型放在 `MSST_RVC_MODEL_DIR`（默认 `<MSST_MODEL_DIR>/rvc_models`，即
`models/rvc_models`）。每个音色是一个扁平的 `<name>.pth` 文件，可搭配同名的
`<name>.index`（按名称自动匹配）：

```
models/rvc_models/
├── alice.pth
├── alice.index
└── bob_v2.pth
```

辅助资源（ContentVec 特征提取器与 RMVPE 音高预测器）会在**首次使用时自动下载**
到 `MSST_RVC_ASSET_DIR`（默认 `<MSST_MODEL_DIR>/rvc_assets`）。FCPE 已内置于
`torchfcpe`，因此 `f0_method=fcpe` 无需额外下载。

### 1. 列出音色

```bash
curl http://localhost:8000/api/rvc/models
curl http://localhost:8000/api/rvc/models/alice
```

### 2. 转换

```bash
curl -X POST http://localhost:8000/api/rvc/inference \
  -F "audio=@input.wav" \
  -F "model=alice" \
  -F "f0_method=rmvpe" \
  -F "pitch=0" \
  -F "output_format=wav" \
  -o output.wav
```

响应体即为转换后的音频，`Content-Type` 随 `output_format` 变化；响应头
`X-RVC-Model`、`X-RVC-Elapsed` 分别标识音色与耗时。

### `POST /api/rvc/inference` 表单字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `audio` | file（必填） | 输入音频，支持 wav/flac/mp3/m4a 等（别名 `file`、`input`） |
| `model` | str | 音色 id/名称（来自 `GET /api/rvc/models`，别名 `model_name`、`pth`） |
| `index` | str | 索引文件名；省略时按音色自动匹配 |
| `device` | str | 默认取 `MSST_DEVICE`（`cuda:0`） |
| `pitch` | int | 变调半音数（默认 `0`） |
| `f0_method` | str | `rmvpe`（默认）/ `crepe` / `crepe-tiny` / `fcpe` |
| `index_rate` | float | 索引混合比例（默认 `0.75`） |
| `volume_envelope` | float | 音量包络混合（默认 `1.0`） |
| `protect` | float | 清辅音保护（默认 `0.5`） |
| `sid` | int | 说话人 id（默认 `0`） |
| `split_audio` | bool | 是否按静音切分长音频 |
| `f0_autotune` / `f0_autotune_strength` | bool / float | 电音修音 |
| `proposed_pitch` / `proposed_pitch_threshold` | bool / float | 自动向目标频率变调 |
| `clean_audio` / `clean_strength` | bool / float | 输出前降噪 |
| `resample_sr` | int | 重采样到指定采样率（`0` 表示关闭） |
| `embedder_model` / `embedder_model_custom` | str / str | 特征提取器（默认 `contentvec`） |
| `formant_shifting` / `formant_qfrency` / `formant_timbre` | bool / float / float | 共振峰偏移 |
| `post_process` | bool | 启用 pedalboard 效果链 |
| 效果参数 | | `reverb`、`pitch_shift`、`limiter`、`gain`、`distortion`、`chorus`、`bitcrush`、`clipping`、`compressor`、`delay` 及各自参数 |
| `output_format` | str | `wav`（默认）/ `mp3` / `flac` / `ogg` / `opus` / `m4a` / `aac` / `aiff` / `ac3` |

> 若想对**分离出的音轨**（如 MSST 的 vocals）做 RVC，先调用
> `/api/msst/inference`，取出 `vocals.wav`，再提交到 `/api/rvc/inference`。

---

## ASR（`asr`）与流式 ASR（`streaming_asr`）

按能力暴露两个接口，支持模型各不相同：

| 能力 | 接口 | 支持模型 |
| --- | --- | --- |
| `asr`（单句） | `POST /api/asr/transcribe` | Qwen3-ASR、PinyinASR、RomajiASR |
| `streaming_asr`（流式） | `WS /api/asr/streaming` | 仅 Qwen3-ASR |

模型包放在 `MSST_ASR_MODEL_DIR`（默认 `<MSST_MODEL_DIR>/asr`）。Qwen3-ASR 首次使用时
从 Hub 自动下载；PinyinASR / RomajiASR 的 ONNX 包则从其 GitHub Release 资产
（`model_onnx.zip`）自动下载并解压。也可直接放入该目录，或通过自定义
`MSST_ASR_CATALOG` 配置镜像源。

```bash
curl http://localhost:8000/api/asr/models
curl "http://localhost:8000/api/asr/models?task=streaming"     # 仅 Qwen3-ASR

curl -X POST http://localhost:8000/api/asr/transcribe \
  -F "audio=@vocal.wav" -F "model=qwen3-asr-1.7b" -F "language=zh"
# PinyinASR：-F "model=pinyin-asr"；RomajiASR：-F "model=romaji-asr"
```

流式接口为 WebSocket：连接
`ws://localhost:8000/api/asr/streaming?model=qwen3-asr-1.7b&language=zh`，发送
float32 小端 PCM（或 `format=pcm_s16le` 的 int16）二进制帧；每个音频块返回
`{"type":"partial",...}`，发送 `finish` 后返回 `{"type":"final",...}`。真正的增量解码
使用 Qwen3-ASR 的 **vLLM** 后端（`pip install "qwen-asr[vllm]"`）；未安装 vLLM 时回退
为对增长缓冲区反复解码（`MSST_ASR_STREAMING_BACKEND=auto|vllm|transformers`）。

---

## Vocal2Midi（`v2m`）

`POST /api/v2m/inference` 复刻 Vocal2Midi 流程——切片 → ASR → 强制对齐（TiFA，或
HubertFA）→ GAME 音符提取 → 量化 → 导出——并以 zip 返回所选产物及 `manifest.json`。

模型包放在 `MSST_V2M_MODEL_DIR`（默认 `<MSST_MODEL_DIR>/v2m`）。全部在首次使用时准备：
GAME、HubertFA、kashi-g2p 从其 GitHub Release 资产解压（经 `MSST_GITHUB_PROXY`，默认
`https://ghfast.top/`）；TiFA 上游发布的是 PyTorch 检查点，其 ONNX 图由内置的 TIFA
`deploy` 代码导出（`MSST_V2M_TIFA_OPSET`，默认 18）；音高曲线复用 RVC/Applio 的 RMVPE
预测器（`MSST_V2M_RMVPE_BACKEND=rvc`）。也可改用 `alignment_engine=hfa`。

**低显存（8 GB）。** 流程保持 `MSST_MAX_LOADED_MODELS=1`，在加载下一个模型前释放上一个
（并完整清空 CUDA 缓存）。GAME 的 estimator 是另一大占用，因此分块被限制在
`MSST_V2M_MAX_CHUNK_SEC`（默认 30 秒）。在空闲的 8 GB 卡上，`chinese_asr_engine=qwen`
（Qwen3-ASR-1.7B，bf16）时峰值约 5.5 GB；可用 `chinese_asr_engine=pinyin` 获得更大余量。

**语言识别。** 不传 `language`（或传 `auto`）时，流程会用 Qwen3-ASR 采样一小段音频，自动识别
`zh` / `en` / `ja` / `yue` 后再选择 ASR、对齐器与 G2P 链（`MSST_V2M_LANGUAGE_DETECT_SEC`、
`MSST_V2M_DEFAULT_LANGUAGE`）。若想跳过识别（例如使用 `chinese_asr_engine=pinyin` 且不想加载
Qwen），请显式传 `language=zh`。

```bash
curl -X POST http://localhost:8000/api/v2m/inference \
  -F "audio=@vocal.wav" \
  -F "language=zh" \
  -F "output_formats=mid,txt,ustx" \
  -o vocal.zip
```

常用字段：`language`（`zh`/`ja`/`en`/`yue`）、`lyric_output_mode`、`original_lyrics`、
`output_formats`（`mid,txt,csv,ustx,vsqx,textgrid,chunks,asr_match_log`）、`output_lyrics`、
`alignment_engine`（`tifa`/`hfa`）、`chinese_asr_engine`（`qwen`/`pinyin`）、
`japanese_asr_engine`（`romaji`/`qwen`）、`slicing_method`、`slice_min_sec`/`slice_max_sec`、
`tempo`、`quantization_step`/`quantization_mode`、各模型 id（`asr_model`、`game_model`、
`tifa_model` 等）、`device`、`source`。

分能力接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v2m/models` | 列出 v2m 模型包 |
| `POST` | `/api/v2m/models/{id}/download` | 预下载模型包 |
| `POST` | `/api/v2m/slice` | 切片，返回分块 wav 的 zip |
| `POST` | `/api/v2m/pitch` | RMVPE 音高曲线（JSON） |
| `POST` | `/api/v2m/notes` | GAME 纯音高音符提取（JSON） |
| `POST` | `/api/v2m/align` | 已知文本的强制对齐（TiFA，JSON） |
| `POST` | `/api/v2m/g2p` | 字形转读音（kashi-g2p / 拼音） |

> **所有**子系统的模型加载与推理（MSST、RVC、ASR、v2m）共用同一套全局 LRU 与并发预算
> （`MSST_MAX_LOADED_MODELS` / `MSST_MAX_CONCURRENCY`），且仅在请求真正需要时才懒加载模型。

---

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MSST_MODEL_DIR` | `./models`（镜像内 `/models`） | **模型存放路径** |
| `MSST_CONFIG_DIR` | 内置配置目录 | 覆盖模型 YAML 配置目录 |
| `MSST_RVC_MODEL_DIR` | `<MSST_MODEL_DIR>/rvc_models` | **RVC 音色模型目录**（扁平 `*.pth` + `*.index`） |
| `MSST_RVC_ASSET_DIR` | `<MSST_MODEL_DIR>/rvc_assets` | RVC 特征提取器 / F0 预测器缓存 |
| `MSST_RVC_EMBEDDER` | `contentvec` | 默认 RVC 特征提取器 |
| `MSST_ASR_MODEL_DIR` | `<MSST_MODEL_DIR>/asr` | **ASR 模型包目录** |
| `MSST_ASR_CATALOG` | 空 | 追加的 ASR 模型 JSON 目录 |
| `MSST_ASR_DEFAULT_MODEL` | `qwen3-asr-1.7b` | 请求未指定 `model` 时使用的 ASR 模型 |
| `MSST_ASR_STREAMING_BACKEND` | `auto` | 流式后端：`auto` / `vllm` / `transformers` |
| `MSST_QWEN_ASR_DTYPE` | `bfloat16` | Qwen3-ASR 加载精度 |
| `MSST_V2M_MODEL_DIR` | `<MSST_MODEL_DIR>/v2m` | **Vocal2Midi 模型包目录** |
| `MSST_V2M_CATALOG` | 空 | 追加的 v2m 模型 JSON 目录 |
| `MSST_V2M_RMVPE_BACKEND` | `rvc` | 音高曲线后端：`rvc`（复用 RVC RMVPE）/ `onnx` |
| `MSST_V2M_TIFA_OPSET` | `18` | TiFA 导出的 ONNX opset（18–20） |
| `MSST_V2M_MAX_CHUNK_SEC` | `30` | 每个 GAME 分块上限（秒），约束显存（0 关闭） |
| `MSST_V2M_LANGUAGE_DETECT_SEC` | `30` | Qwen3-ASR 自动识别语言时采样的音频秒数 |
| `MSST_V2M_DEFAULT_LANGUAGE` | `zh` | 识别不确定/不支持时的回退语言 |
| `MSST_GITHUB_PROXY` | `https://ghfast.top/` | GitHub 发布资产下载的前缀代理（留空则直连） |
| `MSST_HOST` / `MSST_PORT` | `0.0.0.0` / `8000` | 监听地址与端口 |
| `MSST_DEVICE` | `cuda:0` | 默认推理设备 |
| `MSST_ALLOW_CPU` | `false` | 是否允许 CPU 推理（默认仅 GPU） |
| `MSST_MAX_LOADED_MODELS` | `1` | 常驻显存的模型数量（MSST 与 RVC **共享** LRU 淘汰） |
| `MSST_MAX_CONCURRENCY` | `1` | 并发推理数（MSST 与 RVC **共享**） |
| `MSST_MAX_UPLOAD_MB` | `512` | 上传大小上限 |
| `MSST_TEMP_DIR` | `/tmp/msst-api` | 临时目录 |
| `MSST_DOWNLOAD_BACKEND` | `modelscope` | 下载后端偏好：`modelscope` / `huggingface` / `auto` |
| `MSST_MODELSCOPE_MIRROR` | 空 | 指向镜像 HF 目录结构的 ModelScope 仓库 |
| `MSST_STRICT_BACKEND` | `false` | 为 `true` 时不做后端回退 |
| `MSST_HF_ENDPOINT` | 空 | Hugging Face 镜像（如 `https://hf-mirror.com`） |
| `MSST_HF_TOKEN` / `MSST_MODELSCOPE_TOKEN` | 空 | 私有仓库访问令牌 |
| `MSST_VERIFY_DOWNLOAD` | `true` | 下载后校验 sha256 |
| `MSST_DOWNLOAD_TIMEOUT` | `60` | 单次读取超时（秒），`0` 表示不限制 |

完整示例见 [`.env.example`](.env.example)。

---

## 模型下载行为

1. 推理时先在本机 `MSST_MODEL_DIR/<category>/<name>` 查找权重。
2. 找不到时，按 `MSST_DOWNLOAD_BACKEND` 指定的顺序尝试下载：
   - 默认 `modelscope`：优先从 ModelScope 获取；
   - 若该模型在 ModelScope 上没有对应仓库/文件，**自动回退**到 Hugging Face；
   - `MSST_STRICT_BACKEND=true` 可禁止回退。
3. 下载完成后校验文件大小与 sha256（可用 `MSST_VERIFY_DOWNLOAD=false` 关闭）。
4. 下载使用临时文件 + 原子重命名，避免中断产生的坏文件被复用。
5. 无法直连 Hugging Face 时，设置 `MSST_HF_ENDPOINT=https://hf-mirror.com`：
   下载会同时改写 SDK 请求与直链回退到该镜像。

> **关于 ModelScope**：官方 MSST 权重目前托管在 Hugging Face
> (`Sucial/MSST-WebUI`)，ModelScope 上尚无官方镜像。默认后端仍为
> `modelscope`，实际下载时会按上面的规则回退到 Hugging Face。如果你在
> ModelScope 上建立了镜像仓库（目录结构与 HF 一致），设置
> `MSST_MODELSCOPE_MIRROR=你的组织/MSST-WebUI` 即可让 ModelScope 真正生效。

### 配置与权重的自动适配

不同时期发布的 checkpoint 对同一架构参数（例如 mel-band RoFormer 的
`mask_estimator_depth`）的取值不同。服务在加载权重时会先用配置构建模型，
并将 `state_dict` 与 checkpoint 的键和形状逐一比对；若不匹配，则自动尝试
已知的架构变体（如 `mask_estimator_depth ∈ {1,2,3,4}`、
`mlp_expansion_factor ∈ {2,4,8}`），选用与权重完全一致的那一组。这样同一个
模型目录可以同时兼容旧版权重与新版配置，无需为每个 checkpoint 手工维护配置。

已在 GPU 上实测通过的架构（覆盖目录内全部 9 类模型）：`mel_band_roformer`、
`bs_roformer`、`mdx23c`、`htdemucs`（多音轨 + 单音轨 target_instrument）、
`scnet`、`apollo`、`bandit`、`segm_models`、`swin_upernet`。

> 若所在网络无法直连 `huggingface.co`，设置
> `MSST_HF_ENDPOINT=https://hf-mirror.com` 即可让下载（含 URL 回退）走镜像；
> 实测经过镜像可正常下载并校验 sha256。

---

## 本地开发（uv）

```bash
# 创建环境并安装（会自动从 SJTU 镜像拉取 cu128 版 PyTorch）
uv sync --dev

# 启动服务（需要可用 GPU；否则设置 MSST_ALLOW_CPU=true 且 MSST_DEVICE=cpu）
MSST_MODEL_DIR=./models uv run msst-api

# 运行测试
uv run pytest
```

`pyproject.toml` 已配置镜像源：

- PyPI：`https://pypi.tuna.tsinghua.edu.cn/simple`
- PyTorch：`https://mirror.sjtu.edu.cn/pytorch-wheels/cu128`

## 添加自定义模型

- **MSST 目录方式**：在 `src/msst_api/data/configs/<category>/` 放入与权重同名的
  YAML 配置，并在 `registry.json` 增加条目（或使用
  `scripts/build_registry.py` 重新生成）。
- **MSST 临时方式**：推理请求中直接提供 `model_type` + `config`/`config_url`
  + `checkpoint`/`checkpoint_url`，无需修改目录。
- **RVC 音色**：直接把 `<name>.pth`（及可选的 `<name>.index`）放进
  `MSST_RVC_MODEL_DIR`（`models/rvc_models`）即可，无需注册表；服务会在每次
  请求时重新扫描该目录。

---

## 说明与限制

- 默认仅 GPU；容器镜像基于 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`。
- 首次请求会因为下载/加载模型而较慢（可达数分钟），之后命中缓存。
- 权重与音频可能受版权保护，请遵守各模型与素材的许可。
