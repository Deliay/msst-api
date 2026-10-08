# MSST API

[English](README.md) | **简体中文**

面向 [Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training)
的 GPU 推理 HTTP 服务。提交一段音频和一个模型，返回分离后的各个音轨。

- `POST /api/msst/inference`：表单推理，支持指定 MSST 模型及其推理参数。
- 缺失模型在首次推理时**自动下载**，支持 **ModelScope（默认）** 与 **Hugging Face**。
- 模型存放路径通过环境变量 `MSST_MODEL_DIR` 指定。
- 使用 [uv](https://docs.astral.sh/uv/) 管理依赖，默认**仅支持 GPU 推理**。
- 内置 49 个来自 MSST 生态的预训练模型（人声/伴奏、单音轨、多音轨）。

> 参考实现：[MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI)、
> [RVCSVC-API-MSST](https://github.com/sdfsfsk/RVCSVC-API-MSST)。

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

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MSST_MODEL_DIR` | `./models`（镜像内 `/models`） | **模型存放路径** |
| `MSST_CONFIG_DIR` | 内置配置目录 | 覆盖模型 YAML 配置目录 |
| `MSST_HOST` / `MSST_PORT` | `0.0.0.0` / `8000` | 监听地址与端口 |
| `MSST_DEVICE` | `cuda:0` | 默认推理设备 |
| `MSST_ALLOW_CPU` | `false` | 是否允许 CPU 推理（默认仅 GPU） |
| `MSST_MAX_LOADED_MODELS` | `1` | 常驻显存的模型数量（LRU 淘汰） |
| `MSST_MAX_CONCURRENCY` | `1` | 并发推理数 |
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

- **目录方式**：在 `src/msst_api/data/configs/<category>/` 放入与权重同名的
  YAML 配置，并在 `registry.json` 增加条目（或使用
  `scripts/build_registry.py` 重新生成）。
- **临时方式**：推理请求中直接提供 `model_type` + `config`/`config_url`
  + `checkpoint`/`checkpoint_url`，无需修改目录。

---

## 说明与限制

- 默认仅 GPU；容器镜像基于 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`。
- 首次请求会因为下载/加载模型而较慢（可达数分钟），之后命中缓存。
- 权重与音频可能受版权保护，请遵守各模型与素材的许可。
