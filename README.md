# MSST API

**English** | [简体中文](README.zh-CN.md)

A GPU inference HTTP service for
[Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training).
Submit an audio file plus a model, and get the separated stems back.

- `POST /api/msst/inference`: form-based inference; pick any supported MSST model
  and its parameters.
- `POST /api/rvc/inference`: **RVC voice conversion** — convert an audio clip to
  the timbre of an installed RVC voice model.
- Missing MSST models are **downloaded automatically** on first use, from
  **ModelScope (default)** or **Hugging Face**; RVC voice models are read from
  `MSST_RVC_MODEL_DIR` (`models/rvc_models` by default).
- The model storage path is set with the `MSST_MODEL_DIR` environment variable.
- Dependencies are managed with [uv](https://docs.astral.sh/uv/); inference is
  **GPU-only** by default.
- Ships with **51** pretrained MSST models from the MSST ecosystem (vocals /
  instrumental, single-stem, multi-stem).

> References: [MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI),
> [RVCSVC-API-MSST](https://github.com/sdfsfsk/RVCSVC-API-MSST),
> [applio-api-plugin](https://github.com/Deliay/applio-api-plugin),
> [Applio](https://github.com/IAHispano/Applio).

---

## Layout

```
.
├── Dockerfile                     # CUDA 12.8 runtime + uv
├── pyproject.toml / uv.lock       # uv dependencies and lockfile
├── scripts/
│   ├── build_registry.py          # build registry.json from the upstream catalog
│   └── models_info.json           # upstream model catalog (MSST-WebUI)
└── src/msst_api/
    ├── main.py                    # FastAPI app and routes
    ├── separator.py               # MSST inference wrapper + LRU model cache
    ├── rvc_engine.py              # RVC voice discovery + LRU converter cache
    ├── rvc/                       # vendored RVC inference engine (from Applio)
    │   ├── infer/                 # VoiceConverter + conversion pipeline
    │   ├── lib/                   # algorithm, predictors (rmvpe/crepe/fcpe), utils
    │   └── configs/               # sample-rate presets
    ├── download.py                # ModelScope / HuggingFace / URL downloads
    ├── registry.py                # model catalog access
    ├── audio.py                   # audio decoding / encoding
    ├── config.py                  # environment-based configuration
    └── data/
        ├── registry.json          # generated model catalog
        └── configs/               # per-checkpoint YAML configs
```

---

## Quick start (Docker, recommended)

Requirements: an NVIDIA driver and `nvidia-container-toolkit`.

```bash
# Build the image
docker build -t msst-api:latest .

# Run (mount the model directory, use the GPU)
docker run --gpus all -p 8000:8000 \
  -v /data/msst-models:/models \
  -e MSST_DOWNLOAD_BACKEND=modelscope \
  msst-api:latest
```

Open <http://localhost:8000/docs> for the interactive API docs.

> The image already points Ubuntu apt, PyPI and PyTorch at mainland-China
> mirrors (Tsinghua apt/PyPI, SJTU CUDA 12.8).

---

## API usage

### 1. List available models

```bash
curl http://localhost:8000/api/msst/models
curl "http://localhost:8000/api/msst/models?downloaded=false"
```

### 2. Inference (returns a zip by default)

```bash
curl -X POST http://localhost:8000/api/msst/inference \
  -F "audio=@song.wav" \
  -F "model=model_bs_roformer_ep_317_sdr_12.9755.ckpt" \
  -F "output_format=wav" \
  -o stems.zip
```

The returned `stems.zip` contains stems such as `vocals.wav` and
`instrumental.wav`, plus a `manifest.json`.

### 3. Inference (JSON + base64)

```bash
curl -X POST http://localhost:8000/api/msst/inference \
  -F "audio=@song.wav" \
  -F "model=model_bs_roformer_ep_317_sdr_12.9755.ckpt" \
  -F "response_format=json" \
  -F "output_format=flac"
```

### 4. Python client

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

### `POST /api/msst/inference` form fields

| Field | Type | Description |
| --- | --- | --- |
| `audio` | file (required) | Input audio: wav/flac/mp3/m4a, etc. |
| `model` | str | Catalog model id (from `GET /api/msst/models`) |
| `model_type` | str | Architecture key for a custom model (e.g. `bs_roformer`) |
| `config` / `config_url` | file / url | YAML config for a custom model |
| `checkpoint` / `checkpoint_url` | file / url | Weights for a custom model |
| `device` | str | Defaults to `MSST_DEVICE` (`cuda:0`) |
| `batch_size` | int | Overrides `inference.batch_size` |
| `num_overlap` | int | Overrides `inference.num_overlap` (higher = smoother, slower) |
| `chunk_size` | int | Overrides `audio.chunk_size` |
| `normalize` | bool | Overrides `inference.normalize` |
| `use_tta` | bool | Test-time augmentation (slower, usually more stable) |
| `bigshifts` | int | BigShifts shift-averaging count (default 1) |
| `extract_instrumental` | bool | Also output `instrumental` (mix − vocals) |
| `stems` | str | Comma-separated subset of output stems, e.g. `vocals,instrumental` |
| `output_format` | str | `wav` (default) / `flac` / `mp3` |
| `bit_depth` | str | e.g. `PCM_16`, `PCM_24`, `FLOAT`, `320k` |
| `response_format` | str | `zip` (default) / `json` |
| `source` | str | Download backend for this request: `modelscope` / `huggingface` / `auto` |

### Other endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Health, CUDA status, loaded models |
| `GET` | `/api/msst/models` | List models |
| `GET` | `/api/msst/models/{id}` | Model details |
| `POST` | `/api/msst/models/{id}/download` | Pre-download a model |

---

## RVC voice conversion

`POST /api/rvc/inference` converts an input clip to the timbre of an installed
RVC voice model and streams the converted audio back in the response body
(in-memory, no output files on disk).

RVC voice models live in `MSST_RVC_MODEL_DIR` (default
`<MSST_MODEL_DIR>/rvc_models`, i.e. `models/rvc_models`). Each voice is a flat
`<name>.pth` file with an optional sibling `<name>.index` (auto-matched by
name):

```
models/rvc_models/
├── alice.pth
├── alice.index
└── bob_v2.pth
```

Auxiliary assets (the ContentVec embedder and the RMVPE F0 predictor) are
**downloaded on first use** from the Applio Hugging Face repo into
`MSST_RVC_ASSET_DIR` (default `<MSST_MODEL_DIR>/rvc_assets`). FCPE is bundled
inside `torchfcpe`, so `f0_method=fcpe` needs no download.

### 1. List voices

```bash
curl http://localhost:8000/api/rvc/models
curl http://localhost:8000/api/rvc/models/alice
```

### 2. Convert

```bash
curl -X POST http://localhost:8000/api/rvc/inference \
  -F "audio=@input.wav" \
  -F "model=alice" \
  -F "f0_method=rmvpe" \
  -F "pitch=0" \
  -F "output_format=wav" \
  -o output.wav
```

The response body is the converted audio; `Content-Type` follows
`output_format`. `X-RVC-Model` and `X-RVC-Elapsed` headers identify the voice
and the wall-clock seconds.

### `POST /api/rvc/inference` form fields

| Field | Type | Description |
| --- | --- | --- |
| `audio` | file (required) | Input audio: wav/flac/mp3/m4a, etc. (aliases: `file`, `input`) |
| `model` | str | Voice id/name from `GET /api/rvc/models` (alias: `model_name`, `pth`) |
| `index` | str | Index file name; auto-matched from the voice when omitted |
| `device` | str | Defaults to `MSST_DEVICE` (`cuda:0`) |
| `pitch` | int | Pitch shift in semitones (default `0`) |
| `f0_method` | str | `rmvpe` (default) / `crepe` / `crepe-tiny` / `fcpe` |
| `index_rate` | float | Index blend ratio (default `0.75`) |
| `volume_envelope` | float | RMS envelope mix (default `1.0`) |
| `protect` | float | Voiceless-consonant protection (default `0.5`) |
| `sid` | int | Speaker id (default `0`) |
| `split_audio` | bool | Split long audio on silence before conversion |
| `f0_autotune` / `f0_autotune_strength` | bool / float | Autotune the F0 contour |
| `proposed_pitch` / `proposed_pitch_threshold` | bool / float | Auto pitch to a target frequency |
| `clean_audio` / `clean_strength` | bool / float | Noise reduction before output |
| `resample_sr` | int | Resample output to this rate (`0` disables) |
| `embedder_model` / `embedder_model_custom` | str / str | Feature extractor (default `contentvec`) |
| `formant_shifting` / `formant_qfrency` / `formant_timbre` | bool / float / float | Formant shifting |
| `post_process` | bool | Enable the pedalboard effect chain |
| effect params | | `reverb`, `pitch_shift`, `limiter`, `gain`, `distortion`, `chorus`, `bitcrush`, `clipping`, `compressor`, `delay` plus their parameters |
| `output_format` | str | `wav` (default) / `mp3` / `flac` / `ogg` / `opus` / `m4a` / `aac` / `aiff` / `ac3` |

> To convert a **separated stem** (e.g. MSST vocals) with RVC, run
> `/api/msst/inference`, extract `vocals.wav`, then post it to
> `/api/rvc/inference`.

---

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `MSST_MODEL_DIR` | `./models` (`/models` in the image) | **Model storage path** |
| `MSST_CONFIG_DIR` | bundled config dir | Override the model YAML config directory |
| `MSST_RVC_MODEL_DIR` | `<MSST_MODEL_DIR>/rvc_models` | **RVC voice model directory** (flat `*.pth` + `*.index`) |
| `MSST_RVC_ASSET_DIR` | `<MSST_MODEL_DIR>/rvc_assets` | Cache for RVC embedders / F0 predictors |
| `MSST_RVC_EMBEDDER` | `contentvec` | Default RVC feature extractor |
| `MSST_HOST` / `MSST_PORT` | `0.0.0.0` / `8000` | Bind address and port |
| `MSST_DEVICE` | `cuda:0` | Default inference device |
| `MSST_ALLOW_CPU` | `false` | Allow CPU inference (GPU-only by default) |
| `MSST_MAX_LOADED_MODELS` | `1` | Models kept resident across MSST **and** RVC (shared LRU eviction) |
| `MSST_MAX_CONCURRENCY` | `1` | Concurrent inference requests across MSST **and** RVC (shared) |
| `MSST_MAX_UPLOAD_MB` | `512` | Max upload size |
| `MSST_TEMP_DIR` | `/tmp/msst-api` | Temporary directory |
| `MSST_DOWNLOAD_BACKEND` | `modelscope` | Backend preference: `modelscope` / `huggingface` / `auto` |
| `MSST_MODELSCOPE_MIRROR` | empty | ModelScope repo mirroring the HF layout |
| `MSST_STRICT_BACKEND` | `false` | When `true`, disable backend fallback |
| `MSST_HF_ENDPOINT` | empty | Hugging Face mirror (e.g. `https://hf-mirror.com`) |
| `MSST_HF_TOKEN` / `MSST_MODELSCOPE_TOKEN` | empty | Access tokens for private repos |
| `MSST_VERIFY_DOWNLOAD` | `true` | Verify sha256 after download |
| `MSST_DOWNLOAD_TIMEOUT` | `60` | Per-read timeout in seconds; `0` disables it |

See [`.env.example`](.env.example) for a complete example.

---

## Download behaviour

1. On inference, the service first looks for weights at
   `MSST_MODEL_DIR/<category>/<name>`.
2. If they are missing, it downloads them following the backend order set by
   `MSST_DOWNLOAD_BACKEND`:
   - default `modelscope`: try ModelScope first;
   - if the model has no ModelScope repo/file, **fall back automatically** to
     Hugging Face;
   - set `MSST_STRICT_BACKEND=true` to disable the fallback.
3. After downloading, the file size and sha256 are verified (disable with
   `MSST_VERIFY_DOWNLOAD=false`).
4. Downloads use a temporary file plus an atomic rename, so interrupted
   downloads never leave a corrupt file in place.
5. If direct access to Hugging Face is blocked, set
   `MSST_HF_ENDPOINT=https://hf-mirror.com`: both the SDK request and the
   direct-URL fallback are rewritten to that mirror.

> **About ModelScope**: the official MSST weights are hosted on Hugging Face
> (`Sucial/MSST-WebUI`); there is no official ModelScope mirror yet. The default
> backend is still `modelscope`, and the downloader falls back to Hugging Face
> as described above. If you create a ModelScope mirror repo (with the same
> directory layout as HF), set
> `MSST_MODELSCOPE_MIRROR=your-org/MSST-WebUI` to make ModelScope effective.

### Automatic config/checkpoint matching

Checkpoints released at different times sometimes use different values for the
same architecture parameter (for example `mask_estimator_depth` on mel-band
RoFormers). When loading weights, the service builds the model from the config
and compares its `state_dict` keys and shapes against the checkpoint. On a
mismatch it tries the known architecture variants (e.g.
`mask_estimator_depth ∈ {1,2,3,4}`, `mlp_expansion_factor ∈ {2,4,8}`) and keeps
the variant that matches exactly. This lets one catalog serve both legacy
weights and newer configs without hand-maintained per-checkpoint files.

The following architectures have been verified on GPU (all 9 model families in
the catalog): `mel_band_roformer`, `bs_roformer`, `mdx23c`, `htdemucs`
(multi-stem and single-stem target_instrument), `scnet`, `apollo`, `bandit`,
`segm_models`, `swin_upernet`.

> If your network cannot reach `huggingface.co`, set
> `MSST_HF_ENDPOINT=https://hf-mirror.com`; downloads (including the URL
> fallback) then go through the mirror and sha256 verification still applies.

---

## Local development (uv)

```bash
# Create the environment and install (pulls the cu128 PyTorch build from the
# SJTU mirror automatically)
uv sync --dev

# Start the service (needs a GPU; otherwise set MSST_ALLOW_CPU=true and
# MSST_DEVICE=cpu)
MSST_MODEL_DIR=./models uv run msst-api

# Run the tests
uv run pytest
```

`pyproject.toml` already configures the mirrors:

- PyPI: `https://pypi.tuna.tsinghua.edu.cn/simple`
- PyTorch: `https://mirror.sjtu.edu.cn/pytorch-wheels/cu128`

## Adding custom models

- **MSST catalog approach**: put a YAML config named after the weights in
  `src/msst_api/data/configs/<category>/` and add an entry to `registry.json`
  (or regenerate it with `scripts/build_registry.py`).
- **MSST ad-hoc approach**: pass `model_type` + `config`/`config_url` +
  `checkpoint`/`checkpoint_url` directly in the inference request; no catalog
  changes needed.
- **RVC voices**: just drop a `<name>.pth` (and optional `<name>.index`) into
  `MSST_RVC_MODEL_DIR` (`models/rvc_models`). No registry needed; the service
  rescans the directory on every request.

---

## Notes and limitations

- GPU-only by default; the image is based on
  `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`.
- The first request is slow (model download/load, potentially minutes);
  subsequent requests hit the cache.
- Weights and audio may be copyright-protected; respect each model's and
  asset's license.
