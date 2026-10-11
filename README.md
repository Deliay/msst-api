# MSST API

**English** | [简体中文](README.zh-CN.md)

A GPU inference HTTP service for
[Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training).
Submit an audio file plus a model, and get the separated stems back.

- `POST /api/msst/inference`: form-based inference; pick any supported MSST model
  and its parameters.
- `POST /api/rvc/inference`: **RVC voice conversion** — convert an audio clip to
  the timbre of an installed RVC voice model.
- `POST /api/asr/transcribe`: **ASR** — single-sentence recognition
  (Qwen3-ASR / PinyinASR / RomajiASR).
- `WS /api/asr/streaming`: **streaming ASR** — live recognition over a
  WebSocket (Qwen3-ASR only).
- `POST /api/v2m/inference`: **Vocal2Midi** — vocal audio → lyric-aligned MIDI
  (and USTX/VSQX/TextGrid), returned as a zip.
- `POST /api/diffsinger/inference`: **DiffSinger (SVS)** — render an OpenUtau
  `.ustx` score with an OpenUtau-compatible DiffSinger voicebank; auto-pitch
  (``pitd``) is generated and the score can be returned alongside the audio.
- `POST /api/diffsinger/pitch`: **auto-pitch only** — `.ustx` → `.ustx` with a
  generated ``pitd`` curve.
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
    ├── asr/                       # ASR capability (asr + streaming_asr)
    │   ├── catalog.py             # Qwen3-ASR / PinyinASR / RomajiASR catalog
    │   ├── engines.py             # Qwen3-ASR + ONNX CTC engines, streaming sessions
    │   └── manager.py             # shared-cache model manager
    ├── v2m/                       # Vocal2Midi capability
    │   ├── catalog.py             # GAME / TiFA / HubertFA / RMVPE / kashi-g2p
    │   ├── manager.py             # catalog + store + shared cache wiring
    │   ├── pipeline.py            # end-to-end pipeline (zip output)
    │   ├── capabilities.py        # per-capability helpers (slice/pitch/notes/align/g2p)
    │   ├── runtime.py             # shared-cache bridge for the vendored loaders
    │   └── _vendor/               # vendored Vocal2Midi inference runtime (Apache-2.0)
    ├── model_store.py             # multi-file model bundles + snapshot downloads
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

## ASR (`asr`) and streaming ASR (`streaming_asr`)

Two capabilities are exposed, backed by **different engines**:

| Capability | Endpoint | Models |
| --- | --- | --- |
| `asr` (single sentence) | `POST /api/asr/transcribe` | Qwen3-ASR, PinyinASR, RomajiASR |
| `streaming_asr` | `WS /api/asr/streaming` | Qwen3-ASR only |

Model bundles live under `MSST_ASR_MODEL_DIR` (default `<MSST_MODEL_DIR>/asr`).
Qwen3-ASR is downloaded from the Hub on first use, and the PinyinASR /
RomajiASR ONNX bundles are downloaded from their GitHub release assets
(`model_onnx.zip`) and unpacked automatically. You can also place the bundles
there directly or fetch them via a custom `MSST_ASR_CATALOG`.

### 1. List models

```bash
curl http://localhost:8000/api/asr/models
curl "http://localhost:8000/api/asr/models?task=streaming"   # Qwen3-ASR only
```

### 2. Single-sentence transcription

```bash
curl -X POST http://localhost:8000/api/asr/transcribe \
  -F "audio=@vocal.wav" -F "model=qwen3-asr-1.7b" -F "language=zh"
# {"model":"qwen3-asr-1.7b","text":"...","language":"Chinese","elapsed_seconds":...}
```

`PinyinASR` returns toneless pinyin tokens, `RomajiASR` returns romaji mora
tokens (both include a `phonemes` list):

```bash
curl -X POST http://localhost:8000/api/asr/transcribe \
  -F "audio=@vocal.wav" -F "model=pinyin-asr"
```

### 3. Streaming (WebSocket)

Connect to `ws://localhost:8000/api/asr/streaming?model=qwen3-asr-1.7b&language=zh`
and send binary audio frames (float32 little-endian PCM at `sample_rate`, or
int16 with `format=pcm_s16le`). The server replies with
`{"type":"partial","text":...}` after each frame and `{"type":"final",...}`
when you send `finish`.

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect(
        "ws://localhost:8000/api/asr/streaming?model=qwen3-asr-1.7b&language=zh"
    ) as ws:
        print(await ws.recv())              # {"type": "ready", ...}
        await ws.send(pcm_f32_bytes)        # one or more audio frames
        print(await ws.recv())              # {"type": "partial", ...}
        await ws.send("finish")
        print(await ws.recv())              # {"type": "final", ...}

asyncio.run(main())
```

True incremental decoding uses the Qwen3-ASR **vLLM** backend
(`pip install "qwen-asr[vllm]"`). Without vLLM the endpoint falls back to
repeatedly re-decoding a growing buffer with the transformers backend
(`MSST_ASR_STREAMING_BACKEND=auto|vllm|transformers`).

---

## Vocal2Midi (`v2m`)

`POST /api/v2m/inference` mirrors Vocal2Midi's pipeline — slicing → ASR →
forced alignment (TiFA, or HubertFA) → GAME note extraction → quantization →
export — and returns a zip of the requested artifacts plus a `manifest.json`.

Model bundles live under `MSST_V2M_MODEL_DIR` (default `<MSST_MODEL_DIR>/v2m`).
Everything is prepared on first use: GAME, HubertFA and kashi-g2p are unpacked
from their GitHub release assets (through `MSST_GITHUB_PROXY`, default
`https://ghfast.top/`); TiFA's upstream release is a PyTorch checkpoint, so the
ONNX graphs are exported with the vendored TIFA `deploy` code
(`MSST_V2M_TIFA_OPSET`, default 18); and the pitch curve reuses the RVC/Applio
RMVPE predictor (`MSST_V2M_RMVPE_BACKEND=rvc`). `alignment_engine=hfa` remains an
alternative aligner.

**Low VRAM (8 GB).** The pipeline keeps `MSST_MAX_LOADED_MODELS=1` and releases
each model (with a full CUDA cache flush) before loading the next. GAME's
estimator is the other consumer, so chunks are capped at
`MSST_V2M_MAX_CHUNK_SEC` (default 30 s). Peak on an otherwise-empty 8 GB card is
~5.5 GB with `chinese_asr_engine=qwen` (Qwen3-ASR-1.7B, bf16); use
`chinese_asr_engine=pinyin` there for more headroom.

**Language.** Omit `language` (or pass `auto`) and the pipeline samples a short
window with Qwen3-ASR to detect `zh` / `en` / `ja` / `yue` before choosing the
ASR, aligner and G2P chain (`MSST_V2M_LANGUAGE_DETECT_SEC`,
`MSST_V2M_DEFAULT_LANGUAGE`). Pass `language=zh` explicitly to skip detection
(e.g. when using `chinese_asr_engine=pinyin` and you don't want Qwen loaded).

```bash
curl -X POST http://localhost:8000/api/v2m/inference \
  -F "audio=@vocal.wav" \
  -F "language=zh" \
  -F "output_formats=mid,txt,ustx" \
  -o vocal.zip
```

### `POST /api/v2m/inference` form fields

| Field | Type | Description |
| --- | --- | --- |
| `audio` | file (required) | Input vocal audio. |
| `language` | str | `zh`, `ja`, `en`, `yue` (default `zh`). |
| `lyric_output_mode` | str | `auto` (default; romanised: pinyin / Jyutping / romaji) / `hanzi` / `pinyin` / `romaji` / `kana` / `word` / `jyutping`. |
| `original_lyrics` | str | Optional reference lyrics for matching. |
| `output_formats` | str | Comma list: `mid` (default), `txt`, `csv`, `ustx`, `vsqx`, `textgrid`, `chunks`, `asr_match_log`. |
| `output_lyrics` | bool | `true` (default) runs ASR + alignment; `false` extracts pitch only. |
| `alignment_engine` | str | `tifa` (default) or `hfa`. |
| `chinese_asr_engine` | str | `qwen` (default) or `pinyin`. |
| `japanese_asr_engine` | str | `romaji` (default) or `qwen`. |
| `japanese_g2p_engine` | str | `kashi-g2p-onnx` (default) or `pyopenjtalk`. |
| `slicing_method` | str | `auto` / `default` / `smart` / `heuristic` / `grid`. |
| `slice_min_sec` / `slice_max_sec` | float | Slice bounds (defaults 5 / 10). |
| `tempo` | float | Export tempo (default 120). |
| `quantization_step` / `quantization_mode` | int / str | Quantization (default `0` / `smart`). |
| `output_pitch_curve` | bool | Include an RMVPE pitch curve in USTX/VSQX. |
| `asr_model`, `pinyin_asr_model`, `romaji_asr_model` | str | ASR model ids. |
| `game_model`, `tifa_model`, `hfa_model`, `rmvpe_model`, `kashi_g2p_model` | str | v2m model ids. |
| `device` | str | Defaults to `MSST_DEVICE`. |
| `source` | str | Download backend for this request: `modelscope` / `huggingface` / `auto`. |

### Per-capability endpoints

Each pipeline stage is also exposed on its own:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v2m/models` | List v2m bundles (GAME / TiFA / HubertFA / RMVPE / kashi-g2p). |
| `POST` | `/api/v2m/models/{id}/download` | Pre-download a bundle. |
| `POST` | `/api/v2m/slice` | Silence/pitch-based slicing → zip of chunk wavs. |
| `POST` | `/api/v2m/pitch` | RMVPE F0 curve (JSON). |
| `POST` | `/api/v2m/notes` | GAME pitch-only note extraction (JSON). |
| `POST` | `/api/v2m/align` | Forced alignment of a known transcript (TiFA, JSON). |
| `POST` | `/api/v2m/g2p` | Grapheme-to-phoneme reading (kashi-g2p / pinyin). |

> Model loading and inference for **all** subsystems (MSST, RVC, ASR, v2m,
> DiffSinger) share the same global LRU and concurrency budgets
> (`MSST_MAX_LOADED_MODELS` / `MSST_MAX_CONCURRENCY`), and every model is loaded
> lazily only when a request needs it.

---

## DiffSinger singing voice synthesis (`svs`)

`POST /api/diffsinger/inference` renders an OpenUtau `.ustx` score with an
**OpenUtau-compatible DiffSinger voicebank**. The pipeline mirrors OpenUtau:

1. romanize + phonemize the lyrics and predict phoneme durations (the `dsdur`
   linguistic + duration models);
2. run the **auto-pitch** model (`dspitch`) and write the result back into the
   score as a `pitd` (pitch deviation) curve — the `ustx -> pitd + ustx` step;
3. rebuild the phrases and render each with the variance (`dsvariance`),
   acoustic, and vocoder ONNX models, placing the phrase audio on the timeline.

A voicebank is a folder containing `dsconfig.yaml`, `aco.phonemes.json`,
`dsdict*.yaml`, the `dsdur/`, `dspitch/`, `dsvariance/` and `dsvocoder/`
sub-folders. Drop such folders under `MSST_SVS_MODEL_DIR`
(`<MSST_MODEL_DIR>/svs` by default).

### 1. List voicebanks

```bash
curl http://localhost:8000/api/diffsinger/models
```

### 2. Render a score

```bash
curl -X POST http://localhost:8000/api/diffsinger/inference \
  -F "model=MyVoicebank" \
  -F "ustx=@song.ustx" \
  -F "language=auto" \
  -F "output_format=wav" \
  -o out.wav
```

Add `-F "return_ustx=true"` to get a zip containing the audio **and** the
regenerated `pitd` score (`<name>.pitd.ustx`) plus a `manifest.json`.

### 3. Auto-pitch only

```bash
curl -X POST http://localhost:8000/api/diffsinger/pitch \
  -F "model=MyVoicebank" -F "ustx=@song.ustx" -o song.pitd.ustx
```

### `POST /api/diffsinger/inference` form fields

| Field | Default | Description |
| --- | --- | --- |
| `ustx` | — | The OpenUtau score to render (required). |
| `model` | `MSST_SVS_DEFAULT_MODEL` | Voicebank id (folder name) from `/api/diffsinger/models`. |
| `language` | `auto` | `auto` (detect from the lyrics/voicebank) / `ja` / `zh`. |
| `auto_pitch` | `true` | Run the auto-pitch model; when `false` the score's own `pitd` is used. |
| `pitch_steps` | `20` | Sampling steps for the auto-pitch model. |
| `variance_steps` | `20` | Sampling steps for the variance model. |
| `steps` | `20` | Sampling steps for the acoustic model. |
| `depth` | `1.0` | Diffusion depth (clamped by `dsconfig.yaml`'s `max_depth`). |
| `return_ustx` | `false` | Return a zip with the audio + regenerated `pitd` score. |
| `output_format` | `wav` | `wav` / `flac` / `mp3`. |
| `bit_depth` | — | e.g. `PCM_16`, `PCM_24`, `320k`. |
| `device` | `MSST_DEVICE` | ONNX Runtime device, e.g. `cuda:0` or `cpu`. |

> Only the models the voicebank declares are used; missing optional sub-models
> (e.g. no `dspitch/`) degrade gracefully (the score's `pitd` is used as-is).
> Chinese (`zh`) lyrics may be pinyin or hanzi. The v2m endpoint emits pinyin
> by default (`lyric_output_mode=auto`), so a v2m score feeds straight in;
> hanzi lyrics need the optional
> [`pypinyin`](https://pypi.org/project/pypinyin/) package.
> See [`docs/diffsinger-autopitch.md`](docs/diffsinger-autopitch.md) for how the
> auto-pitch step works in OpenUtau.

---

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `MSST_MODEL_DIR` | `./models` (`/models` in the image) | **Model storage path** |
| `MSST_CONFIG_DIR` | bundled config dir | Override the model YAML config directory |
| `MSST_RVC_MODEL_DIR` | `<MSST_MODEL_DIR>/rvc_models` | **RVC voice model directory** (flat `*.pth` + `*.index`) |
| `MSST_RVC_ASSET_DIR` | `<MSST_MODEL_DIR>/rvc_assets` | Cache for RVC embedders / F0 predictors |
| `MSST_RVC_EMBEDDER` | `contentvec` | Default RVC feature extractor |
| `MSST_ASR_MODEL_DIR` | `<MSST_MODEL_DIR>/asr` | **ASR model bundle directory** |
| `MSST_ASR_CATALOG` | empty | Extra JSON catalog for ASR models |
| `MSST_ASR_DEFAULT_MODEL` | `qwen3-asr-1.7b` | ASR model used when a request omits `model` |
| `MSST_ASR_STREAMING_BACKEND` | `auto` | Streaming backend: `auto` / `vllm` / `transformers` |
| `MSST_QWEN_ASR_DTYPE` | `bfloat16` | dtype used to load Qwen3-ASR |
| `MSST_V2M_MODEL_DIR` | `<MSST_MODEL_DIR>/v2m` | **Vocal2Midi model bundle directory** |
| `MSST_V2M_CATALOG` | empty | Extra JSON catalog for v2m models |
| `MSST_V2M_RMVPE_BACKEND` | `rvc` | Pitch-curve backend: `rvc` (reuse RVC RMVPE) / `onnx` |
| `MSST_V2M_TIFA_OPSET` | `18` | ONNX opset (18–20) for the TiFA export |
| `MSST_V2M_MAX_CHUNK_SEC` | `30` | Cap on each GAME chunk; bounds estimator VRAM (0 = off) |
| `MSST_V2M_LANGUAGE_DETECT_SEC` | `30` | Audio sampled by Qwen3-ASR to auto-detect the language |
| `MSST_V2M_DEFAULT_LANGUAGE` | `zh` | Fallback when detection is inconclusive / unsupported |
| `MSST_SVS_MODEL_DIR` | `<MSST_MODEL_DIR>/svs` | **DiffSinger voicebank directory** |
| `MSST_SVS_DEFAULT_MODEL` | empty | Voicebank used when a request omits `model` |
| `MSST_GITHUB_PROXY` | `https://ghfast.top/` | Prefix proxy for GitHub release downloads (empty = direct) |
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
- `onnxruntime-gpu` is pinned to `<1.27` (see `pyproject.toml`). ORT 1.27+
  publishes CUDA 13 builds that require `libcublasLt.so.13`, which is not
  present in the CUDA 12.8 runtime image; 1.26.x is the last CUDA 12.8 line.
- The first request is slow (model download/load, potentially minutes);
  subsequent requests hit the cache.
- Weights and audio may be copyright-protected; respect each model's and
  asset's license.
