# DoDuyTTS 🌍

DoDuyTTS is a massively multilingual zero-shot text-to-speech (TTS) model supporting 600+ languages. Built on a diffusion language model-style architecture, it generates high-quality speech with fast inference, supporting voice cloning and voice design.

**Contents**: [Key Features](#key-features) | [Installation](#installation) | [Quick Start](#quick-start) | [Python API](#python-api) | [Command-Line Tools](#command-line-tools) | [Disclaimer](#disclaimer)

## Key Features

- **600+ Languages Supported**: broad language coverage for zero-shot TTS.
- **Voice Cloning**: clone a voice from a short reference audio clip.
- **Voice Design**: control voice attributes (gender, age, pitch, dialect/accent, whisper, etc.) without any reference audio.
- **Auto Voice**: let the model pick a voice automatically.
- **Fine-grained Control**: non-verbal symbols (e.g. `[laughter]`) and pronunciation correction via pinyin or phonemes.
- **Fast Inference**: RTF as low as 0.025 (~40x faster than real-time), with optional [FlashInfer](#flashinfer-acceleration) acceleration on NVIDIA GPUs.

---

## Installation

Requires Python >= 3.10. Choose **one** of the following methods.

### uv (recommended)

```bash
uv sync
```

> Tip: use a mirror with `uv sync --default-index "https://mirrors.aliyun.com/pypi/simple"`

### pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .
```

**GPU-specific PyTorch** (installed automatically by the steps above on CPU/Apple Silicon; for CUDA or Intel Arc, install PyTorch first):

<details>
<summary>NVIDIA GPU (CUDA)</summary>

```bash
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```
See the [PyTorch site](https://pytorch.org/get-started/locally/) for other CUDA versions.
</details>

<details>
<summary>Apple Silicon (MPS)</summary>

```bash
pip install torch==2.8.0 torchaudio==2.8.0
```
Inference automatically uses the `mps` backend when available.
</details>

<details>
<summary>Intel Arc GPU (XPU)</summary>

```bash
pip install torch torchaudio --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
Requires the [Intel GPU driver](https://dgpu-docs.intel.com/driver/installation.html). `flash_attn` is not available on XPU; the model falls back to SDPA.
</details>

### Optional extras

```bash
pip install -e ".[tn]"     # text normalization (numbers, dates, currency)
pip install -e ".[lora]"   # LoRA finetuning / adapter loading
pip install -e ".[eval]"   # WER / speaker-similarity evaluation tools
```

---

## Quick Start

Activate the environment first (skip if using `uv run`):

```bash
source .venv/bin/activate
```

Then either launch the web demo or run a single inference from the command line:

```bash
# Web UI (voice cloning + voice design)
doduytts-demo --ip 0.0.0.0 --port 8001

# Single-item inference, auto voice, saved to outputs/hello.wav
doduytts-infer \
    --model k2-fsa/OmniVoice \
    --text "Hello, this is a test of text to speech." \
    --output outputs/hello.wav
```

`--model` accepts a HuggingFace repo id (downloaded and cached automatically) or a local checkpoint directory. If you have trouble reaching HuggingFace, set `export HF_ENDPOINT="https://hf-mirror.com"` before running.

Device is auto-detected (`cuda` > `xpu` > `mps` > `cpu`); override with `--device`.

---

## Python API

```python
from doduytts import DoDuyTTS
import soundfile as sf
import torch

model = DoDuyTTS.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",   # or "mps" (Apple Silicon), "xpu" (Intel Arc), "cpu"
    dtype=torch.float16,
)

# Voice cloning: provide ref_audio (+ optional ref_text, auto-transcribed via Whisper if omitted)
audio = model.generate(
    text="Hello, this is a test of zero-shot voice cloning.",
    ref_audio="ref.wav",
    ref_text="Transcription of the reference audio.",
)

# Voice design: describe the voice, no reference audio needed
audio = model.generate(
    text="Hello, this is a test of zero-shot voice design.",
    instruct="female, low pitch, british accent",
)

# Auto voice: no reference audio or instruction
audio = model.generate(text="This is a sentence without any voice prompt.")

sf.write("out.wav", audio[0], 24000)  # audio[0] is np.ndarray, shape (T,), 24 kHz
```

**Useful `generate()` options**: `num_step` (diffusion steps, default 32; try 16 for faster inference), `speed`, `duration` (fixed output length, overrides `speed`), `normalize_text=True` (requires the `tn` extra).

### Reusing a cloned voice across sessions

```python
prompt = model.create_voice_clone_prompt(ref_audio="ref.wav", ref_text="Transcription.")
prompt.save("my_voice.pt")

# Later:
from doduytts import VoiceClonePrompt
prompt = VoiceClonePrompt.load("my_voice.pt")
audio = model.generate(text="Hello again!", voice_clone_prompt=prompt)
```

---

## Command-Line Tools

| Command | Description | Source |
|---|---|---|
| `doduytts-demo` | Interactive Gradio web demo | [doduytts/cli/demo.py](doduytts/cli/demo.py) |
| `doduytts-infer` | Single-item inference | [doduytts/cli/infer.py](doduytts/cli/infer.py) |
| `doduytts-infer-batch` | Batch inference across multiple GPUs | [doduytts/cli/infer_batch.py](doduytts/cli/infer_batch.py) |
| `doduytts-merge-lora` | Merge a LoRA adapter into the base model | [doduytts/cli/merge_lora.py](doduytts/cli/merge_lora.py) |
| `doduytts-serve` | Streaming TTS server (LLM text in, audio out, micro-batching) | [doduytts/server/app.py](doduytts/server/app.py) |

Run any command with `--help` for the full list of options.

### Batch Inference

```bash
doduytts-infer-batch \
    --model k2-fsa/OmniVoice \
    --test_list test.jsonl \
    --res_dir results/
```

`test.jsonl` is one JSON object per line: `id` and `text` are required; `ref_audio`/`ref_text` for voice cloning; `instruct` for voice design; optional `language_id`, `duration`, `speed`.

### Streaming TTS Server

`doduytts-serve` accepts streaming text (as an LLM produces it), chunks it into sentences incrementally, and synthesizes each sentence as soon as it is complete. Concurrent requests are gathered by a dynamic micro-batcher (default: up to 4 requests per batch, 20 ms window) into single batched `generate()` calls.

```bash
doduytts-serve --port 8001                 # real model (auto device)
doduytts-serve --port 8001 --mock          # pipeline test without weights
doduytts-serve --num-step 32 --max-batch 4 --window-ms 20
```

- **WebSocket `/ws/tts`** — send `{"type":"config", "instruct"|"language"|"ref_audio"+"ref_text": ...}` once (optional), then `{"type":"delta","text":...}` per LLM token, then `{"type":"end"}`. The server replies with `sentence` events as sentences are detected and ordered `audio` events (base64 WAV, 24 kHz).
- **SSE `GET /demo/stream?text=...&delay_ms=25`** — self-contained demo driven by a simulated LLM token stream.
- **`GET /stats`** — batch count / average batch size.

Demo client (simulates an LLM streaming word-by-word; `--concurrent N` opens N sessions to exercise batching):

```bash
python examples/stream_client.py --concurrent 4
```

### FlashInfer Acceleration

~2-2.9x lossless speedup on NVIDIA GPUs:

```bash
pip install flashinfer-python==0.6.15.post1 "flashinfer-jit-cache==0.6.15.post1+cu128" \
    --extra-index-url https://flashinfer.ai/whl/cu128/

doduytts-infer-batch --model k2-fsa/OmniVoice --test_list test.jsonl --res_dir results/ \
    --batch_size 8 --enable_flashinfer true
```

---

## Disclaimer

Users are strictly prohibited from using this model for unauthorized voice cloning, voice impersonation, fraud, scams, or any other illegal or unethical activities. All users shall ensure full compliance with applicable local laws, regulations, and ethical standards. The developers assume no liability for any misuse of this model and advocate for responsible AI development and use.
