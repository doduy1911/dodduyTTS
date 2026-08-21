"""Model wrapper used by the server's micro-batcher.

``synthesize_batch`` receives a mixed micro-batch and splits it into
homogeneous sub-batches (auto voice / voice design / voice clone), because
``DoDuyTTS.generate()`` expects per-item argument lists of one kind. All
model access is serialized behind a lock — the model is not thread-safe.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

from .batcher import TTSRequest

logger = logging.getLogger("doduytts.server")


class TTSEngine:
    def __init__(
        self,
        model_path: str = "doduy1911/dodduyTTS",
        device: Optional[str] = None,
        num_step: int = 8,
        flashinfer: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.num_step = num_step
        self.flashinfer = flashinfer
        self.model = None
        self.gen_config = None
        self._lock = threading.Lock()

    @property
    def sampling_rate(self) -> int:
        return self.model.sampling_rate

    def load(self) -> None:
        import torch

        from doduytts import DoDuyTTS, DoDuyTTSGenerationConfig
        from doduytts.utils.common import get_best_device

        device = self.device or get_best_device()
        dtype = torch.float32 if device == "cpu" else torch.float16
        logger.info("loading %s on %s (%s)...", self.model_path, device, dtype)
        t0 = time.perf_counter()
        self.model = DoDuyTTS.from_pretrained(
            self.model_path, device_map=device, dtype=dtype, load_asr=False
        )
        if self.flashinfer:
            from doduytts.models.doduytts_flashinfer import apply_flashinfer

            apply_flashinfer(self.model)
            logger.info("flashinfer acceleration enabled")
        self.gen_config = DoDuyTTSGenerationConfig(
            num_step=self.num_step,
            # no inserted silence between sentence chunks in a stream
            pad_duration=0.0,
            fade_duration=0.0,
        )
        logger.info("model loaded in %.1fs", time.perf_counter() - t0)
        self.warmup()

    def warmup(self) -> None:
        t0 = time.perf_counter()
        self.synthesize_batch([TTSRequest(text="Hello, this is a warmup sentence.")])
        logger.info("warmup done in %.1fs", time.perf_counter() - t0)

    def create_voice_clone_prompt(self, ref_audio: str, ref_text: str):
        with self._lock:
            return self.model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text
            )

    def synthesize_batch(self, requests: list[TTSRequest]) -> list[np.ndarray]:
        # Group into homogeneous sub-batches, preserving original order.
        groups: dict[tuple, list[int]] = {}
        for idx, req in enumerate(requests):
            if req.voice_clone_prompt is not None:
                key = ("clone", req.language)
            elif req.instruct is not None:
                key = ("design", req.language)
            else:
                key = ("auto", req.language)
            groups.setdefault(key, []).append(idx)

        results: list[Optional[np.ndarray]] = [None] * len(requests)
        with self._lock:
            for (mode, language), indices in groups.items():
                subset = [requests[i] for i in indices]
                kwargs = {}
                if language is not None:
                    kwargs["language"] = language
                if mode == "clone":
                    kwargs["voice_clone_prompt"] = [
                        r.voice_clone_prompt for r in subset
                    ]
                elif mode == "design":
                    kwargs["instruct"] = [r.instruct for r in subset]
                audios = self.model.generate(
                    text=[r.text for r in subset],
                    generation_config=self.gen_config,
                    **kwargs,
                )
                for i, audio in zip(indices, audios):
                    results[i] = audio
        return results  # type: ignore[return-value]


class MockEngine:
    """Stands in for the real model: returns a short tone per sentence.

    Lets the whole pipeline (chunker, batcher, websocket protocol) be
    exercised quickly without loading 3GB of weights.
    """

    sampling_rate = 24000

    def __init__(self, latency_s: float = 0.3):
        self.latency_s = latency_s

    def load(self) -> None:
        pass

    def create_voice_clone_prompt(self, ref_audio: str, ref_text: str):
        return object()

    def synthesize_batch(self, requests: list[TTSRequest]) -> list[np.ndarray]:
        time.sleep(self.latency_s)  # simulate one batched forward pass
        out = []
        for req in requests:
            dur = max(0.3, min(len(req.text) * 0.06, 8.0))
            t = np.linspace(0, dur, int(self.sampling_rate * dur), endpoint=False)
            freq = 220 + (hash(req.session) % 8) * 55
            out.append((0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32))
        return out
