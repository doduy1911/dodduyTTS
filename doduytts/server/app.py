"""Streaming TTS server.

Accepts streaming text (as an LLM produces it), chunks it into sentences
incrementally, and synthesizes speech through a dynamic micro-batcher
(default: up to 4 requests per batch, 20 ms gather window).

Endpoints:
    WS  /ws/tts        — bidirectional streaming (text deltas in, audio out)
    GET /demo/stream   — SSE demo driven by a simulated LLM token stream
    GET /health, /stats

Run:
    doduytts-serve --port 8001                # real model
    doduytts-serve --port 8001 --mock         # pipeline test without weights
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .batcher import MicroBatcher, TTSRequest
from .chunker import SentenceChunker
from .engine import MockEngine, TTSEngine
from .llm_sim import DEFAULT_TEXT, stream_tokens

logger = logging.getLogger("doduytts.server")


@dataclass
class Settings:
    model: str = "doduy1911/dodduyTTS"
    device: str | None = None
    num_step: int = 8
    max_batch_size: int = 4
    window_ms: float = 20.0
    mock: bool = False
    flashinfer: bool = False


settings = Settings()
engine: TTSEngine | MockEngine | None = None
batcher: MicroBatcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, batcher
    engine = MockEngine() if settings.mock else TTSEngine(
        model_path=settings.model,
        device=settings.device,
        num_step=settings.num_step,
        flashinfer=settings.flashinfer,
    )
    await asyncio.to_thread(engine.load)
    batcher = MicroBatcher(
        engine.synthesize_batch,
        max_batch_size=settings.max_batch_size,
        window_ms=settings.window_ms,
    )
    batcher.start()
    logger.info(
        "server ready (mock=%s, max_batch=%d, window=%.0fms)",
        settings.mock, settings.max_batch_size, settings.window_ms,
    )
    yield
    await batcher.stop()


app = FastAPI(title="DoDuyTTS streaming server", lifespan=lifespan)


def _wav_b64(audio: np.ndarray, sample_rate: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/health")
async def health():
    return {"status": "ok", "mock": settings.mock}


@app.get("/stats")
async def stats():
    return {
        "batches": batcher.batches,
        "requests": batcher.requests,
        "avg_batch_size": round(batcher.requests / batcher.batches, 2)
        if batcher.batches else 0.0,
    }


# --------------------------------------------------------------------------
# WebSocket: real streaming input (e.g. piped from an actual LLM)
# --------------------------------------------------------------------------
#
# Client -> server messages (JSON):
#   {"type": "config", "language": ..., "instruct": ...,
#    "ref_audio": <server-side path>, "ref_text": ...}      (optional, first)
#   {"type": "delta", "text": "..."}                        (repeat)
#   {"type": "end"}          -> flush pending text, then server sends "done"
#
# Server -> client messages:
#   {"type": "sentence", "seq": n, "text": ...}             (sentence detected)
#   {"type": "audio", "seq": n, "text": ..., "sample_rate": sr,
#    "latency_ms": ..., "data": <base64 wav>}               (in order)
#   {"type": "done", "count": n} / {"type": "error", "message": ...}

@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    await ws.accept()
    session = f"ws-{id(ws):x}"
    chunker = SentenceChunker()
    config: dict = {}
    seq = 0
    # (seq, text, task) in sentence order; None marks end-of-stream
    pending: asyncio.Queue = asyncio.Queue()

    async def sender():
        done_count = 0
        while True:
            item = await pending.get()
            if item is None:
                await ws.send_text(json.dumps({"type": "done", "count": done_count}))
                continue
            s, text, task, t_submit = item
            try:
                audio = await task
            except Exception as exc:
                await ws.send_text(json.dumps({"type": "error", "seq": s, "message": str(exc)}))
                continue
            await ws.send_text(json.dumps({
                "type": "audio",
                "seq": s,
                "text": text,
                "sample_rate": engine.sampling_rate,
                "latency_ms": round((time.perf_counter() - t_submit) * 1000),
                "data": _wav_b64(audio, engine.sampling_rate),
            }))
            done_count += 1

    async def submit_sentence(text: str):
        nonlocal seq
        s = seq
        seq += 1
        await ws.send_text(json.dumps({"type": "sentence", "seq": s, "text": text}))
        req = TTSRequest(
            text=text,
            language=config.get("language"),
            instruct=config.get("instruct"),
            voice_clone_prompt=config.get("_prompt"),
            session=session,
        )
        task = asyncio.create_task(batcher.submit(req))
        await pending.put((s, text, task, time.perf_counter()))

    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            mtype = msg.get("type")
            if mtype == "config":
                config.update({k: v for k, v in msg.items() if k != "type"})
                if msg.get("ref_audio"):
                    config["_prompt"] = await asyncio.to_thread(
                        engine.create_voice_clone_prompt,
                        msg["ref_audio"], msg.get("ref_text"),
                    )
                await ws.send_text(json.dumps({"type": "ready"}))
            elif mtype == "delta":
                for sentence in chunker.feed(msg.get("text", "")):
                    await submit_sentence(sentence)
            elif mtype == "end":
                for sentence in chunker.flush():
                    await submit_sentence(sentence)
                await pending.put(None)
            else:
                await ws.send_text(json.dumps(
                    {"type": "error", "message": f"unknown message type: {mtype}"}
                ))
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()


# --------------------------------------------------------------------------
# SSE demo: a simulated LLM streams text through the same pipeline
# --------------------------------------------------------------------------

@app.get("/demo/stream")
async def demo_stream(
    text: str = DEFAULT_TEXT,
    delay_ms: float = 25.0,
    language: str | None = None,
    instruct: str | None = None,
):
    session = f"sse-{time.monotonic_ns():x}"

    async def events():
        chunker = SentenceChunker()
        out: asyncio.Queue = asyncio.Queue()

        async def produce():
            seq = 0

            async def emit(sentence: str):
                nonlocal seq
                req = TTSRequest(
                    text=sentence, language=language,
                    instruct=instruct, session=session,
                )
                task = asyncio.create_task(batcher.submit(req))
                await out.put(("sentence", seq, sentence, task, time.perf_counter()))
                seq += 1

            async for token in stream_tokens(text, delay_ms):
                for sentence in chunker.feed(token):
                    await emit(sentence)
            for sentence in chunker.flush():
                await emit(sentence)
            await out.put(None)

        producer = asyncio.create_task(produce())
        try:
            count = 0
            while True:
                item = await out.get()
                if item is None:
                    yield f"event: done\ndata: {json.dumps({'count': count})}\n\n"
                    break
                _, s, sentence, task, t_submit = item
                yield f"event: sentence\ndata: {json.dumps({'seq': s, 'text': sentence})}\n\n"
                audio = await task
                payload = {
                    "seq": s,
                    "text": sentence,
                    "sample_rate": engine.sampling_rate,
                    "latency_ms": round((time.perf_counter() - t_submit) * 1000),
                    "data": _wav_b64(audio, engine.sampling_rate),
                }
                yield f"event: audio\ndata: {json.dumps(payload)}\n\n"
                count += 1
        finally:
            producer.cancel()

    return StreamingResponse(events(), media_type="text/event-stream")


def main():
    parser = argparse.ArgumentParser(description="DoDuyTTS streaming TTS server")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="doduy1911/dodduyTTS")
    parser.add_argument("--device", default=None, help="cuda | xpu | mps | cpu (auto)")
    parser.add_argument("--num-step", type=int, default=8,
                        help="diffusion steps (8 = fast, 32 = best quality)")
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--mock", action="store_true",
                        help="use a mock engine (no model load) to test the pipeline")
    parser.add_argument("--flashinfer", action="store_true",
                        help="enable FlashInfer acceleration (NVIDIA GPU only)")
    args = parser.parse_args()

    settings.model = args.model
    settings.device = args.device
    settings.num_step = args.num_step
    settings.max_batch_size = args.max_batch
    settings.window_ms = args.window_ms
    settings.mock = args.mock
    settings.flashinfer = args.flashinfer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import uvicorn
    uvicorn.run(app, host=args.ip, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
