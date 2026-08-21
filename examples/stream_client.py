"""Demo client for the DoDuyTTS streaming server.

Simulates an LLM streaming text word-by-word into the websocket, receives
per-sentence audio back, and saves the WAV files. With --concurrent N it
opens N sessions at once so the server's micro-batching kicks in.

Usage:
    python examples/stream_client.py                       # one session
    python examples/stream_client.py --concurrent 6       # show batching
    python examples/stream_client.py --text "Xin chào. Đây là thử nghiệm."
"""

import argparse
import asyncio
import base64
import json
import re
import time
from pathlib import Path

import websockets

DEFAULT_TEXT = (
    "Hello there! This is a live streaming test. "
    "The text arrives word by word, just like an LLM would produce it. "
    "Each finished sentence is synthesized immediately. "
    "Latency should stay low even with several sessions at once."
)


async def run_session(idx: int, url: str, text: str, delay_ms: float, out_dir: Path):
    t_start = time.perf_counter()
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "config"}))
        await ws.recv()  # ready

        async def send_tokens():
            for token in re.findall(r"\S+\s*", text):
                await ws.send(json.dumps({"type": "delta", "text": token}))
                await asyncio.sleep(delay_ms / 1000.0)
            await ws.send(json.dumps({"type": "end"}))

        sender = asyncio.create_task(send_tokens())
        try:
            while True:
                msg = json.loads(await ws.recv())
                if msg["type"] == "sentence":
                    print(f"[s{idx}] sentence {msg['seq']}: {msg['text']!r}")
                elif msg["type"] == "audio":
                    path = out_dir / f"session{idx}_seq{msg['seq']}.wav"
                    path.write_bytes(base64.b64decode(msg["data"]))
                    print(
                        f"[s{idx}] audio    {msg['seq']}: {msg['latency_ms']} ms "
                        f"-> {path}"
                    )
                elif msg["type"] == "error":
                    print(f"[s{idx}] ERROR: {msg['message']}")
                elif msg["type"] == "done":
                    print(
                        f"[s{idx}] done: {msg['count']} sentences in "
                        f"{time.perf_counter() - t_start:.1f}s"
                    )
                    break
        finally:
            sender.cancel()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8001/ws/tts")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--delay-ms", type=float, default=25.0)
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--out", default="outputs/stream_client")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(*[
        run_session(i, args.url, args.text, args.delay_ms, out_dir)
        for i in range(args.concurrent)
    ])


if __name__ == "__main__":
    asyncio.run(main())
