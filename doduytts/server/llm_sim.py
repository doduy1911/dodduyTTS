"""Simulated LLM token stream for demos and testing."""

import asyncio
import re
from typing import AsyncIterator

DEFAULT_TEXT = (
    "Hello! This is a streaming text to speech demo. "
    "Sentences are synthesized as soon as they are complete, "
    "so playback can start long before the language model finishes. "
    "Requests from concurrent sessions are gathered into micro batches "
    "of up to four, which keeps the model busy and the latency low."
)


async def stream_tokens(
    text: str = DEFAULT_TEXT, delay_ms: float = 25.0
) -> AsyncIterator[str]:
    """Yield the text word by word, like an LLM emitting tokens."""
    for token in re.findall(r"\S+\s*", text):
        yield token
        await asyncio.sleep(delay_ms / 1000.0)
