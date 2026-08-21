"""Incremental sentence chunker for streaming LLM text.

Feed text deltas as they arrive from the LLM; complete sentences are emitted
as early as possible so TTS generation can start before the LLM finishes.
"""

# Characters that can terminate a sentence (Latin + CJK fullwidth).
_SENTENCE_END = set(".!?…。！？;；\n")

# Words that end with a period without ending the sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "no", "vs", "etc",
    "e.g", "i.e", "approx", "dept", "est", "fig", "vol",
    # Vietnamese
    "tp", "q", "p", "tr", "ths", "pgs", "gs",
}


class SentenceChunker:
    """Accumulates streamed text and yields complete sentences incrementally.

    Args:
        min_chars: sentences shorter than this are held and merged with the
            following text (avoids sending tiny fragments like "OK." alone
            when more text is coming; ``flush()`` always drains everything).
        max_chars: force a split at the last comma/space once the buffer
            grows past this length, so one huge unpunctuated clause cannot
            stall the pipeline.
    """

    def __init__(self, min_chars: int = 8, max_chars: int = 200):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        """Add a text delta; return any sentences completed by it."""
        self._buf += delta
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            if sentence:
                out.append(sentence)
        out.extend(self._force_splits())
        return out

    def flush(self) -> list[str]:
        """Drain whatever remains in the buffer (call when the stream ends)."""
        rest = self._buf.strip()
        self._buf = ""
        return [rest] if rest else []

    @property
    def pending(self) -> str:
        return self._buf

    # -- internals ---------------------------------------------------------

    def _find_cut(self) -> int | None:
        """Index just past a confirmed sentence boundary, or None."""
        buf = self._buf
        for i, ch in enumerate(buf):
            if ch not in _SENTENCE_END:
                continue
            # Need to have seen the next character (or a newline) to be sure
            # the punctuation is final — "3." might continue as "3.5".
            if ch != "\n":
                if i + 1 >= len(buf):
                    return None  # wait for more text
                if not buf[i + 1].isspace() and buf[i + 1] not in _SENTENCE_END:
                    continue
            if ch == "." and self._is_abbreviation(buf, i):
                continue
            # Consume any run of trailing punctuation ("?!", "..." already
            # handled since we scan left-to-right and require whitespace next).
            end = i + 1
            candidate = buf[:end].strip()
            if not any(c.isalnum() for c in candidate):
                continue  # bare punctuation — keep merging
            if len(candidate) < self.min_chars:
                continue  # too short — merge with the next sentence
            return end
        return None

    @staticmethod
    def _is_abbreviation(buf: str, i: int) -> bool:
        """True if the period at buf[i] belongs to an abbreviation/decimal."""
        if 0 < i < len(buf) - 1 and buf[i - 1].isdigit() and buf[i + 1].isdigit():
            return True  # decimal number
        start = i - 1
        while start >= 0 and (buf[start].isalnum() or buf[start] == "."):
            start -= 1
        word = buf[start + 1 : i].lower()
        if len(word) == 1 and word.isalpha():
            return True  # initials like "J."
        return word in _ABBREVIATIONS

    def _force_splits(self) -> list[str]:
        out = []
        while len(self._buf) > self.max_chars:
            window = self._buf[: self.max_chars]
            cut = max(window.rfind(","), window.rfind("，"), window.rfind(" "))
            if cut <= 0:
                cut = self.max_chars
            else:
                cut += 1  # keep the comma with the emitted chunk
            piece = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            if piece:
                out.append(piece)
        return out
