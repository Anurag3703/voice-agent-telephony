"""
Response phrase / sentence chunker – Stage 5 tightened version.

Goals:
- Emit the *first* usable phrase as early as possible (aggressive first flush)
- Prefer natural boundaries (sentence > clause > comma > word)
- Avoid emitting tiny trailing fragments
- Support continuous streaming so LLM and TTS stay overlapped
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_SENTENCE_END = re.compile(r"([.!?])(\s+|$)")
_CLAUSE = re.compile(r"([,;:\n—–])(\s+|$)")


@dataclass
class ResponseChunker:
    # First chunk can be shorter so TTS starts sooner
    first_min_chars: int = 6
    min_chars: int = 16
    max_chars: int = 90
    min_final_chars: int = 4

    _buffer: str = field(default="", init=False)
    _emitted_first: bool = field(default=False, init=False)

    def reset(self) -> None:
        self._buffer = ""
        self._emitted_first = False

    def _min_for_next(self) -> int:
        return self.first_min_chars if not self._emitted_first else self.min_chars

    def add(self, text: str) -> list[str]:
        if not text:
            return []
        self._buffer += text
        return self._drain(force=False)

    def flush(self) -> list[str]:
        chunks = self._drain(force=True)
        leftover = self._buffer.strip()
        self._buffer = ""
        if leftover and (len(leftover) >= self.min_final_chars or not chunks):
            chunks.append(leftover)
            self._emitted_first = True
        return chunks

    def _drain(self, force: bool) -> list[str]:
        chunks: list[str] = []
        min_c = self._min_for_next()

        while True:
            buf = self._buffer
            if len(buf) < min_c and not force:
                break
            if not buf.strip():
                self._buffer = ""
                break

            window = buf[: self.max_chars]

            # 1. Sentence boundary
            sent = None
            for m in _SENTENCE_END.finditer(window):
                if m.end() >= min_c:
                    sent = m
            if sent:
                end = sent.end()
                chunk = buf[:end].strip()
                if chunk:
                    chunks.append(chunk)
                    self._emitted_first = True
                self._buffer = buf[end:]
                min_c = self._min_for_next()
                continue

            # 2. Clause / comma
            if len(buf) >= min_c:
                clause = None
                for m in _CLAUSE.finditer(window):
                    if m.end() >= min_c:
                        clause = m
                if clause:
                    end = clause.end()
                    chunk = buf[:end].strip()
                    if chunk:
                        chunks.append(chunk)
                        self._emitted_first = True
                    self._buffer = buf[end:]
                    min_c = self._min_for_next()
                    continue

            # 3. Forced word-boundary split
            if len(buf) >= self.max_chars or force:
                if len(buf) <= self.max_chars and force:
                    chunk = buf.strip()
                    if chunk:
                        chunks.append(chunk)
                        self._emitted_first = True
                    self._buffer = ""
                    break

                split_at = buf.rfind(" ", 0, self.max_chars)
                if split_at < min_c:
                    split_at = self.max_chars
                chunk = buf[:split_at].strip()
                if chunk:
                    chunks.append(chunk)
                    self._emitted_first = True
                self._buffer = buf[split_at:].lstrip()
                min_c = self._min_for_next()
                continue

            break

        return chunks
