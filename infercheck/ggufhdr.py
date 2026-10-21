"""Read the metadata (key/value) section of a GGUF file without downloading its tensors,
and write it back out as a vocab-only GGUF that llama.cpp can load with vocab_only.

A GGUF file starts with: magic "GGUF", version (u32), tensor count (u64), key/value count (u64),
then the key/value pairs, then tensor infos, then tensor data. The tokenizer (tokens, merges,
token types, pre-tokenizer name) lives in the key/value pairs, so the first few megabytes of a
file are enough to rebuild its tokenizer exactly.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# GGUF value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
_FIXED = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", BOOL: "<?",
          U64: "<Q", I64: "<q", F64: "<d"}


class GGUFFormatError(ValueError):
    pass


class _Reader:
    """Sequential reader over a byte source that is fetched in chunks on demand."""

    def __init__(self, fetch: Callable[[int, int], bytes], chunk: int = 4 << 20, limit: int = 256 << 20):
        self.fetch, self.chunk, self.limit = fetch, chunk, limit
        self.buf = bytearray()
        self.pos = 0
        self.eof = False

    def _need(self, n: int) -> None:
        while len(self.buf) - self.pos < n:
            if self.eof:
                raise GGUFFormatError("unexpected end of file in metadata")
            if len(self.buf) >= self.limit:
                raise GGUFFormatError(f"metadata larger than {self.limit} bytes")
            data = self.fetch(len(self.buf), self.chunk)
            if not data:
                self.eof = True
            self.buf += data
            if len(data) < self.chunk:
                self.eof = True

    def take(self, n: int) -> bytes:
        self._need(n)
        out = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return out

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))[0]


def _read_value(r: _Reader, vtype: int, depth: int = 0):
    if vtype in _FIXED:
        return r.unpack(_FIXED[vtype])
    if vtype == STR:
        n = r.unpack("<Q")
        return r.take(n).decode("utf-8", "surrogateescape")
    if vtype == ARR:
        if depth > 2:
            raise GGUFFormatError("nested arrays too deep")
        sub = r.unpack("<I")
        count = r.unpack("<Q")
        if sub in _FIXED:
            fmt = _FIXED[sub]
            size = struct.calcsize(fmt)
            raw = r.take(size * count)
            return list(struct.unpack("<" + fmt[1:] * count, raw)) if count else []
        return [_read_value(r, sub, depth + 1) for _ in range(count)]
    raise GGUFFormatError(f"unknown value type {vtype}")


@dataclass
class GGUFHeader:
    version: int
    n_tensors: int
    kv: dict = field(default_factory=dict)
    kv_bytes: bytes = b""          # the raw key/value section, exactly as in the file
    n_kv: int = 0
    kv_spans: list = field(default_factory=list)  # (key, start, end) of each pair within kv_bytes

    def get(self, key: str, default=None):
        return self.kv.get(key, default)

    @property
    def arch(self) -> str | None:
        return self.kv.get("general.architecture")

    @property
    def tokenizer_model(self) -> str | None:
        return self.kv.get("tokenizer.ggml.model")

    @property
    def pre(self) -> str | None:
        return self.kv.get("tokenizer.ggml.pre")


def read_header(fetch: Callable[[int, int], bytes], **kw) -> GGUFHeader:
    """Parse magic, counts and every key/value pair. ``fetch(offset, size)`` returns file bytes."""
    r = _Reader(fetch, **kw)
    if r.take(4) != b"GGUF":
        raise GGUFFormatError("not a GGUF file")
    version = r.unpack("<I")
    if version < 2:
        raise GGUFFormatError(f"GGUF version {version} not supported")
    n_tensors = r.unpack("<Q")
    n_kv = r.unpack("<Q")
    start = r.pos
    kv = {}
    spans = []
    for _ in range(n_kv):
        p0 = r.pos - start
        klen = r.unpack("<Q")
        key = r.take(klen).decode("utf-8", "surrogateescape")
        vtype = r.unpack("<I")
        kv[key] = _read_value(r, vtype)
        spans.append((key, p0, r.pos - start))
    return GGUFHeader(version, n_tensors, kv, bytes(r.buf[start:r.pos]), n_kv, spans)


def write_vocab_only(header: GGUFHeader, path: str | Path, alignment: int = 32) -> Path:
    """Write a GGUF with the same key/value pairs and no tensors (what --vocab-only produces).
    The split.* keys of a sharded model are left out: with them, llama.cpp looks for the other shards."""
    path = Path(path)
    spans = header.kv_spans or [(None, 0, len(header.kv_bytes))]
    keep = [(a, b) for k, a, b in spans if not (k or "").startswith("split.")]
    n_kv = header.n_kv - (len(spans) - len(keep)) if header.kv_spans else header.n_kv
    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", header.version, 0, n_kv)
    for a, b in keep:
        out += header.kv_bytes[a:b]
    pad = (-len(out)) % int(header.kv.get("general.alignment", alignment) or alignment)
    out += b"\0" * pad
    path.write_bytes(bytes(out))
    return path


def file_fetcher(path: str | Path) -> Callable[[int, int], bytes]:
    p = Path(path)

    def fetch(offset: int, size: int) -> bytes:
        with open(p, "rb") as fh:
            fh.seek(offset)
            return fh.read(size)
    return fetch
