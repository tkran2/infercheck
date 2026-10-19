"""Tokenizer engines behind one interface: encode many strings, return ids and decoded text.

- ``HFTokenizer``: Hugging Face ``tokenizers`` (the reference),
- ``OllamaTokenizer``: Ollama's MLX-runner tokenizer, through the Go harness in harness/ollama_tok,
- ``LlamaCppTokenizer``: llama.cpp's tokenizer, through the C++ harness in harness/llama_tok.

Every engine encodes without adding BOS/EOS and matches added tokens that appear in the text,
which is what Hugging Face ``encode(text, add_special_tokens=False)`` does.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Result:
    ids: list[int] | None
    decoded: str | None
    error: str | None = None


@dataclass
class Engine:
    name: str
    path: str
    meta: dict = field(default_factory=dict)

    def encode_many(self, texts: list[str]) -> list[Result]:  # pragma: no cover - interface
        raise NotImplementedError


def clean_up_tokenization(text: str) -> str:
    """transformers' clean_up_tokenization: what decode does when clean_up_tokenization_spaces is true."""
    return (text.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",").replace(" ' ", "'")
            .replace(" n't", "n't").replace(" 'm", "'m").replace(" 's", "'s").replace(" 've", "'ve")
            .replace(" 're", "'re"))


class HFTokenizer(Engine):
    """Hugging Face tokenizers, decoding the way transformers does for this model: with
    clean_up_tokenization applied exactly when tokenizer_config.json sets
    clean_up_tokenization_spaces to true (transformers 5 defaults it to false)."""

    def __init__(self, tokenizer_json: str):
        super().__init__("hf", tokenizer_json)
        import tokenizers

        self.tok = tokenizers.Tokenizer.from_file(tokenizer_json)
        self.tok.no_padding()
        self.tok.no_truncation()
        cfg_path = Path(tokenizer_json).with_name("tokenizer_config.json")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        self.clean_spaces = bool(cfg.get("clean_up_tokenization_spaces", False))
        self.meta = {"tokenizers": tokenizers.__version__, "clean_up_tokenization_spaces": self.clean_spaces}

    def decode(self, ids: list[int]) -> str:
        text = self.tok.decode(ids, skip_special_tokens=False)
        return clean_up_tokenization(text) if self.clean_spaces else text

    def encode_many(self, texts: list[str]) -> list[Result]:
        out = []
        for t in texts:
            try:
                ids = self.tok.encode(t, add_special_tokens=False).ids
                out.append(Result(list(ids), self.decode(ids)))
            except Exception as exc:  # the reference failing is itself worth knowing
                out.append(Result(None, None, f"{type(exc).__name__}: {exc}"))
        return out


def _run(cmd: list[str], payload: str, timeout: float) -> tuple[str, str, int]:
    proc = subprocess.run(cmd, input=payload.encode("utf-8", "surrogatepass"), capture_output=True, timeout=timeout)
    return proc.stdout.decode("utf-8", "replace"), proc.stderr.decode("utf-8", "replace"), proc.returncode


class OllamaTokenizer(Engine):
    BIN = Path(os.environ.get("INFERCHECK_OLLAMATOK", ROOT / "harness" / "ollama_tok" / "ollamatok"))

    def __init__(self, tokenizer_json: str):
        super().__init__("ollama-mlx", tokenizer_json)

    def encode_many(self, texts: list[str], timeout: float = 600) -> list[Result]:
        payload = "".join(json.dumps({"id": str(i), "text": t}, ensure_ascii=True) + "\n" for i, t in enumerate(texts))
        out, err, code = _run([str(self.BIN), self.path], payload, timeout)
        if code != 0:
            return [Result(None, None, f"exit {code}: {err.strip()[:300]}") for _ in texts]
        rows = [json.loads(line) for line in out.split("\n") if line.strip()]
        res = []
        for r in rows:
            if r.get("error"):
                res.append(Result(None, None, r["error"]))
            else:
                res.append(Result([int(x) for x in r["ids"]], r["decoded"]))
        if len(res) != len(texts):
            raise RuntimeError(f"ollamatok returned {len(res)} rows for {len(texts)} inputs: {err[:300]}")
        return res


def _ollama_decode_ids(self, seqs: list[list[int]], timeout: float = 600) -> list[str | None]:
    payload = "".join(json.dumps({"id": str(i), "ids": ids}) + "\n" for i, ids in enumerate(seqs))
    out, err, code = _run([str(self.BIN), self.path, "-decode"], payload, timeout)
    if code != 0:
        return [None] * len(seqs)
    rows = [json.loads(line) for line in out.split("\n") if line.strip()]
    return [None if r.get("error") else r["decoded"] for r in rows]


OllamaTokenizer.decode_ids = _ollama_decode_ids


class LlamaCppTokenizer(Engine):
    BIN = Path(os.environ.get("INFERCHECK_LLAMATOK", ROOT / "harness" / "llama_tok" / "llamatok"))

    def __init__(self, gguf: str):
        super().__init__("llama.cpp", gguf)

    def encode_many(self, texts: list[str], timeout: float = 600) -> list[Result]:
        payload = "".join(t.encode("utf-8", "surrogatepass").hex() + "\n" for t in texts)
        out, err, code = _run([str(self.BIN), self.path], payload, timeout)
        if code != 0:
            return [Result(None, None, f"exit {code}: {err.strip()[:300]}") for _ in texts]
        lines = out.split("\n")[:len(texts)]
        if len(lines) != len(texts):
            raise RuntimeError(f"llamatok returned {len(lines)} rows for {len(texts)} inputs: {err[:300]}")
        res = []
        for line in lines:
            ids_s, _, dec = line.partition("\t")
            if dec == "ERROR":
                res.append(Result(None, None, "llama_tokenize failed"))
                continue
            ids = [int(x) for x in ids_s.split()] if ids_s else []
            res.append(Result(ids, bytes.fromhex(dec).decode("utf-8", "replace")))
        return res


def _llama_decode_ids(self, seqs: list[list[int]], timeout: float = 600) -> list[str | None]:
    payload = "".join(" ".join(map(str, ids)) + "\n" for ids in seqs)
    out, err, code = _run([str(self.BIN), self.path, "--decode"], payload, timeout)
    if code != 0:
        return [None] * len(seqs)
    lines = out.split("\n")[:len(seqs)]
    return [None if line == "ERROR" else bytes.fromhex(line).decode("utf-8", "replace") for line in lines]


LlamaCppTokenizer.decode_ids = _llama_decode_ids


def source_dir(name: str) -> Path:
    """Where an engine's source checkout is: next to this repo, or in ../ext/."""
    for p in (ROOT.parent / name, ROOT.parent.parent / "ext" / name):
        if p.exists():
            return p
    return ROOT.parent / name


def engine_versions() -> dict:
    """Versions of the code under test, for the record."""
    v = {}
    for name in ("ollama", "llama.cpp"):
        env_path = os.environ.get(f"INFERCHECK_{name.replace('.', '').upper()}_SRC")
        p = Path(env_path) if env_path else source_dir(name)
        try:
            v[name] = subprocess.run(["git", "-C", str(p), "log", "-1", "--format=%H %cd", "--date=short"],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            v[name] = None
    return v
