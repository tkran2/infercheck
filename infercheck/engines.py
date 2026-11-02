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
import re
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
    # GGML_NO_BACKTRACE: on an abort ggml would otherwise attach gdb for a backtrace, which is slow and can
    # write to stdout
    env = {**os.environ, "GGML_NO_BACKTRACE": "1"}
    proc = subprocess.run(cmd, input=payload.encode("utf-8", "surrogatepass"), capture_output=True, timeout=timeout,
                          env=env)
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

    def __init__(self, gguf: str, bin_path: str | Path | None = None):
        super().__init__("llama.cpp", gguf)
        if bin_path is not None:  # a harness built against another llama.cpp commit (bench)
            self.BIN = Path(bin_path)

    def info(self, timeout: float = 120) -> dict:
        """Special-token behavior of the vocabulary: harness --info (see llamatok.cpp)."""
        out, err, code = _run([str(self.BIN), self.path, "--info"], "", timeout)
        if code != 0:
            hint = " (the harness predates --info: rebuild it)" if "usage:" in err else ""
            raise RuntimeError(f"llamatok --info failed with exit {code}{hint}: {err.strip()[:300]}")
        try:
            return json.loads(out.strip().split("\n")[0])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"llamatok --info printed no JSON: {out[:200]!r}") from exc

    MAX_RESTARTS = 1000
    last_crashes = 0

    LINE = {"": re.compile(r"^[0-9 ]*\t(?:[0-9a-f]*|ERROR)$"), "--decode": re.compile(r"^(?:[0-9a-f]*|ERROR)$")}

    def _batch(self, args: list[str], lines_in: list[str], timeout: float) -> tuple[list[str | None], list[str]]:
        """Feed lines to the harness and collect one output line per input. If the harness crashes (an abort or
        a segfault on one input), the input it died on gets None, the harness restarts after it, and the error
        is kept; the harness flushes every line, so nothing before the crash is lost. A failure to load the
        model (exit 3) or too many crashes marks the remaining inputs None."""
        out_lines: list[str | None] = []
        errors: list[str] = []
        start = 0
        self.last_crashes = 0
        silent = 0  # consecutive crashes with no output at all: the harness cannot start
        while start < len(lines_in):
            chunk = lines_in[start:]
            out, err, code = _run([str(self.BIN), self.path, *args], "".join(x + "\n" for x in chunk), timeout)
            done = out.split("\n")[:-1][:len(chunk)]  # complete lines only
            if code != 0:  # keep only well-formed result lines from before the crash
                pat = self.LINE["--decode" if "--decode" in args else ""]
                n = next((i for i, x in enumerate(done) if not pat.match(x)), len(done))
                done = done[:n]
            out_lines += done
            if code == 0:
                if len(done) != len(chunk):
                    raise RuntimeError(f"llamatok returned {len(done)} rows for {len(chunk)} inputs: {err[:300]}")
                break
            msg = f"exit {code}: {err.strip()[:300]}"
            errors.append(msg)
            silent = silent + 1 if not done else 0
            if code == 3 or silent >= 3 or self.last_crashes >= self.MAX_RESTARTS:
                out_lines += [None] * (len(chunk) - len(done))
                break
            self.last_crashes += 1
            out_lines.append(None)  # the input it died on
            start += len(done) + 1
        return out_lines, errors

    def encode_many(self, texts: list[str], timeout: float = 600) -> list[Result]:
        lines, errors = self._batch([], [t.encode("utf-8", "surrogatepass").hex() for t in texts], timeout)
        res = []
        e = iter(errors)
        last = None
        for line in lines:
            if line is None:
                last = next(e, last)
                res.append(Result(None, None, last or "harness failed"))
                continue
            ids_s, _, dec = line.partition("\t")
            if dec == "ERROR":
                res.append(Result(None, None, "llama_tokenize failed"))
                continue
            ids = [int(x) for x in ids_s.split()] if ids_s else []
            res.append(Result(ids, bytes.fromhex(dec).decode("utf-8", "replace")))
        return res


def _llama_decode_ids(self, seqs: list[list[int]], timeout: float = 600) -> list[str | None]:
    """Decode each id sequence; None where the harness failed or crashed (self.last_crashes counts crashes)."""
    lines, _ = self._batch(["--decode"], [" ".join(map(str, ids)) for ids in seqs], timeout)
    return [None if line is None or line == "ERROR" else bytes.fromhex(line).decode("utf-8", "replace") for line in lines]


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
