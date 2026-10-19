"""Compare engines against the reference, shrink each disagreement, and group the findings.

A disagreement is one of:

- ``encode``: the engine's token ids differ from the reference's,
- ``decode``: the ids agree but decoding them gives different text,
- ``error``: the engine failed (error or crash) where the reference did not.

Each disagreement is shrunk with delta debugging over code points to a string that still
disagrees in the same way, removing any one code point of which makes it agree (1-minimal).
Findings with the same shrunk string are one finding.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import asdict, dataclass

from .engines import Engine, HFTokenizer, LlamaCppTokenizer, OllamaTokenizer, Result


class Persistent:
    """Hold an engine's harness process open and query it one string at a time."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.lock = threading.Lock()
        if isinstance(engine, HFTokenizer):
            self.proc = None
            return
        if isinstance(engine, OllamaTokenizer):
            cmd = [str(engine.BIN), engine.path]
        elif isinstance(engine, LlamaCppTokenizer):
            cmd = [str(engine.BIN), engine.path]
        else:
            raise TypeError(engine)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def query(self, text: str) -> Result:
        if self.proc is None:
            return self.engine.encode_many([text])[0]
        with self.lock:
            if self.proc.poll() is not None:  # crashed earlier: restart
                self.__init__(self.engine)
            try:
                if isinstance(self.engine, OllamaTokenizer):
                    self.proc.stdin.write((json.dumps({"id": "0", "text": text}) + "\n").encode())
                    self.proc.stdin.flush()
                    line = self.proc.stdout.readline()
                    if not line:
                        raise RuntimeError(self.proc.stderr.read().decode()[:300])
                    r = json.loads(line)
                    if r.get("error"):
                        return Result(None, None, r["error"])
                    return Result([int(x) for x in r["ids"]], r["decoded"])
                self.proc.stdin.write(text.encode("utf-8", "surrogatepass").hex().encode() + b"\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError(self.proc.stderr.read().decode()[:300])
                ids_s, _, dec = line.decode().rstrip("\n").partition("\t")
                if dec == "ERROR":
                    return Result(None, None, "llama_tokenize failed")
                return Result([int(x) for x in ids_s.split()] if ids_s else [],
                              bytes.fromhex(dec).decode("utf-8", "replace"))
            except Exception as exc:
                try:
                    self.proc.kill()
                except Exception:
                    pass
                return Result(None, None, f"crash: {type(exc).__name__}: {str(exc)[:200]}")

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


def kind_of(ref: Result, got: Result) -> str | None:
    """The kind of disagreement between the reference and an engine, or None if they agree."""
    if ref.error is not None:
        return None  # nothing to compare against
    if got.error is not None:
        return "error"
    if got.ids != ref.ids:
        return "encode"
    if got.decoded != ref.decoded:
        return "decode"
    return None


def ddmin_chars(text: str, still_fails) -> str:
    """Zeller and Hildebrandt's ddmin over code points: a 1-minimal substring sequence."""
    chars = list(text)
    n = 2
    while len(chars) >= 2:
        size = max(1, len(chars) // n)
        chunks = [chars[i:i + size] for i in range(0, len(chars), size)]
        reduced = False
        for i in range(len(chunks)):
            subset = chunks[i]
            if still_fails("".join(subset)):
                chars, n, reduced = subset, 2, True
                break
        if not reduced:
            for i in range(len(chunks)):
                complement = [c for j, ch in enumerate(chunks) if j != i for c in ch]
                if complement and still_fails("".join(complement)):
                    chars, n, reduced = complement, max(n - 1, 2), True
                    break
        if not reduced:
            if n >= len(chars):
                break
            n = min(len(chars), 2 * n)
    return "".join(chars)


@dataclass
class Finding:
    engine: str
    kind: str
    case_id: str
    text: str
    minimal: str
    ref_ids: list[int] | None
    got_ids: list[int] | None
    ref_decoded: str | None
    got_decoded: str | None
    got_error: str | None
    min_ref_ids: list[int] | None = None
    min_got_ids: list[int] | None = None
    min_ref_decoded: str | None = None
    min_got_decoded: str | None = None
    min_got_error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def compare(ref: HFTokenizer, engines: list[Engine], cases, shrink: bool = True, max_shrink: int = 200,
            log=print) -> tuple[list[Finding], dict]:
    texts = [c.text for c in cases]
    ref_res = ref.encode_many(texts)
    findings: list[Finding] = []
    stats = {"cases": len(cases), "reference_errors": sum(r.error is not None for r in ref_res), "engines": {}}
    for eng in engines:
        got_res = eng.encode_many(texts)
        kinds = [kind_of(r, g) for r, g in zip(ref_res, got_res)]
        counts = {k: kinds.count(k) for k in ("encode", "decode", "error")}
        stats["engines"][eng.name] = counts
        log(f"  {eng.name}: {counts['encode']} encode, {counts['decode']} decode, {counts['error']} error disagreements "
            f"in {len(cases)} cases")
        bad = [(c, r, g, k) for c, r, g, k in zip(cases, ref_res, got_res, kinds) if k]
        p_eng, p_ref = (Persistent(eng), Persistent(ref)) if shrink and bad else (None, None)
        seen_min: set[str] = set()
        try:
            for i, (c, r, g, k) in enumerate(bad):
                f = Finding(eng.name, k, c.id, c.text, c.text, r.ids, g.ids, r.decoded, g.decoded, g.error)
                if shrink and i < max_shrink:
                    def still(s: str, k=k) -> bool:
                        return kind_of(p_ref.query(s), p_eng.query(s)) == k
                    f.minimal = ddmin_chars(c.text, still)
                    mr, mg = p_ref.query(f.minimal), p_eng.query(f.minimal)
                    f.min_ref_ids, f.min_got_ids = mr.ids, mg.ids
                    f.min_ref_decoded, f.min_got_decoded, f.min_got_error = mr.decoded, mg.decoded, mg.error
                if f.minimal in seen_min:
                    continue
                seen_min.add(f.minimal)
                findings.append(f)
        finally:
            for p in (p_eng, p_ref):
                if p:
                    p.close()
    return findings, stats
