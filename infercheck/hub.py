"""Audit GGUF files on the Hugging Face Hub without downloading the models.

For each GGUF repo:

1. read only the metadata section of one GGUF file (HTTP range requests; a few MB),
2. rebuild it as a vocab-only GGUF and tokenize the InferCheck corpus with llama.cpp,
3. tokenize the same corpus with the tokenizer.json of the model the GGUF was converted from,
4. record every disagreement (shrunk to a minimal string),
5. also compare that source tokenizer with its own base model's tokenizer ("drift": fine-tunes
   that publish a tokenizer.json different from the model they were trained from).

Results go to OUT/records.jsonl (one line per repo, resumable) and OUT/findings/*.jsonl.
`report` turns them into a Markdown table sorted by downloads.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

from .ggufhdr import GGUFFormatError, read_header, write_vocab_only

HF = "https://huggingface.co"
SKIP_FILE = re.compile(r"mmproj|vae|clip|encoder|projector|vision|audio|tts|imatrix", re.I)
PROBES = [
    "भारत एक विशाल देश है और यहाँ कई भाषाएँ बोली जाती हैं।",
    "আমি বাংলায় কথা বলতে ভালোবাসি।",
    "தமிழ் ஒரு பழமையான மொழி ஆகும்.",
    "ภาษาไทยเป็นภาษาที่มีวรรณยุกต์",
    "اَلْعَرَبِيَّةُ لُغَةٌ جَمِيلَةٌ",
    "Tiếng Việt có nhiều dấu thanh.",
    "Café vs Café, naïve résumé",
    "def f(x):\r\n    return x  # 1234567\r\n",
    "The quick brown fox jumps over the lazy dog.",
    "I'VE 'RE 'Ll we'Ve 3.14159 1,000,000",
]


# ----------------------------------------------------------------------------- Hub access

def _session():
    import requests
    from huggingface_hub.utils import build_hf_headers

    s = requests.Session()
    s.headers.update(build_hf_headers())  # uses the token from `hf auth login` or HF_TOKEN, if any
    return s


_S = None


def session():
    global _S
    if _S is None:
        _S = _session()
    return _S


def api(path: str, params: dict | None = None, retries: int = 3):
    for i in range(retries):
        r = session().get(f"{HF}/api/{path}", params=params, timeout=60)
        if r.status_code == 429 and i < retries - 1:
            time.sleep(10 * (i + 1))
            continue
        r.raise_for_status()
        return r.json(), r.headers
    return None, {}


def top_gguf_repos(n: int) -> list[dict]:
    """The n most-downloaded (last 30 days) GGUF repos."""
    out, url, params = [], f"{HF}/api/models", {"filter": "gguf", "sort": "downloads", "direction": "-1", "limit": 100}
    while url and len(out) < n:
        r = session().get(url, params=params, timeout=60)
        r.raise_for_status()
        out += [{"id": m["id"], "downloads": m.get("downloads", 0)} for m in r.json()]
        nxt = r.links.get("next", {}).get("url")
        url, params = nxt, None
    return out[:n]


def model_info(repo: str) -> dict:
    info, _ = api(f"models/{repo}")
    return info or {}


def card_base_models(info: dict) -> list[str]:
    bm = (info.get("cardData") or {}).get("base_model")
    if isinstance(bm, str):
        bm = [bm]
    return [b for b in (bm or []) if isinstance(b, str) and "/" in b]


def pick_gguf(repo: str, revision: str = "main") -> str | None:
    tree, _ = api(f"models/{repo}/tree/{revision}", {"recursive": "true"})
    files = [f["path"] for f in tree or [] if f.get("type") == "file" and f["path"].endswith(".gguf")
             and not SKIP_FILE.search(f["path"])]
    if not files:
        return None
    first_split = [p for p in files if re.search(r"-00001-of-\d+\.gguf$", p)]
    singles = [p for p in files if not re.search(r"-\d{5}-of-\d+\.gguf$", p)]
    for pref in (r"Q4_K_M", r"Q8_0", r"Q4_0", r"BF16|F16"):
        for pool in (singles, first_split):
            hit = [p for p in pool if re.search(pref, p, re.I)]
            if hit:
                return sorted(hit, key=len)[0]
    return (singles or first_split or files)[0]


def range_fetcher(repo: str, path: str, revision: str | None = None):
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo, path, revision=revision)

    def fetch(offset: int, size: int) -> bytes:
        for attempt in range(3):
            try:
                r = session().get(url, headers={"Range": f"bytes={offset}-{offset + size - 1}"}, stream=True,
                                  timeout=120)
                if r.status_code not in (200, 206):
                    r.raise_for_status()
                if r.status_code == 200 and offset:  # server ignored the range
                    raise RuntimeError("range requests not honored")
                data = bytearray()
                for chunk in r.iter_content(1 << 20):
                    data += chunk
                    if len(data) >= size:
                        break
                r.close()
                return bytes(data[:size])
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)
        return b""
    return fetch


def fetch_tokenizer(repo: str, cache: Path, revision: str | None = None) -> Path | None:
    """tokenizer.json (+ tokenizer_config.json and generation_config.json if present) of a repo, or None
    if unavailable. The files land in one snapshot folder, so the others sit next to tokenizer.json."""
    from huggingface_hub import hf_hub_download

    try:
        tj = Path(hf_hub_download(repo, "tokenizer.json", cache_dir=str(cache), revision=revision))
    except Exception:
        return None
    # special_tokens_map.json and config.json let transformers' AutoTokenizer load the folder; chat_template.jinja
    # (newer repos keep the template out of tokenizer_config.json) says whether chat prompts start with BOS (special.py)
    for name in ("tokenizer_config.json", "generation_config.json", "special_tokens_map.json", "config.json",
                 "chat_template.jinja"):
        try:
            hf_hub_download(repo, name, cache_dir=str(cache), revision=revision)
        except Exception:
            pass
    return tj


def hf_repo_from_url(url) -> str | None:
    if not isinstance(url, str):
        return None
    m = re.match(r"https?://huggingface\.co/([^/\s]+/[^/\s?#]+)", url)
    return m.group(1) if m else None


# ----------------------------------------------------------------------------- checks

def classify(minimal: str, special: list[str]) -> str:
    """Rough cause of a finding, from its minimal string, for triage across many repos."""
    if any(t and t in minimal for t in special):
        return "special-token"
    if "\r" in minimal:
        return "crlf"
    if unicodedata.normalize("NFC", minimal) != minimal:
        return "nfc"
    if any(unicodedata.category(c).startswith("M") for c in minimal):
        return "combining-mark"
    if any(c.isdigit() for c in minimal):
        return "digits"
    if "'" in minimal:
        return "apostrophe"
    if all(c.isspace() or unicodedata.category(c) in ("Zs", "Cc", "Cf") for c in minimal):
        return "whitespace"
    if any(c.isspace() for c in minimal):
        return "whitespace+text"
    return "other"


def _llama_cleanup(s: str) -> str:
    """What llama.cpp's detokenizer does when clean_spaces is on (llama-vocab.cpp, three passes)."""
    s = re.sub(r" ([?!.,])", r"\1", s)
    s = s.replace(" ' ", "'")
    return re.sub(r" '(s|m|re|ve)", r"'\1", s)


def cause_of(f, special: list[str]) -> str:
    """Cause of one finding. Unlike classify(), this checks the two causes that explain most findings across
    the Hub first: the reference normalizes to NFC (llama.cpp's BPE path does not), and llama.cpp's detokenizer
    cleans up spaces before punctuation when the source tokenizer does not."""
    if f.kind == "error":
        return "engine-error"
    if f.kind == "decode":
        ref = f.min_ref_decoded if f.min_ref_decoded is not None else f.ref_decoded
        got = f.min_got_decoded if f.min_got_decoded is not None else f.got_decoded
        if ref is not None and got is not None and got != ref and _llama_cleanup(ref) == got:
            return "decode-cleanup"
    text = f.minimal or f.text
    if unicodedata.normalize("NFC", text) != text:
        return "nfc"
    return classify(text, special)


CAUSES = {
    "nfc": "the source tokenizer normalizes text to NFC; llama.cpp's BPE tokenizer does not "
           "(ggml-org/llama.cpp#30747, fix in review in #30708)",
    "decode-cleanup": "llama.cpp's detokenizer removes the space before . , ! ? and 's 'm 're 've "
                      "(clean_up_tokenization_spaces) although the source tokenizer leaves it off",
    "special-token": "text next to added or special tokens (lstrip/rstrip and normalization flags)",
    "control-text": "text llama.cpp parses as an end-of-generation control token although the source tokenizer treats "
                    "it as ordinary text (Qwen's </s> since llama.cpp #18825)",
    "crlf": "carriage returns",
    "digits": "digit grouping",
    "combining-mark": "combining marks",
    "apostrophe": "apostrophes and contractions",
    "whitespace": "runs of whitespace",
    "whitespace+text": "whitespace followed by text",
    "other": "other",
}


def _load_failed(r: dict) -> bool:
    """llama.cpp could not load the vocabulary (every case errored), as opposed to tokenizing differently."""
    e = r.get("engine") or {}
    return r.get("status") == "engine-error" or (bool(e.get("cases")) and (e.get("error") or 0) >= 0.5 * e["cases"])


def _sha(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:16]


def tokenizer_key(header) -> str:
    keys = sorted(k for k in header.kv if k.startswith("tokenizer.") and k != "tokenizer.chat_template")
    blob = json.dumps([header.arch] + [[k, header.kv[k]] for k in keys], ensure_ascii=False, default=str)
    return _sha(blob.encode("utf-8", "surrogatepass"))


def _pretok_signature(tokenizer_json: Path) -> dict:
    tj = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    cfg_p = tokenizer_json.with_name("tokenizer_config.json")
    cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
    return {"normalizer": tj.get("normalizer"), "pre_tokenizer": tj.get("pre_tokenizer"),
            "clean_up_tokenization_spaces": cfg.get("clean_up_tokenization_spaces"),
            "tokenizer_class": cfg.get("tokenizer_class"), "vocab_size": len((tj.get("model") or {}).get("vocab") or {})}


def drift(source_tj: Path, base_tj: Path) -> dict | None:
    """How a fine-tune's tokenizer differs from its base model's: fields and probe-text token ids."""
    from tokenizers import Tokenizer

    a, b = _pretok_signature(source_tj), _pretok_signature(base_tj)
    fields = [k for k in ("normalizer", "pre_tokenizer", "clean_up_tokenization_spaces") if a[k] != b[k]]
    ta, tb = Tokenizer.from_file(str(source_tj)), Tokenizer.from_file(str(base_tj))
    probes = [p for p in PROBES if ta.encode(p, add_special_tokens=False).ids != tb.encode(p, add_special_tokens=False).ids]
    if not fields and not probes:
        return None
    return {"fields": fields, "probes_differ": len(probes), "probes": len(PROBES), "example": probes[:1],
            "source_class": a["tokenizer_class"], "base_class": b["tokenizer_class"]}


def refdecode_check(ref, eng, texts: list[str], examples: int = 5, keys: bool = False) -> dict:
    """Decode the reference's token ids with the engine and compare with the reference's decoding.
    This reaches token ids the engine's own tokenizer never produces (byte-fallback tokens, for
    instance), so it catches detokenizer bugs and crashes that encode-and-decode round trips miss.
    With keys, also list the keys of the strings compared and of those that differ (see diff.text_key)."""
    from .diff import text_key

    ref_res = ref.encode_many(texts)
    pairs = [(t, r) for t, r in zip(texts, ref_res) if r.error is None and r.ids]
    if not pairs:
        return {"compared": 0, "mismatch": 0, "error": 0}
    got = eng.decode_ids([r.ids for _, r in pairs])
    out = {"compared": len(pairs), "mismatch": 0, "error": 0, "examples": []}
    if getattr(eng, "last_crashes", 0) or (got and all(g is None for g in got)):
        out["crash"] = True  # the harness aborted or segfaulted while decoding some of these
    bad = set()
    for (t, r), g in zip(pairs, got):
        if g is None:
            out["error"] += 1
            bad.add(text_key(t))
        elif g != r.decoded:
            out["mismatch"] += 1
            bad.add(text_key(t))
            if len(out["examples"]) < examples:
                out["examples"].append({"text": t[:80], "ids": r.ids[:16], "ref": r.decoded[:80], "got": g[:80]})
    if keys:
        out["corpus"] = sorted({text_key(t) for t, _ in pairs})
        out["bad"] = sorted(bad)
    return out


def stress_check(ref, eng) -> dict:
    """Long inputs (tens of thousands of repeated characters), where regex backtracking overflows the stack."""
    from .corpus import stress_cases
    from .diff import kind_of

    cases = stress_cases()
    texts = [c.text for c in cases]
    ref_res = ref.encode_many(texts)
    got = eng.encode_many(texts)
    if all(g.error for g in got):  # one input took the batch down: rerun them one by one to see which
        got = [eng.encode_many([t])[0] for t in texts]
    kinds = [kind_of(r, g) for r, g in zip(ref_res, got)]
    bad = [{"case": c.id, "kind": k, "error": (g.error or "")[:120]} for c, g, k in zip(cases, got, kinds) if k]
    return {"cases": len(cases), "mismatch": sum(k in ("encode", "decode") for k in kinds),
            "error": kinds.count("error"), "failures": bad,
            "corpus": [c.id for c, r in zip(cases, ref_res) if r.error is None]}


def special_check(eng, ref_tj: Path) -> dict:
    """Start/stop-token comparison (see special.py)."""
    from .special import compare_special, reference_special

    try:
        info = eng.info()
    except Exception as exc:
        return {"error": str(exc)[:300]}
    try:
        ref = reference_special(ref_tj)
    except Exception as exc:
        return {"error": f"reference: {exc}"[:300]}
    issues = compare_special(ref, info)
    return {"issues": issues, "add_bos": info.get("add_bos"), "add_eos": info.get("add_eos"), "ref_impl": ref.get("impl"),
            "eog": [e["id"] for e in info.get("eog", [])], "ref_stops": sorted(int(k) for k in ref["stops"])}


def run_engine_check(vocab_gguf: Path, ref_tj: Path, n_gen: int, seed: int, max_shrink: int,
                     bin_path: Path | None = None, extra: bool = True, keys: bool = False) -> tuple[dict, list]:
    """Encode/decode differential test of llama.cpp (the harness at bin_path, or the default build)
    against the reference tokenizer. With extra, also run refdecode, stress and special checks; their
    results are under the "refdecode", "stress" and "special" keys. With keys, record which strings
    disagree (for comparing two runs string by string; bench uses this, hub-scan does not)."""
    from .cli import added_tokens
    from .corpus import fixed_cases, generated_cases, special_cases
    from .diff import compare
    from .engines import HFTokenizer, LlamaCppTokenizer

    ref = HFTokenizer(str(ref_tj))
    eng = LlamaCppTokenizer(str(vocab_gguf), bin_path=bin_path)
    eng.name = "llama.cpp"
    special = added_tokens(ref_tj)[:300]
    engine_only: list[str] = []
    if extra:
        # also the text of every token llama.cpp treats as end-of-generation: llama.cpp parses those as control
        # tokens in prompts, so one that is ordinary text in the reference (Qwen's "</s>") tokenizes differently
        try:
            engine_only = [e["text"] for e in eng.info().get("eog", [])
                           if e.get("text") and e["text"] not in special and len(e["text"]) > 1]
            special += engine_only
        except Exception:
            pass  # a harness without --info: the corpus is as before
    cases = fixed_cases() + special_cases(special) + generated_cases(n_gen, seed, special)
    if extra:
        from .corpus import vocab_pieces, word_cases
        try:
            size = ref.tok.get_vocab_size(with_added_tokens=False)
            pieces = vocab_pieces(lambda ids: [ref.tok.decode([i], skip_special_tokens=True) for i in ids], size, seed=seed)
            cases += word_cases(pieces, seed=seed)
        except Exception:
            pass
    findings, stats = compare(ref, [eng], cases, shrink=max_shrink > 0, max_shrink=max_shrink, log=lambda *a, **k: None,
                              record_keys=keys)
    s = stats["engines"].get("llama.cpp", {})
    classes: dict = {}
    causes: dict = {}
    for f in findings:
        c = classify(f.minimal, special)
        classes[c] = classes.get(c, 0) + 1
        f.cls = cause_of(f, special)
        if f.cls == "special-token" and any(t in (f.minimal or f.text) for t in engine_only):
            f.cls = "control-text"
        causes[f.cls] = causes.get(f.cls, 0) + 1
    result = {"cases": stats["cases"], **s, "reference_errors": stats["reference_errors"], "classes": classes, "causes": causes}
    if keys:
        result["corpus"] = stats.get("corpus", [])
    err = next((f.got_error or f.min_got_error for f in findings if f.kind == "error" and (f.got_error or f.min_got_error)), None)
    if err:
        result["error_sample"] = err[:300]
    if extra and not (_load_failed({"engine": result}) and (result.get("error_sample") or "").startswith("exit 3")):
        for key, fn in (("refdecode", lambda: refdecode_check(ref, eng, [c.text for c in cases], keys=keys)),
                        ("stress", lambda: stress_check(ref, eng)),
                        ("special", lambda: special_check(eng, ref_tj))):
            try:
                result[key] = fn()
            except Exception as exc:
                result[key] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return result, findings


def source_candidates(repo: str, header, info: dict) -> list[str]:
    """Repos that may hold the tokenizer a GGUF was converted from, most likely first:
    general.source.repo_url; the same organization's repo without the -GGUF suffix (a first-party upload such
    as Qwen/Qwen3-Embedding-0.6B-GGUF); the GGUF repo card's base_model (the source, for quantizer repos);
    and last general.base_model.0, which the converter copies from the source's own card and so names the
    model the source was trained from (ibm-granite/granite-4.2-8b-GGUF names granite-4.1-8b-base there)."""
    org, _, name = repo.partition("/")
    stripped = re.sub(r"[-_.](gguf|GGUF)$", "", name)
    cands = [hf_repo_from_url(header.get("general.source.repo_url")),
             f"{org}/{stripped}" if stripped != name else None]
    cands += card_base_models(info)
    cands.append(hf_repo_from_url(header.get("general.base_model.0.repo_url")))
    return [c for i, c in enumerate(cands) if c and c not in cands[:i] and c != repo]


def check_repo(repo: str, downloads: int, out: Path, cache: Path, memo: dict, n_gen: int = 300, seed: int = 0,
               max_shrink: int = 15) -> dict:
    rec = {"repo": repo, "downloads": downloads, "checked": time.strftime("%Y-%m-%dT%H:%M:%S")}
    info = model_info(repo)
    path = pick_gguf(repo)
    if not path:
        return {**rec, "status": "no-gguf"}
    rec["file"] = path
    try:
        header = read_header(range_fetcher(repo, path))
    except (GGUFFormatError, Exception) as exc:
        return {**rec, "status": "header-error", "error": str(exc)[:200]}
    rec.update(arch=header.arch, tokenizer_model=header.tokenizer_model, pre=header.pre)
    if not header.tokenizer_model:
        return {**rec, "status": "no-tokenizer"}

    # the model this GGUF was converted from, then its base model (for drift)
    cands = source_candidates(repo, header, info)
    source, src_tj = None, None
    for c in cands:
        src_tj = fetch_tokenizer(c, cache)
        if src_tj:
            source = c
            break
    rec["source_candidates"] = cands
    if not source:
        return {**rec, "status": "no-reference"}
    rec["source"] = source

    base_tj = None
    try:
        base_info = model_info(source)
        bases = [b for b in card_base_models(base_info) if b != source]
        if bases:
            base_tj = fetch_tokenizer(bases[0], cache)
            rec["base"] = bases[0]
            if base_tj:
                rec["drift"] = drift(src_tj, base_tj)
    except Exception as exc:
        rec["drift_error"] = str(exc)[:200]

    key = (tokenizer_key(header), _sha(src_tj.read_bytes()))
    if key in memo:
        rec.update(memo[key], status="ok", cached=True)
        return rec
    with tempfile.TemporaryDirectory() as td:
        vg = write_vocab_only(header, Path(td) / "vocab.gguf")
        try:
            stats, findings = run_engine_check(vg, src_tj, n_gen, seed, max_shrink)
        except Exception as exc:
            return {**rec, "status": "engine-error", "error": str(exc)[:300]}
        if _load_failed({"engine": stats}):
            return {**rec, "status": "engine-error", "engine": stats,
                    "error": "llama.cpp could not load the vocabulary: " + (stats.get("error_sample") or "?")}
        base_stats = None
        if rec.get("drift") and base_tj:  # llama.cpp may match the base model rather than the drifted source
            try:
                base_stats, _ = run_engine_check(vg, base_tj, n_gen, seed, 0, extra=False)
            except Exception as exc:
                base_stats = {"error": str(exc)[:200]}
    examples = [{"kind": f.kind, "class": getattr(f, "cls", None), "minimal": f.minimal,
                 "ref": f.min_ref_ids or f.ref_ids, "got": f.min_got_ids or f.got_ids} for f in findings[:8]]
    result = {"engine": stats, "examples": examples}
    if base_stats is not None:
        result["engine_vs_base"] = base_stats
    memo[key] = result
    if findings:
        fdir = out / "findings"
        fdir.mkdir(parents=True, exist_ok=True)
        with open(fdir / (repo.replace("/", "__") + ".jsonl"), "w", encoding="utf-8") as fh:
            for f in findings:
                fh.write(f.to_json() + "\n")
    return {**rec, **result, "status": "ok"}


# ----------------------------------------------------------------------------- commands

RETRY = {"engine-error", "error", "header-error"}


def _latest(recs_path: Path) -> list[dict]:
    """One record per repo (the most recent line wins), in first-seen order."""
    if not recs_path.exists():
        return []
    by_repo: dict = {}
    for line in recs_path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            by_repo[r["repo"]] = r
    return list(by_repo.values())


def engine_problem() -> str | None:
    """Return a message if the llama.cpp harness cannot run, so a scan does not record hundreds of engine errors."""
    from .engines import LlamaCppTokenizer
    b = LlamaCppTokenizer.BIN
    if not b.exists():
        return (f"llama.cpp harness not found at {b}.\nBuild it first (see README, 'Auditing GGUF files on Hugging Face'), "
                "or point INFERCHECK_LLAMATOK at an existing build.")
    try:
        p = subprocess.run([str(b)], input="", capture_output=True, text=True, timeout=30)
    except OSError as exc:
        return f"llama.cpp harness at {b} does not run: {exc}"
    if p.returncode < 0 or "dyld" in p.stderr or "Library not loaded" in p.stderr:
        return f"llama.cpp harness at {b} does not run: {p.stderr.strip()[:300]}"
    return None


def scan(top: int, out: Path, n_gen: int, seed: int, repos: list[str] | None = None, max_shrink: int = 15) -> int:
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "hf-cache"
    recs_path = out / "records.jsonl"
    problem = engine_problem()
    if problem:
        print(problem, file=sys.stderr)
        return 2
    # resume: skip repos already finished; retry the ones that failed for engine or network reasons
    done = {r["repo"] for r in _latest(recs_path) if r["status"] not in RETRY and not _load_failed(r)}
    targets = [{"id": r, "downloads": 0} for r in repos] if repos else top_gguf_repos(top)
    memo: dict = {}
    t0 = time.time()
    with open(recs_path, "a", encoding="utf-8") as fh:
        for i, t in enumerate(targets, 1):
            if t["id"] in done:
                continue
            try:
                rec = check_repo(t["id"], t["downloads"], out, cache, memo, n_gen, seed, max_shrink)
            except Exception as exc:
                rec = {"repo": t["id"], "downloads": t["downloads"], "status": "error", "error": str(exc)[:300]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            e = rec.get("engine", {})
            print(f"[{i}/{len(targets)} {time.time() - t0:5.0f}s] {t['id']}: {rec['status']}"
                  + (f" pre={rec.get('pre')} encode={e.get('encode')} decode={e.get('decode')}" if e else "")
                  + (" DRIFT" if rec.get("drift") else "")
                  + (f" ({rec['error'][:240]})" if rec["status"] in RETRY and rec.get("error") else ""), flush=True)
    return 0


def _fmt_ex(ex: dict) -> str:
    s = ex["minimal"]
    shown = "".join(c if (c.isprintable() and not c.isspace() and not unicodedata.category(c).startswith("M")) or c == " "
                    else f"<U+{ord(c):04X}>" for c in s)
    shown = shown if len(shown) <= 30 else shown[:30] + "…"
    return f"`{shown}` ({ex.get('class') or ex['kind']})"


def _causes(r: dict, out: Path, by_signature: dict) -> dict:
    """Causes for a record. Records written before causes were recorded get them from the findings file,
    or from another repo with the same tokenizer result when the result was reused from memory."""
    e = r["engine"]
    if e.get("causes") is not None:
        return e["causes"]
    f = out / "findings" / (r["repo"].replace("/", "__") + ".jsonl")
    if f.exists():
        from types import SimpleNamespace
        causes: dict = {}
        for line in f.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                c = cause_of(SimpleNamespace(**json.loads(line)), [])
                causes[c] = causes.get(c, 0) + 1
        return causes
    return by_signature.get(_signature(r), e.get("classes") or {})


def _signature(r: dict):
    e = r["engine"]
    return (r.get("pre"), e.get("encode"), e.get("decode"), tuple(sorted((e.get("classes") or {}).items())))


def _cell(s: str) -> str:
    """Text for a Markdown table cell: pipes escaped (token names like <|im_end|> keep their pipes), one line."""
    return s.replace("|", "\\|").replace("\n", " ")


def _extra_sections(ok: list[dict]) -> list[str]:
    """Report sections for the refdecode, stress and start/stop-token checks (records from older scans
    have none of these and are skipped)."""
    lines: list[str] = []
    special = [(r, [i for i in ((r["engine"].get("special") or {}).get("issues") or []) if i["severity"] == "high"])
               for r in ok if (r["engine"].get("special") or {}).get("issues") is not None]
    flagged = [(r, iss) for r, iss in special if iss]
    if special:
        lines += ["", "## Start and stop tokens", "",
                  f"Checked in {len(special)} repos. **{len(flagged)}** differ from their source model in the tokens "
                  "added around a prompt or the tokens that end generation "
                  f"({sum(r['downloads'] for r, _ in flagged):,} downloads).", ""]
        if flagged:
            lines += ["| repo | downloads | check | difference |", "|---|---:|---|---|"]
            for r, iss in sorted(flagged, key=lambda x: -x[0]["downloads"]):
                for i in iss[:4]:
                    lines.append(f"| {r['repo']} | {r['downloads']:,} | {i['check']} | {_cell(i['detail'][:220])} |")
    rd = [r for r in ok if isinstance(r["engine"].get("refdecode"), dict) and "compared" in r["engine"]["refdecode"]]
    rd_bad = [r for r in rd if r["engine"]["refdecode"]["mismatch"] + r["engine"]["refdecode"]["error"]]
    if rd:
        lines += ["", "## Decoding the source tokenizer's token ids", "",
                  f"Checked in {len(rd)} repos. llama.cpp decodes the source tokenizer's ids differently in "
                  f"**{len(rd_bad)}** ({sum(r['downloads'] for r in rd_bad):,} downloads).", ""]
        if rd_bad:
            lines += ["| repo | downloads | ids decoded differently | errors | example |", "|---|---:|---:|---:|---|"]
            for r in sorted(rd_bad, key=lambda r: -r["downloads"]):
                x = r["engine"]["refdecode"]
                ex = (x.get("examples") or [{}])[0]
                exs = f"`{ex.get('ref', '')!r}` → `{ex.get('got', '')!r}`" if ex else ("crash" if x.get("crash") else "")
                lines.append(f"| {r['repo']} | {r['downloads']:,} | {x['mismatch']} | {x['error']} | {_cell(exs[:160])} |")
    st = [r for r in ok if isinstance(r["engine"].get("stress"), dict) and "cases" in r["engine"]["stress"]]
    st_bad = [r for r in st if r["engine"]["stress"]["mismatch"] + r["engine"]["stress"]["error"]]
    if st:
        lines += ["", "## Long inputs", "",
                  f"Checked in {len(st)} repos. llama.cpp fails or disagrees on long inputs in **{len(st_bad)}**.", ""]
        if st_bad:
            lines += ["| repo | downloads | failures |", "|---|---:|---|"]
            for r in sorted(st_bad, key=lambda r: -r["downloads"]):
                f = r["engine"]["stress"]["failures"]
                lines.append(f"| {r['repo']} | {r['downloads']:,} | "
                             f"{', '.join(x['case'] + ' ' + x['kind'] for x in f[:5])} |")
    return lines


def report(out: Path) -> str:
    recs = _latest(out / "records.jsonl")
    failed = [r for r in recs if _load_failed(r)]
    ok = [r for r in recs if r["status"] == "ok" and not _load_failed(r)]
    bad = [r for r in ok if (r["engine"].get("encode") or 0) + (r["engine"].get("decode") or 0)]
    drifted = [r for r in recs if r.get("drift")]
    statuses: dict = {}
    for r in recs:
        st = "engine-error" if _load_failed(r) else r["status"]
        statuses[st] = statuses.get(st, 0) + 1

    by_signature: dict = {}
    for r in bad:
        f = out / "findings" / (r["repo"].replace("/", "__") + ".jsonl")
        if r["engine"].get("causes") is not None or f.exists():
            by_signature.setdefault(_signature(r), _causes(r, out, {}))
    causes_of = {r["repo"]: _causes(r, out, by_signature) for r in bad}

    lines = ["# GGUF tokenizer audit", "",
             f"- repos scanned: {len(recs)} ({', '.join(f'{k}: {v}' for k, v in sorted(statuses.items()))})",
             f"- llama.cpp could not load the vocabulary: {len(failed)} repos (listed below; usually an architecture "
             "newer than the llama.cpp build)",
             f"- checked against their source tokenizer: {len(ok)}",
             f"- llama.cpp tokenizes or decodes differently from the source tokenizer: **{len(bad)}** repos, "
             f"{sum(r['downloads'] for r in bad):,} downloads in the last 30 days",
             f"- source tokenizer differs from its base model's (drift): **{len(drifted)}** repos, "
             f"{sum(r['downloads'] for r in drifted):,} downloads", ""]

    agg: dict = {}
    for r in bad:
        for c in causes_of[r["repo"]]:
            a = agg.setdefault(c, {"repos": [], "downloads": 0})
            a["repos"].append(r)
            a["downloads"] += r["downloads"]
    lines += ["## Causes", "",
              "A repo is counted once for every cause among its findings.", "",
              "| cause | repos | downloads (30 days) | what it is | largest repos |", "|---|---:|---:|---|---|"]
    for c, a in sorted(agg.items(), key=lambda kv: -kv[1]["downloads"]):
        top = ", ".join(x["repo"] for x in sorted(a["repos"], key=lambda x: -x["downloads"])[:3])
        lines.append(f"| {c} | {len(a['repos'])} | {a['downloads']:,} | {CAUSES.get(c, c)} | {top} |")

    lines += ["", "## llama.cpp vs source tokenizer", "",
              "| repo | downloads | arch | pre | cases | encode | decode | vs base | causes | examples |",
              "|---|---:|---|---|---:|---:|---:|---|---|---|"]
    for r in sorted(bad, key=lambda r: -r["downloads"]):
        e = r["engine"]
        vb = r.get("engine_vs_base")
        vb_s = "-" if not vb else (f"{vb.get('encode')}/{vb.get('decode')}" if "encode" in vb else "error")
        causes = ", ".join(f"{k} {v}" for k, v in sorted(causes_of[r["repo"]].items(), key=lambda kv: -kv[1]))
        lines.append(f"| {r['repo']} | {r['downloads']:,} | {r.get('arch')} | {r.get('pre')} | {e.get('cases')} | "
                     f"{e.get('encode')} | {e.get('decode')} | {vb_s} | {causes} | "
                     f"{'; '.join(_fmt_ex(x) for x in r.get('examples', [])[:3])} |")
    groups: dict = {}
    for r in bad:
        g = groups.setdefault((r.get("arch"), r.get("pre")), {"repos": 0, "downloads": 0})
        g["repos"] += 1
        g["downloads"] += r["downloads"]
    lines += ["", "## By architecture and pre-tokenizer", "", "| arch | pre | repos | downloads |", "|---|---|---:|---:|"]
    for (a, p), g in sorted(groups.items(), key=lambda kv: -kv[1]["downloads"]):
        lines.append(f"| {a} | {p} | {g['repos']} | {g['downloads']:,} |")
    lines += _extra_sections(ok)
    if failed:
        lines += ["", "## llama.cpp could not load the vocabulary", "",
                  "| repo | downloads | arch | pre | error |", "|---|---:|---|---|---|"]
        for r in sorted(failed, key=lambda r: -r["downloads"]):
            err = _cell(r.get("error") or (r.get("engine") or {}).get("error_sample") or "")
            lines.append(f"| {r['repo']} | {r['downloads']:,} | {r.get('arch')} | {r.get('pre')} | {err[:160]} |")
    lines += ["", "## Tokenizer drift: source repo vs its base model", "",
              "| GGUF repo | downloads | source | base | differs in | probes changed |", "|---|---:|---|---|---|---:|"]
    for r in sorted(drifted, key=lambda r: -r["downloads"]):
        d = r["drift"]
        lines.append(f"| {r['repo']} | {r['downloads']:,} | {r.get('source')} | {r.get('base')} | "
                     f"{', '.join(d['fields']) or 'ids only'} | {d['probes_differ']}/{d['probes']} |")
    return "\n".join(lines) + "\n"
