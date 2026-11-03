"""Start and stop tokens: what llama.cpp puts around a prompt and when it stops generating,
compared with the model the GGUF was converted from.

Two checks, both from the vocabulary alone (no weights):

- **wrap**: the tokens added around a prompt. The reference is transformers' ``AutoTokenizer`` when
  transformers is installed (some tokenizer classes add BOS themselves, whatever the post-processor
  says), otherwise Hugging Face ``tokenizers`` ``encode(text, add_special_tokens=True)`` (the
  tokenizer.json post-processor); the engine is ``llama_tokenize(text, add_special=true)``. A missing
  BOS, a doubled BOS or an EOS appended where the reference appends a different token all show up here.
- **stop**: the tokens that end generation. The reference stop set is ``generation_config.json``
  ``eos_token_id`` plus ``tokenizer_config.json`` ``eos_token``; the engine's is every token
  ``llama_vocab_is_eog`` accepts. A reference stop token the engine does not stop on means the
  model can run past the end of its answer; an ordinary text token the engine stops on means it
  can stop in the middle of one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROBE = "Hello world"   # must match PROBE in harness/llama_tok/llamatok.cpp
ATTR_CONTROL, ATTR_USER_DEFINED = 1 << 3, 1 << 4


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def _token_content(v) -> str | None:
    if isinstance(v, str):
        return v
    if isinstance(v, dict) and isinstance(v.get("content"), str):
        return v["content"]
    return None


def split_wrap(with_special: list[int], plain: list[int]) -> tuple[list[int], list[int]] | None:
    """The tokens added before and after ``plain`` in ``with_special``, or None if ``plain`` is not
    a contiguous part of it (then the added tokens changed the text's own tokens too)."""
    n, m = len(with_special), len(plain)
    for start in range(0, n - m + 1):
        if with_special[start:start + m] == plain:
            return with_special[:start], with_special[start + m:]
    return None


def transformers_ids(folder: Path) -> tuple[list[int], list[int], str] | None:
    """PROBE encoded by transformers' AutoTokenizer without and with special tokens, or None if transformers
    is not installed or cannot load the folder. transformers is what users of the original model run: some
    tokenizer classes add BOS from tokenizer_config.json even when tokenizer.json's post-processor does not
    (Gemma), so it is the better reference for the tokens around a prompt."""
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        # never run a repo's own Python (and never stop at transformers' prompt asking whether to): such
        # tokenizers fall back to the tokenizers library
        tok = AutoTokenizer.from_pretrained(str(folder), trust_remote_code=False)
        plain = list(tok(PROBE, add_special_tokens=False)["input_ids"])
        full = list(tok(PROBE)["input_ids"])
    except Exception:
        return None
    return plain, full, f"transformers {transformers.__version__}"


def chat_template(folder: Path, cfg: dict) -> str | None:
    """The default chat template: tokenizer_config.json "chat_template" (a string, or a list of named
    templates) or chat_template.jinja."""
    t = cfg.get("chat_template")
    if isinstance(t, list):
        t = next((x.get("template") for x in t if isinstance(x, dict) and x.get("name") == "default"),
                 t[0].get("template") if t and isinstance(t[0], dict) else None)
    if isinstance(t, str) and t:
        return t
    p = folder / "chat_template.jinja"
    try:
        return p.read_text(encoding="utf-8") if p.exists() else None
    except OSError:
        return None


# what can come before a template's first output without producing any: comments, {% set x = ... %}, macro definitions
_PREAMBLE = re.compile(r"\s*(?:\{#.*?#\}|\{%-?\s*set\s[^%]*?=.*?%\}|\{%-?\s*macro\b.*?\{%-?\s*endmacro\s*-?%\})", re.S)
_BOS_OUT = re.compile(r"\s*\{\{-?\s*bos_token\s*-?\}\}")


def template_starts_with_bos(template: str, bos_text: str | None) -> bool:
    """Whether the first thing a chat template outputs is the BOS token ({{ bos_token }} or its literal text),
    after leading comments, {% set %} statements and macro definitions. A BOS inside a loop or a condition
    does not count."""
    pos = 0
    while True:
        m = _PREAMBLE.match(template, pos)
        if not m:
            break
        pos = m.end()
    rest = template[pos:]
    if _BOS_OUT.match(rest):
        return True
    return bool(bos_text) and rest.lstrip().startswith(bos_text)


def reference_special(tokenizer_json: Path) -> dict:
    """Wrap tokens and stop tokens of a Hugging Face tokenizer (tokenizer.json next to
    tokenizer_config.json and, if present, generation_config.json)."""
    from tokenizers import Tokenizer
    import tokenizers

    tok = Tokenizer.from_file(str(tokenizer_json))
    cfg = _json(tokenizer_json.with_name("tokenizer_config.json"))
    gen = _json(tokenizer_json.with_name("generation_config.json"))
    tj = _json(tokenizer_json)
    special_ids = {t["id"] for t in tj.get("added_tokens", []) if t.get("special")}
    added_ids = {t["id"] for t in tj.get("added_tokens", [])}

    tf = transformers_ids(tokenizer_json.parent)
    if tf:
        plain, full, impl = tf
    else:
        plain = tok.encode(PROBE, add_special_tokens=False).ids
        full = tok.encode(PROBE, add_special_tokens=True).ids
        impl = f"tokenizers {tokenizers.__version__}"
    wrap = split_wrap(list(full), list(plain))

    stops: dict[int, str] = {}
    g = gen.get("eos_token_id")
    for i in ([g] if isinstance(g, int) else g if isinstance(g, list) else []):
        if isinstance(i, int):
            stops[i] = "generation_config.json eos_token_id"
    eos = _token_content(cfg.get("eos_token"))
    if eos is not None:
        i = tok.token_to_id(eos)
        if i is not None:
            stops.setdefault(i, "tokenizer_config.json eos_token")

    # a template that starts with BOS gives every chat prompt a BOS even when plain encoding adds none
    bos = _token_content(cfg.get("bos_token"))
    tmpl = chat_template(tokenizer_json.parent, cfg)
    template_bos = tok.token_to_id(bos) if bos and tmpl and template_starts_with_bos(tmpl, bos) else None

    return {
        "plain": list(plain), "full": list(full), "impl": impl,
        "prefix": wrap[0] if wrap else None, "suffix": wrap[1] if wrap else None,
        "config_add_bos": cfg.get("add_bos_token") if isinstance(cfg.get("add_bos_token"), bool) else None,
        "config_add_eos": cfg.get("add_eos_token") if isinstance(cfg.get("add_eos_token"), bool) else None,
        "stops": stops, "has_generation_config": bool(gen),
        "special_ids": sorted(special_ids), "added_ids": sorted(added_ids), "template_bos": template_bos,
        "text": {i: tok.id_to_token(i) for i in set(stops) | set(full) | ({template_bos} - {None})},
    }


def _name(ref: dict, eng: dict, i: int) -> str:
    t = ref.get("text", {}).get(i)
    if t is None:
        t = next((e["text"] for e in eng.get("eog", []) if e["id"] == i), None)
    return f"{t!r} ({i})" if t is not None else str(i)


def compare_special(ref: dict, eng: dict) -> list[dict]:
    """Issues between reference and engine start/stop behavior. Severity "high" for differences that
    change what the model sees or when it stops; "info" for ones that rarely matter."""
    issues: list[dict] = []
    # --- wrap
    if eng.get("probe_plain") != ref.get("plain"):
        issues.append({"check": "wrap", "severity": "info",
                       "detail": f"the probe text itself tokenizes differently ({eng.get('probe_plain')} vs "
                                 f"{ref.get('plain')}); see the encode findings"})
    ew = split_wrap(eng.get("probe_special") or [], eng.get("probe_plain") or [])
    if ref.get("prefix") is None or ew is None:
        issues.append({"check": "wrap", "severity": "info", "detail": "could not separate the added tokens"})
    else:
        (rp, rs), (ep, es) = (ref["prefix"], ref["suffix"]), ew
        # without transformers, a config that disagrees with the post-processor leaves the true behavior unknown
        uncertain = not str(ref.get("impl", "")).startswith("transformers") and (
            (ref.get("config_add_bos") is True and not rp) or (ref.get("config_add_bos") is False and rp))
        for where, r, e in (("start", rp, ep), ("end", rs, es)):
            if r == e:
                continue
            names = lambda ids: "[" + ", ".join(_name(ref, eng, i) for i in ids) + "]"
            # llama.cpp's convention: add BOS when tokenizing and drop the template's own BOS, so a model whose
            # reference template starts with BOS gets one BOS per chat prompt either way; only plain-text
            # prompts differ, and there llama.cpp's BOS is deliberate (Gemma 4, llama.cpp #21500)
            via_template = (where == "start" and not r and ref.get("template_bos") is not None
                            and e == [ref["template_bos"]])
            note = (" (tokenizer_config.json add_bos_token disagrees with the reference's post-processor, so "
                    "transformers' behavior depends on the tokenizer class)" if where == "start" and uncertain
                    else " (the reference adds it through its chat template instead, so chat prompts agree; only "
                         "plain-text prompts differ)" if via_template else "")
            sev = "info" if note else "high"
            kind = ("missing" if r and not e else "extra" if e and not r else "different")
            issues.append({"check": "wrap", "severity": sev, "kind": f"{where}-{kind}",
                           "detail": f"tokens added at the {where} of a prompt: llama.cpp {names(e)}, "
                                     f"reference {names(r)}" + note})
    # --- stop
    eog = {e["id"]: e for e in eng.get("eog", [])}
    stops = {int(k): v for k, v in (ref.get("stops") or {}).items()}
    for i, src in sorted(stops.items()):
        if i not in eog:
            issues.append({"check": "stop", "severity": "high", "kind": "missing", "token": i,
                           "detail": f"{_name(ref, eng, i)} ends generation in the reference ({src}) but not in llama.cpp; "
                                     "the model can run past the end of its answer"})
    special = set(ref.get("special_ids") or [])
    added = set(ref.get("added_ids") or []) | special
    for i, e in sorted(eog.items()):
        if i in stops:
            continue
        # Whether the token is ordinary text comes from the reference only: llama.cpp sets the CONTROL bit
        # itself on every token it auto-detects as end-of-generation ("control-looking token ... its type
        # will be overridden"), so the engine's attribute cannot tell. Added tokens that are not marked
        # special (Qwen's <|fim_pad|>, <|repo_name|>, <|file_sep|>) are still single atomic tokens, and
        # llama.cpp stops on fill-in-the-middle tokens on purpose, so only plain vocabulary tokens count.
        if i not in added:
            sev, why = "high", " and treats as ordinary text: generation can stop mid-answer"
        elif i not in special:
            sev, why = "info", " (an added token the reference does not mark special)"
        else:
            sev, why = "info", ""
        issues.append({"check": "stop", "severity": sev, "kind": "extra", "token": i,
                       "detail": f"llama.cpp stops on {_name(ref, eng, i)}, which the reference does not stop on" + why})
    return issues


def score(issues: list[dict], check: str, strict: bool = False) -> int:
    """How many issues of one check: all of them (for comparing two versions of the same model,
    where shared baseline issues cancel) or only high-severity ones (strict, for a single version)."""
    return sum(1 for i in issues if i["check"] == check and (i["severity"] == "high" or not strict)
               and i.get("kind") is not None)
