#!/usr/bin/env python3
"""Replay a llama.cpp engine bug offline, against llama.cpp's own reference tokenizations.

llama.cpp keeps test vocabularies with Hugging Face's expected output next to them
(models/ggml-vocab-NAME.gguf, .gguf.inp, .gguf.out). For each harness given (llamatok built at
different llama.cpp commits, e.g. by scripts/build_llama_at.sh), this prints how many of the test
strings it tokenizes differently from the expected ids (encode), how many of the expected ids it
decodes to something other than the input text or crashes on (refdecode), and its end-of-generation
tokens. No network and no Python packages needed.

  scripts/bench_offline.py models/ggml-vocab-gemma-4.gguf BROKEN/llamatok FIXED/llamatok

With --pair, it instead runs InferCheck's own corpus (fixed probes, special-token probes, 300
generated strings and the long stress inputs) through two builds and counts where the build before
the fix differs from the build after it: different ids (encode), different text from the same ids
(decode), decoding the fixed build's ids (refdecode), long inputs (stress) and end-of-generation
tokens. The fixed build stands in for the reference, which is what `bench run` compares on top of.

  scripts/bench_offline.py --pair models/ggml-vocab-gemma-4.gguf BROKEN/llamatok FIXED/llamatok
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SEP = "\n__ggml_vocab_test__\n"


def load_tests(vocab: Path) -> list[tuple[str, list[int]]]:
    inp = Path(str(vocab) + ".inp").read_text(encoding="utf-8")
    if inp.endswith(SEP):
        inp = inp[: -len(SEP)]
    texts = inp.split(SEP)
    outs = Path(str(vocab) + ".out").read_text(encoding="utf-8").split("\n")[: len(texts)]
    return [(t, [int(x) for x in o.split()]) for t, o in zip(texts, outs)]


def run(bin_path: str, vocab: Path, args: list[str], payload: str) -> tuple[list[str], int, str]:
    p = subprocess.run([bin_path, str(vocab), *args], input=payload.encode(), capture_output=True, timeout=600)
    return p.stdout.decode("utf-8", "replace").split("\n"), p.returncode, p.stderr.decode("utf-8", "replace")[-300:]


def check(bin_path: str, vocab: Path, tests: list[tuple[str, list[int]]]) -> dict:
    lines, code, err = run(bin_path, vocab, [], "".join(t.encode().hex() + "\n" for t, _ in tests))
    enc_bad, examples = 0, []
    if code != 0:
        enc = {"crash": f"exit {code}: {err.strip()}"}
    else:
        for (t, want), line in zip(tests, lines):
            got = [int(x) for x in line.partition("\t")[0].split()]
            if got != want:
                enc_bad += 1
                if len(examples) < 3:
                    examples.append({"text": t[:40], "want": want[:12], "got": got[:12]})
        enc = {"mismatch": enc_bad, "examples": examples}
    seqs = [(t, ids) for t, ids in tests if ids]
    lines, code, err = run(bin_path, vocab, ["--decode"], "".join(" ".join(map(str, ids)) + "\n" for _, ids in seqs))
    if code != 0:
        dec = {"crash": f"exit {code}: {err.strip()}"}
    else:
        bad = [(t, bytes.fromhex(line).decode("utf-8", "replace")) for (t, _), line in zip(seqs, lines)
               if bytes.fromhex(line).decode("utf-8", "replace") != t]
        dec = {"mismatch": len(bad), "examples": [{"text": t[:40], "got": g[:40]} for t, g in bad[:3]]}
    info_lines, code, err = run(bin_path, vocab, ["--info"], "")
    eog = [(e["id"], e["text"]) for e in json.loads(info_lines[0])["eog"]] if code == 0 else f"exit {code}"
    return {"encode": enc, "refdecode": dec, "eog": eog}


def pair(vocab: Path, broken: str, fixed: str) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from infercheck.corpus import fixed_cases, generated_cases, special_cases, stress_cases, vocab_pieces, word_cases
    from infercheck.engines import LlamaCppTokenizer

    b, f = LlamaCppTokenizer(str(vocab), bin_path=broken), LlamaCppTokenizer(str(vocab), bin_path=fixed)
    infos = {}
    for name, e in (("broken", b), ("fixed", f)):
        try:
            infos[name] = e.info()
        except RuntimeError as exc:
            infos[name] = {"error": str(exc)}
    eog_text = [x["text"] for x in infos["fixed"].get("eog", []) if len(x["text"]) > 1]
    pieces = vocab_pieces(lambda ids: f.decode_ids([[i] for i in ids]), infos["fixed"].get("n_tokens", 0))
    groups = {"probes": fixed_cases() + special_cases(eog_text) + generated_cases(300, 0, eog_text),
              "words": word_cases(pieces)}
    out = {"cases": {g: len(c) for g, c in groups.items()}, "encode": {}, "decode": {}, "error": {},
           "fixed_side_errors": {}, "examples": []}
    for g, cases in groups.items():
        texts = [c.text for c in cases]
        rb, rf = b.encode_many(texts), f.encode_many(texts)
        for k in ("encode", "decode", "error", "fixed_side_errors"):
            out[k][g] = 0
        for t, x, y in zip(texts, rb, rf):
            if y.error:
                out["fixed_side_errors"][g] += 1   # the fix crashes here: a regression, not the bug under test
                continue
            k = "error" if x.error else "encode" if x.ids != y.ids else "decode" if x.decoded != y.decoded else None
            if k:
                out[k][g] += 1
                if len(out["examples"]) < 4:
                    out["examples"].append({"kind": k, "text": t[:40], "broken": x.ids if x.error is None else x.error[:90], "fixed": y.ids})
    texts = [c.text for c in groups["probes"]]
    rf = f.encode_many(texts)
    seqs = [(y.ids, y.decoded) for y in rf if not y.error and y.ids]
    dec = b.decode_ids([s for s, _ in seqs])
    out["refdecode"] = {"differ": sum(1 for d, (_, want) in zip(dec, seqs) if d is not None and d != want),
                        "failed": sum(1 for d in dec if d is None), "crashes": b.last_crashes}
    stress = [c.text for c in stress_cases()]
    sb = [b.encode_many([t])[0] for t in stress]
    sf = [f.encode_many([t])[0] for t in stress]
    out["stress"] = sum(1 for x, y in zip(sb, sf) if not y.error and (x.error or x.ids != y.ids))
    eog = {n: sorted((x["id"], x["text"]) for x in i.get("eog", [])) for n, i in infos.items()}
    out["eog_only_broken"] = [x for x in eog["broken"] if x not in eog["fixed"]]
    out["eog_only_fixed"] = [x for x in eog["fixed"] if x not in eog["broken"]]
    out["wrap"] = None if "error" in infos["broken"] else (
        [infos[n]["probe_special"] for n in ("broken", "fixed")]
        if infos["broken"]["probe_special"] != infos["fixed"]["probe_special"] else 0)
    return out


def main() -> int:
    if len(sys.argv) >= 5 and sys.argv[1] == "--pair":
        r = pair(Path(sys.argv[2]), sys.argv[3], sys.argv[4])
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    vocab = Path(sys.argv[1])
    tests = load_tests(vocab)
    print(f"{vocab.name}: {len(tests)} reference tokenizations")
    for b in sys.argv[2:]:
        r = check(b, vocab, tests)
        commit = (Path(b).parent / "COMMIT").read_text().strip() if (Path(b).parent / "COMMIT").exists() else b
        print(f"\n== {commit}")
        print("  encode:   ", r["encode"].get("crash") or f"{r['encode']['mismatch']} of {len(tests)} differ")
        for x in r["encode"].get("examples", []):
            print(f"     {x['text']!r}: expected {x['want']} got {x['got']}")
        print("  refdecode:", r["refdecode"].get("crash") or f"{r['refdecode']['mismatch']} decode differently")
        for x in r["refdecode"].get("examples", []):
            print(f"     {x['text']!r} -> {x['got']!r}")
        print("  eog:      ", r["eog"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
