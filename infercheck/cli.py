"""InferCheck command line.

  infercheck tok    --data DIR --out results/   differential tokenization against Hugging Face
  infercheck vocab  --data DIR --out results/   decode every token id alone, against Hugging Face
  infercheck list   --data DIR                  what is available to test
  infercheck compare BEFORE AFTER               per-tokenizer disagreement counts of two tok runs (Markdown)

DIR is the unpacked data/ folder from scripts/mac_fetch.sh: tokenizers/<name>/tokenizer.json,
tokenizers/ggml-vocabs/**.gguf and ollama-gguf/*.gguf.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .corpus import fixed_cases, generated_cases, special_cases
from .engines import HFTokenizer, LlamaCppTokenizer, OllamaTokenizer, engine_versions

# llama.cpp vocabulary files (ggml-org/vocabs or local Ollama models) for a tokenizer name.
GGUF_FOR = {
    "gpt2": ["ggml-vocab-gpt-2.gguf"],
    "qwen1.5": ["ggml-vocab-qwen2.gguf"],
    "qwen3.5": ["ggml-vocab-qwen35.gguf"],
    "deepseek-llm": ["ggml-vocab-deepseek-llm.gguf"],
    "deepseek-coder": ["ggml-vocab-deepseek-coder.gguf"],
    "falcon": ["ggml-vocab-falcon.gguf"],
    "mpt": ["ggml-vocab-mpt.gguf"],
    "starcoder2": ["ggml-vocab-starcoder.gguf"],
    "refact": ["ggml-vocab-refact.gguf"],
    "phi-3": ["ggml-vocab-phi-3.gguf"],
    "bge-small": ["ggml-vocab-bert-bge.gguf"],
    "llama3": ["ggml-vocab-llama-bpe.gguf"],
    "llama2": ["ggml-vocab-llama-spm.gguf"],
    "command-r": ["ggml-vocab-command-r.gguf"],
    "qwen3": ["qwen3_8b.gguf"],
    "gpt-oss": ["gpt-oss_20b.gguf"],
    # Ollama's registry copies of these tokenizers match llama.cpp's reference vocabularies
    "ollama-gemma4": ["ggml-vocab-gemma-4.gguf"],
}


def find_gguf(data: Path, name: str, extra_dirs: list[Path]) -> list[Path]:
    """GGUF files to test for a tokenizer: <name>.gguf in a --gguf-dir (written by
    scripts/convert_vocabs.sh), else the files GGUF_FOR names, searched --gguf-dir first."""
    for d in extra_dirs:
        if (d / f"{name}.gguf").exists():
            return [d / f"{name}.gguf"]
    found = []
    for fname in GGUF_FOR.get(name, []):
        for d in [*extra_dirs, data / "ollama-gguf", data / "tokenizers" / "ggml-vocabs"]:
            hits = sorted(d.rglob(fname)) if d.exists() else []
            if hits:
                found.append(hits[0])
                break
    return found


def tokenizer_dirs(data: Path, only: list[str] | None) -> list[Path]:
    root = data / "tokenizers"
    dirs = sorted(p for p in root.iterdir() if (p / "tokenizer.json").exists())
    if only:
        dirs = [d for d in dirs if d.name in only]
    return dirs


def added_tokens(tokenizer_json: Path) -> list[str]:
    tj = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    return [t["content"] for t in tj.get("added_tokens", []) if t.get("content")]


def cmd_list(args) -> int:
    data = Path(args.data)
    for d in tokenizer_dirs(data, args.only):
        gg = find_gguf(data, d.name, [Path(p) for p in args.gguf_dir])
        print(f"{d.name:20s} tokenizer.json  gguf: {', '.join(p.name for p in gg) or '-'}")
    return 0


def cmd_tok(args) -> int:
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "versions": engine_versions(), "seed": args.seed,
               "generated": args.n, "tokenizers": {}}
    for d in tokenizer_dirs(data, args.only):
        tj = d / "tokenizer.json"
        print(f"== {d.name}", flush=True)
        try:
            ref = HFTokenizer(str(tj))
        except Exception as exc:
            print(f"  reference failed to load: {exc}")
            summary["tokenizers"][d.name] = {"reference_error": str(exc)[:300]}
            continue
        summary["versions"]["tokenizers"] = ref.meta["tokenizers"]
        engines = []
        if not args.no_ollama:
            o = OllamaTokenizer(str(tj))
            probe = o.encode_many(["a"])[0]
            if probe.error and probe.error.startswith("exit 3"):
                print(f"  ollama-mlx: does not load ({probe.error[8:120]})")
            else:
                engines.append(o)
        for g in ([] if args.no_llama else find_gguf(data, d.name, [Path(p) for p in args.gguf_dir])):
            e = LlamaCppTokenizer(str(g))
            e.name = f"llama.cpp[{g.name}]"
            engines.append(e)
        if not engines:
            print("  no engine to compare")
            continue
        special = added_tokens(tj)
        cases = fixed_cases() + special_cases(special) + generated_cases(args.n, args.seed, special)
        from .diff import compare

        findings, stats = compare(ref, engines, cases, shrink=not args.no_shrink, max_shrink=args.max_shrink)
        summary["tokenizers"][d.name] = stats
        with open(out / f"{d.name}.findings.jsonl", "w", encoding="utf-8") as fh:
            for f in findings:
                fh.write(f.to_json() + "\n")
        print(f"  {len(findings)} distinct findings", flush=True)
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


def cmd_vocab(args) -> int:
    """Decode each token id alone with each engine and compare with Hugging Face."""
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for d in tokenizer_dirs(data, args.only):
        tj = d / "tokenizer.json"
        try:
            ref = HFTokenizer(str(tj))
        except Exception as exc:
            print(f"== {d.name}: reference failed to load: {exc}")
            continue
        size = ref.tok.get_vocab_size(with_added_tokens=True)
        ids = list(range(size))
        ref_txt = [ref.decode([i]) for i in ids]
        rows = {}
        engines = [] if args.no_ollama else [OllamaTokenizer(str(tj))]
        for g in ([] if args.no_llama else find_gguf(data, d.name, [Path(p) for p in args.gguf_dir])):
            e = LlamaCppTokenizer(str(g))
            e.name = f"llama.cpp[{g.name}]"
            engines.append(e)
        for eng in engines:
            got = eng.decode_ids([[i] for i in ids]) if hasattr(eng, "decode_ids") else None
            if got is None:
                continue
            bad = []
            for i, (r, g) in enumerate(zip(ref_txt, got)):
                if g is None or "�" in r or "�" in g:
                    continue
                if r != g:
                    bad.append({"id": i, "ref": r, "got": g})
            rows[eng.name] = {"compared": size, "mismatches": len(bad), "examples": bad[:50]}
            print(f"== {d.name} {eng.name}: {len(bad)} of {size} single-token decodes differ", flush=True)
        report[d.name] = rows
    (out / "vocab_decode.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


def compare_runs(before: Path, after: Path) -> str:
    """A Markdown table of disagreement counts per tokenizer and engine, before and after a change."""
    b = json.loads((before / "summary.json").read_text())["tokenizers"]
    a = json.loads((after / "summary.json").read_text())["tokenizers"]
    rows = ["| tokenizer | engine | cases | encode before | encode after | decode before | decode after |",
            "|---|---|---:|---:|---:|---:|---:|"]
    tot = [0, 0, 0, 0]
    for name in sorted(b):
        for eng, x in b[name].get("engines", {}).items():
            y = a.get(name, {}).get("engines", {}).get(eng)
            if y is None:
                continue
            vals = [x["encode"], y["encode"], x["decode"], y["decode"]]
            tot = [t + v for t, v in zip(tot, vals)]
            if any(vals):
                rows.append(f"| {name} | {eng} | {b[name]['cases']} | " + " | ".join(map(str, vals)) + " |")
    rows.append("| **total** | | | " + " | ".join(f"**{t}**" for t in tot) + " |")
    return "\n".join(rows) + "\n"


def cmd_compare(args) -> int:
    print(compare_runs(Path(args.before), Path(args.after)), end="")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="infercheck", description="Conformance tests for LLM inference engines.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "tok", "vocab"):
        p = sub.add_parser(name)
        p.add_argument("--data", default="data")
        p.add_argument("--out", default="results")
        p.add_argument("--only", nargs="*")
        p.add_argument("--gguf-dir", nargs="*", default=[])
        p.add_argument("--no-ollama", action="store_true")
        p.add_argument("--no-llama", action="store_true")
        if name == "tok":
            p.add_argument("--n", type=int, default=20000, help="generated cases per tokenizer")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--no-shrink", action="store_true")
            p.add_argument("--max-shrink", type=int, default=300)
    p = sub.add_parser("compare")
    p.add_argument("before")
    p.add_argument("after")
    args = ap.parse_args(argv)
    return {"list": cmd_list, "tok": cmd_tok, "vocab": cmd_vocab, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
