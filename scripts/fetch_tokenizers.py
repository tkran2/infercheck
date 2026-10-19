"""Download tokenizer files for differential testing (run on a machine with internet access).

Writes data/tokenizers/<name>/ with whatever of tokenizer.json, tokenizer_config.json,
special_tokens_map.json, config.json, tokenizer.model and chat_template.jinja the source has,
plus data/tokenizers/fetch.json describing every attempt. Sources:

- Hugging Face repositories (gated ones are skipped unless you are logged in and accepted
  the license),
- the Ollama registry, for the tokenizer.json that Ollama's MLX runner loads,
- the ggml-org/vocabs repository: vocabulary-only GGUF files that llama.cpp loads.

No model weights are downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HF_REPOS = {
    # name: repository
    "gpt2": "openai-community/gpt2",
    "gpt-4o": "Xenova/gpt-4o",
    "gpt-oss": "openai/gpt-oss-20b",
    "qwen1.5": "Qwen/Qwen1.5-7B",
    "qwen2.5": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
    "qwen3-embedding": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3.5": "Qwen/Qwen3.5-9B-Instruct",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1-qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-llm": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-base",
    "phi-2": "microsoft/phi-2",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "phi-4": "microsoft/phi-4",
    "smollm": "HuggingFaceTB/SmolLM-135M",
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "falcon": "tiiuae/falcon-7b",
    "falcon3": "tiiuae/Falcon3-7B-Base",
    "falcon-h1": "tiiuae/Falcon-H1-0.5B-Base",
    "mpt": "mosaicml/mpt-7b",
    "starcoder2": "bigcode/starcoder2-3b",
    "refact": "smallcloudai/Refact-1_6-base",
    "olmo": "allenai/OLMo-1.7-7B-hf",
    "olmo2": "allenai/OLMo-2-1124-7B",
    "superbpe": "UW/OLMo2-8B-SuperBPE-t180k",
    "tekken": "mistralai/Mistral-Nemo-Base-2407",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "glm4": "THUDM/glm-4-9b-hf",
    "glm4.5": "zai-org/GLM-4.5-Air",
    "glm4.7": "zai-org/GLM-4.7-Flash",
    "minimax-m2": "MiniMaxAI/MiniMax-M2",
    "hunyuan": "tencent/Hunyuan-A13B-Instruct",
    "seed-coder": "ByteDance-Seed/Seed-Coder-8B-Base",
    "lfm2": "LiquidAI/LFM2.5-350M",
    "granite-docling": "ibm-granite/granite-docling-258M",
    "exaone4": "LGAI-EXAONE/EXAONE-4.0-32B",
    "bailingmoe2": "inclusionAI/Ling-mini-base-2.0",
    "trillion": "trillionlabs/Trillion-7B-preview",
    "stablelm2": "stabilityai/stablelm-2-zephyr-1_6b",
    "mellum": "JetBrains/Mellum-4b-base",
    "laguna": "poolside/Laguna-XS.2",
    "north-mini-code": "CohereLabs/North-Mini-Code-1.0",
    "tiny-aya": "CohereLabs/tiny-aya-base",
    "gemma4-dfm": "danish-foundation-models/DFM-Mimir",
    "grok-2": "alvarobartt/grok-2-tokenizer",
    "solar-open": "upstage/Solar-Open-100B",
    "kanana2": "kakaocorp/kanana-2-30b-a3b-instruct-2601",
    "minicpm5": "openbmb/MiniCPM5-1B",
    "modernbert": "answerdotai/ModernBERT-base",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "t5": "google-t5/t5-small",
    # usually gated: fetched only if you are logged in and accepted the license
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3.2": "meta-llama/Llama-3.2-1B-Instruct",
    "llama2": "meta-llama/Llama-2-7b-hf",
    "gemma2": "google/gemma-2-9b",
    "gemma3": "google/gemma-3-1b-it",
    "command-r": "CohereForAI/c4ai-command-r-v01",
    "muse-spark": "meta-models/Muse-Spark-Open-816B-A42B",
}

HF_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json",
            "tokenizer.model", "chat_template.jinja")

# Models whose tokenizer.json Ollama's MLX runner tests against (mlxrunner/tokenizer).
OLLAMA_MODELS = {
    "ollama-qwen3.5": "qwen3.5:0.8b-mxfp8",
    "ollama-qwen3.8": "qwen3.8:27b-nvfp4",
    "ollama-nemotron3.5": "nemotron-3.5-lightning:30b-a3b-nvfp4",
    "ollama-nimble": "nimble:9b-nvfp4",
    "ollama-north": "north-mini-code-1.0:mlx-nvfp4",
    "ollama-laguna": "laguna-xs.2:nvfp4",
    "ollama-gemma4": "gemma4:e2b-nvfp4",
    "ollama-muse": "muse-glimmer:30b-nvfp4",
}
REGISTRY = "https://registry.ollama.ai/v2/"
LIST_TYPE = "application/vnd.ollama.manifest.list.v2+json"
MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(url: str, accept: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def fetch_hf(name: str, repo: str, out: Path) -> dict:
    from huggingface_hub import hf_hub_download, list_repo_files

    rec = {"source": "hf", "name": name, "repo": repo, "files": {}}
    try:
        files = set(list_repo_files(repo))
    except Exception as exc:  # gated, missing, offline
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return rec
    for fname in HF_FILES:
        if fname not in files:
            continue
        try:
            path = Path(hf_hub_download(repo, fname))
        except Exception as exc:
            rec["files"][fname] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            continue
        data = path.read_bytes()
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / fname).write_bytes(data)
        rec["files"][fname] = {"bytes": len(data), "sha256": sha256(data)}
    if "tokenizer.json" not in rec["files"]:
        rec.setdefault("error", "no tokenizer.json")
    return rec


def fetch_ollama(name: str, model: str, out: Path) -> dict:
    rec = {"source": "ollama-registry", "name": name, "model": model, "files": {}}
    repo, tag = model.split(":", 1)
    repo = repo if "/" in repo else f"library/{repo}"
    try:
        m = json.loads(get(f"{REGISTRY}{repo}/manifests/{tag}", f"{LIST_TYPE}, {MANIFEST_TYPE}"))
        if m.get("mediaType") == LIST_TYPE:
            child = next((c for c in m.get("manifests", []) if c.get("runner", "").lower() == "mlx"), None)
            if child is None:
                raise RuntimeError("no MLX manifest in list")
            m = json.loads(get(f"{REGISTRY}{repo}/blobs/{child['digest']}"))
        for layer in m.get("layers", []):
            fname = layer.get("name")
            if layer.get("mediaType") != "application/vnd.ollama.image.json" or fname not in HF_FILES:
                continue
            data = get(f"{REGISTRY}{repo}/blobs/{layer['digest']}")
            if "sha256:" + sha256(data) != layer["digest"]:
                raise RuntimeError(f"digest mismatch for {fname}")
            (out / name).mkdir(parents=True, exist_ok=True)
            (out / name / fname).write_bytes(data)
            rec["files"][fname] = {"bytes": len(data), "sha256": sha256(data)}
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if "tokenizer.json" not in rec["files"]:
        rec.setdefault("error", "no tokenizer.json")
    return rec


def fetch_vocabs(out: Path) -> dict:
    from huggingface_hub import snapshot_download

    rec = {"source": "hf", "name": "ggml-vocabs", "repo": "ggml-org/vocabs"}
    try:
        path = snapshot_download("ggml-org/vocabs", allow_patterns=["*.gguf", "*.inp", "*.out"],
                                 local_dir=str(out / "ggml-vocabs"))
        files = [p for p in Path(path).rglob("*") if p.is_file() and ".cache" not in p.parts]
        rec["files"] = len(files)
        rec["bytes"] = sum(p.stat().st_size for p in files)
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/tokenizers")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name, repo in HF_REPOS.items():
        rec = fetch_hf(name, repo, out)
        records.append(rec)
        print(f"{'ok  ' if 'error' not in rec else 'skip'} {name:18s} {repo}  {rec.get('error', '')[:90]}", flush=True)
    for name, model in OLLAMA_MODELS.items():
        rec = fetch_ollama(name, model, out)
        records.append(rec)
        print(f"{'ok  ' if 'error' not in rec else 'skip'} {name:18s} {model}  {rec.get('error', '')[:90]}", flush=True)
    rec = fetch_vocabs(out)
    records.append(rec)
    print(f"{'ok  ' if 'error' not in rec else 'skip'} ggml-org/vocabs  {rec.get('files', '')} files  {rec.get('error', '')[:90]}")
    try:
        import tokenizers
        tok_version = tokenizers.__version__
    except ImportError:
        tok_version = None
    (out / "fetch.json").write_text(json.dumps({"fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                "tokenizers": tok_version, "records": records}, indent=1))
    ok = sum("error" not in r for r in records)
    print(f"{ok} of {len(records)} sources fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
