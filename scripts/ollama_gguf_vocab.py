"""Copy the metadata of locally installed Ollama GGUF models into vocabulary-only GGUF files.

Ollama serves library models such as qwen3:8b as GGUF files that llama.cpp loads. Their
tokenizer is whatever Ollama's converter wrote into the file's metadata, so testing those
files tests what users actually run. This writes data/ollama-gguf/<model>.gguf with every
metadata key of the original and no tensors, which llama.cpp can load with vocab_only.

Usage: python scripts/ollama_gguf_vocab.py qwen3:8b gpt-oss:20b
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

LIST_TYPE = "application/vnd.ollama.manifest.list.v2+json"


def models_dir() -> Path:
    return Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))


def blob(digest: str) -> Path:
    return models_dir() / "blobs" / digest.replace(":", "-")


def model_blob(model: str) -> Path:
    name, tag = model.split(":", 1) if ":" in model else (model, "latest")
    repo = name if "/" in name else f"library/{name}"
    path = models_dir() / "manifests" / "registry.ollama.ai" / repo / tag
    m = json.loads(path.read_text())
    if m.get("mediaType") == LIST_TYPE:
        child = next((c for c in m.get("manifests", [])
                      if c.get("runner", "").lower() in ("llamacpp", "ggml")), None)
        if child is None:
            raise RuntimeError(f"{model}: no GGUF manifest in the list")
        m = json.loads(blob(child["digest"]).read_text())
    layer = next((x for x in m.get("layers", []) if x.get("mediaType") == "application/vnd.ollama.image.model"), None)
    if layer is None:
        raise RuntimeError(f"{model}: no model layer")
    return blob(layer["digest"])


def write_vocab_only(src: Path, dst: Path) -> int:
    from gguf import GGUFReader, GGUFValueType, GGUFWriter

    r = GGUFReader(str(src))
    arch = r.fields["general.architecture"].contents()
    w = GGUFWriter(str(dst), arch)
    n = 0
    for key, field in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        vtype = GGUFValueType(field.types[0])
        sub = GGUFValueType(field.types[-1]) if vtype == GGUFValueType.ARRAY else None
        w.add_key_value(key, field.contents(), vtype, sub_type=sub)
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return n


def main() -> int:
    out = Path("data/ollama-gguf")
    out.mkdir(parents=True, exist_ok=True)
    for model in sys.argv[1:] or ["qwen3:8b", "gpt-oss:20b"]:
        try:
            src = model_blob(model)
            with open(src, "rb") as fh:
                if fh.read(4) != b"GGUF":
                    raise RuntimeError(f"{src} is not a GGUF file")
            dst = out / (model.replace(":", "_").replace("/", "_") + ".gguf")
            n = write_vocab_only(src, dst)
            print(f"ok   {model}: {n} metadata keys -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:
            print(f"skip {model}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
