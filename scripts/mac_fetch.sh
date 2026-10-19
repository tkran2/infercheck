#!/bin/sh
# Gather everything the cloud side of InferCheck needs but cannot download itself:
#   1. tokenizer files from Hugging Face, the Ollama registry and ggml-org/vocabs,
#   2. vocabulary-only GGUF copies of the Ollama models installed on this Mac,
#   3. Python wheels (tokenizers, transformers, ...) for the Linux test machine.
# Everything lands in data/ and is packed into infercheck-data.tar.gz. No model weights
# are downloaded. Safe to rerun.
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null 2>&1 || PY=python3

if [ ! -d .venv ]; then
    "$PY" -m venv .venv
fi
. .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q tokenizers transformers huggingface_hub gguf jinja2

echo "== tokenizers"
python scripts/fetch_tokenizers.py --out data/tokenizers

echo "== vocabulary-only GGUF copies of local Ollama models"
python scripts/ollama_gguf_vocab.py qwen3:8b gpt-oss:20b

echo "== wheels for the Linux test machine (CPython 3.13, x86_64)"
mkdir -p data/wheels
python -m pip download -q --only-binary=:all: --python-version 3.13 --implementation cp --abi cp313 \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
    -d data/wheels \
    tokenizers transformers jinja2 regex sentencepiece protobuf

python -c "import tokenizers, transformers; print('Mac versions: tokenizers', tokenizers.__version__, '| transformers', transformers.__version__)" \
    | tee data/versions.txt

echo "== packing"
tar czf infercheck-data.tar.gz data
ls -lh infercheck-data.tar.gz
echo "== done. Tell Claude the archive is ready."
