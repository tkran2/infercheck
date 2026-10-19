#!/bin/sh
# Convert each fetched Hugging Face tokenizer to a vocabulary-only GGUF with llama.cpp's own
# converter (convert_hf_to_gguf.py --vocab-only), so llama.cpp is tested on exactly the
# metadata its converter writes today. Needs no weights. Usage:
#   scripts/convert_vocabs.sh DATA_DIR LLAMA_CPP_DIR OUT_DIR [PYTHON]
# Without PyTorch installed, put scripts/torchstub on PYTHONPATH (vocab-only reads no tensors).
data=$1; llama=$2; out=$3; py=${4:-python3}
mkdir -p "$out"
for d in "$data"/tokenizers/*/; do
    name=$(basename "$d")
    [ -f "$d/tokenizer.json" ] && [ -f "$d/config.json" ] || continue
    [ -f "$out/$name.gguf" ] && continue
    if timeout 300 "$py" "$llama/convert_hf_to_gguf.py" "$d" --vocab-only --outfile "$out/$name.gguf" > "$out/$name.log" 2>&1; then
        echo "ok    $name"
    else
        echo "fail  $name: $(grep -v '^\[transformers\]' "$out/$name.log" | tail -1 | cut -c1-160)"
        rm -f "$out/$name.gguf"
    fi
done
