# InferCheck

Differential conformance testing for the tokenizers inside LLM inference engines.

An inference engine has to turn text into exactly the token ids the model was trained on. If it doesn't, the model gets input it never saw in training, prompts get longer, and output text comes back wrong. Engines reimplement Hugging Face tokenizers by hand: llama.cpp in C++, Ollama's MLX runner in Go. InferCheck checks those reimplementations against the Hugging Face `tokenizers` library on hundreds of thousands of adversarial strings. It shrinks every disagreement to a minimal reproducer, which is usually one or two characters.

**Upstream status:** the combining-marks fix is merged into llama.cpp and shipped in release [b11753](https://github.com/ggml-org/llama.cpp/releases/tag/b11753). The Ollama fix is under review. The NFC bug below (fix 2) was also reported independently in [#30747](https://github.com/ggml-org/llama.cpp/issues/30747), and a broader fix for it is open in [#30708](https://github.com/ggml-org/llama.cpp/pull/30708).

## Results

The runs below used llama.cpp `f74e2fa` and Ollama `d1ed3fd` (both 2026-10-16), with Hugging Face `tokenizers` 0.23.3 as the reference.

| engine | tokenizers | test strings | encode disagreements | after the fixes below |
|---|---:|---:|---:|---:|
| llama.cpp, vocab-only GGUFs from its own `convert_hf_to_gguf.py` | 52 | 387,413 | 14,809 | 5,818 |
| Ollama MLX runner tokenizer | 43 | 322,416 | 8,030 (decode: 15,347) | 8,030 (decode: 10,150) |

### Bugs found and fixed

| # | engine | bug | affected | effect | fix and status |
|---|---|---|---|---|---|
| 1 | llama.cpp | The GPT-4o and Tekken pre-tokenizer rewrites leave `\p{M}` out of their lookaheads, so combining marks split off their letters | gpt-oss, Llama 4, Muse Spark, Kanana-2, MiniMax-M2, Mistral NeMo (Tekken) | Hindi `दी` is 2 tokens instead of 1. gpt-oss uses +87% tokens on Hindi, +100% on Bengali, +88% on Telugu | 2 lines, regex only. Issue [#30787](https://github.com/ggml-org/llama.cpp/issues/30787), **merged** in [#30845](https://github.com/ggml-org/llama.cpp/pull/30845), released in [b11753](https://github.com/ggml-org/llama.cpp/releases/tag/b11753); regression tests in [ggml-org/vocabs](https://huggingface.co/ggml-org/vocabs) |
| 2 | llama.cpp | The BPE tokenizer ignores `tokenizer.ggml.normalizer.nfc`, which the converter has written since 2026-10-14 | Qwen1.5/2.5/3/3.5, DeepSeek-R1-Qwen, Bailing, Seed-Coder, OLMo, ModernBERT, MiniMax-M2 | `e` + U+0301 becomes `e` plus a lone accent token instead of `é`. 11 tokenizers have about 660 disagreements each | 16 lines (local patch). Reported independently in [#30747](https://github.com/ggml-org/llama.cpp/issues/30747); upstream fix in progress in [#30708](https://github.com/ggml-org/llama.cpp/pull/30708) |
| 3 | Ollama | The MLX runner's `Decode` byte-maps added tokens, garbling any that contain non-Latin-1 characters | Laguna, DeepSeek-style `<｜…｜>` tokens | Laguna's `〈|EOS|〉` decodes to `"\b|EOS|\t"` in streamed text, logprobs and xgrammar pieces | 1 function + test. PR [ollama/ollama#19123](https://github.com/ollama/ollama/pull/19123), under review |

With fixes 1 and 2, 15 tokenizers go from hundreds of disagreements to zero, ModernBERT drops from 661 to 6, and no other tokenizer changes ([table](results/llama.cpp-master-vs-fixed.md)). All of llama.cpp's own `test-tokenizer-0` vocabularies still pass. Fix 3 removes 5,197 decode disagreements in Ollama ([table](results/ollama-master-vs-fixed.md)).

Token counts on ordinary sentences, gpt-oss, Hugging Face vs llama.cpp before fix 1 ([data](results/token_inflation.json)):

| | Hindi | Bengali | Telugu | Tamil | Thai | Khmer | English |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hugging Face | 15 | 9 | 8 | 11 | 12 | 17 | 10 |
| llama.cpp | 28 | 18 | 15 | 17 | 16 | 19 | 10 |

### Other disagreements found (not yet fixed)

- **llama.cpp, Laguna:** `\r\n` becomes two tokens. Its custom `[^\n]+|[\n]+` splitter cuts CRLF apart, while Hugging Face's `(?:\r?\n)+` keeps it whole.
- **llama.cpp, Tiny Aya and Cohere North:** the digit-grouping splitter merges the text before a number into the first digit group (`"  1"`). It also ignores the `\b` in `\d{1,3}(?=(?:\d{3})*\b)` (`1234a`).
- **llama.cpp, Nemotron 3 / 3.5 ([#30894](https://github.com/ggml-org/llama.cpp/issues/30894)):** their tokenizer hashes the same as Pixtral's, so the GGUFs (NVIDIA's own included) are tagged `pixtral` and get the Llama 3 regex instead of Tekken. Hindi uses 27 tokens instead of 16.
- **llama.cpp, added-token flags:** `lstrip`/`rstrip` from tokenizer.json are not carried into GGUF (Phi-3, Phi-4, ModernBERT `[MASK]`).
- **llama.cpp, `llama_detokenize` space clean-up:** clean-up is fixed per pre-tokenizer instead of following `clean_up_tokenization_spaces` (GLM-4, LFM2, EXAONE 4, Phi-4, Kanana-2).
- **Ollama:** Sequence normalizers are silently ignored (Phi-3: 7,091 of 7,150 strings wrong). `lstrip`/`rstrip` are ignored, NUL is dropped on decode, `clean_up_tokenization_spaces` is ignored, and SentencePiece decode keeps the leading space.

Minimal reproducers for every class are in `results/*/findings.jsonl`.

## How it works

1. **Corpus** (`infercheck/corpus.py`). More than 1,000 hand-written probes, one rule per string: whitespace runs before words, digit grouping, contractions, combining marks, normalization forms, invisible and control characters, and emoji sequences. Each tokenizer's added tokens are placed at the boundaries where matching rules differ. Seeded random strings are mixed from the same pools, so a finding reproduces from `(seed, index)`.
2. **Engines** (`infercheck/engines.py`, `harness/`). Every engine runs behind the same line protocol:
   - `harness/llama_tok` links libllama, loads a GGUF with `vocab_only` and calls `llama_tokenize` / `llama_detokenize`.
   - `harness/ollama_tok` vendors Ollama's `mlxrunner/tokenizer` package unmodified.
   - The reference is `tokenizers` with the model's own `tokenizer_config.json` decode settings.
3. **Diff and shrink** (`infercheck/diff.py`). Each disagreement is classified as `encode` (different ids), `decode` (same ids, different text) or `error`. Delta debugging (ddmin) then shrinks it over code points to a 1-minimal string that still disagrees the same way.
4. **GGUFs as users get them** (`scripts/convert_vocabs.sh`). llama.cpp is tested on vocab-only GGUFs written by its current `convert_hf_to_gguf.py`, plus the GGUFs Ollama's registry ships. No model weights are needed.

## Auditing GGUF files on Hugging Face

`hub-scan` checks published GGUF repos without downloading the models. For each repo it reads only the metadata section of one GGUF file (a few MB, by HTTP range requests), rebuilds it as a vocab-only GGUF, and runs the corpus through llama.cpp and through the `tokenizer.json` of the model the GGUF was converted from. It also compares that tokenizer with its own base model's, which catches fine-tunes that publish a changed tokenizer.

```sh
python -m infercheck.cli hub-scan --top 500 --out results/hub     # resumable; one line per repo in records.jsonl
python -m infercheck.cli hub-scan --repos nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF --out results/one
python -m infercheck.cli hub-report results/hub                   # writes results/hub/report.md
```

The report lists every repo where llama.cpp disagrees with the source tokenizer (sorted by downloads, with a rough cause for each finding: combining marks, NFC, digits, whitespace, CRLF, special tokens), grouped by architecture and pre-tokenizer, plus every repo whose source tokenizer differs from its base model. Gated base models need `hf auth login` with access granted.

## Reproduce

Hugging Face downloads happen on a machine with internet access (`scripts/mac_fetch.sh`). The test machine needs no network.

```sh
scripts/mac_fetch.sh                      # tokenizers, vocab-only GGUFs of local Ollama models, wheels
python -m venv .venv && . .venv/bin/activate && pip install tokenizers transformers

# engines
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp && cmake -S ../llama.cpp -B ../llama.cpp/build && cmake --build ../llama.cpp/build --target llama -j
L=../llama.cpp; g++ -O2 -std=c++17 harness/llama_tok/llamatok.cpp -I$L/include -I$L/ggml/include -L$L/build/bin \
    -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o harness/llama_tok/llamatok
(cd harness/ollama_tok && go mod tidy && go build -o ollamatok .)

# llama.cpp's converter on every fetched tokenizer (scripts/torchstub stands in for PyTorch: vocab-only reads no tensors)
PYTHONPATH=scripts/torchstub:$L/gguf-py scripts/convert_vocabs.sh data $L conv python

python -m infercheck.cli tok   --data data --out results/run --gguf-dir conv $L/models   # encode + decode, shrunk findings
python -m infercheck.cli vocab --data data --out results/run                              # every token id decoded alone
python -m infercheck.cli compare results/before results/after                            # Markdown table
python scripts/token_counts.py data/tokenizers/gpt-oss/tokenizer.json conv/gpt-oss.gguf          # token counts on sample sentences
python -m unittest discover -s tests
```

To compare two builds of llama.cpp, set `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) to the other build's library folder. `INFERCHECK_LLAMATOK` and `INFERCHECK_OLLAMATOK` point the harness at other binaries.

## Limits

- **The reference is the model's `tokenizer.json`.** Where the original tokenizer is SentencePiece, the fast Hugging Face tokenizer can itself differ from it. For example, Mistral-7B's `" "` gives `▁` in Hugging Face and `▁▁` in sentencepiece. Those cases are reported, but not counted as engine bugs.
- **Only tokenization is tested.** Chat templates, sampling and model outputs are not.
- **Unicode versions.** Engines and the reference can disagree on code points assigned after an engine's Unicode tables were generated.

## Layout

```
infercheck/   corpus, engines, diff + ddmin, CLI, hub audit (hub.py, ggufhdr.py)
harness/      llama_tok (C++ over libllama), ollama_tok (Go, vendored Ollama tokenizer, MIT)
scripts/      data fetch, GGUF conversion, digest of results
results/      run summaries, shrunk findings, before/after tables
tests/        unit tests (corpus, shrinking, CLI, both harnesses)
```
