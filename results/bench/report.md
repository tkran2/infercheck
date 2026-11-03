# InferCheck bug benchmark

Real tokenizer and special-token bugs from llama.cpp's history and from published GGUF files, replayed: each case runs InferCheck on the version with the bug and the version with the fix. A check catches a bug when the fix repairs strings (or start/stop tokens) that it flagged with the bug.

Run on darwin; reference for the tokens around a prompt: transformers 5.22.1.

**InferCheck caught 10 of 10 replayed bugs** (2 of 2 bugs in GGUF files, 8 of 8 in llama.cpp), from the GGUF metadata alone.

3 of the 10 were first found by InferCheck itself; on the 7 bugs found by others it caught 7.

- caught: 10, missed: 0
- not covered yet (chat templates, model metadata): 4
- inconclusive (could not build, find or load a side): 0
- not run here (the bug only shows on another platform): 2
- controls (no known bug): 1 clean, 3 flagged

## Which checks caught them

| check | what it compares | bugs caught |
|---|---|---:|
| encode | token ids differ from the reference | 7 |
| decode | same ids, different decoded text | 2 |
| refdecode | decoding the reference's ids gives different text or crashes | 4 |
| wrap | different tokens added around a prompt (BOS/EOS) | 2 |
| stop | different end-of-generation tokens | 1 |
| stress | fails or differs on long inputs | 0 |

2 of the 10 were caught only by the refdecode, wrap, stop or stress checks.

## Cases

Counts are problems found with the bug → with the fix, for each check that fired.

| case | date | where | category | result | checks that fired | trigger reproduces |
|---|---|---|---|---|---|---|
| gemma4-loaded-as-spm | 2026-04-03 | engine | tokenizer | caught | encode 120→891 (82 repaired, 853 new) | - |
| gemma4-newline-split | 2026-04-04 | engine | crash | other-platform | - | - |
| gemma4-byte-token-decode | 2026-04-06 | engine | detokenizer | caught | refdecode 887→24 | yes |
| gemma4-stop-tokens | 2026-04-08 | engine | stop-tokens | caught | encode 890→866, stop 2→0 | - |
| gemma4-byte-fallback-encode | 2026-04-09 | engine | tokenizer | caught | encode 866→0 | yes |
| gemma4-softcapping | 2026-04-04 | engine | metadata | not-covered | - | - |
| tekken-combining-marks | 2026-10-20 | engine | tokenizer | caught | encode 51→0 | yes |
| pixtral-pretokenizer | 2026-10-22 | engine | tokenizer | caught | encode 58→0, decode 24→0, refdecode 24→0 | - |
| clean-spaces-decode | 2026-10-22 | engine | detokenizer | caught | decode 24→0, refdecode 24→0 | yes |
| tilde-symbol-class | 2026-08-18 | engine | tokenizer | caught | encode 4→0 | yes |
| qwen2-long-input-crash | 2026-04-07 | engine | crash | other-platform | - | - |
| llama3-missing-pretokenizer | 2024-04-29 | file | tokenizer | caught | encode 674→-, refdecode 1→-, wrap 1→- | yes |
| gemma4-add-bos | 2026-04-06 | file | wrap | not-covered | - | - |
| glm4-0414-conversion | 2025-04-23 | file | metadata | not-covered | - | - |
| qwen3-embedding-eos | 2025-06-21 | file | wrap | caught | wrap 1→0 | - |
| gemma4-chat-template | 2026-04-11 | file | template | not-covered | - | - |
| control-gpt-oss-20b |  | control | control | flagged | encode 1 | - |
| control-llama-3.1-8b |  | control | control | flagged | decode 1, refdecode 1 | - |
| control-qwen3-8b |  | control | control | flagged | encode 117, stop 1 | - |
| control-granite-4.2 |  | control | control | clean | - | - |

## Details

### gemma4-loaded-as-spm: Gemma 4's BPE vocabulary was tokenized with the SentencePiece algorithm

The first Gemma 4 support loaded its vocabulary as SPM; #21343 switched it to BPE with newline-only pre-splitting and raw UTF-8 bytes. The switch fixed words like " discards" but started dropping characters that need byte-fallback tokens (fixed in #21534) and crashing on some newline runs (#21406), so the fixed side has bugs of its own: only the comparison against the reference tokenizer can say which side is closer.

- broken: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp 0c58ba3365d2, reference google/gemma-4-E2B-it
  - stop: '<eos>' (1) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: '<|tool_response>' (50) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: llama.cpp stops on '</s>' (212), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer
  - encode (other): `'sgo'`
  - encode (other): `'tgo'`
  - encode (other): `'rgo'`
- fixed: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp b069b10ab48f, reference google/gemma-4-E2B-it
  - stop: '<eos>' (1) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: '<|tool_response>' (50) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: llama.cpp stops on '</s>' (212), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer
  - decode (decode-cleanup): `' !'`
  - encode (crlf): `'\r'`
  - encode (crlf): `'a\r\rb'`
  - stress x5: error exit -6: /Users/thomas/Projects/infercheck/results/bench/builds/llama-b069b10ab48f/src/src/llama-vocab.cpp:3715: GGML_AS
- note: the fix also introduced new problems: encode 853, decode 19, refdecode 887, stress 1
- sources: https://github.com/ggml-org/llama.cpp/pull/21343

### gemma4-newline-split: Gemma 4 crashed on long inputs: its newline pre-split regex fell back to std::regex

The pre-tokenizer pattern [^\n]+|[\n]+ had no custom implementation, so it went through std::regex, which segfaults (stack overflow) on about 50,000 repeated characters. #21406 added a custom splitter. Checked offline with scripts/bench_offline.py: llama.cpp d0068583 segfaults on 6 of the 10 long inputs and b7ad48eb on none. The stack overflow is in libstdc++'s std::regex (GCC, so Linux); Apple's libc++ does not overflow on the same input, so the case only runs on Linux and a macOS run reports it as other-platform.

- broken: wrong-platform, ?, llama.cpp d006858316d4
- fixed: wrong-platform, ?, llama.cpp b7ad48ebda22
- note: this bug reproduces only on linux (this run: darwin)
- sources: https://github.com/ggml-org/llama.cpp/pull/21406

### gemma4-byte-token-decode: Gemma 4's byte-fallback tokens decoded to nothing (emoji vanished from output)

token_to_piece had no branch for byte tokens in BPE vocabularies, so <0xF0><0x9F><0xA6><0x99> (🦙) decoded to an empty string; token_to_byte would have aborted. llama.cpp's own encoder dropped those characters at the time (fixed three days later), so an encode-and-decode round trip never produced the tokens: only decoding the reference's ids shows the bug. Checked offline with scripts/bench_offline.py: llama.cpp 941146b3 decodes ' this is 🦙.cpp' as ' this is.cpp', 4aa962e2 gets it right, and InferCheck's corpus run through both builds finds no difference.

- broken: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp 941146b3f1eb, reference google/gemma-4-E2B-it
  - stop: '<eos>' (1) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: llama.cpp stops on '</s>' (212), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer
  - decode (decode-cleanup): `' !'`
  - encode (crlf): `'\r'`
  - encode (crlf): `'a\r\rb'`
- fixed: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp 4aa962e2b0d0, reference google/gemma-4-E2B-it
  - stop: '<eos>' (1) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: llama.cpp stops on '</s>' (212), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer
  - decode (decode-cleanup): `' !'`
  - encode (crlf): `'\r'`
  - encode (crlf): `'a\r\rb'`
- sources: https://github.com/ggml-org/llama.cpp/pull/21488

### gemma4-stop-tokens: Gemma 4 stopped on the text token </s>

llama.cpp's end-of-generation auto-detection matched </s>; #21492 removes it from the EOG list (and makes it a normal token again) when the vocabulary has <|tool_response>.

- broken: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp 4a05e0c566cf, reference google/gemma-4-E2B-it
  - stop: '<eos>' (1) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: llama.cpp stops on '</s>' (212), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer
  - decode (decode-cleanup): `' !'`
  - encode (crlf): `'\r'`
  - encode (crlf): `'a\r\rb'`
- fixed: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp d9a12c82f0c8, reference google/gemma-4-E2B-it
  - decode (decode-cleanup): `' !'`
  - encode (crlf): `'\r'`
  - encode (crlf): `'a\r\rb'`
- sources: https://github.com/ggml-org/llama.cpp/pull/21492

### gemma4-byte-fallback-encode: Gemma 4 silently dropped characters that need byte-fallback tokens

For BPE without GPT-2 byte encoding, the fallback looked up a one-byte string instead of <0xXX>, found nothing and emitted nothing.

- broken: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp 243532e55687, reference google/gemma-4-E2B-it
  - decode (decode-cleanup): `' !'`
  - encode (crlf): `'\r'`
  - encode (crlf): `'a\r\rb'`
- fixed: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp 0ec191e1d761, reference google/gemma-4-E2B-it
  - decode (decode-cleanup): `' !'`
  - decode (decode-cleanup): `" 's"`
  - decode (decode-cleanup): `" 're"`
- note: the fix also introduced new problems: decode 5
- sources: https://github.com/ggml-org/llama.cpp/pull/21534

### gemma4-softcapping: Gemma 4's final_logit_softcapping was not read

Changes the logits, not the tokenizer: InferCheck has no check for it yet (a logits comparison would).

- broken: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp b7ad48ebda22, reference google/gemma-4-E2B-it
  - stop: '<eos>' (1) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: '<|tool_response>' (50) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: llama.cpp stops on '</s>' (212), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer
  - decode (decode-cleanup): `' !'`
  - encode (crlf): `'\r'`
  - encode (crlf): `'a\r\rb'`
- fixed: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp 650bf14eb9a9, reference google/gemma-4-E2B-it
  - stop: '<eos>' (1) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: '<|tool_response>' (50) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: llama.cpp stops on '</s>' (212), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer
  - decode (decode-cleanup): `' !'`
  - encode (crlf): `'\r'`
  - encode (crlf): `'a\r\rb'`
- sources: https://github.com/ggml-org/llama.cpp/pull/21390

### tekken-combining-marks: Tekken and GPT-4o pre-tokenizers split words at combining marks

Found with InferCheck (tkran2). The regex lookaheads allowed letters but not marks (\p{M}).

- broken: ok, bartowski/Mistral-Nemo-Instruct-2407-GGUF@a2dd64a0a7, llama.cpp 7879491cf4c1, reference mistralai/Mistral-Nemo-Instruct-2407
  - encode (combining-mark): `'दी'`
  - encode (combining-mark): `'न्'`
  - encode (combining-mark): `'วั'`
- fixed: ok, bartowski/Mistral-Nemo-Instruct-2407-GGUF@a2dd64a0a7, llama.cpp d8d92973c8c2, reference mistralai/Mistral-Nemo-Instruct-2407
- sources: https://github.com/ggml-org/llama.cpp/issues/30787, https://github.com/ggml-org/llama.cpp/pull/30845

### pixtral-pretokenizer: GGUFs tagged "pixtral" got the Llama 3 pre-tokenizer instead of Tekken

Found with InferCheck (tkran2): affected NVIDIA Nemotron 3/3.5 GGUFs.

- broken: ok, nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF@ba223d14e4, llama.cpp bf039c84969b, reference nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8
  - wrap: tokens added at the start of a prompt: llama.cpp [1], reference []
  - decode (decode-cleanup): `' !'`
  - decode (decode-cleanup): `" 's"`
  - decode (decode-cleanup): `" 're"`
- fixed: ok, nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF@ba223d14e4, llama.cpp e197cef4554d, reference nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8
  - wrap: tokens added at the start of a prompt: llama.cpp [1], reference []
- sources: https://github.com/ggml-org/llama.cpp/issues/30894, https://github.com/ggml-org/llama.cpp/pull/30960

### clean-spaces-decode: llama.cpp removed the space before punctuation when decoding Gemma 4 and others

Found with InferCheck (tkran2): x != y decoded as x!= y. #30940 adds a clean_spaces GGUF key and turns the clean-up off by default for these pre-tokenizers.

- broken: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp e197cef4554d, reference google/gemma-4-E2B-it
  - decode (decode-cleanup): `' !'`
  - decode (decode-cleanup): `" 's"`
  - decode (decode-cleanup): `" 're"`
- fixed: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp d88040c704c8, reference google/gemma-4-E2B-it
- sources: https://github.com/ggml-org/llama.cpp/issues/30905, https://github.com/ggml-org/llama.cpp/pull/30940

### tilde-symbol-class: " ~" was split in two, preventing DeepSeek V4's Ġ~ merge

The collapsed \p{S} symbol class was missing '~'. The repos are found by search: check resolved.json.

- broken: ok, unsloth/DeepSeek-V4.1-Flash-1017-GGUF@8a6ec9cf11, llama.cpp 169e4a7ff201, reference deepseek-ai/DeepSeek-V4-Flash-0731
  - encode (whitespace+text): `' ~'`
  - encode (whitespace+text): `'~\n'`
- fixed: ok, unsloth/DeepSeek-V4.1-Flash-1017-GGUF@8a6ec9cf11, llama.cpp afd439df1f08, reference deepseek-ai/DeepSeek-V4-Flash-0731
- sources: https://github.com/ggml-org/llama.cpp/pull/26972

### qwen2-long-input-crash: Qwen2-style tokenizers crashed llama.cpp on long runs of one character

The Qwen2 regex fell through to std::regex, whose recursive backtracking overflowed the stack (43K 'A's). InferCheck's corpus had nothing that long, so this case motivated the stress check. The stack overflow is in libstdc++'s std::regex (GCC, so Linux); Apple's libc++ does not overflow on the same input, so the case only runs on Linux and a macOS run reports it as other-platform.

- broken: wrong-platform, ?, llama.cpp a8ec0df46176
- fixed: wrong-platform, ?, llama.cpp 0d049d6a9245
- note: this bug reproduces only on linux (this run: darwin)
- sources: https://github.com/ggml-org/llama.cpp/issues/21113, https://github.com/ggml-org/llama.cpp/pull/21257

### llama3-missing-pretokenizer: Llama 3 GGUFs converted before the BPE pre-tokenizer fix

Files without tokenizer.ggml.pre still load, with a warning, using the default (GPT-2) pre-tokenizer. The repo's maintainer said V1 was made before the fix. No fixed upload of the same repo, so this case counts only findings with the causes listed.

- broken: ok, QuantFactory/Meta-Llama-3-8B-GGUF@1ca85c857d, llama.cpp current build, reference NousResearch/Meta-Llama-3-8B, tokenizer.ggml.pre = None
  - wrap: tokens added at the start of a prompt: llama.cpp [], reference ['<|begin_of_text|>' (128000)]
  - encode (whitespace+text): `' !'`
  - encode (whitespace+text): `'\tb'`
  - encode (whitespace+text): `'\tw'`
- sources: https://github.com/ggml-org/llama.cpp/pull/6920, https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF-v2/discussions/1

### gemma4-add-bos: Gemma 4 GGUFs from launch week had add_bos_token = false

The converter wrote add_bos_token = false, so llama.cpp gave plain-text prompts no BOS; #21500 made the converter write true and made llama.cpp override old files. This case was first scored as a wrap bug, but the run showed the original tokenizer behaves like the file before the fix: transformers adds no BOS to plain Gemma 4 text (Google's tokenizer_config.json does not set add_bos_token, and transformers 5.22.1 then adds none), and chat prompts get their BOS from the chat template in both. #21500 is llama.cpp choosing to give plain-text prompts a BOS too, which a comparison against the original tokenizer cannot call a bug, so the case is not covered.

- broken: ok, unsloth/gemma-4-E2B-it-GGUF@e18a8a4803, llama.cpp f51fd36d79a0, reference google/gemma-4-E2B-it, tokenizer.ggml.add_bos_token = False
  - stop: '<eos>' (1) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: llama.cpp stops on '</s>' (212), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer
  - decode (decode-cleanup): `' !'`
  - encode (crlf): `'\r'`
  - encode (crlf): `'a\r\rb'`
- fixed: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp f51fd36d79a0, reference google/gemma-4-E2B-it, tokenizer.ggml.add_bos_token = True
  - stop: '<eos>' (1) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - stop: llama.cpp stops on '</s>' (212), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer
  - decode (decode-cleanup): `' !'`
  - encode (crlf): `'\r'`
  - encode (crlf): `'a\r\rb'`
- note: the fix also introduced new problems: wrap 1
- sources: https://github.com/ggml-org/llama.cpp/pull/21500, https://unsloth.ai/docs/new/changelog

### glm4-0414-conversion: GLM-4-0414 GGUFs converted before the rotary-dimension fix

#13021 fixed three things in the GLM-4-0414 converter: the rotary dimension (half the head dimension; the old files used all of it, which gives wrong logits), the end tokens and the BOS. This case was first scored as a stop-token bug, but the run showed the end-token change made no difference: the old files took <|user|> as EOS from tokenizer_config.json and llama.cpp stops on <|endoftext|> by name, so both files stop on the same two tokens, and [gMASK] comes from the chat template either way. What users saw was the rotary dimension, which changes the logits: not covered. Both files also lack <|observation|> as a stop token; today's converter marks it (eom), so only files converted before that change are affected.

- broken: ok, bartowski/THUDM_GLM-4-9B-0414-GGUF@04ba18da36, llama.cpp current build, reference THUDM/GLM-4-9B-0414, tokenizer.ggml.eot_token_id = None
  - stop: '<|observation|>' (151338) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - decode (decode-cleanup): `' !'`
  - decode (decode-cleanup): `" 's"`
  - decode (decode-cleanup): `" 're"`
- fixed: ok, bartowski/THUDM_GLM-4-9B-0414-GGUF@211924cd94, llama.cpp current build, reference THUDM/GLM-4-9B-0414, tokenizer.ggml.eot_token_id = 151336
  - stop: '<|observation|>' (151338) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer
  - decode (decode-cleanup): `' !'`
  - decode (decode-cleanup): `" 's"`
  - decode (decode-cleanup): `" 're"`
- sources: https://github.com/ggml-org/llama.cpp/pull/13021

### qwen3-embedding-eos: Qwen3-Embedding GGUFs appended the wrong end token

The tokenizer's post-processor appends <|endoftext|>; the converter kept the config's EOS instead. Needs an upload before and after 2025-06-22; otherwise inconclusive.

- broken: ok, Qwen/Qwen3-Embedding-0.6B-GGUF@48f36f50b4, llama.cpp current build, reference Qwen/Qwen3-Embedding-0.6B, tokenizer.ggml.add_eos_token = None
  - wrap: tokens added at the end of a prompt: llama.cpp [], reference ['<|endoftext|>' (151643)]
  - stop: llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer
  - encode (nfc): `'\u2000'`
  - encode (nfc): `'\u2000\u20001'`
  - encode (nfc): `'\u2000\u2000!'`
- fixed: ok, Qwen/Qwen3-Embedding-0.6B-GGUF@370f27d755, llama.cpp current build, reference Qwen/Qwen3-Embedding-0.6B, tokenizer.ggml.add_eos_token = True
  - stop: llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer
  - encode (nfc): `'\u2000'`
  - encode (nfc): `'\u2000\u20001'`
  - encode (nfc): `'\u2000\u2000!'`
- sources: https://github.com/ggml-org/llama.cpp/pull/14314

### gemma4-chat-template: Gemma 4 GGUFs updated with Google's chat template fixes (tool calling)

A chat template change: InferCheck does not render templates yet, so this is expected to show nothing.

- broken: ok, unsloth/gemma-4-E2B-it-GGUF@f7c65a52de, llama.cpp current build, reference google/gemma-4-E2B-it
  - decode (decode-cleanup): `' !'`
  - decode (decode-cleanup): `" 's"`
  - decode (decode-cleanup): `" 're"`
- fixed: ok, unsloth/gemma-4-E2B-it-GGUF@0314792d7f, llama.cpp current build, reference google/gemma-4-E2B-it
  - decode (decode-cleanup): `' !'`
  - decode (decode-cleanup): `" 's"`
  - decode (decode-cleanup): `" 're"`
- sources: https://unsloth.ai/docs/new/changelog

### control-gpt-oss-20b: Control: gpt-oss-20b

- only: ok, unsloth/gpt-oss-20b-GGUF@d449b42d93, llama.cpp current build, reference openai/gpt-oss-20b
  - encode (other): `'ÇA'`
- sources: https://huggingface.co/openai/gpt-oss-20b

### control-llama-3.1-8b: Control: Llama 3.1 8B Instruct

- only: ok, bartowski/Meta-Llama-3.1-8B-Instruct-GGUF@bf5b95e96d, llama.cpp current build, reference unsloth/Meta-Llama-3.1-8B-Instruct
  - decode (apostrophe): `" n't"`
- sources: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

### control-qwen3-8b: Control: Qwen3 8B

Expected to be flagged: since llama.cpp #18825 (2026-02-20) the ordinary Qwen token "</s>" (128247) is an end-of-generation control token, so prompts containing </s> tokenize differently. Found while building this benchmark; not yet reported.

- only: ok, Qwen/Qwen3-8B-GGUF@7c41481f57, llama.cpp current build, reference Qwen/Qwen3-8B
  - stop: llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer
  - encode (nfc): `'\u2000'`
  - encode (nfc): `'\u2000\u20001'`
  - encode (nfc): `'\u2000\u2000!'`
- sources: https://github.com/ggml-org/llama.cpp/pull/18825

### control-granite-4.2: Control: Granite 4.2 (an intentional tokenizer change)

Granite 4.2 tokenizes numbers and blank lines differently from 4.1 on purpose (IBM confirmed). The GGUF should match its own source model, so nothing should be flagged here.

- only: ok, ibm-granite/granite-4.2-8b-GGUF@93f3f6a893, llama.cpp current build, reference ibm-granite/granite-4.2-8b
- sources: https://huggingface.co/ibm-granite/granite-4.2-8b/discussions

