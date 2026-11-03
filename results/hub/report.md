# GGUF tokenizer audit

- repos scanned: 500 (engine-error: 10, header-error: 7, no-gguf: 13, no-reference: 48, no-tokenizer: 85, ok: 337)
- llama.cpp could not load the vocabulary: 10 repos (listed below; usually an architecture newer than the llama.cpp build)
- checked against their source tokenizer: 337
- llama.cpp tokenizes or decodes differently from the source tokenizer: **255** repos, 141,490,828 downloads in the last 30 days
- source tokenizer differs from its base model's (drift): **32** repos, 16,259,911 downloads

## Causes

A repo is counted once for every cause among its findings.

| cause | repos | downloads (30 days) | what it is | largest repos |
|---|---:|---:|---|---|
| nfc | 238 | 137,204,377 | the source tokenizer normalizes text to NFC; llama.cpp's BPE tokenizer does not (ggml-org/llama.cpp#30747, fix in review in #30708) | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, unsloth/Qwen3.8-27B-GGUF, ornith-ai/Ornith-1.5-9B-GGUF |
| crlf | 46 | 24,164,239 | carriage returns | JonathanColetti/Qwen3.8-27B-Uncensored-GGUF, ornith-ai/Ornith-1.0-9B-GGUF, mradermacher/Ornith-1.5-9B-uncensored-GGUF |
| special-token | 53 | 23,387,638 | text next to added or special tokens (lstrip/rstrip and normalization flags) | JonathanColetti/Qwen3.8-27B-Uncensored-GGUF, ornith-ai/Ornith-1.0-9B-GGUF, mradermacher/Ornith-1.5-9B-uncensored-GGUF |
| combining-mark | 41 | 21,216,702 | combining marks | JonathanColetti/Qwen3.8-27B-Uncensored-GGUF, ornith-ai/Ornith-1.0-9B-GGUF, mradermacher/Ornith-1.5-9B-uncensored-GGUF |
| control-text | 60 | 17,702,277 | text llama.cpp parses as an end-of-generation control token although the source tokenizer treats it as ordinary text (Qwen's </s> since llama.cpp #18825) | Qwen/Qwen3-VL-8B-Instruct-GGUF, bartowski/Qwen2.5-32B-Instruct-GGUF, Qwen/Qwen3-8B-GGUF |
| other | 17 | 5,195,173 | other | unsloth/Muse-Glimmer-30B-GGUF, unsloth/gpt-oss-20b-GGUF, togatogah/jinen-v1-small.gguf |
| apostrophe | 17 | 4,937,710 | apostrophes and contractions | tencent/Hy-MT2-7B-GGUF, togatogah/jinen-v1-small.gguf, datalab-to/surya-ocr-2-gguf |
| whitespace+text | 15 | 4,673,779 | whitespace followed by text | tencent/Hy-MT2-7B-GGUF, togatogah/jinen-v1-small.gguf, datalab-to/surya-ocr-2-gguf |
| digits | 14 | 4,546,798 | digit grouping | tencent/Hy-MT2-7B-GGUF, togatogah/jinen-v1-small.gguf, datalab-to/surya-ocr-2-gguf |
| whitespace | 6 | 1,669,300 | runs of whitespace | togatogah/jinen-v1-small.gguf, datalab-to/surya-ocr-2-gguf, llm-jp/llm-jp-4-33b-thinking-gguf |
| decode-cleanup | 4 | 1,399,661 | llama.cpp's detokenizer removes the space before . , ! ? and 's 'm 're 've (clean_up_tokenization_spaces) although the source tokenizer leaves it off | togatogah/jinen-v1-small.gguf, datalab-to/surya-ocr-2-gguf, second-state/All-MiniLM-L6-v2-Embedding-GGUF |

## llama.cpp vs source tokenizer

| repo | downloads | arch | pre | cases | encode | decode | vs base | causes | examples |
|---|---:|---|---|---:|---:|---:|---|---|---|
| ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF | 7,668,149 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3.8-27B-GGUF | 6,272,560 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ornith-ai/Ornith-1.5-9B-GGUF | 5,426,513 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ornith-ai/Ornith-1.5-35B-A3B-GGUF | 4,038,732 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF | 3,157,401 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP | 2,959,788 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| JonathanColetti/Qwen3.8-27B-Uncensored-GGUF | 2,598,773 | qwen35 | qwen35 | 3116 | 165 | 0 | 103/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ornith-ai/Ornith-1.0-9B-GGUF | 2,414,835 | qwen35 | qwen35 | 3116 | 165 | 0 | - | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF | 2,380,949 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ggml-org/Qwen3.8-27B-GGUF | 2,339,954 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF | 2,279,692 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| byteshape/Qwen3.8-27B-GGUF | 2,011,277 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF | 1,965,614 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| prism-ml/Ternary-Bonsai-2-27B-gguf | 1,851,236 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mradermacher/Ornith-1.5-9B-uncensored-GGUF | 1,713,226 | qwen35 | qwen35 | 3116 | 165 | 0 | 100/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF | 1,581,931 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3.5-9B-GGUF | 1,554,745 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ornith-ai/Ornith-1.0-35B-GGUF | 1,538,701 | qwen35moe | qwen35 | 3116 | 165 | 0 | - | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3.6-35B-A3B-MTP-GGUF | 1,529,269 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF | 1,451,434 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen3-VL-8B-Instruct-GGUF | 1,406,262 | qwen3vl | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3.6-35B-A3B-GGUF | 1,400,527 | qwen35moe | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF | 1,349,805 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3.5-4B-GGUF | 1,323,712 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF | 1,310,650 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| empero-ai/Qwen3.8-35B-A3B-Distill-GGUF | 1,268,674 | qwen35moe | qwen35 | 3116 | 165 | 0 | - | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ornith-ai/Ornith-1.5-397B-GGUF | 1,254,698 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mudler/Laguna-XS-2.1-APEX-GGUF | 1,215,528 | laguna | laguna | 3491 | 54 | 0 | - | crlf 36, nfc 4 | `<U+000D><U+000A>` (crlf); `word<U+000D><U+000A><U+000D><U…` (crlf); `<U+000D><U+000A><U+000D><U+000…` (crlf) |
| lmstudio-community/Qwen3.8-27B-GGUF | 1,152,075 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3.8-Flash-Next-GGUF | 1,149,929 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF | 1,149,119 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive | 1,106,939 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| tencent/Hy-MT2-7B-GGUF | 1,057,596 | hunyuan-dense | hunyuan | 3494 | 0 | 27 | - | whitespace+text 8, apostrophe 5, crlf 4, digits 2, nfc 2 | ` !` (whitespace+text); ` 's` (apostrophe); ` 're` (apostrophe) |
| bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF | 974,177 | qwen35 | qwen35 | 3116 | 165 | 0 | - | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Jackrong/Qwopus3.8-27B-Flash-V2-GGUF | 938,577 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mradermacher/Cyber-Ornith-1.5-9B-OBLITERATED-i1-GGUF | 935,184 | qwen35 | qwen35 | 3116 | 165 | 0 | - | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP | 934,917 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF | 930,328 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| agentionai/Qwen3.8-27B-AP-GGUF | 886,741 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Qwen2.5-32B-Instruct-GGUF | 868,558 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF | 861,399 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Muse-Glimmer-30B-GGUF | 812,668 | muse-glimmer | llama4 | 3494 | 2 | 0 | - | other 2 | `ÔN` (other); `ÕES` (other) |
| Accio-Lab/occamy-1.0-GGUF | 797,020 | qwen35moe | qwen2 | 3116 | 103 | 0 | 115/0 | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| jegreene/Swift-1.5-Qwen3.8-27B-heretic-GSQ-RCO-GGUF | 769,010 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF | 767,975 | qwen4exp | qwen35 | 3116 | 165 | 0 | 103/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen3-8B-GGUF | 733,491 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mradermacher/Simplicio-27B-GGUF | 729,914 | qwen35 | qwen35 | 3116 | 165 | 0 | 103/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| DavidAU/Qwen3.8-27B-Turbo-Brilliance-50X-Reasoning-Instruct-NEO-MAX-MTP-GGUF | 720,584 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3.5-35B-A3B-GGUF | 717,987 | qwen35moe | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mudler/KAT-Coder-V2.5-Dev-APEX-GGUF | 710,647 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen2.5-7B-Instruct-GGUF | 691,536 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ConwayResearch/Underdog-Saluki-27B-1.0 | 679,079 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mradermacher/Qwen3-VL-8B-Instruct-abliterated-GGUF | 677,420 | qwen3vl | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF | 665,980 | qwen3vl | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive | 662,378 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| agentionai/Qwen3.8-Flash-Next-Gyro-GGUF | 643,896 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen3-VL-4B-Instruct-GGUF | 638,421 | qwen3vl | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mradermacher/AdaVaR-3B-GGUF | 618,484 | qwen2vl | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| lmstudio-community/Qwen3.5-9B-GGUF | 617,620 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen3-4B-GGUF | 617,066 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| agentionai/Qwen3.8-Flash-Next-Gyro-abliterated-Huihui-GGUF | 607,127 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ornith-ai/Ornith-1.5-122B-GGUF | 595,019 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| prism-ml/Ternary-Bonsai-2-9B-gguf | 592,665 | dflash | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mradermacher/Qwen3.8-27B-Fable-5-Coding-Distilled-GGUF | 586,259 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3-4B-Instruct-2507-GGUF | 586,058 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/gpt-oss-20b-GGUF | 579,462 | gpt-oss | gpt-4o | 2948 | 1 | 0 | - | other 1 | `ÇA` (other) |
| empero-ai/Qwen3.8-4B-Distill-GGUF | 579,324 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| togatogah/jinen-v1-small.gguf | 558,147 | gpt2 | default | 3494 | 1837 | 14 | - | special-token 677, whitespace+text 349, other 215, digits 143, combining-mark 126, nfc 103, whitespace 97, apostrophe 95, crlf 20, decode-cleanup 13 | `a ` (whitespace+text); `d ` (whitespace+text); ` !` (decode-cleanup) |
| datalab-to/surya-ocr-2-gguf | 543,884 | qwen35 | default | 3393 | 2067 | 18 | - | other 536, whitespace+text 462, special-token 393, combining-mark 228, whitespace 142, nfc 115, digits 94, apostrophe 46, crlf 29, decode-cleanup 16 | ` !` (decode-cleanup); `<U+0301>` (combining-mark); `é` (other) |
| unsloth/gemma-4-E2B-it-qat-mobile-GGUF | 539,138 | gemma4-assistant | None | 2990 | 15 | 0 | - | special-token 1 | `<|video|>` (special-token) |
| philbert440/Qwen3.8-27B-Uncensored-Cyber-GGUF | 538,634 | qwen35 | qwen35 | 3116 | 165 | 0 | 103/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| nomic-ai/nomic-embed-text-v1.5-GGUF | 536,182 | nomic-bert | None | 2724 | 99 | 1966 | - | other 619, whitespace+text 520, digits 387, apostrophe 168, combining-mark 134, nfc 89, special-token 67, crlf 63 | `a` (other); `w` (other); `1` (digits) |
| empero-ai/Qwen3.8-9B-Distill-GGUF | 532,267 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3-4B-GGUF | 531,614 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mradermacher/Qwen3.5-9B-Kimi-k3-Distilled-GGUF | 530,801 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| FINAL-Bench/POCKET-35B-GGUF | 527,066 | qwen35moe | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ukisai/Swift-Qwen3.8-27B-GGUF | 517,539 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF | 511,578 | qwen3moe | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| OBLITERATUS/Qwen3.8-27B-OBLITERATED | 507,560 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| AtomicChat/Qwen3.8-Flash-Next-GGUF | 502,221 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mradermacher/B1-9B-GGUF | 497,702 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| empero-ai/Qwen3.8-2B-Distill-GGUF | 491,921 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF | 471,274 | qwen3moe | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| prism-ml/Ternary-Bonsai-2-2B-gguf | 462,067 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF | 455,382 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF | 449,426 | qwen4exp | qwen35 | 3116 | 165 | 0 | 103/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| abenzerps/Apodex-1.1-mini-GGUF | 441,313 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| migtissera/Tess-4.5-27B-GGUF | 434,507 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF | 432,769 | qwen35 | qwen35 | 3116 | 165 | 0 | 100/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3.5-2B-GGUF | 426,284 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen2.5-1.5B-Instruct-GGUF | 426,120 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| jimmy31chen/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-IQ3_S-Strata | 424,213 | qwen4exp | qwen35 | 3116 | 165 | 0 | 103/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| z-lab/Qwen3.8-27B-DFlash2-GGUF | 411,570 | dflash | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3-VL-4B-Instruct-GGUF | 400,864 | qwen3vl | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen2.5-0.5B-Instruct-GGUF | 399,704 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| prism-ml/Ternary-Bonsai-27B-gguf | 399,632 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF | 396,783 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Ornith-1.0-9B-GGUF | 388,215 | qwen35 | qwen35 | 3116 | 165 | 0 | - | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| poolside/Laguna-S-2.1-GGUF | 382,072 | laguna | laguna | 3491 | 54 | 0 | - | crlf 36, nfc 4 | `<U+000D><U+000A>` (crlf); `word<U+000D><U+000A><U+000D><U…` (crlf); `<U+000D><U+000A><U+000D><U+000…` (crlf) |
| FINAL-Bench/POCKET-Darwin-180B-GGUF | 357,189 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF | 357,177 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen3-14B-GGUF | 353,321 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| InternScience/Sage-A1.5-Q4_K_M-GGUF | 344,686 | qwen35moe | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| DavidAU/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-MTP-GGUF | 337,984 | qwen35 | qwen35 | 3116 | 165 | 0 | 103/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/thomsonreuters_Thomson-1.0-Small-GGUF | 331,476 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| migtissera/Tess-4-27B-GGUF | 328,714 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| tencent/Hy-MT2-1.8B-GGUF | 325,655 | hunyuan-dense | hunyuan-dense | 3494 | 33 | 24 | - | crlf 19, whitespace+text 14, apostrophe 5, nfc 3, digits 2, special-token 1 | ` !` (whitespace+text); `<U+000A><U+000D><U+000A>` (crlf); `a<U+000D><U+000A><U+000D><U+00…` (crlf) |
| Qwen/Qwen2.5-Coder-7B-Instruct-GGUF | 325,382 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3.6-27B-GGUF | 324,861 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| OBLITERATUS/Ornith-1.5-9B-OBLITERATED | 323,893 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| empero-ai/Qwythos-9B-v2-GGUF | 323,534 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| janhq/Jan-v3.5-4B-gguf | 319,723 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| openbmb/MiniCPM-V-4.7-35B-A3B-gguf | 318,881 | qwen35moe | qwen35 | 3494 | 157 | 0 | - | nfc 86, combining-mark 52, special-token 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mudler/Ornith-1.5-35B-A3B-APEX-GGUF | 311,816 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF | 307,829 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| OsaurusAI/Diplodocus-preview-GGUF | 300,054 | k2-horizon | k2-horizon | 3494 | 104 | 0 | - | nfc 90 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Qwen3.8-27B-GGUF | 299,398 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ggml-org/gpt-oss-20b-GGUF | 298,349 | eagle3 | gpt-4o | 2948 | 1 | 0 | - | other 1 | `ÇA` (other) |
| ttindall/Qwen3.8-27B-Defiance-Uncensored-MTP-GGUF | 294,520 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF | 290,441 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF | 285,365 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3.5-0.8B-GGUF | 284,620 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| abhishek085/spark-s1-4b-v6-GGUF | 283,123 | qwen35 | qwen35 | 3116 | 165 | 0 | - | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| empero-ai/Qwen3.8-27B-Ridge-GGUF | 282,362 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Ornith-1.5-122B-GGUF | 279,921 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF | 271,349 | qwen35 | qwen35 | 3116 | 165 | 0 | 103/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| llm-jp/llm-jp-4-33b-thinking-gguf | 270,460 | llama | default | 3494 | 744 | 0 | - | other 264, digits 159, whitespace+text 117, special-token 74, apostrophe 44, combining-mark 36, nfc 19, whitespace 10, crlf 10 | ` w` (whitespace+text); ` b` (whitespace+text); ` o` (whitespace+text) |
| ProCreations/grug-27b-gguf | 265,912 | qwen35 | qwen35 | 3116 | 165 | 0 | 103/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF | 265,004 | qwen35moe | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| prism-ml/Ternary-Bonsai-8B-gguf | 264,648 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen2.5-3B-Instruct-GGUF | 263,681 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/endless-frontier_BigBang-v1-GGUF | 260,134 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| InternScience/Sage-A1.5-Q8_0-GGUF | 258,434 | qwen35moe | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Qwen2.5-7B-Instruct-GGUF | 251,218 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| prism-ml/Bonsai-27B-gguf | 251,054 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| webAI-Official/TwIL-LM3 | 250,621 | smollm3 | smaug-bpe | 3494 | 2 | 1 | - | whitespace+text 2, apostrophe 1 | ` n't` (apostrophe); ` کلاس` (whitespace+text); ` údaj` (whitespace+text) |
| Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF | 249,457 | qwen35moe | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF | 245,689 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bloomer010/Ling-3.0-tiny-GGUF | 245,465 | bailingmoe3 | bailingmoe2 | 3494 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| meta-models/Muse-Glimmer-30B-GGUF | 244,803 | dflash | llama4 | 3494 | 2 | 0 | - | other 2 | `ÔN` (other); `ÕES` (other) |
| mradermacher/Hemmingway-1-GGUF | 244,096 | qwen35 | qwen35 | 3116 | 165 | 0 | 103/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3-1.7B-GGUF | 242,630 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| lmstudio-community/Bonsai-27B-GGUF | 240,229 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen2.5-VL-7B-Instruct-GGUF | 240,086 | qwen2vl | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/North-Mini-Code-1.0-GGUF | 239,740 | cohere2moe | tiny_aya | 3172 | 58 | 0 | - | digits 40, combining-mark 6, nfc 3, special-token 2, crlf 1 | `  1` (digits); `  2` (digits); `<U+0009><U+0009>1` (digits) |
| AtomicChat/Ling-3.0-flash-GGUF | 238,714 | bailingmoe3 | bailingmoe2 | 3494 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF | 237,373 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| superwhisper/s1-mini-GGUF | 234,269 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ggml-org/gpt-oss-120b-GGUF | 226,796 | eagle3 | gpt-4o | 2948 | 1 | 0 | - | other 1 | `ÇA` (other) |
| Qwen/Qwen3-0.6B-GGUF | 222,170 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen-AgentWorld-35B-A3B-GGUF | 221,200 | qwen35moe | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| AnkitAI/Parable-Qwen3-8B-Claude-Fable-5-GGUF | 220,251 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| AnkitAI/Parable-Qwen3-4B-Claude-Fable-5-GGUF | 220,063 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| empero-ai/Qwythos-27B-v1-GGUF | 216,751 | qwen35 | qwen35 | 3116 | 165 | 0 | 100/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3-VL-8B-Instruct-GGUF | 215,179 | qwen3vl | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| KyleHessling1/Qwopus3.6-27B-Fusion-GGUF | 214,825 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mradermacher/Qwen3.8-Flash-Next-Uncensored-i1-GGUF | 213,984 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/gemma-4-12B-it-qat-GGUF | 212,855 | gemma4-assistant | None | 2990 | 15 | 0 | - | special-token 1 | `<|video|>` (special-token) |
| ChristianAzinn/mxbai-embed-large-v1-gguf | 209,469 | bert | None | 2724 | 99 | 1966 | - | other 619, whitespace+text 520, digits 387, apostrophe 168, combining-mark 134, nfc 89, special-token 67, crlf 63 | `a` (other); `w` (other); `1` (digits) |
| unsloth/bge-small-en-v1.5-GGUF | 207,876 | bert | jina-v2-en | 2724 | 99 | 1966 | - | other 619, whitespace+text 520, digits 387, apostrophe 168, combining-mark 134, nfc 89, special-token 67, crlf 63 | `a` (other); `w` (other); `1` (digits) |
| unsloth/Qwen3.5-27B-GGUF | 201,831 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Qwen_Qwen3.5-9B-GGUF | 199,826 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Llama-3.2-3B-Instruct-GGUF | 197,414 | llama | llama-bpe | 3494 | 0 | 1 | - | apostrophe 1 | ` n't` (apostrophe) |
| Qwen/Qwen3-VL-2B-Instruct-GGUF | 197,323 | qwen3vl | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| turintech/Qwen3.5-0.8B-Artemis-GGUF | 195,968 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen2.5-Coder-14B-Instruct-GGUF | 193,090 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF | 187,658 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/gemma-4-26B-A4B-it-qat-GGUF | 182,962 | gemma4-assistant | None | 2990 | 15 | 0 | - | special-token 1 | `<|video|>` (special-token) |
| InternScience/Agents-A1-4B-Q8_0-GGUF | 178,539 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ukisai/Swift-1.5-Qwen3.8-27B-GGUF | 178,321 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| AtomicChat/MiniCPM-V-4.7-35B-A3B-Uncensored-GGUF | 174,737 | qwen35moe | qwen35 | 3494 | 157 | 0 | - | nfc 86, combining-mark 52, special-token 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Qwen_Qwen3.6-35B-A3B-GGUF | 174,030 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| lmstudio-community/Qwen3.5-4B-GGUF | 173,138 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive | 170,126 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen3-Embedding-0.6B-GGUF | 167,454 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3-Coder-Next-GGUF | 165,809 | qwen3next | qwen2 | 3032 | 117 | 0 | 100/0 | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| AngelSlim/Hy3-GGUF | 164,853 | hy_v3 | hunyuan-dense | 3494 | 33 | 0 | - | crlf 15, nfc 3, special-token 1 | `<U+000A><U+000D><U+000A>` (crlf); `<U+000D><U+000A><U+000D><U+000…` (crlf); `<U+000D><U+000A><U+000D><U+000…` (crlf) |
| azukivc/Qwen3.8-27B-Uncensored-GGUF | 163,559 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF | 162,668 | qwen4exp | qwen35 | 3116 | 165 | 0 | 103/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| poolside/Laguna-XS-2.1-GGUF | 162,655 | laguna | laguna | 3491 | 54 | 0 | - | crlf 36, nfc 4 | `<U+000D><U+000A>` (crlf); `word<U+000D><U+000A><U+000D><U…` (crlf); `<U+000D><U+000A><U+000D><U+000…` (crlf) |
| lmstudio-community/Qwen3.6-35B-A3B-GGUF | 162,417 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Kujira/Underdog-Saluki-27B-1.0-MTP-Abliterated-GGUF | 161,171 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Qwen2.5-14B-Instruct-GGUF | 160,896 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| InternScience/Agents-A1-4B-Q4_K_M-GGUF | 160,793 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ggml-org/Qwen2.5-Omni-7B-GGUF | 159,946 | qwen2vl | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF | 159,542 | qwen3 | qwen2 | 3060 | 113 | 0 | - | nfc 88, control-text 11 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen3-1.7B-GGUF | 157,743 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| second-state/All-MiniLM-L6-v2-Embedding-GGUF | 156,488 | bert | None | 2724 | 99 | 1974 | - | other 619, whitespace+text 520, digits 387, apostrophe 168, combining-mark 134, nfc 89, special-token 67, crlf 63, decode-cleanup 8 | `a` (other); `w` (other); `1` (digits) |
| unsloth/Qwen3-0.6B-GGUF | 155,466 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Myric/Laguna-S-2.1-APEX-GGUF | 148,752 | dflash | laguna | 3491 | 54 | 0 | - | crlf 36, nfc 4 | `<U+000D><U+000A>` (crlf); `word<U+000D><U+000A><U+000D><U…` (crlf); `<U+000D><U+000A><U+000D><U+000…` (crlf) |
| ggml-org/Qwen3.6-35B-A3B-GGUF | 146,443 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3-8B-GGUF | 143,705 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| leliuga/all-MiniLM-L6-v2-GGUF | 141,142 | bert | None | 2724 | 99 | 1974 | - | other 619, whitespace+text 520, digits 387, apostrophe 168, combining-mark 134, nfc 89, special-token 67, crlf 63, decode-cleanup 8 | `a` (other); `w` (other); `1` (digits) |
| ggml-org/Qwen3-1.7B-GGUF | 139,297 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF | 136,516 | qwen2 | deepseek-r1-qwen | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ggml-org/Qwen2.5-VL-3B-Instruct-GGUF | 135,505 | qwen2vl | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/gemma-4-E4B-it-qat-GGUF | 130,276 | gemma4-assistant | None | 2990 | 15 | 0 | - | special-token 1 | `<|video|>` (special-token) |
| batiai/Qwen3.6-27B-GGUF | 129,981 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| openbmb/MiniCPM-V-4.7-1B-gguf | 128,106 | qwen35 | qwen35 | 3354 | 102 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Qwen_Qwen3-0.6B-GGUF | 128,028 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Hcompany/Holo-3.1-35B-A3B-GGUF | 126,066 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/gemma-3-4b-it-GGUF | 123,745 | gemma3 | default | 3494 | 8 | 0 | - | special-token 1 | `▁▁` (special-token) |
| ProCreations/grug-27b-mtp-gguf | 123,680 | qwen35 | qwen35 | 3116 | 165 | 0 | 103/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Qwen_Qwen3.5-4B-GGUF | 122,834 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| axi0mX/Qwen3.8-Flash-Next-Uncensored-GGUF | 121,863 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mradermacher/Huihui-Qwen3.8-Flash-Next-abliterated-i1-GGUF | 120,007 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive | 119,744 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF | 118,991 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen3-30B-A3B-GGUF | 116,961 | qwen3moe | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| michaelw9999/Qwen3.6-35B-A3B-NVFP4-MTP-GGUF | 116,711 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Phi-4-mini-instruct-GGUF | 116,100 | phi3 | gpt-4o | 2850 | 54 | 0 | - | special-token 37, nfc 3, other 2, whitespace+text 1 | ` �` (whitespace+text); `��` (other); `ï¿½` (special-token) |
| prism-ml/Bonsai-1.7B-gguf | 113,954 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| michaelw9999/Qwopus3.6-27B-v2-MTP-NVFP4-GGUF | 113,700 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| michaelw9999/Qwen3.6-27B-NVFP4-MTP-GGUF | 112,537 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF | 111,576 | qwen2 | deepseek-r1-qwen | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF | 111,018 | qwen35moe | qwen35 | 3116 | 165 | 0 | 100/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Jackrong/Qwopus3.6-27B-Coder-Compat-MTP-GGUF | 110,024 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen3-32B-GGUF | 109,761 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| antirez/qwen3.8-flash-next-gguf | 108,061 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| sh0wie/Qwen3.8-Flash-Next-REAP-288-GGUF | 107,623 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Qwen_Qwen3-4B-GGUF | 105,951 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Nanbeige_Nanbeige4.3-33B-GGUF | 105,801 | llama | default | 2794 | 442 | 0 | - | whitespace+text 201, special-token 121, whitespace 27, combining-mark 27, apostrophe 21, digits 18, nfc 13 | ` ` (whitespace); `   !` (whitespace+text); `   <U+0301>` (combining-mark) |
| unsloth/Qwen3.5-4B-MTP-GGUF | 104,543 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| lmstudio-community/gpt-oss-20b-GGUF | 103,744 | gpt-oss | gpt-4o | 2948 | 1 | 0 | - | other 1 | `ÇA` (other) |
| SoAIHQ/Qwen3.5-4B-GGUF | 103,496 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| SupraLabs/Supra2-100M-Instruct | 101,461 | qwen3 | qwen2 | 3060 | 59 | 0 | - | digits 16, whitespace+text 9, special-token 9, combining-mark 6, crlf 6, apostrophe 4, nfc 3 | `  1` (digits); `  2` (digits); `<U+000A><U+000A>b` (whitespace+text) |
| prism-ml/Bonsai-8B-gguf | 100,445 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| openbmb/MiniCPM-o-4_5-gguf | 100,215 | qwen3 | qwen2 | 3494 | 102 | 0 | - | nfc 88 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ggml-org/Qwen2.5-Omni-3B-GGUF | 100,202 | qwen2vl | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ThorOdinson246/nl2sh-1.5b-Q4_K_M | 99,391 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Qwen3.8-Flash-Next-GGUF | 98,794 | qwen4exp | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/Qwen2.5-0.5B-Instruct-GGUF | 98,285 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| culturerevolt/gemma-4-12b-heretic-abliterated-GGUF | 98,204 | gemma4 | None | 2990 | 15 | 0 | - | special-token 1 | `<|video|>` (special-token) |
| lmstudio-community/Qwen3-4B-Instruct-2507-GGUF | 98,148 | qwen3 | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| PaddlePaddle/PaddleOCR-VL-1.6-GGUF | 98,111 | paddleocr | default | 3474 | 7 | 0 | - | whitespace 1 | `<U+FEFF>` (whitespace) |
| mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-i1-GGUF | 97,518 | qwen3moe | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-GGUF | 97,342 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| Qwen/Qwen2.5-Coder-3B-Instruct-GGUF | 97,322 | qwen2 | qwen2 | 2976 | 120 | 0 | - | nfc 90, control-text 14 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF | 97,185 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| IsValorum/Qwen3.8-35B-A3B-Distill-MTP-APEX-I-MiniPlus-V2.1-Abliterated-GGUF | 96,973 | qwen35moe | qwen35 | 3116 | 165 | 0 | - | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| bartowski/microsoft_Phi-4-mini-instruct-GGUF | 96,706 | phi3 | gpt-4o | 2822 | 40 | 0 | - | special-token 34, other 1 | `<|assistant|> ` (special-token); `<|assistant|><U+000A>` (special-token); `<|end|> ` (special-token) |
| unsloth/Qwen3.5-9B-MTP-GGUF | 96,245 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| mradermacher/Huihui-Qwen3.6-35B-A3B-abliterated-GGUF | 95,947 | qwen35moe | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Qwen3-VL-2B-Instruct-GGUF | 95,722 | qwen3vl | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| TheDrummer/Cydonia-24B-v4.3-GGUF | 95,465 | llama | tekken | 3494 | 58 | 0 | - | combining-mark 36, apostrophe 5, nfc 5, special-token 4, crlf 1 | `'Tx` (apostrophe); `'REx` (apostrophe); `'Vex` (apostrophe) |
| Kujira/Underdog-Saluki-27B-1.0-MTP-GGUF | 95,339 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF | 93,538 | qwen3vl | qwen2 | 3032 | 117 | 0 | 100/0 | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF | 92,897 | mistral3 | tekken | 3494 | 2692 | 0 | 24/0 | whitespace+text 625, other 518, special-token 472, digits 332, combining-mark 264, whitespace 193, nfc 103, crlf 89, apostrophe 74 | ` ` (whitespace); `  1` (digits); `  !` (whitespace+text) |
| unsloth/gemma-3-270m-it-GGUF | 90,597 | gemma3 | default | 3494 | 8 | 0 | - | special-token 1 | `▁▁` (special-token) |
| ajgazin/Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-GGUF | 90,409 | qwen35 | qwen35 | 3116 | 103 | 0 | - | nfc 89 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| rapatsky/Hy-Memory-30B-A3B-GGUF | 89,845 | hy_v3 | hunyuan-dense | 3494 | 33 | 0 | - | crlf 15, nfc 3, special-token 1 | `<U+000A><U+000D><U+000A>` (crlf); `<U+000D><U+000A><U+000D><U+000…` (crlf); `<U+000D><U+000A><U+000D><U+000…` (crlf) |
| lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF | 89,768 | qwen3moe | qwen2 | 3032 | 117 | 0 | - | nfc 88, control-text 15 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ggml-org/Clef-Flash-GGUF | 88,650 | clef | qwen35 | 3116 | 165 | 0 | 100/0 | nfc 89, combining-mark 54, special-token 5, crlf 3 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| ggml-org/Qwen3.6-27B-GGUF | 87,815 | qwen35 | qwen35 | 3018 | 100 | 0 | - | nfc 86 | `<U+2000>` (nfc); `<U+2000><U+2000>1` (nfc); `<U+2000><U+2000>!` (nfc) |
| unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF | 87,152 | llama | tekken | 3494 | 58 | 0 | - | combining-mark 36, apostrophe 5, nfc 5, special-token 4, crlf 1 | `'Tx` (apostrophe); `'REx` (apostrophe); `'Vex` (apostrophe) |

## By architecture and pre-tokenizer

| arch | pre | repos | downloads |
|---|---|---:|---:|
| qwen35 | qwen35 | 89 | 67,912,277 |
| qwen35moe | qwen35 | 39 | 26,687,766 |
| qwen4exp | qwen35 | 19 | 15,118,484 |
| qwen3 | qwen2 | 27 | 6,445,465 |
| qwen3vl | qwen2 | 9 | 4,390,709 |
| qwen2 | qwen2 | 13 | 4,112,556 |
| laguna | laguna | 3 | 1,760,255 |
| qwen3moe | qwen2 | 5 | 1,287,099 |
| qwen2vl | qwen2 | 5 | 1,254,223 |
| gemma4-assistant | None | 4 | 1,065,231 |
| hunyuan-dense | hunyuan | 1 | 1,057,596 |
| dflash | qwen35 | 2 | 1,004,235 |
| muse-glimmer | llama4 | 1 | 812,668 |
| qwen35moe | qwen2 | 1 | 797,020 |
| gpt-oss | gpt-4o | 2 | 683,206 |
| gpt2 | default | 1 | 558,147 |
| qwen35 | default | 1 | 543,884 |
| nomic-bert | None | 1 | 536,182 |
| eagle3 | gpt-4o | 2 | 525,145 |
| bert | None | 3 | 507,099 |
| bailingmoe3 | bailingmoe2 | 2 | 484,179 |
| llama | default | 2 | 376,261 |
| hunyuan-dense | hunyuan-dense | 1 | 325,655 |
| k2-horizon | k2-horizon | 1 | 300,054 |
| hy_v3 | hunyuan-dense | 2 | 254,698 |
| smollm3 | smaug-bpe | 1 | 250,621 |
| qwen2 | deepseek-r1-qwen | 2 | 248,092 |
| dflash | llama4 | 1 | 244,803 |
| cohere2moe | tiny_aya | 1 | 239,740 |
| gemma3 | default | 2 | 214,342 |
| phi3 | gpt-4o | 2 | 212,806 |
| bert | jina-v2-en | 1 | 207,876 |
| llama | llama-bpe | 1 | 197,414 |
| llama | tekken | 2 | 182,617 |
| qwen3next | qwen2 | 1 | 165,809 |
| dflash | laguna | 1 | 148,752 |
| gemma4 | None | 1 | 98,204 |
| paddleocr | default | 1 | 98,111 |
| mistral3 | tekken | 1 | 92,897 |
| clef | qwen35 | 1 | 88,650 |

## Start and stop tokens

Checked in 337 repos. **75** differ from their source model in the tokens added around a prompt or the tokens that end generation (22,714,450 downloads).

| repo | downloads | check | difference |
|---|---:|---|---|
| Qwen/Qwen3-VL-8B-Instruct-GGUF | 1,406,262 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| antirez/deepseek-v4.1-flash-gguf | 1,280,228 | wrap | tokens added at the start of a prompt: llama.cpp [0], reference [] |
| tencent/Hy-MT2-7B-GGUF | 1,057,596 | stop | '<\|extra_5\|>' (127967) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer |
| tencent/Hy-MT2-7B-GGUF | 1,057,596 | stop | llama.cpp stops on '$' (3), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| bartowski/Qwen2.5-32B-Instruct-GGUF | 868,558 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen3-8B-GGUF | 733,491 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen2.5-7B-Instruct-GGUF | 691,536 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| mradermacher/Qwen3-VL-8B-Instruct-abliterated-GGUF | 677,420 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF | 665,980 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen3-VL-4B-Instruct-GGUF | 638,421 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| mradermacher/AdaVaR-3B-GGUF | 618,484 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen3-4B-GGUF | 617,066 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/Qwen3-4B-Instruct-2507-GGUF | 586,058 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| togatogah/jinen-v1-small.gguf | 558,147 | wrap | tokens added at the end of a prompt: llama.cpp [], reference ['</s>' (3)] |
| unsloth/Qwen3-4B-GGUF | 531,614 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF | 511,578 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF | 471,274 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen2.5-1.5B-Instruct-GGUF | 426,120 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/Qwen3-VL-4B-Instruct-GGUF | 400,864 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen2.5-0.5B-Instruct-GGUF | 399,704 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen3-14B-GGUF | 353,321 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen2.5-Coder-7B-Instruct-GGUF | 325,382 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| openbmb/MiniCPM5-2B-GGUF | 320,315 | wrap | tokens added at the start of a prompt: llama.cpp [], reference ['<s>' (0)] |
| janhq/Jan-v3.5-4B-gguf | 319,723 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF | 281,538 | wrap | tokens added at the start of a prompt: llama.cpp [2], reference [] |
| llm-jp/llm-jp-4-33b-thinking-gguf | 270,460 | wrap | tokens added at the start of a prompt: llama.cpp [1], reference [] |
| prism-ml/Ternary-Bonsai-8B-gguf | 264,648 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen2.5-3B-Instruct-GGUF | 263,681 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| bartowski/Qwen2.5-7B-Instruct-GGUF | 251,218 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/Qwen3-1.7B-GGUF | 242,630 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/Qwen2.5-VL-7B-Instruct-GGUF | 240,086 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/flint-GGUF | 239,139 | stop | '<\|developer\|>' (150018) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer |
| Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF | 237,373 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| superwhisper/s1-mini-GGUF | 234,269 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen3-0.6B-GGUF | 222,170 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| AnkitAI/Parable-Qwen3-8B-Claude-Fable-5-GGUF | 220,251 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| AnkitAI/Parable-Qwen3-4B-Claude-Fable-5-GGUF | 220,063 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/Qwen3-VL-8B-Instruct-GGUF | 215,179 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen3-VL-2B-Instruct-GGUF | 197,323 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-GGUF | 196,725 | wrap | tokens added at the start of a prompt: llama.cpp [], reference ['<s>' (0)] |
| Qwen/Qwen2.5-Coder-14B-Instruct-GGUF | 193,090 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen3-Embedding-0.6B-GGUF | 167,454 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/Qwen3-Coder-Next-GGUF | 165,809 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| bartowski/Qwen2.5-14B-Instruct-GGUF | 160,896 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| ggml-org/Qwen2.5-Omni-7B-GGUF | 159,946 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF | 159,542 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen3-1.7B-GGUF | 157,743 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/Qwen3-0.6B-GGUF | 155,466 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Myric/Laguna-S-2.1-APEX-GGUF | 148,752 | stop | '</assistant>' (24) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer |
| unsloth/Qwen3-8B-GGUF | 143,705 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Abiray/Nemotron-3-Embed-8B-GGUF | 141,290 | wrap | tokens added at the start of a prompt: llama.cpp [1], reference [] |
| ggml-org/Qwen3-1.7B-GGUF | 139,297 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF | 136,516 | wrap | tokens added at the start of a prompt: llama.cpp [151646], reference [] |
| bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF | 136,516 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| ggml-org/Qwen2.5-VL-3B-Instruct-GGUF | 135,505 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| bartowski/Qwen_Qwen3-0.6B-GGUF | 128,028 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF | 118,991 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF | 118,615 | wrap | tokens added at the start of a prompt: llama.cpp [0], reference [] |
| Qwen/Qwen3-30B-A3B-GGUF | 116,961 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| lmstudio-community/flint-GGUF | 116,085 | stop | '<\|developer\|>' (150018) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer |
| prism-ml/Bonsai-1.7B-gguf | 113,954 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF | 111,576 | wrap | tokens added at the start of a prompt: llama.cpp [151646], reference [] |
| unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF | 111,576 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen3-32B-GGUF | 109,761 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| bartowski/Qwen_Qwen3-4B-GGUF | 105,951 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| prism-ml/Bonsai-8B-gguf | 100,445 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| openbmb/MiniCPM-o-4_5-gguf | 100,215 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| ggml-org/Qwen2.5-Omni-3B-GGUF | 100,202 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| ThorOdinson246/nl2sh-1.5b-Q4_K_M | 99,391 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| bartowski/Qwen2.5-0.5B-Instruct-GGUF | 98,285 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| lmstudio-community/Qwen3-4B-Instruct-2507-GGUF | 98,148 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-i1-GGUF | 97,518 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| Qwen/Qwen2.5-Coder-3B-Instruct-GGUF | 97,322 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/Qwen3-VL-2B-Instruct-GGUF | 95,722 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF | 93,538 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |
| unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF | 92,897 | wrap | tokens added at the start of a prompt: llama.cpp [1], reference [] |
| LiquidAI/LFM3-Omni-2B-GGUF | 90,171 | stop | '<\|text_end\|>' (125014) ends generation in the reference (generation_config.json eos_token_id) but not in llama.cpp; the model can run past the end of its answer |
| lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF | 89,768 | stop | llama.cpp stops on '</s>' (128247), which the reference does not stop on and treats as ordinary text: generation can stop mid-answer |

## Decoding the source tokenizer's token ids

Checked in 337 repos. llama.cpp decodes the source tokenizer's ids differently in **13** (4,375,482 downloads).

| repo | downloads | ids decoded differently | errors | example |
|---|---:|---:|---:|---|
| tencent/Hy-MT2-7B-GGUF | 1,057,596 | 27 | 0 | `'!'` → `' !'` |
| togatogah/jinen-v1-small.gguf | 558,147 | 370 | 0 | `' !'` → `'!'` |
| datalab-to/surya-ocr-2-gguf | 543,884 | 24 | 0 | `' !'` → `'!'` |
| nomic-ai/nomic-embed-text-v1.5-GGUF | 536,182 | 2046 | 0 | `'a b'` → `' a b'` |
| tencent/Hy-MT2-1.8B-GGUF | 325,655 | 27 | 0 | `'!'` → `' !'` |
| webAI-Official/TwIL-LM3 | 250,621 | 1 | 0 | `"n't"` → `" n't"` |
| ChristianAzinn/mxbai-embed-large-v1-gguf | 209,469 | 2046 | 0 | `'a b'` → `' a b'` |
| unsloth/bge-small-en-v1.5-GGUF | 207,876 | 2046 | 0 | `'a b'` → `' a b'` |
| unsloth/Llama-3.2-3B-Instruct-GGUF | 197,414 | 1 | 0 | `"n't"` → `" n't"` |
| second-state/All-MiniLM-L6-v2-Embedding-GGUF | 156,488 | 2056 | 0 | `'a b'` → `' a b'` |
| leliuga/all-MiniLM-L6-v2-GGUF | 141,142 | 2056 | 0 | `'a b'` → `' a b'` |
| PaddlePaddle/PaddleOCR-VL-1.6-GGUF | 98,111 | 7 | 0 | `'\ufeff'` → `'[EMPTY_96148]'` |
| unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF | 92,897 | 703 | 0 | `'xé'` → `'x�'` |

## Long inputs

Checked in 337 repos. llama.cpp fails or disagrees on long inputs in **9**.

| repo | downloads | failures |
|---|---:|---|
| datalab-to/surya-ocr-2-gguf | 543,884 | x5 encode, x6 encode, x7 encode, x9 encode |
| nomic-ai/nomic-embed-text-v1.5-GGUF | 536,182 | x0 encode, x1 encode, x3 encode, x6 encode, x7 decode |
| llm-jp/llm-jp-4-33b-thinking-gguf | 270,460 | x0 encode, x1 encode, x2 encode, x4 encode |
| ChristianAzinn/mxbai-embed-large-v1-gguf | 209,469 | x0 encode, x1 encode, x3 encode, x6 encode, x7 decode |
| unsloth/bge-small-en-v1.5-GGUF | 207,876 | x0 encode, x1 encode, x3 encode, x6 encode, x7 decode |
| second-state/All-MiniLM-L6-v2-Embedding-GGUF | 156,488 | x0 encode, x1 encode, x3 encode, x6 encode, x7 decode |
| leliuga/all-MiniLM-L6-v2-GGUF | 141,142 | x0 encode, x1 encode, x3 encode, x6 encode, x7 decode |
| bartowski/Nanbeige_Nanbeige4.3-33B-GGUF | 105,801 | x2 encode |
| unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF | 92,897 | x2 encode, x5 encode, x6 encode, x7 encode, x8 encode |

## llama.cpp could not load the vocabulary

| repo | downloads | arch | pre | error |
|---|---:|---|---|---|
| handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf | 2,261,012 | parakeet | None | llama.cpp could not load the vocabulary: exit 3: llama.cpp failed to load the model: llama_model_load: error loading model: unknown model architecture: 'parakee |
| handy-computer/parakeet-tdt-0.6b-v3-gguf | 953,691 | parakeet | None | llama.cpp could not load the vocabulary: exit 3: llama.cpp failed to load the model: llama_model_load: error loading model: unknown model architecture: 'parakee |
| handy-computer/whisper-medium-gguf | 735,389 | whisper | gpt2 | llama.cpp could not load the vocabulary: exit 3: llama.cpp failed to load the model: llama_model_load: error loading model: unknown model architecture: 'whisper |
| handy-computer/whisper-large-v3-turbo-gguf | 602,914 | whisper | gpt2 | llama.cpp could not load the vocabulary: exit 3: llama.cpp failed to load the model: llama_model_load: error loading model: unknown model architecture: 'whisper |
| unsloth/Inkling-Small-GGUF | 566,054 | inkling | inkling | llama.cpp could not load the vocabulary: exit 3: llama.cpp failed to load the model: llama_model_load: error loading model: unknown model architecture: 'inkling |
| handy-computer/whisper-large-v3-gguf | 456,963 | whisper | gpt2 | llama.cpp could not load the vocabulary: exit 3: llama.cpp failed to load the model: llama_model_load: error loading model: unknown model architecture: 'whisper |
| unsloth/inkling-GGUF | 291,813 | inkling | inkling | llama.cpp could not load the vocabulary: exit 3: llama.cpp failed to load the model: llama_model_load: error loading model: unknown model architecture: 'inkling |
| cstr/ark-asr-3b-GGUF | 197,766 | arkasr | None | llama.cpp could not load the vocabulary: exit 3: llama.cpp failed to load the model: llama_model_load: error loading model: unknown model architecture: 'arkasr' |
| handy-computer/canary-1b-v2-gguf | 108,529 | canary | None | llama.cpp could not load the vocabulary: exit 3: llama.cpp failed to load the model: llama_model_load: error loading model: unknown model architecture: 'canary' |
| handy-computer/parakeet-ultra-gguf | 94,053 | parakeet | None | llama.cpp could not load the vocabulary: exit 3: llama.cpp failed to load the model: llama_model_load: error loading model: unknown model architecture: 'parakee |

## Tokenizer drift: source repo vs its base model

| GGUF repo | downloads | source | base | differs in | probes changed |
|---|---:|---|---|---|---:|
| JonathanColetti/Qwen3.8-27B-Uncensored-GGUF | 2,598,773 | JonathanColetti/Qwen3.8-27B-Uncensored | Qwen/Qwen3.8-27B | pre_tokenizer | 5/10 |
| mradermacher/Ornith-1.5-9B-uncensored-GGUF | 1,713,226 | junafinity/Ornith-1.5-9B-uncensored | ornith-ai/Ornith-1.5-9B | pre_tokenizer | 5/10 |
| empero-ai/Qwen3.8-35B-A3B-Distill-GGUF | 1,268,674 | empero-ai/Qwen3.8-35B-A3B-Distill | Qwen/Qwen3.6-35B-A3B | pre_tokenizer | 5/10 |
| bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF | 974,177 | XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B | Qwen/Qwen3.5-9B | pre_tokenizer | 5/10 |
| mradermacher/Cyber-Ornith-1.5-9B-OBLITERATED-i1-GGUF | 935,184 | DuoNeural/Cyber-Ornith-1.5-9B-OBLITERATED | OBLITERATUS/Ornith-1.5-9B-OBLITERATED | pre_tokenizer | 5/10 |
| Accio-Lab/occamy-1.0-GGUF | 797,020 | Accio-Lab/occamy-1.0 | Qwen/Qwen3.6-35B-A3B | pre_tokenizer | 5/10 |
| ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF | 767,975 | ukisai/Swift-Qwen3.8-Flash-Next | Qwen/Qwen3.8-Flash-Next | pre_tokenizer | 5/10 |
| mradermacher/Simplicio-27B-GGUF | 729,914 | wesleysimplicio/Simplicio-27B | Qwen/Qwen3.8-27B | pre_tokenizer | 5/10 |
| philbert440/Qwen3.8-27B-Uncensored-Cyber-GGUF | 538,634 | philbert440/Qwen3.8-27B-Uncensored-Cyber | philbert440/Qwen3.8-27B-Uncensored-Aggressive | pre_tokenizer | 5/10 |
| DavidAU/LFM2.5-24B-A2B-Turbo-Brilliance-50X-NEO-MAX-Imatrix-GGUF | 470,282 | LiquidAI/LFM2.5-24B-A2B | LiquidAI/LFM2.5-24B-A2B-Base | pre_tokenizer | 0/10 |
| SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF | 449,426 | ukisai/Swift-Qwen3.8-Flash-Next | Qwen/Qwen3.8-Flash-Next | pre_tokenizer | 5/10 |
| DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF | 432,769 | DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-MTP | Qwen/Qwen3.6-27B | pre_tokenizer | 5/10 |
| jimmy31chen/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-IQ3_S-Strata | 424,213 | 0bserverx/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored | Qwen/Qwen3.8-Flash-Next | pre_tokenizer | 5/10 |
| LiquidAI/LFM2.5-2.6B-GGUF | 409,542 | LiquidAI/LFM2.5-2.6B | LiquidAI/LFM2.5-2.6B-Base | pre_tokenizer | 0/10 |
| DavidAU/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-MTP-GGUF | 337,984 | DavidAU/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1 | Qwen/Qwen3.8-27B | pre_tokenizer | 5/10 |
| AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF | 309,727 | LiquidAI/LFM2.5-24B-A2B | LiquidAI/LFM2.5-24B-A2B-Base | pre_tokenizer | 0/10 |
| abhishek085/spark-s1-4b-v6-GGUF | 283,123 | abhishek085/spark-s1-4b-v6 | Qwen/Qwen3.5-4B | pre_tokenizer | 5/10 |
| esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF | 271,349 | unsloth/Qwen3.8-27B-NVFP4 | Qwen/Qwen3.8-27B | pre_tokenizer | 5/10 |
| ibm-granite/granite-4.2-3b-GGUF | 269,920 | ibm-granite/granite-4.2-3b | ibm-granite/granite-4.1-3b-base | pre_tokenizer | 5/10 |
| ibm-granite/granite-4.2-8b-GGUF | 268,281 | ibm-granite/granite-4.2-8b | ibm-granite/granite-4.1-8b-base | pre_tokenizer | 5/10 |
| ProCreations/grug-27b-gguf | 265,912 | ProCreations/grug-27b | Qwen/Qwen3.6-27B | pre_tokenizer | 5/10 |
| mradermacher/Hemmingway-1-GGUF | 244,096 | HemmingwayAI/Hemmingway-1 | Qwen/Qwen3.8-27B | pre_tokenizer | 5/10 |
| ibm-granite/granite-4.2-30b-GGUF | 242,364 | ibm-granite/granite-4.2-30b | ibm-granite/granite-4.1-30b-base | pre_tokenizer | 5/10 |
| LiquidAI/LFM2.5-24B-A2B-GGUF | 229,042 | LiquidAI/LFM2.5-24B-A2B | LiquidAI/LFM2.5-24B-A2B-Base | pre_tokenizer | 0/10 |
| empero-ai/Qwythos-27B-v1-GGUF | 216,751 | empero-ai/Qwythos-27B-v1 | Qwen/Qwen3.5-27B | pre_tokenizer | 5/10 |
| unsloth/Qwen3-Coder-Next-GGUF | 165,809 | unsloth/Qwen3-Coder-Next | Qwen/Qwen3-Coder-Next | pre_tokenizer | 0/10 |
| ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF | 162,668 | ukisai/Swift-Qwen3.8-Flash-Next | Qwen/Qwen3.8-Flash-Next | pre_tokenizer | 5/10 |
| llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF | 111,018 | llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved | Qwen/Qwen3.6-35B-A3B | pre_tokenizer | 5/10 |
| IsValorum/Qwen3.8-35B-A3B-Distill-MTP-APEX-I-MiniPlus-V2.1-Abliterated-GGUF | 96,973 | empero-ai/Qwen3.8-35B-A3B-Distill | Qwen/Qwen3.6-35B-A3B | pre_tokenizer | 5/10 |
| pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF | 93,538 | pottokao/Qwen-Image-2.1-Text-Encoder-Heretic | Qwen/Qwen3-VL-8B-Instruct | pre_tokenizer | 0/10 |
| unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF | 92,897 | unsloth/Devstral-Small-2-24B-Instruct-2512 | mistralai/Devstral-Small-2-24B-Instruct-2512 | pre_tokenizer, clean_up_tokenization_spaces | 10/10 |
| ggml-org/Clef-Flash-GGUF | 88,650 | Cloudflare/clef-flash | Qwen/Qwen3.5-9B | pre_tokenizer | 5/10 |
