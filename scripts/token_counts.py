"""Token counts of a few ordinary sentences, Hugging Face vs llama.cpp.

Usage: python scripts/token_counts.py TOKENIZER_JSON GGUF [LIB_DIR ...]

Each LIB_DIR is a folder holding another libllama build (it is put on LD_LIBRARY_PATH /
DYLD_LIBRARY_PATH), so one call can compare llama.cpp before and after a change.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infercheck.engines import LlamaCppTokenizer  # noqa: E402

SENTENCES = {
    "Hindi": "भारत एक विशाल देश है और यहाँ कई भाषाएँ बोली जाती हैं।",
    "Bengali": "আমি বাংলায় কথা বলতে ভালোবাসি।",
    "Tamil": "தமிழ் ஒரு பழமையான மொழி ஆகும்.",
    "Telugu": "తెలుగు భాష చాలా అందమైనది",
    "Thai": "ภาษาไทยเป็นภาษาที่มีวรรณยุกต์",
    "Khmer": "ភាសាខ្មែរគឺជាភាសាផ្លូវការ",
    "Arabic (vowelled)": "اَلْعَرَبِيَّةُ لُغَةٌ جَمِيلَةٌ",
    "Hebrew (niqqud)": "שָׁלוֹם עֲלֵיכֶם",
    "Vietnamese": "Tiếng Việt có nhiều dấu thanh.",
    "English": "The quick brown fox jumps over the lazy dog.",
}


def main() -> int:
    import tokenizers

    tj, gguf, *libs = sys.argv[1:]
    ref = tokenizers.Tokenizer.from_file(tj)
    texts = list(SENTENCES.values())
    cols = {"hf": [len(ref.encode(t, add_special_tokens=False).ids) for t in texts]}
    for label, lib in [("llama.cpp", None)] + [(f"llama.cpp[{Path(d).name}]", d) for d in libs]:
        saved = {k: os.environ.get(k) for k in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")}
        if lib:
            os.environ["LD_LIBRARY_PATH"] = os.environ["DYLD_LIBRARY_PATH"] = lib
        try:
            cols[label] = [len(r.ids or []) for r in LlamaCppTokenizer(gguf).encode_many(texts)]
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    print("| text | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for i, name in enumerate(SENTENCES):
        print(f"| {name} | " + " | ".join(str(c[i]) for c in cols.values()) + " |")
    print(json.dumps({name: {k: c[i] for k, c in cols.items()} for i, name in enumerate(SENTENCES)}, ensure_ascii=False),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
