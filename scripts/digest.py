"""Shrink a `infercheck tok` results folder into a small, publishable digest.

Keeps summary.json and, per tokenizer, up to N shrunk findings with only the minimal string and
the ids/text each side produced for it.

Usage: python scripts/digest.py RESULTS_DIR OUT_DIR [N]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "summary.json", out / "summary.json")
    with open(out / "findings.jsonl", "w", encoding="utf-8") as fh:
        for path in sorted(src.glob("*.findings.jsonl")):
            kept = 0
            for line in open(path, encoding="utf-8"):
                f = json.loads(line)
                if f.get("min_ref_ids") is None and f.get("min_ref_decoded") is None:
                    continue  # not shrunk
                row = {"tokenizer": path.name.removesuffix(".findings.jsonl"), "engine": f["engine"], "kind": f["kind"],
                       "minimal": f["minimal"], "codepoints": " ".join(f"U+{ord(c):04X}" for c in f["minimal"]),
                       "hf_ids": f["min_ref_ids"], "engine_ids": f["min_got_ids"],
                       "hf_text": f["min_ref_decoded"], "engine_text": f["min_got_decoded"], "engine_error": f["min_got_error"]}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                if kept >= n:
                    break
    print(f"{out}: summary.json, findings.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
