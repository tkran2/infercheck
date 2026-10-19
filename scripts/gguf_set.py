"""Copy a GGUF file's metadata (no tensors) into a new vocabulary-only file, setting some keys.

Usage: python scripts/gguf_set.py SRC DST key=bool:true [key=str:value ...]

Used to test how llama.cpp behaves when a converter writes a key that an older file lacks,
for example tokenizer.ggml.normalizer.nfc, which gguf-py writes since llama.cpp #30540.
"""

from __future__ import annotations

import sys


def parse(spec: str):
    from gguf import GGUFValueType

    key, _, rest = spec.partition("=")
    kind, _, value = rest.partition(":")
    if kind == "bool":
        return key, value.lower() in ("1", "true", "yes"), GGUFValueType.BOOL
    if kind == "str":
        return key, value, GGUFValueType.STRING
    if kind == "u32":
        return key, int(value), GGUFValueType.UINT32
    raise SystemExit(f"unknown type in {spec!r}: use bool, str or u32")


def main() -> int:
    from gguf import GGUFReader, GGUFValueType, GGUFWriter

    src, dst, *specs = sys.argv[1:]
    override = {k: (v, t) for k, v, t in map(parse, specs)}
    r = GGUFReader(src)
    w = GGUFWriter(dst, r.fields["general.architecture"].contents())
    for key, field in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture" or key in override:
            continue
        vtype = GGUFValueType(field.types[0])
        sub = GGUFValueType(field.types[-1]) if vtype == GGUFValueType.ARRAY else None
        w.add_key_value(key, field.contents(), vtype, sub_type=sub)
    for key, (value, vtype) in override.items():
        w.add_key_value(key, value, vtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{dst}: {len(r.fields)} keys copied, {len(override)} set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
