"""Harness checks that need no Hugging Face install: corpus, shrinking, both engine harnesses."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infercheck.corpus import fixed_cases, generated_cases, special_cases
from infercheck.diff import ddmin_chars, kind_of
from infercheck.engines import LlamaCppTokenizer, OllamaTokenizer, Result, source_dir

LLAMA_VOCABS = source_dir("llama.cpp") / "models"


def mini_tokenizer(path: Path) -> None:
    """A byte-level BPE tokenizer.json small enough to write inline."""
    chars = [chr(c) for c in range(0x21, 0x7f)] + ["Ġ"]
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["Ġt"] = len(vocab)
    vocab["he"] = len(vocab)
    vocab["Ġthe"] = len(vocab)
    tj = {"version": "1.0", "added_tokens": [{"id": 500, "content": "<|x|>", "special": True, "normalized": False,
                                              "lstrip": False, "rstrip": False, "single_word": False}],
          "normalizer": None,
          "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
          "decoder": {"type": "ByteLevel"},
          "model": {"type": "BPE", "vocab": dict(vocab, **{"<|x|>": 500}), "merges": ["Ġ t", "h e", "Ġt he"]}}
    path.write_text(json.dumps(tj))


class Corpus(unittest.TestCase):
    def test_deterministic_and_unique_ids(self):
        a, b = generated_cases(200, seed=3), generated_cases(200, seed=3)
        self.assertEqual([c.text for c in a], [c.text for c in b])
        self.assertNotEqual([c.text for c in a], [c.text for c in generated_cases(200, seed=4)])
        cases = fixed_cases() + special_cases(["<|x|>"]) + a
        self.assertEqual(len({c.id for c in cases}), len(cases))
        self.assertGreater(len(fixed_cases()), 1000)


class Shrinking(unittest.TestCase):
    def test_ddmin_finds_the_pair(self):
        text = "abcXdefYghi"
        calls = []

        def fails(s):
            calls.append(s)
            return "X" in s and "Y" in s

        self.assertEqual(ddmin_chars(text, fails), "XY")

    def test_ddmin_single_char(self):
        self.assertEqual(ddmin_chars("hellóworld", lambda s: "́" in s), "́")

    def test_kind_of(self):
        ok = Result([1, 2], "ab")
        self.assertIsNone(kind_of(ok, Result([1, 2], "ab")))
        self.assertEqual(kind_of(ok, Result([1, 3], "ab")), "encode")
        self.assertEqual(kind_of(ok, Result([1, 2], "a b")), "decode")
        self.assertEqual(kind_of(ok, Result(None, None, "panic")), "error")
        self.assertIsNone(kind_of(Result(None, None, "ref failed"), Result([1], "a")))


class Cli(unittest.TestCase):
    def test_find_gguf_prefers_converted_file(self):
        from infercheck.cli import find_gguf

        with tempfile.TemporaryDirectory() as d:
            data, conv, models = Path(d) / "data", Path(d) / "conv", Path(d) / "models"
            for p in (data / "ollama-gguf", conv, models):
                p.mkdir(parents=True)
            (models / "ggml-vocab-qwen2.gguf").write_bytes(b"GGUF")
            self.assertEqual(find_gguf(data, "qwen1.5", [conv, models]), [models / "ggml-vocab-qwen2.gguf"])
            (conv / "qwen1.5.gguf").write_bytes(b"GGUF")
            self.assertEqual(find_gguf(data, "qwen1.5", [conv, models]), [conv / "qwen1.5.gguf"])
            self.assertEqual(find_gguf(data, "no-such", [conv, models]), [])

    def test_compare_runs(self):
        from infercheck.cli import compare_runs

        def run(path, enc, dec):
            path.mkdir()
            (path / "summary.json").write_text(json.dumps({"tokenizers": {
                "t1": {"cases": 10, "engines": {"e": {"encode": enc, "decode": dec, "error": 0}}},
                "t2": {"cases": 10, "engines": {"e": {"encode": 0, "decode": 0, "error": 0}}}}}))

        with tempfile.TemporaryDirectory() as d:
            run(Path(d) / "a", 5, 2)
            run(Path(d) / "b", 0, 2)
            table = compare_runs(Path(d) / "a", Path(d) / "b")
            self.assertIn("| t1 | e | 10 | 5 | 0 | 2 | 2 |", table)
            self.assertNotIn("| t2 |", table)
            self.assertIn("**5**", table)


@unittest.skipUnless(OllamaTokenizer.BIN.exists(), "build harness/ollama_tok first")
class OllamaHarness(unittest.TestCase):
    def test_encode_decode_and_special(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tokenizer.json"
            mini_tokenizer(p)
            o = OllamaTokenizer(str(p))
            r = o.encode_many(["the", " the<|x|>!", "\u0085line"])
            self.assertEqual(r[0].decoded, "the")
            self.assertEqual(r[1].ids[-2:], [500, 0])
            self.assertEqual(r[1].decoded, " the<|x|>!")
            self.assertEqual(len(r), 3)  # NEL inside a decoded string must not split the output
            self.assertEqual(o.decode_ids([[500], [0]]), ["<|x|>", "!"])


@unittest.skipUnless(LlamaCppTokenizer.BIN.exists() and (LLAMA_VOCABS / "ggml-vocab-qwen2.gguf").exists(),
                     "build harness/llama_tok and clone llama.cpp first")
class LlamaHarness(unittest.TestCase):
    def test_matches_llama_cpp_reference_files(self):
        """llama.cpp ships Hugging Face outputs for a few dozen strings per vocabulary."""
        for name in ("qwen2", "gpt-2", "llama-spm", "phi-3"):
            gguf = LLAMA_VOCABS / f"ggml-vocab-{name}.gguf"
            texts = (LLAMA_VOCABS / f"ggml-vocab-{name}.gguf.inp").read_text(encoding="utf-8").split("\n__ggml_vocab_test__\n")
            outs = (LLAMA_VOCABS / f"ggml-vocab-{name}.gguf.out").read_text(encoding="utf-8").split("\n")
            texts = texts[:len(outs)]
            got = LlamaCppTokenizer(str(gguf)).encode_many(texts)
            for t, o, g in zip(texts, outs, got):
                self.assertEqual(g.ids, [int(x) for x in o.split()], f"{name}: {t!r}")

    def test_decode_mode(self):
        l = LlamaCppTokenizer(str(LLAMA_VOCABS / "ggml-vocab-qwen2.gguf"))
        ids = l.encode_many(["Hello world"])[0].ids
        self.assertEqual(l.decode_ids([ids, [10 ** 9]]), ["Hello world", None])


if __name__ == "__main__":
    unittest.main()
