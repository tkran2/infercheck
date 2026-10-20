"""Offline tests for the GGUF header reader and the hub audit helpers (no network)."""

import struct
import tempfile
import unittest
from pathlib import Path

from infercheck.ggufhdr import ARR, STR, U32, GGUFFormatError, file_fetcher, read_header, write_vocab_only
from infercheck.hub import classify, hf_repo_from_url


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def synthetic_gguf(n_tensors: int = 1) -> bytes:
    kv = b""
    kv += _s("general.architecture") + struct.pack("<I", STR) + _s("llama")
    kv += _s("tokenizer.ggml.model") + struct.pack("<I", STR) + _s("gpt2")
    kv += _s("tokenizer.ggml.pre") + struct.pack("<I", STR) + _s("pixtral")
    kv += _s("tokenizer.ggml.tokens") + struct.pack("<I", ARR) + struct.pack("<IQ", STR, 3) + _s("a") + _s("b") + _s("दी")
    kv += _s("tokenizer.ggml.token_type") + struct.pack("<I", ARR) + struct.pack("<IQ", U32, 3) + struct.pack("<3I", 1, 1, 1)
    head = b"GGUF" + struct.pack("<IQQ", 3, n_tensors, 5)
    tensor_info = _s("w") + struct.pack("<I", 1) + struct.pack("<Q", 4) + struct.pack("<I", 0) + struct.pack("<Q", 0)
    return head + kv + tensor_info + b"\0" * 64


class TestHeader(unittest.TestCase):
    def test_read_and_rewrite(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "m.gguf"
            src.write_bytes(synthetic_gguf())
            h = read_header(file_fetcher(src), chunk=7)  # tiny chunks exercise the on-demand reader
            self.assertEqual(h.n_tensors, 1)
            self.assertEqual((h.arch, h.tokenizer_model, h.pre), ("llama", "gpt2", "pixtral"))
            self.assertEqual(h.get("tokenizer.ggml.tokens"), ["a", "b", "दी"])
            self.assertEqual(h.get("tokenizer.ggml.token_type"), [1, 1, 1])
            out = write_vocab_only(h, Path(td) / "v.gguf")
            h2 = read_header(file_fetcher(out))
            self.assertEqual(h2.n_tensors, 0)
            self.assertEqual(h2.kv, h.kv)
            self.assertEqual(out.stat().st_size % 32, 0)

    def test_not_gguf(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.bin"
            p.write_bytes(b"NOPE" + b"\0" * 32)
            with self.assertRaises(GGUFFormatError):
                read_header(file_fetcher(p))

    def test_truncated(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.gguf"
            p.write_bytes(synthetic_gguf()[:60])
            with self.assertRaises(GGUFFormatError):
                read_header(file_fetcher(p))


class TestHelpers(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify("दी", []), "combining-mark")
        self.assertEqual(classify("é", []), "nfc")
        self.assertEqual(classify("\r\n", []), "crlf")
        self.assertEqual(classify("  1", []), "digits")
        self.assertEqual(classify("<|end|> ", ["<|end|>"]), "special-token")
        self.assertEqual(classify("'Tx", []), "apostrophe")

    def test_repo_from_url(self):
        self.assertEqual(hf_repo_from_url("https://huggingface.co/Qwen/Qwen3-8B"), "Qwen/Qwen3-8B")
        self.assertIsNone(hf_repo_from_url("https://github.com/x/y"))
        self.assertIsNone(hf_repo_from_url(None))


if __name__ == "__main__":
    unittest.main()
