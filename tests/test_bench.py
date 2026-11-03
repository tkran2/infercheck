"""Offline tests for the bug benchmark and the start/stop-token check (no network, no llama.cpp)."""

import json
import os
import stat
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from infercheck import bench, hub
from infercheck.engines import LlamaCppTokenizer, Result
from infercheck.special import compare_special, score, split_wrap

try:
    import tokenizers  # noqa: F401
    HAVE_TOKENIZERS = True
except ImportError:
    HAVE_TOKENIZERS = False

SHA_A, SHA_B = "a" * 40, "b" * 40


class TestCatalog(unittest.TestCase):
    def test_shipped_catalog_is_valid(self):
        cases = bench.load_cases()
        self.assertEqual(bench.validate(cases), [])
        self.assertGreaterEqual(len(cases), 15)
        kinds = {c["kind"] for c in cases}
        self.assertEqual(kinds, {"engine", "file", "control"})

    def test_validate_catches_mistakes(self):
        bad = [{"id": "x", "title": "t", "kind": "engine", "category": "c", "detector": "encode", "sources": [],
                "model": {"gguf": ["a/b"]}, "engine": {"broken": "abc", "fixed": SHA_B}, "trigger": [3]},
               {"id": "x", "title": "t", "kind": "file", "category": "c", "detector": "nope", "sources": [],
                "model": {"gguf": []}, "file": {"broken_before": "April 1"}}]
        errs = "\n".join(bench.validate(bad))
        for needle in ("engine.broken must be", "a trigger is", "duplicate id", "detector must be", "model.gguf is empty",
                       "must be YYYY-MM-DD"):
            self.assertIn(needle, errs)

    def test_side_specs(self):
        eng = {"kind": "engine", "model": {"gguf": ["a/b"]}, "engine": {"broken": SHA_A, "fixed": SHA_B}}
        s = bench.side_specs(eng)
        self.assertEqual((s["broken"]["engine"], s["fixed"]["engine"], s["fixed"]["revision"]), (SHA_A, SHA_B, "main"))
        f = {"kind": "file", "model": {"gguf": ["a/b"]}, "engine": {"at": SHA_A}, "file": {"broken_before": "2026-04-07"}}
        s = bench.side_specs(f)
        self.assertEqual(s["broken"], {"engine": SHA_A, "gguf": ["a/b"], "revision": {"before": "2026-04-07"}})
        self.assertEqual(s["fixed"]["revision"], "main")
        only_broken = {"kind": "file", "model": {"gguf": ["a/b"]}, "file": {"broken_repo": "q/v1"}}
        s = bench.side_specs(only_broken)
        self.assertEqual(list(s), ["broken"])
        self.assertEqual((s["broken"]["engine"], s["broken"]["gguf"]), ("current", ["q/v1"]))
        ctl = {"kind": "control", "model": {"gguf": ["a/b"]}}
        self.assertEqual(list(bench.side_specs(ctl)), ["only"])

    def test_expand_triggers(self):
        self.assertEqual(bench.expand_triggers(["x", {"repeat": "ab", "times": 3}]), ["x", "ababab"])


class TestRevisions(unittest.TestCase):
    commits = [(datetime(2026, 4, 11, 9, tzinfo=timezone.utc), "c3" * 20),
               (datetime(2026, 4, 8, 12, tzinfo=timezone.utc), "c2" * 20),
               (datetime(2026, 4, 3, 18, tzinfo=timezone.utc), "c1" * 20)]

    def test_pick(self):
        p = bench.pick_revision
        self.assertEqual(p(self.commits, "main"), "c3" * 20)
        self.assertEqual(p(self.commits, {"before": "2026-04-07"}), "c1" * 20)
        self.assertEqual(p(self.commits, {"before": "2026-04-08"}), "c1" * 20)   # same-day uploads are not "before"
        self.assertEqual(p(self.commits, {"after": "2026-04-08"}), "c2" * 20)
        self.assertIsNone(p(self.commits, {"before": "2026-04-01"}))
        self.assertEqual(p(self.commits, "f" * 40), "f" * 40)
        naive = [(datetime(2026, 4, 3), "n1" * 20)]
        self.assertEqual(p(naive, {"before": "2026-04-04"}), "n1" * 20)


def side(**kw):
    base = {"status": "ok", "engine": {"cases": 100, "encode": 0, "decode": 0, "error": 0, "causes": {}},
            "refdecode": {"compared": 100, "mismatch": 0, "error": 0},
            "stress": {"cases": 10, "mismatch": 0, "error": 0, "failures": []},
            "special": {"issues": []}, "repo": "a/b", "revision": "r1", "engine_commit": "current"}
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


class TestScoring(unittest.TestCase):
    eng_case = {"id": "e", "kind": "engine", "detector": "refdecode", "category": "detokenizer",
                "model": {"gguf": ["a/b"]}, "engine": {"broken": SHA_A, "fixed": SHA_B}}

    def test_caught_by_differential(self):
        b = side(engine_commit=SHA_A, refdecode={"mismatch": 0, "error": 100, "crash": True}, engine={"encode": 7})
        f = side(engine_commit=SHA_B, engine={"encode": 7})   # the shared 7 encode findings cancel
        s = bench.score_case(self.eng_case, {"broken": b, "fixed": f})
        self.assertEqual(s["outcome"], "caught")
        self.assertEqual(s["fired"], ["refdecode"])
        self.assertEqual(s["metrics"]["encode"], [7, 7, 0, 0])

    def test_missed_and_inconclusive(self):
        b, f = side(engine_commit=SHA_A), side(engine_commit=SHA_B)
        self.assertEqual(bench.score_case(self.eng_case, {"broken": b, "fixed": f})["outcome"], "missed")
        s = bench.score_case(self.eng_case, {"broken": side(status="build-error", error="boom"), "fixed": f})
        self.assertEqual(s["outcome"], "inconclusive")
        self.assertIn("build-error", s["notes"][0])
        self.assertEqual(bench.score_case(self.eng_case, {"broken": b})["outcome"], "inconclusive")  # fixed not run

    def test_same_file_is_inconclusive(self):
        fc = {"id": "f", "kind": "file", "detector": "wrap", "category": "wrap", "model": {"gguf": ["a/b"]},
              "file": {"broken_before": "2026-04-07"}}
        s = bench.score_case(fc, {"broken": side(), "fixed": side()})
        self.assertEqual(s["outcome"], "inconclusive")

    def test_wrap_and_stop_scores(self):
        bos = {"check": "wrap", "severity": "high", "kind": "start-missing", "detail": ""}
        info = {"check": "stop", "severity": "info", "kind": "extra", "detail": ""}
        fc = {"id": "f", "kind": "file", "detector": "wrap", "category": "wrap", "model": {"gguf": ["a/b"]},
              "file": {"broken_before": "2026-04-07"}}
        s = bench.score_case(fc, {"broken": side(revision="r1", special={"issues": [bos, info]}),
                                  "fixed": side(revision="r2", special={"issues": [info]})})
        self.assertEqual((s["outcome"], s["fired"]), ("caught", ["wrap"]))

    def test_single_side_uses_causes(self):
        fc = {"id": "l3", "kind": "file", "detector": "encode", "category": "tokenizer", "causes": ["digits"],
              "model": {"gguf": ["q/v1"]}, "file": {"broken_repo": "q/v1"}}
        unrelated = side(engine={"encode": 5, "causes": {"nfc": 5}})
        self.assertEqual(bench.score_case(fc, {"broken": unrelated})["outcome"], "missed")
        related = side(engine={"encode": 5, "causes": {"nfc": 2, "digits": 3}})
        s = bench.score_case(fc, {"broken": related})
        self.assertEqual((s["outcome"], s["metrics"]["encode"]), ("caught", [3, None]))

    def test_trigger_and_not_covered(self):
        nc = dict(self.eng_case, detector="none", trigger=["x"])
        b = side(engine_commit=SHA_A, trigger={"total": 0})
        f = side(engine_commit=SHA_B, trigger={"total": 0})
        s = bench.score_case(nc, {"broken": b, "fixed": f})
        self.assertEqual(s["outcome"], "not-covered")
        self.assertFalse(s["trigger"]["reproduces"])
        self.assertTrue(any("does not reproduce" in n for n in s["notes"]))

    def test_controls(self):
        ctl = {"id": "c", "kind": "control", "category": "control", "model": {"gguf": ["a/b"]}}
        self.assertEqual(bench.score_case(ctl, {"only": side()})["outcome"], "clean")
        info_only = side(special={"issues": [{"check": "stop", "severity": "info", "kind": "extra", "detail": ""}]})
        self.assertEqual(bench.score_case(ctl, {"only": info_only})["outcome"], "clean")
        s = bench.score_case(ctl, {"only": side(engine={"encode": 2})})
        self.assertEqual((s["outcome"], s["fired"]), ("flagged", ["encode"]))


def _tiny_gguf(arch="llama") -> bytes:
    import struct
    from infercheck.ggufhdr import STR

    def s(t):
        b = t.encode()
        return struct.pack("<Q", len(b)) + b
    kv = s("tokenizer.ggml.model") + struct.pack("<I", STR) + s("gpt2")
    if arch:
        kv = s("general.architecture") + struct.pack("<I", STR) + s(arch) + kv
    return b"GGUF" + struct.pack("<IQQ", 3, 0, 2 if arch else 1) + kv + b"\0" * 32


class HubStandIns(unittest.TestCase):
    """The Hub and llama.cpp replaced by local stand-ins, for run_side and run."""

    def setUp(self):
        from infercheck.ggufhdr import file_fetcher
        self.td = tempfile.TemporaryDirectory()
        d = Path(self.td.name)
        (d / "m.gguf").write_bytes(_tiny_gguf())
        (d / "tokenizer.json").write_text("{}")
        self.calls = []
        self.stats = {"cases": 100, "encode": 2, "decode": 0, "error": 0, "causes": {"digits": 2},
                      "refdecode": {"compared": 10, "mismatch": 0, "error": 0}, "stress": {"cases": 10, "mismatch": 0,
                      "error": 0, "failures": []}, "special": {"issues": []}}
        self.saved = {k: getattr(hub, k) for k in ("model_info", "pick_gguf", "range_fetcher", "fetch_tokenizer",
                                                   "run_engine_check")}
        self.saved_commits = bench._commits
        hub.model_info = lambda repo: {"id": repo} if repo != "gone/repo" else (_ for _ in ()).throw(RuntimeError("404"))
        hub.pick_gguf = lambda repo, rev="main": "m.gguf"
        hub.range_fetcher = lambda repo, path, rev=None: (self.calls.append((repo, rev)), file_fetcher(d / "m.gguf"))[1]
        hub.fetch_tokenizer = lambda repo, cache, revision=None: d / "tokenizer.json"
        hub.run_engine_check = lambda vg, tj, n, seed, ms, bin_path=None, extra=True, keys=False: (
            json.loads(json.dumps(self.stats)), [])
        bench._commits = lambda repo: [(datetime(2026, 4, 11, tzinfo=timezone.utc), "c" * 40),
                                       (datetime(2026, 4, 3, tzinfo=timezone.utc), "a" * 40)]

        class B:
            def harness(self, engine):
                return Path("/bin/true")
        self.builds = B()

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(hub, k, v)
        bench._commits = self.saved_commits
        self.td.cleanup()


class TestRunSide(HubStandIns):
    def test_file_case_pins_revisions(self):
        case = {"id": "f", "kind": "file", "detector": "encode", "category": "t",
                "model": {"gguf": ["gone/repo", "ok/repo"], "reference": ["ref/repo"]}, "file": {"broken_before": "2026-04-07"}}
        resolved = {}
        recs = {s: bench.run_side(case, s, spec, resolved, self.builds, Path(self.td.name), 1, 0, 0)
                for s, spec in bench.side_specs(case).items()}
        self.assertEqual((recs["broken"]["status"], recs["fixed"]["status"]), ("ok", "ok"))
        self.assertEqual((recs["broken"]["repo"], recs["broken"]["revision"], recs["fixed"]["revision"]),
                         ("ok/repo", "a" * 40, "c" * 40))
        self.assertEqual(recs["broken"]["reference"], "ref/repo")
        self.assertEqual(recs["broken"]["refdecode"]["compared"], 10)
        self.assertNotIn("refdecode", recs["broken"]["engine"])
        self.assertIn("gone/repo", resolved["f/broken"]["notes"][0])

    def test_engine_case_fixed_side_reuses_the_file(self):
        case = {"id": "e", "kind": "engine", "detector": "encode", "category": "t",
                "model": {"gguf": ["ok/repo"], "reference": ["ref/repo"]},
                "engine": {"broken": SHA_A, "fixed": SHA_B}}
        resolved = {"e/broken": {"repo": "ok/repo", "revision": "a" * 40, "file": "m.gguf"}}
        rec = bench.run_side(case, "fixed", bench.side_specs(case)["fixed"], resolved, self.builds, Path(self.td.name), 1, 0, 0)
        self.assertEqual((rec["status"], rec["revision"]), ("ok", "a" * 40))

    def test_load_failure_vs_crash(self):
        case = {"id": "e", "kind": "engine", "detector": "encode", "category": "t",
                "model": {"gguf": ["ok/repo"], "reference": ["ref/repo"]},
                "engine": {"broken": SHA_A, "fixed": SHA_B}}
        spec = bench.side_specs(case)["broken"]
        self.stats.update(error=100, error_sample="exit 3: llama.cpp failed to load the model: unknown pre-tokenizer")
        rec = bench.run_side(case, "broken", spec, {}, self.builds, Path(self.td.name), 1, 0, 0)
        self.assertEqual(rec["status"], "load-error")
        self.stats.update(error=100, error_sample="exit -6: GGML_ASSERT failed")
        rec = bench.run_side(case, "broken", spec, {}, self.builds, Path(self.td.name), 1, 0, 0)
        self.assertEqual(rec["status"], "ok")
        self.assertIn("GGML_ASSERT", rec["crashed"])
        self.assertEqual(bench.metric(rec, "encode"), 102)

    def test_no_revision_before_date(self):
        case = {"id": "f", "kind": "file", "detector": "encode", "category": "t", "model": {"gguf": ["ok/repo"]},
                "file": {"broken_before": "2026-01-01"}}
        rec = bench.run_side(case, "broken", bench.side_specs(case)["broken"], {}, self.builds, Path(self.td.name), 1, 0, 0)
        self.assertEqual(rec["status"], "no-revision")


class TestReport(unittest.TestCase):
    def test_report(self):
        cases = {"version": 1, "cases": [
            {"id": "e1", "title": "decode crash", "date": "2026-04-06", "kind": "engine", "category": "detokenizer",
             "detector": "refdecode", "model": {"gguf": ["a/b"]}, "engine": {"broken": SHA_A, "fixed": SHA_B},
             "sources": ["https://example.com/1"]},
            {"id": "e2", "title": "missed one", "date": "2026-04-07", "kind": "engine", "category": "tokenizer",
             "detector": "encode", "model": {"gguf": ["a/b"]}, "engine": {"broken": SHA_A, "fixed": SHA_B},
             "sources": ["https://example.com/2"], "found_by_infercheck": True},
            {"id": "c1", "title": "control", "kind": "control", "category": "control", "model": {"gguf": ["a/b"]},
             "sources": ["https://example.com/3"]},
            {"id": "c2", "title": "flagged control", "kind": "control", "category": "control", "model": {"gguf": ["a/b"]},
             "sources": ["https://example.com/4"]}]}
        recs = [{"case": "e1", "side": "broken", **side(engine_commit=SHA_A, refdecode={"error": 50, "crash": True})},
                {"case": "e1", "side": "fixed", **side(engine_commit=SHA_B)},
                {"case": "e2", "side": "broken", **side(engine_commit=SHA_A)},
                {"case": "e2", "side": "fixed", **side(engine_commit=SHA_B)},
                {"case": "c1", "side": "only", **side()},
                {"case": "c2", "side": "only", **side(engine={"encode": 2})}]
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "cases.json"
            cp.write_text(json.dumps(cases))
            (Path(td) / "sides.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
            text = bench.report(Path(td), cp)
        self.assertIn("**InferCheck caught 1 of 2 replayed bugs**", text)
        self.assertIn("| e1 | 2026-04-06 | engine | detokenizer | caught | refdecode 50→0 | - |", text)
        self.assertIn("| e2 | 2026-04-07 | engine | tokenizer | missed | - | - |", text)
        self.assertIn("controls (no known bug): 1 clean, 1 flagged", text)
        self.assertIn("| c2 |  | control | control | flagged | encode 2 | - |", text)   # one side: no "→"
        self.assertIn("1 of the 1 were caught only by the refdecode", text)
        self.assertIn("1 of the 2 were first found by InferCheck itself; on the 1 bugs found by others it caught 1.", text)

    def test_run_refuses_bad_catalog(self):
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "cases.json"
            cp.write_text(json.dumps({"cases": [{"id": "x"}]}))
            self.assertEqual(bench.run(cp, Path(td) / "out", None, None, 1, 0, 0), 2)


class TestSpecial(unittest.TestCase):
    def ref(self, **kw):
        r = {"plain": [10, 11], "full": [1, 10, 11], "prefix": [1], "suffix": [], "config_add_bos": None,
             "stops": {2: "generation_config.json eos_token_id", 7: "generation_config.json eos_token_id"},
             "special_ids": [1, 2, 7, 9], "text": {1: "<s>", 2: "</s>", 7: "<|im_end|>"}}
        r.update(kw)
        return r

    def eng(self, **kw):
        e = {"probe_plain": [10, 11], "probe_special": [1, 10, 11],
             "eog": [{"id": 2, "text": "</s>", "attr": 8}, {"id": 7, "text": "<|im_end|>", "attr": 8}]}
        e.update(kw)
        return e

    def test_split_wrap(self):
        self.assertEqual(split_wrap([1, 10, 11, 2], [10, 11]), ([1], [2]))
        self.assertEqual(split_wrap([10, 11], [10, 11]), ([], []))
        self.assertIsNone(split_wrap([1, 12], [10, 11]))

    def test_agreement(self):
        self.assertEqual([i for i in compare_special(self.ref(), self.eng()) if i["severity"] == "high"], [])

    def test_missing_bos(self):
        iss = compare_special(self.ref(), self.eng(probe_special=[10, 11]))
        high = [i for i in iss if i["severity"] == "high"]
        self.assertEqual([(i["check"], i["kind"]) for i in high], [("wrap", "start-missing")])
        self.assertEqual(score(iss, "wrap", strict=True), 1)

    def test_bos_uncertain_when_config_disagrees(self):
        iss = compare_special(self.ref(prefix=[], full=[10, 11], config_add_bos=True), self.eng())
        self.assertEqual([(i["kind"], i["severity"]) for i in iss if i["check"] == "wrap"], [("start-extra", "info")])

    def test_wrong_eos_appended(self):
        iss = compare_special(self.ref(full=[10, 11, 2], suffix=[2]), self.eng(probe_special=[1, 10, 11, 7]))
        self.assertEqual([i["kind"] for i in iss if i["severity"] == "high"], ["end-different"])

    def test_stop_missing_and_extra(self):
        eng = self.eng(eog=[{"id": 2, "text": "</s>", "attr": 8},
                            {"id": 9, "text": "<|endoftext|>", "attr": 8},
                            {"id": 151662, "text": "<|fim_pad|>", "attr": 8},
                            {"id": 128247, "text": "</s>", "attr": 12}])   # llama.cpp sets CONTROL itself
        iss = compare_special(self.ref(added_ids=[1, 2, 7, 9, 151662]), eng)
        got = [(i["check"], i.get("kind"), i.get("token"), i["severity"]) for i in iss if i["check"] == "stop"]
        self.assertIn(("stop", "missing", 7, "high"), got)
        self.assertIn(("stop", "extra", 9, "info"), got)          # a special token: rarely matters
        self.assertIn(("stop", "extra", 151662, "info"), got)     # added but not special (Qwen's FIM tokens)
        self.assertIn(("stop", "extra", 128247, "high"), got)     # ordinary text in the reference
        self.assertEqual(score(iss, "stop", strict=True), 2)
        self.assertEqual(score(iss, "stop"), 4)

    def test_report_keeps_pipes_in_token_names(self):
        self.assertEqual(hub._cell("stops on '<|fim_pad|>'\nx"), "stops on '<\\|fim_pad\\|>' x")

    @unittest.skipUnless(HAVE_TOKENIZERS, "needs the tokenizers package")
    def test_reference_special_from_tokenizer_json(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors

        from infercheck.special import reference_special
        vocab = {"<s>": 0, "</s>": 1, "<|im_end|>": 2, "Hello": 3, "world": 4}
        tok = Tokenizer(models.WordLevel(vocab, unk_token="</s>"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.add_special_tokens(["<s>", "</s>", "<|im_end|>"])
        tok.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 0)])
        with tempfile.TemporaryDirectory() as td:
            tok.save(str(Path(td) / "tokenizer.json"))
            (Path(td) / "tokenizer_config.json").write_text(json.dumps({"eos_token": "</s>"}))
            (Path(td) / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 2]}))
            r = reference_special(Path(td) / "tokenizer.json")
        self.assertEqual((r["plain"], r["prefix"], r["suffix"]), ([3, 4], [0], []))
        self.assertEqual(sorted(r["stops"]), [1, 2])


class FakeRef:
    def __init__(self, table):
        self.table = table

    def encode_many(self, texts):
        return [Result(*self.table[t]) if t in self.table else Result(None, None, "no") for t in texts]


class FakeEng:
    def __init__(self, decoded=None, crash=False, encode=None):
        self.decoded, self.crash, self.encode = decoded or {}, crash, encode or {}

    def decode_ids(self, seqs):
        if self.crash:
            return [None] * len(seqs)
        return [self.decoded.get(tuple(s)) for s in seqs]

    def encode_many(self, texts):
        return [self.encode.get(t, Result(None, None, "exit -11: segfault")) for t in texts]


class TestExtraChecks(unittest.TestCase):
    def test_refdecode(self):
        ref = FakeRef({"ab": ([1, 2], "ab"), "a b": ([1, 3, 2], "a b"), "": ([], "")})
        r = hub.refdecode_check(ref, FakeEng({(1, 2): "ab", (1, 3, 2): "a  b"}), ["ab", "a b", ""])
        self.assertEqual((r["compared"], r["mismatch"], r["error"]), (2, 1, 0))
        self.assertEqual(r["examples"][0]["got"], "a  b")
        c = hub.refdecode_check(ref, FakeEng(crash=True), ["ab", "a b"])
        self.assertEqual((c["error"], c.get("crash")), (2, True))

    def test_stress(self):
        from infercheck.corpus import stress_cases
        texts = [c.text for c in stress_cases()]
        ref = FakeRef({t: ([len(t)], t) for t in texts})
        ok = {t: Result([len(t)], t) for t in texts}
        self.assertEqual(hub.stress_check(ref, FakeEng(encode=ok))["error"], 0)
        crash_on_A = {t: r for t, r in ok.items() if not t.startswith("A")}
        r = hub.stress_check(ref, FakeEng(encode=crash_on_A))
        self.assertEqual((r["error"], r["failures"][0]["case"]), (1, "x0"))

    def test_info_from_fake_harness(self):
        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "llamatok"
            fake.write_text("#!/bin/sh\nif [ \"$2\" = --info ]; then echo '{\"add_bos\": true, \"eog\": []}'; "
                            "else echo \"usage: llamatok MODEL.gguf [--decode] < lines\" >&2; exit 2; fi\n")
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            self.assertEqual(LlamaCppTokenizer("v.gguf", bin_path=fake).info(), {"add_bos": True, "eog": []})
            old = Path(td) / "old"
            old.write_text("#!/bin/sh\necho \"usage: llamatok MODEL.gguf [--decode] < lines\" >&2\nexit 2\n")
            old.chmod(old.stat().st_mode | stat.S_IEXEC)
            with self.assertRaisesRegex(RuntimeError, "rebuild"):
                LlamaCppTokenizer("v.gguf", bin_path=old).info()

    def _fake_harness(self, td, body: str) -> Path:
        p = Path(td) / "fake_llamatok"
        p.write_text("#!" + sys.executable + "\nimport os, sys\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return p

    def test_crash_on_one_input_keeps_the_rest(self):
        body = ("for line in sys.stdin:\n"
                "    t = bytes.fromhex(line.strip()).decode()\n"
                "    if t == 'boom':\n"
                "        sys.stdout.write('[Thread debugging garbage]\\n'); sys.stdout.flush(); os.abort()\n"
                "    sys.stdout.write(str(len(t)) + '\\t' + t.encode().hex() + '\\n'); sys.stdout.flush()\n")
        with tempfile.TemporaryDirectory() as td:
            eng = LlamaCppTokenizer("v.gguf", bin_path=self._fake_harness(td, body))
            res = eng.encode_many(["ab", "boom", "c", "boom", "dd"])
            self.assertEqual([r.ids for r in res], [[2], None, [1], None, [2]])
            self.assertTrue(res[1].error.startswith("exit -6"))
            self.assertEqual(eng.last_crashes, 2)

    def test_load_failure_does_not_restart(self):
        body = "sys.stderr.write('llama.cpp failed to load the model\\n'); sys.exit(3)\n"
        with tempfile.TemporaryDirectory() as td:
            eng = LlamaCppTokenizer("v.gguf", bin_path=self._fake_harness(td, body))
            res = eng.encode_many(["a", "b", "c"])
            self.assertTrue(all(r.error and r.error.startswith("exit 3") for r in res))
            self.assertEqual(eng.last_crashes, 0)

    def test_crash_at_startup_gives_up(self):
        with tempfile.TemporaryDirectory() as td:
            eng = LlamaCppTokenizer("v.gguf", bin_path=self._fake_harness(td, "os.abort()\n"))
            res = eng.encode_many([str(i) for i in range(50)])
            self.assertEqual(len(res), 50)
            self.assertTrue(all(r.error for r in res))
            self.assertLessEqual(eng.last_crashes, 3)

    def test_word_cases(self):
        from infercheck.corpus import vocab_pieces, word_cases
        pieces = vocab_pieces(lambda ids: [f" w{i}" for i in ids], 100, k=20)
        self.assertEqual(len(pieces), 20)
        cases = word_cases(pieces + ["<|im_end|>", "�", "   "], n=50)
        self.assertEqual(len(cases), 50)
        self.assertTrue(all("<|im_end|>" not in c.text and "�" not in c.text for c in cases))
        self.assertEqual([c.text for c in cases], [c.text for c in word_cases(pieces, n=50)])  # seeded

    def test_hub_report_sections(self):
        recs = [{"repo": "q/qwen", "downloads": 100, "status": "ok", "pre": "qwen2",
                 "engine": {"cases": 10, "encode": 0, "decode": 0, "error": 0, "classes": {}, "causes": {},
                            "special": {"issues": [{"check": "stop", "severity": "high", "kind": "extra",
                                                    "detail": "llama.cpp stops on '</s>' (128247)"}]},
                            "refdecode": {"compared": 10, "mismatch": 1, "error": 0,
                                          "examples": [{"ref": "a b", "got": "a  b"}]},
                            "stress": {"cases": 10, "mismatch": 0, "error": 1,
                                       "failures": [{"case": "x0", "kind": "error"}]}}}]
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "records.jsonl").write_text(json.dumps(recs[0]) + "\n")
            text = hub.report(Path(td))
        self.assertIn("## Start and stop tokens", text)
        self.assertIn("**1** differ from their source model", text)
        self.assertIn("| q/qwen | 100 | stop | llama.cpp stops on '</s>' (128247) |", text)
        self.assertIn("decodes the source tokenizer's ids differently in **1**", text)
        self.assertIn("| q/qwen | 100 | x0 error |", text)


def keyed(bad_encode=(), corpus=("s1", "s2", "s3", "s4", "s5", "s6"), **kw):
    """A side whose encode findings are recorded string by string."""
    return side(engine={"encode": len(bad_encode), "corpus": list(corpus),
                        "bad": {"encode": list(bad_encode), "decode": [], "error": []}}, **kw)


class TestStringByString(unittest.TestCase):
    eng_case = {"id": "e", "kind": "engine", "detector": "encode", "category": "tokenizer",
                "model": {"gguf": ["a/b"]}, "engine": {"broken": SHA_A, "fixed": SHA_B}}

    def test_fix_that_also_breaks_other_strings_is_caught(self):
        # same count on both sides: counting alone calls this a miss
        b = keyed(["s1", "s2", "s3"], engine_commit=SHA_A)
        f = keyed(["s3", "s4", "s5"], engine_commit=SHA_B)
        self.assertEqual(bench.compare_sides(b, f, "encode"), [3, 3, 2, 2])
        s = bench.score_case(self.eng_case, {"broken": b, "fixed": f})
        self.assertEqual((s["outcome"], s["fired"]), ("caught", ["encode"]))
        self.assertTrue(any("introduced new problems: encode 2" in n for n in s["notes"]))
        self.assertEqual(bench._fired_cell("encode", s["metrics"]["encode"]), "encode 3→3 (2 repaired, 2 new)")
        self.assertEqual(bench._fired_cell("encode", [3, 0, 3, 0]), "encode 3→0")

    def test_strings_only_one_side_compared_do_not_count(self):
        # s9 failed on the broken side but the fixed side never ran it (its corpus differs): not a repair
        b = keyed(["s9"], corpus=["s1", "s9"], engine_commit=SHA_A)
        f = keyed([], corpus=["s1"], engine_commit=SHA_B)
        self.assertEqual(bench.compare_sides(b, f, "encode"), [1, 0, 0, 0])
        self.assertEqual(bench.score_case(self.eng_case, {"broken": b, "fixed": f})["outcome"], "missed")

    def test_old_records_fall_back_to_counts(self):
        b, f = side(engine={"encode": 5}), keyed(["s1"])
        self.assertIsNone(bench.bad_set(b, "encode"))
        self.assertEqual(bench.compare_sides(b, f, "encode"), [5, 1, 4, 0])

    def test_other_checks(self):
        b = side(refdecode={"mismatch": 2, "corpus": ["k1", "k2", "k3"], "bad": ["k1", "k2"]},
                 stress={"error": 1, "corpus": ["x0", "x1"], "failures": [{"case": "x0", "kind": "error"}]},
                 special={"issues": [{"check": "stop", "severity": "high", "kind": "missing", "token": 7, "detail": ""}]})
        f = side(refdecode={"mismatch": 1, "corpus": ["k1", "k2", "k3"], "bad": ["k3"]},
                 stress={"corpus": ["x0", "x1"]}, special={"issues": []})
        self.assertEqual(bench.compare_sides(b, f, "refdecode"), [2, 1, 2, 1])
        self.assertEqual(bench.compare_sides(b, f, "stress"), [1, 0, 1, 0])
        self.assertEqual(bench.compare_sides(b, f, "stop"), [1, 0, 1, 0])

    def test_compare_records_keys(self):
        from infercheck.corpus import Case
        from infercheck.diff import compare, text_key

        ref = FakeRef({"a": ([1], "a"), "b": ([2], "b"), "c": ([3], "c")})
        eng = FakeEng(encode={"a": Result([1], "a"), "b": Result([9], "b")})   # "c" crashes
        eng.name = "llama.cpp"
        cases = [Case(f"t{i}", t, "x") for i, t in enumerate(["a", "b", "c", "zz"])]   # the reference fails on "zz"
        _, stats = compare(ref, [eng], cases, shrink=False, log=lambda *a, **k: None, record_keys=True)
        self.assertEqual(stats["corpus"], sorted(text_key(t) for t in "abc"))
        bad = stats["engines"]["llama.cpp"]["bad"]
        self.assertEqual((bad["encode"], bad["error"], bad["decode"]), ([text_key("b")], [text_key("c")], []))
        _, plain = compare(ref, [eng], cases, shrink=False, log=lambda *a, **k: None)
        self.assertNotIn("corpus", plain)

    def test_refdecode_keys(self):
        from infercheck.diff import text_key
        ref = FakeRef({"ab": ([1, 2], "ab"), "a b": ([1, 3, 2], "a b")})
        r = hub.refdecode_check(ref, FakeEng({(1, 2): "ab", (1, 3, 2): "a  b"}), ["ab", "a b"], keys=True)
        self.assertEqual((r["corpus"], r["bad"]), (sorted([text_key("ab"), text_key("a b")]), [text_key("a b")]))


class TestBugHeader(unittest.TestCase):
    def case(self, **bh):
        return {"id": "f", "kind": "file", "detector": "wrap", "category": "wrap", "model": {"gguf": ["a/b"]},
                "file": {"broken_before": "2026-04-07", "bug_header": {"key": "tokenizer.ggml.add_bos_token", **bh}}}

    def rec(self, value, present=True, **kw):
        return side(bug_header={"key": "tokenizer.ggml.add_bos_token", "present": present, "value": value}, **kw)

    def test_holds(self):
        h = bench.bug_header_holds
        self.assertTrue(h(self.case(equals=False), self.rec(False)))
        self.assertFalse(h(self.case(equals=False), self.rec(None, present=False)))
        self.assertTrue(h(self.case(missing=True), self.rec(None, present=False)))
        self.assertFalse(h(self.case(missing=True), self.rec(True)))
        self.assertTrue(h(self.case(not_equals=True), self.rec(None, present=False)))
        self.assertFalse(h(self.case(not_equals=True), self.rec(True)))
        self.assertIsNone(h({"id": "x", "kind": "file", "file": {}}, self.rec(True)))

    def test_scoring_needs_the_bug_in_the_files(self):
        bos = {"check": "wrap", "severity": "high", "kind": "start-missing", "detail": ""}
        c = self.case(equals=False)
        good = {"broken": self.rec(False, revision="r1", special={"issues": [bos]}),
                "fixed": self.rec(True, revision="r2")}
        self.assertEqual(bench.score_case(c, good)["outcome"], "caught")
        no_bug = dict(good, broken=self.rec(True, revision="r1"))
        s = bench.score_case(c, no_bug)
        self.assertEqual(s["outcome"], "inconclusive")
        self.assertIn("before the fix does not have the bug's metadata", s["notes"][0])
        not_fixed = dict(good, fixed=self.rec(False, revision="r2"))
        self.assertIn("after the fix still has", bench.score_case(c, not_fixed)["notes"][0])

    def test_validate(self):
        base = {"id": "f", "title": "t", "kind": "file", "category": "c", "detector": "wrap", "sources": [],
                "model": {"gguf": ["a/b"]}}
        bad = [dict(base, id="p", platform="macos", file={"broken_before": "2026-04-07"}),
               dict(base, id="b1", file={"broken_before": "2026-04-07", "bug_header": {"key": "k"}}),
               dict(base, id="b2", file={"broken_before": "2026-04-07",
                                         "bug_header": {"key": "k", "equals": 1, "missing": True}}),
               dict(base, id="b3", file={"broken_before": "2026-04-07", "bug_header": "k"}),
               dict(base, id="b4", file={"broken_before": "2026-04-07", "bug_header": {"key": "k", "missing": False}})]
        errs = "\n".join(bench.validate(bad))
        self.assertIn("p: platform must be", errs)
        for cid in ("b1", "b2", "b3"):
            self.assertIn(f"{cid}: file.bug_header is", errs)
        self.assertIn("b4: file.bug_header.missing must be true", errs)
        ok = dict(base, id="ok", platform="linux",
                  file={"broken_before": "2026-04-07", "bug_header": {"key": "k", "not_equals": True}})
        self.assertEqual(bench.validate([ok]), [])


OTHER_PLATFORM = "win32" if not sys.platform.startswith("win") else "linux"


class TestPlatformAndCandidates(HubStandIns):
    def test_wrong_platform_is_not_a_miss(self):
        case = {"id": "e", "kind": "engine", "detector": "stress", "category": "crash", "platform": OTHER_PLATFORM,
                "model": {"gguf": ["ok/repo"], "reference": ["ref/repo"]}, "engine": {"broken": SHA_A, "fixed": SHA_B}}
        recs = {s: bench.run_side(case, s, spec, {}, self.builds, Path(self.td.name), 1, 0, 0)
                for s, spec in bench.side_specs(case).items()}
        self.assertEqual({r["status"] for r in recs.values()}, {"wrong-platform"})
        self.assertEqual(self.calls, [])   # nothing downloaded
        s = bench.score_case(case, recs)
        self.assertEqual(s["outcome"], "other-platform")
        self.assertIn(f"only on {OTHER_PLATFORM}", s["notes"][0])
        here = dict(case, platform=sys.platform[:5])
        self.assertEqual(bench.run_side(here, "broken", bench.side_specs(here)["broken"], {}, self.builds,
                                        Path(self.td.name), 1, 0, 0)["status"], "ok")

    def test_skips_a_repo_whose_file_has_no_tokenizer(self):
        from infercheck.ggufhdr import file_fetcher
        d = Path(self.td.name)
        (d / "bare.gguf").write_bytes(_tiny_gguf(arch=None))
        hub.range_fetcher = lambda repo, path, rev=None: (self.calls.append((repo, rev)),
                                                          file_fetcher(d / ("bare.gguf" if repo == "bad/repo" else "m.gguf")))[1]
        case = {"id": "c", "kind": "control", "category": "control",
                "model": {"gguf": ["bad/repo", "ok/repo"], "reference": ["ref/repo"]}}
        resolved = {}
        rec = bench.run_side(case, "only", bench.side_specs(case)["only"], resolved, self.builds, d, 1, 0, 0)
        self.assertEqual((rec["status"], rec["repo"]), ("ok", "ok/repo"))
        self.assertEqual([r for r, _ in self.calls], ["bad/repo", "ok/repo"])
        self.assertIn("bad/repo: skipped", resolved["c/only"]["notes"][0])

    def test_records_the_bug_header(self):
        case = {"id": "f", "kind": "file", "detector": "encode", "category": "t",
                "model": {"gguf": ["ok/repo"], "reference": ["ref/repo"]},
                "file": {"broken_before": "2026-04-07", "bug_header": {"key": "tokenizer.ggml.pre", "missing": True}}}
        rec = bench.run_side(case, "broken", bench.side_specs(case)["broken"], {}, self.builds, Path(self.td.name), 1, 0, 0)
        self.assertEqual(rec["bug_header"], {"key": "tokenizer.ggml.pre", "present": False, "value": None})
        self.assertTrue(bench.bug_header_holds(case, rec))

    def test_retry_reruns_the_whole_case_from_scratch(self):
        case = {"id": "e", "title": "t", "kind": "engine", "detector": "encode", "category": "t", "sources": [],
                "model": {"gguf": ["ok/repo"]}, "engine": {"broken": SHA_A, "fixed": SHA_B}}
        out = Path(self.td.name) / "out"
        out.mkdir()
        cp = Path(self.td.name) / "cases.json"
        cp.write_text(json.dumps({"cases": [case]}))
        (out / "sides.jsonl").write_text(json.dumps({"case": "e", "side": "broken", "status": "header-error"}) + "\n"
                                         + json.dumps({"case": "e", "side": "fixed", **side()}) + "\n")
        (out / "resolved.json").write_text(json.dumps({"e/broken": {"repo": "stale/repo"}, "e/fixed": {"repo": "stale/repo"}}))
        seen = []
        saved = bench.run_side

        def fake(c, s, spec, resolved, *a):
            seen.append((s, dict(resolved.get(f"e/{s}", {}))))
            return {"case": c["id"], "side": s, **side()}
        bench.run_side = fake
        try:
            self.assertEqual(bench.run(cp, out, None, None, 1, 0, 0), 0)
        finally:
            bench.run_side = saved
        self.assertEqual(seen, [("broken", {}), ("fixed", {})])   # both sides, without the stale pins
        seen.clear()
        bench.run_side = fake
        try:
            bench.run(cp, out, None, None, 1, 0, 0)   # nothing left to retry
        finally:
            bench.run_side = saved
        self.assertEqual(seen, [])


class TestTemplateBos(unittest.TestCase):
    def test_template_starts_with_bos(self):
        from infercheck.special import template_starts_with_bos as t
        self.assertTrue(t("{{ bos_token }}{% for m in messages %}", "<bos>"))
        self.assertTrue(t("{%- set sys = none -%}\n{# comment #}\n{{- bos_token -}}<start_of_turn>", "<bos>"))
        self.assertTrue(t("  <bos>{% for m in messages %}", "<bos>"))
        self.assertFalse(t("[gMASK]<sop>{%- if tools -%}", None))
        self.assertFalse(t("{% for m in messages %}{{ bos_token }}{{ m.content }}", "<bos>"))
        self.assertFalse(t("<|im_start|>system", "<bos>"))

    def test_chat_template_sources(self):
        from infercheck.special import chat_template
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self.assertIsNone(chat_template(d, {}))
            self.assertEqual(chat_template(d, {"chat_template": "A"}), "A")
            named = [{"name": "tool_use", "template": "T"}, {"name": "default", "template": "D"}]
            self.assertEqual(chat_template(d, {"chat_template": named}), "D")
            (d / "chat_template.jinja").write_text("J")
            self.assertEqual(chat_template(d, {}), "J")

    def ref(self, **kw):
        r = {"plain": [10, 11], "full": [10, 11], "prefix": [], "suffix": [], "config_add_bos": None,
             "impl": "transformers 5.22.1", "stops": {}, "special_ids": [2], "text": {2: "<bos>"}}
        r.update(kw)
        return r

    def test_bos_from_the_template_is_not_a_bug(self):
        eng = {"probe_plain": [10, 11], "probe_special": [2, 10, 11], "eog": []}
        via = [i for i in compare_special(self.ref(template_bos=2), eng) if i["check"] == "wrap"]
        self.assertEqual([(i["kind"], i["severity"]) for i in via], [("start-extra", "info")])
        self.assertIn("chat template", via[0]["detail"])
        plain = [i for i in compare_special(self.ref(template_bos=None), eng) if i["check"] == "wrap"]
        self.assertEqual([i["severity"] for i in plain], ["high"])
        other = {"probe_plain": [10, 11], "probe_special": [7, 10, 11], "eog": []}   # not the template's BOS
        self.assertEqual([i["severity"] for i in compare_special(self.ref(template_bos=2), other)
                          if i["check"] == "wrap"], ["high"])

    @unittest.skipUnless(HAVE_TOKENIZERS, "needs the tokenizers package")
    def test_reference_reads_the_template(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        from infercheck.special import reference_special
        tok = Tokenizer(models.WordLevel({"<pad>": 0, "<eos>": 1, "<bos>": 2, "Hello": 3, "world": 4}, unk_token="<pad>"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.add_special_tokens(["<pad>", "<eos>", "<bos>"])
        with tempfile.TemporaryDirectory() as td:
            tok.save(str(Path(td) / "tokenizer.json"))
            (Path(td) / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<bos>", "eos_token": "<eos>"}))
            (Path(td) / "chat_template.jinja").write_text("{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}")
            r = reference_special(Path(td) / "tokenizer.json")
        self.assertEqual((r["prefix"], r["template_bos"], r["text"][2]), ([], 2, "<bos>"))


class TestUnknownArch(HubStandIns):
    def _harness(self, stderr: str, code: int) -> Path:
        p = Path(self.td.name) / f"h{code}"
        p.write_text(f"#!/bin/sh\necho \"{stderr}\" >&2\nexit {code}\n")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return p

    def test_detects_only_an_unknown_architecture(self):
        from infercheck.ggufhdr import file_fetcher, read_header
        h = read_header(file_fetcher(Path(self.td.name) / "m.gguf"))
        arch = self._harness("llama.cpp failed to load the model: llama_model_load: error loading model: "
                             "unknown model architecture: 'deepseek41'", 3)
        self.assertTrue(bench.unknown_arch(h, arch))
        pre = self._harness("llama.cpp failed to load the model: unknown pre-tokenizer type: 'x'", 3)
        self.assertFalse(bench.unknown_arch(h, pre))   # can be the bug itself
        self.assertFalse(bench.unknown_arch(h, Path("/bin/true")))

    def test_run_side_skips_to_the_next_repo(self):
        from infercheck.ggufhdr import file_fetcher
        d = Path(self.td.name)
        (d / "fork.gguf").write_bytes(_tiny_gguf(arch="deepseek41"))
        hub.range_fetcher = lambda repo, path, rev=None: (self.calls.append((repo, rev)),
                                                          file_fetcher(d / ("fork.gguf" if repo == "fork/repo" else "m.gguf")))[1]
        saved = bench.unknown_arch
        bench.unknown_arch = lambda h, b: h.arch == "deepseek41"
        try:
            case = {"id": "c", "kind": "control", "category": "control",
                    "model": {"gguf": ["fork/repo", "ok/repo"], "reference": ["ref/repo"]}}
            resolved = {}
            rec = bench.run_side(case, "only", bench.side_specs(case)["only"], resolved, self.builds, d, 1, 0, 0)
            self.assertEqual((rec["status"], rec["repo"], rec["arch"]), ("ok", "ok/repo", "llama"))
            self.assertIn("does not know the architecture 'deepseek41'", resolved["c/only"]["notes"][0])
            only_fork = dict(case, model={"gguf": ["fork/repo"], "reference": ["ref/repo"]})
            rec = bench.run_side(only_fork, "only", bench.side_specs(only_fork)["only"], {}, self.builds, d, 1, 0, 0)
            self.assertEqual(rec["status"], "load-error")
        finally:
            bench.unknown_arch = saved


class TestSourceCandidates(unittest.TestCase):
    class H:
        def __init__(self, kv):
            self.kv = kv

        def get(self, k, default=None):
            return self.kv.get(k, default)

    def test_order(self):
        card = {"cardData": {"base_model": "ibm-granite/granite-4.2-8b"}}
        h = self.H({"general.base_model.0.repo_url": "https://huggingface.co/ibm-granite/granite-4.1-8b-base"})
        self.assertEqual(hub.source_candidates("ibm-granite/granite-4.2-8b-GGUF", h, card),
                         ["ibm-granite/granite-4.2-8b", "ibm-granite/granite-4.1-8b-base"])
        h = self.H({"general.source.repo_url": "https://huggingface.co/Qwen/Qwen3-8B"})
        self.assertEqual(hub.source_candidates("bartowski/Qwen3-8B-GGUF", h, {"cardData": {"base_model": ["Qwen/Qwen3-8B"]}}),
                         ["Qwen/Qwen3-8B", "bartowski/Qwen3-8B"])
        self.assertEqual(hub.source_candidates("Qwen/Qwen3-Embedding-0.6B-GGUF", self.H({}),
                                               {"cardData": {"base_model": "Qwen/Qwen3-0.6B-Base"}}),
                         ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-0.6B-Base"])
        self.assertEqual(hub.source_candidates("a/model", self.H({}), {}), [])


class TestTransformersReference(unittest.TestCase):
    def setUp(self):
        self.saved = sys.modules.get("transformers", "absent")

    def tearDown(self):
        if self.saved == "absent":
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = self.saved

    def test_uses_transformers_when_installed(self):
        import types

        from infercheck.special import PROBE, transformers_ids

        class Tok:
            def __call__(self, text, add_special_tokens=True):
                assert text == PROBE
                return {"input_ids": ([2] if add_special_tokens else []) + [10, 11]}

        fake = types.ModuleType("transformers")
        fake.__version__ = "9.9"
        calls = []
        fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda folder, **kw: (calls.append(kw), Tok())[1])
        sys.modules["transformers"] = fake
        self.assertEqual(transformers_ids(Path(".")), ([10, 11], [2, 10, 11], "transformers 9.9"))
        self.assertEqual(calls, [{"trust_remote_code": False}])   # never a repo's own code
        fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda folder, **kw: 1 / 0)
        self.assertIsNone(transformers_ids(Path(".")))   # cannot load the folder: fall back
        sys.modules["transformers"] = None                # not installed
        self.assertIsNone(transformers_ids(Path(".")))

    def test_start_token_is_certain_with_transformers(self):
        r = {"plain": [10, 11], "full": [2, 10, 11], "prefix": [2], "suffix": [], "config_add_bos": False,
             "stops": {}, "special_ids": [2], "text": {2: "<bos>"}}
        e = {"probe_plain": [10, 11], "probe_special": [10, 11], "eog": []}
        sev = lambda impl: [i["severity"] for i in compare_special(dict(r, impl=impl), e) if i["check"] == "wrap"]
        self.assertEqual(sev("tokenizers 0.22"), ["info"])      # config and post-processor disagree: unknown
        self.assertEqual(sev("transformers 5.0"), ["high"])     # transformers says what users get


if __name__ == "__main__":
    unittest.main()
