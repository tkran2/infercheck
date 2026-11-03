"""Replay bugs that were reported and fixed, and measure how many of them InferCheck catches.

bench/cases.json lists real bugs in llama.cpp and in published GGUF files. Every case has two sides:

- **engine bugs**: the same GGUF file, run through llama.cpp built at the commit before the fix
  ("broken") and at the fix ("fixed");
- **file bugs**: the GGUF file as uploaded before the fix and after it (the Hub keeps every
  revision), run through one llama.cpp build;
- **controls**: a single side, a popular GGUF with no known bug. Anything flagged there is either a
  false alarm or an unknown bug, and needs a look.

Each side gets every InferCheck check (see DETECTORS). The two sides are compared string by string:
a check catches a bug when strings it flagged on the broken side pass on the fixed side, so what
both sides share (other bugs, known differences) cancels out, and a fix that repairs one thing while
breaking another still counts as caught (the report notes the new problems). A case whose fixed side
cannot be found falls back to counting high-severity problems on the broken side alone, restricted
to the causes the case names. File cases can name the metadata the bug lives in (file.bug_header):
a pair of files that does not show it is reported as inconclusive rather than missed. Cases that
reproduce on one platform only ("platform") are skipped elsewhere.

  infercheck bench list                         the catalog
  infercheck bench run --llama-src ~/Projects/llama.cpp --out results/bench
  infercheck bench report results/bench
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .engines import ROOT, LlamaCppTokenizer

CASES = ROOT / "bench" / "cases.json"
DETECTORS = ("encode", "decode", "refdecode", "wrap", "stop", "stress")
DETECTOR_HELP = {
    "encode": "token ids differ from the reference",
    "decode": "same ids, different decoded text",
    "refdecode": "decoding the reference's ids gives different text or crashes",
    "wrap": "different tokens added around a prompt (BOS/EOS)",
    "stop": "different end-of-generation tokens",
    "stress": "fails or differs on long inputs",
}
KINDS = ("engine", "file", "control")
PLATFORMS = ("linux", "darwin", "win32")   # sys.platform prefixes
SHA = re.compile(r"^[0-9a-f]{40}$")


# ----------------------------------------------------------------------------- catalog

def load_cases(path: Path = CASES) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"]


def validate(cases: list[dict]) -> list[str]:
    """Problems with the catalog itself (run before anything is downloaded or built)."""
    errs, seen = [], set()
    for c in cases:
        cid = c.get("id", "?")
        if cid in seen:
            errs.append(f"{cid}: duplicate id")
        seen.add(cid)
        for k in ("id", "title", "kind", "category", "model", "sources") + (("detector",) if c.get("kind") != "control" else ()):
            if k not in c:
                errs.append(f"{cid}: missing {k}")
        if c.get("kind") not in KINDS:
            errs.append(f"{cid}: kind must be one of {KINDS}")
        if c.get("detector") not in DETECTORS + ("none",) and c.get("kind") != "control":
            errs.append(f"{cid}: detector must be one of {DETECTORS} or none")
        if not (c.get("model") or {}).get("gguf"):
            errs.append(f"{cid}: model.gguf is empty")
        if c.get("kind") == "engine":
            e = c.get("engine") or {}
            for side in ("broken", "fixed"):
                if not SHA.match(str(e.get(side, ""))):
                    errs.append(f"{cid}: engine.{side} must be a full 40-character llama.cpp commit")
        if c.get("kind") == "file":
            f = c.get("file") or {}
            if not (f.get("broken_before") or f.get("broken_repo") or f.get("broken_revision")):
                errs.append(f"{cid}: file needs broken_before, broken_repo or broken_revision")
            at = (c.get("engine") or {}).get("at", "current")
            if at != "current" and not SHA.match(at):
                errs.append(f"{cid}: engine.at must be 'current' or a full commit")
            for k in ("broken_before", "fixed_after"):
                if k in f and not re.match(r"^\d{4}-\d{2}-\d{2}$", f[k]):
                    errs.append(f"{cid}: file.{k} must be YYYY-MM-DD")
        for t in c.get("trigger") or []:
            if not (isinstance(t, str) or (isinstance(t, dict) and isinstance(t.get("repeat"), str)
                                           and isinstance(t.get("times"), int))):
                errs.append(f"{cid}: a trigger is a string or {{\"repeat\": str, \"times\": int}}")
        if "found_by_infercheck" in c and not isinstance(c["found_by_infercheck"], bool):
            errs.append(f"{cid}: found_by_infercheck must be true or false")
        if "platform" in c and c["platform"] not in PLATFORMS:
            errs.append(f"{cid}: platform must be one of {PLATFORMS}")
        bh = (c.get("file") or {}).get("bug_header")
        if bh is not None:
            tests = [k for k in ("missing", "equals", "not_equals") if isinstance(bh, dict) and k in bh]
            if not isinstance(bh, dict) or not isinstance(bh.get("key"), str) or len(tests) != 1:
                errs.append(f"{cid}: file.bug_header is {{\"key\": str}} plus one of missing, equals, not_equals")
            elif "missing" in bh and bh["missing"] is not True:
                errs.append(f"{cid}: file.bug_header.missing must be true")
    return errs


def expand_triggers(triggers: list) -> list[str]:
    """Trigger strings; {"repeat": s, "times": n} stands for s * n (long inputs stay readable in JSON)."""
    return [t if isinstance(t, str) else t["repeat"] * int(t["times"]) for t in triggers]


def side_specs(case: dict) -> dict[str, dict]:
    """What to run for each side: {side: {engine, gguf candidates, revision}}."""
    m = case["model"]
    if case["kind"] == "engine":
        rev = m.get("revision", "main")
        return {s: {"engine": case["engine"][s], "gguf": m["gguf"], "revision": rev} for s in ("broken", "fixed")}
    if case["kind"] == "control":
        return {"only": {"engine": "current", "gguf": m["gguf"], "revision": m.get("revision", "main")}}
    f, at = case["file"], (case.get("engine") or {}).get("at", "current")
    broken = {"engine": at, "gguf": [f["broken_repo"]] if f.get("broken_repo") else m["gguf"],
              "revision": f.get("broken_revision") or ({"before": f["broken_before"]} if f.get("broken_before") else "main")}
    specs = {"broken": broken}
    if f.get("fixed_repo") or f.get("fixed_revision") or f.get("fixed_after") or f.get("broken_before"):
        specs["fixed"] = {"engine": at, "gguf": [f["fixed_repo"]] if f.get("fixed_repo") else m["gguf"],
                          "revision": f.get("fixed_revision") or ({"after": f["fixed_after"]} if f.get("fixed_after") else "main")}
    if f.get("no_fixed"):
        specs.pop("fixed", None)
    return specs


# ----------------------------------------------------------------------------- resolving repos and revisions

def resolve_repo(cands: list, gguf: bool = True) -> tuple[str | None, list[str]]:
    """The first candidate repo that exists ({"search": q} entries query the Hub). Returns (repo, notes)."""
    from . import hub

    notes = []
    for c in cands:
        if isinstance(c, dict) and c.get("search"):
            params = {"search": c["search"], "sort": "downloads", "direction": "-1", "limit": 10}
            if gguf:
                params["filter"] = "gguf"
            try:
                hits, _ = hub.api("models", params)
            except Exception as exc:
                notes.append(f"search {c['search']!r}: {exc}")
                continue
            hits = [h["id"] for h in hits or [] if gguf or "gguf" not in h["id"].lower()]
            if hits:
                return hits[0], notes + [f"search {c['search']!r} -> {hits[0]}"]
            notes.append(f"search {c['search']!r}: no results")
            continue
        try:
            info = hub.model_info(c)
            if info.get("id"):
                return info["id"], notes
        except Exception as exc:
            notes.append(f"{c}: {str(exc)[:120]}")
    return None, notes


def repo_candidates(cands: list, limit: int = 5) -> tuple[list[str], list[str]]:
    """Every GGUF repo the case allows, in order: listed repos that exist, then up to `limit` hits of each
    {"search": ...}. run_side takes the first one whose file it can read and that has a tokenizer, so a
    broken upload among the search hits does not sink the case. Returns (repos, notes)."""
    from . import hub

    out, notes = [], []
    for c in cands:
        if isinstance(c, dict) and c.get("search"):
            try:
                hits, _ = hub.api("models", {"search": c["search"], "filter": "gguf", "sort": "downloads",
                                             "direction": "-1", "limit": limit})
            except Exception as exc:
                notes.append(f"search {c['search']!r}: {exc}")
                continue
            out += [h["id"] for h in hits or [] if h["id"] not in out]
            continue
        try:
            info = hub.model_info(c)
            if info.get("id") and info["id"] not in out:
                out.append(info["id"])
        except Exception as exc:
            notes.append(f"{c}: {str(exc)[:120]}")
    return out, notes


def _commits(repo: str) -> list[tuple[datetime, str]]:
    """(date, commit) of a repo's main branch, newest first."""
    from huggingface_hub import list_repo_commits

    return [(c.created_at, c.commit_id) for c in list_repo_commits(repo)]


def pick_revision(commits: list[tuple[datetime, str]], rev) -> str | None:
    """A commit for a revision spec: a commit id, "main" (newest), {"before": date} (the newest commit
    before that day) or {"after": date} (the oldest commit on or after it)."""
    if isinstance(rev, str) and SHA.match(rev):
        return rev
    if rev in (None, "main"):
        return commits[0][1] if commits else None
    day = datetime.strptime(rev.get("before") or rev.get("after"), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dated = [(d if d.tzinfo else d.replace(tzinfo=timezone.utc), c) for d, c in commits]
    if "before" in rev:
        older = [c for d, c in dated if d < day]
        return older[0] if older else None
    newer = [c for d, c in dated if d >= day]
    return newer[-1] if newer else None


# ----------------------------------------------------------------------------- llama.cpp builds

class BuildError(RuntimeError):
    pass


class Builds:
    """Harness binaries per llama.cpp commit, built on demand with scripts/build_llama_at.sh."""

    def __init__(self, llama_src: Path | None, root: Path, log=print):
        self.src, self.root, self.log = llama_src, root, log

    def harness(self, engine: str) -> Path:
        if engine == "current":
            b = LlamaCppTokenizer.BIN
            if not b.exists():
                raise BuildError(f"the default harness {b} is missing; build it (README, Reproduce)")
            return b
        out = self.root / f"llama-{engine[:12]}"
        if (out / "llamatok").exists():
            return out / "llamatok"
        if not self.src:
            raise BuildError(f"llama.cpp {engine[:12]} is not built yet; pass --llama-src (a llama.cpp clone)")
        self.log(f"building llama.cpp {engine[:12]} (a few minutes) ...", flush=True)
        t0 = time.time()
        p = subprocess.run(["bash", str(ROOT / "scripts" / "build_llama_at.sh"), str(self.src), engine, str(out)],
                           capture_output=True, text=True)
        if p.returncode != 0 or not (out / "llamatok").exists():
            raise BuildError(f"build of {engine[:12]} failed: {(p.stderr or p.stdout).strip()[-600:]}")
        self.log(f"  built in {time.time() - t0:.0f}s", flush=True)
        return out / "llamatok"


# ----------------------------------------------------------------------------- running one side

def trigger_check(vocab_gguf: Path, ref_tj: Path, bin_path: Path, triggers: list[str]) -> dict:
    """Run the case's known trigger strings (from the bug report) on one side, without shrinking."""
    from .corpus import Case
    from .diff import compare
    from .engines import HFTokenizer
    from .hub import refdecode_check

    ref = HFTokenizer(str(ref_tj))
    eng = LlamaCppTokenizer(str(vocab_gguf), bin_path=bin_path)
    eng.name = "llama.cpp"
    cases = [Case(f"t{i}", t, "trigger") for i, t in enumerate(triggers)]
    findings, stats = compare(ref, [eng], cases, shrink=False, log=lambda *a, **k: None)
    rd = refdecode_check(ref, eng, triggers)
    s = stats["engines"]["llama.cpp"]
    return {"encode": s["encode"], "decode": s["decode"], "error": s["error"],
            "refdecode": rd["mismatch"] + rd["error"], "total": s["encode"] + s["decode"] + s["error"] + rd["mismatch"] + rd["error"]}


def unknown_arch(header, bin_path: Path) -> bool:
    """Whether llama.cpp at this build refuses the file's vocabulary because it does not know its architecture.
    Any other load failure (an unknown pre-tokenizer, say) can be the bug itself, so it is not checked here."""
    from .ggufhdr import write_vocab_only

    with tempfile.TemporaryDirectory() as td:
        vg = write_vocab_only(header, Path(td) / "vocab.gguf")
        try:
            LlamaCppTokenizer(str(vg), bin_path=bin_path).info()
        except RuntimeError as exc:
            return "unknown model architecture" in str(exc)
        except Exception:
            return False
    return False


def run_side(case: dict, side: str, spec: dict, resolved: dict, builds: Builds, cache: Path,
             n_gen: int, seed: int, max_shrink: int) -> dict:
    from . import hub
    from .ggufhdr import read_header, write_vocab_only

    rec = {"case": case["id"], "side": side, "engine_commit": spec["engine"], "platform": sys.platform,
           "checked": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if case.get("platform") and not sys.platform.startswith(case["platform"]):
        return {**rec, "status": "wrong-platform",
                "error": f"this bug reproduces only on {case['platform']} (this run: {sys.platform})"}
    try:
        bin_path = builds.harness(spec["engine"])
    except BuildError as exc:
        return {**rec, "status": "build-error", "error": str(exc)}

    key = f"{case['id']}/{side}"
    r = resolved.setdefault(key, {})
    if case["kind"] == "engine" and side == "fixed" and not r:
        # an engine case compares two llama.cpp builds on one file: pin the file the broken side used
        r.update({k: v for k, v in resolved.get(f"{case['id']}/broken", {}).items() if k in ("repo", "revision", "file")})
    if "repo" in r:
        repos, notes = [r["repo"]], []
    else:
        repos, notes = repo_candidates(spec["gguf"])
        if not repos:
            return {**rec, "status": "no-repo", "error": "; ".join(notes)[:400]}
    header, last = None, ("no-repo", "no candidate repo")
    for repo in repos:
        try:
            rev = r["revision"] if r.get("repo") == repo and "revision" in r else pick_revision(_commits(repo), spec["revision"])
        except Exception as exc:
            last = ("error", f"{repo}: commit list: {exc}"[:300])
            continue
        if not rev:
            last = ("no-revision", f"no revision of {repo} matches {spec['revision']}")
            continue
        path = r["file"] if r.get("repo") == repo and "file" in r else (case["model"].get("file") or hub.pick_gguf(repo, rev))
        if not path:
            last = ("no-gguf", f"{repo}@{rev[:10]}: no GGUF file")
            continue
        try:
            h = hub.read_header(hub.range_fetcher(repo, path, rev))
        except Exception as exc:
            last = ("header-error", f"{repo}: {exc}"[:300])
            continue
        if not (h.arch and h.tokenizer_model):
            notes.append(f"{repo}: skipped (no architecture or tokenizer in the GGUF metadata)")
            last = ("header-error", f"{repo}: no architecture or tokenizer in the GGUF metadata")
            continue
        if unknown_arch(h, bin_path):
            # a fork's GGUF or a model newer than this build: no use for a tokenizer test, try the next repo
            build = "the current build" if spec["engine"] == "current" else spec["engine"][:12]
            why = f"llama.cpp {build} does not know the architecture {h.arch!r}"
            notes.append(f"{repo}: skipped ({why})")
            last = ("load-error", f"{repo}: {why}")
            continue
        header = h
        r.update(repo=repo, revision=rev, file=path, notes=notes)
        break
    if header is None:
        return {**rec, "status": last[0], "error": last[1]}
    repo, rev = r["repo"], r["revision"]
    rec.update(repo=repo, revision=rev, file=r["file"], arch=header.arch, pre=header.pre,
               tokenizer_key=hub.tokenizer_key(header), add_bos_key=header.get("tokenizer.ggml.add_bos_token"))
    bh = (case.get("file") or {}).get("bug_header")
    if bh:
        rec["bug_header"] = {"key": bh["key"], "present": bh["key"] in header.kv, "value": header.get(bh["key"])}

    refs = list(case["model"].get("reference") or [])
    refs += [hub.hf_repo_from_url(header.get("general.source.repo_url")),
             hub.hf_repo_from_url(header.get("general.base_model.0.repo_url"))]
    try:  # the GGUF repo's model card names its base model, like hub-scan uses
        refs += hub.card_base_models(hub.model_info(repo))
    except Exception:
        pass
    ref_tj, ref_repo = None, None
    for c in [x for i, x in enumerate(refs) if x and x not in refs[:i]]:
        if isinstance(c, dict):
            c, _ = resolve_repo([c], gguf=False)
            if not c:
                continue
        ref_tj = hub.fetch_tokenizer(c, cache, revision=case["model"].get("reference_revision"))
        if ref_tj:
            ref_repo = c
            break
    if not ref_tj:
        return {**rec, "status": "no-reference", "error": f"no tokenizer.json in {refs}"}
    rec["reference"] = ref_repo

    with tempfile.TemporaryDirectory() as td:
        vg = write_vocab_only(header, Path(td) / "vocab.gguf")
        try:
            stats, findings = hub.run_engine_check(vg, ref_tj, n_gen, seed, max_shrink, bin_path=bin_path, keys=True)
        except Exception as exc:
            return {**rec, "status": "engine-error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        # the harness exits with 3 when llama.cpp cannot load the vocabulary; any other failure on most of the
        # corpus is a crash while tokenizing, which is a finding (a crash takes the rest of the batch with it)
        if hub._load_failed({"engine": stats}) and (stats.get("error_sample") or "").startswith("exit 3"):
            return {**rec, "status": "load-error", "engine": stats,
                    "error": "llama.cpp could not load the vocabulary: " + (stats.get("error_sample") or "?")}
        if hub._load_failed({"engine": stats}):
            rec["crashed"] = (stats.get("error_sample") or "")[:200]
        if case.get("trigger"):
            try:
                rec["trigger"] = trigger_check(vg, ref_tj, bin_path, expand_triggers(case["trigger"]))
            except Exception as exc:
                rec["trigger"] = {"error": str(exc)[:200]}
    for k in ("refdecode", "stress", "special"):
        rec[k] = stats.pop(k, None)
    rec["engine"] = stats
    rec["examples"] = [{"kind": f.kind, "cause": getattr(f, "cls", None), "minimal": f.minimal[:60]} for f in findings[:6]]
    return {**rec, "status": "ok"}


# ----------------------------------------------------------------------------- scoring

def metric(rec: dict, d: str, strict: bool = False, causes: list[str] | None = None) -> int:
    """How much one check finds on one side. strict: only high-severity problems and, if the case names
    causes, only findings with those causes (used when there is no fixed side to compare with)."""
    from .special import score

    e = rec.get("engine") or {}
    if d in ("encode", "decode") and strict and causes:
        return sum(v for k, v in (e.get("causes") or {}).items() if k in causes) if d == "encode" else 0
    if d == "encode":
        return (e.get("encode") or 0) + (e.get("error") or 0)
    if d == "decode":
        return e.get("decode") or 0
    if d in ("refdecode", "stress"):
        x = rec.get(d) or {}
        return (x.get("mismatch") or 0) + (x.get("error") or 0)
    if d in ("wrap", "stop"):
        return score((rec.get("special") or {}).get("issues") or [], d, strict=strict)
    raise ValueError(d)


def bad_set(rec: dict, d: str) -> tuple[set, set | None] | None:
    """(keys of what check d found on one side, keys of everything it compared) for a string-by-string
    comparison of two sides, or None if the record has no keys (older runs: fall back to counts).
    For wrap and stop the "strings" are the issues themselves and every issue is comparable."""
    e = rec.get("engine") or {}
    if d in ("encode", "decode"):
        b = e.get("bad")
        if b is None or e.get("corpus") is None:
            return None
        keys = set(b["encode"]) | set(b["error"]) if d == "encode" else set(b["decode"])
        return keys, set(e["corpus"])
    if d == "refdecode":
        x = rec.get("refdecode") or {}
        return (set(x["bad"]), set(x["corpus"])) if "bad" in x and "corpus" in x else None
    if d == "stress":
        x = rec.get("stress") or {}
        if "corpus" not in x:
            return None
        return {f["case"] for f in x.get("failures") or []}, set(x["corpus"])
    issues = (rec.get("special") or {}).get("issues")
    if issues is None:
        return None
    return {f"{i['kind']}:{i.get('token', i['detail'])}" for i in issues if i["check"] == d and i.get("kind")}, None


def compare_sides(b: dict, f: dict, d: str) -> list:
    """[found with the bug, found with the fix, repaired by the fix, broken by the fix] for check d.
    Repaired: problems on the broken side, on strings both sides compared, that the fixed side no longer
    has. A fix that also breaks other strings still gets credit for what it repaired."""
    sb, sf = bad_set(b, d), bad_set(f, d)
    if sb is None or sf is None:
        mb, mf = metric(b, d), metric(f, d)
        return [mb, mf, max(mb - mf, 0), max(mf - mb, 0)]
    (kb, ub), (kf, uf) = sb, sf
    repaired = {k for k in kb - kf if uf is None or k in uf}
    regressed = {k for k in kf - kb if ub is None or k in ub}
    return [len(kb), len(kf), len(repaired), len(regressed)]


def bug_header_holds(case: dict, rec: dict) -> bool | None:
    """Whether a file side's GGUF metadata shows the bug (file.bug_header), or None if the case names none."""
    bh = (case.get("file") or {}).get("bug_header")
    h = rec.get("bug_header")
    if not bh or h is None:
        return None
    if bh.get("missing"):
        return not h["present"]
    if "equals" in bh:
        return h["present"] and h["value"] == bh["equals"]
    if "not_equals" in bh:
        return h["value"] != bh["not_equals"]
    return None


def score_case(case: dict, sides: dict[str, dict]) -> dict:
    out = {"id": case["id"], "expected": case.get("detector"), "fired": [], "metrics": {}, "notes": []}
    other = [r for r in sides.values() if r.get("status") == "wrong-platform"]
    if other:  # not a miss: the bug cannot show on this machine
        out.update(outcome="other-platform", notes=[other[0].get("error", "")])
        return out
    bad = [f"{s}: {r.get('status')} ({(r.get('error') or '')[:160]})" for s, r in sides.items() if r.get("status") != "ok"]
    missing = [s for s in side_specs(case) if s not in sides]
    if bad or missing:
        out.update(outcome="inconclusive", notes=bad + [f"{s}: not run" for s in missing])
        return out
    if case["kind"] == "file":
        bh = (case.get("file") or {}).get("bug_header", {})
        hb = bug_header_holds(case, sides["broken"])
        hf = bug_header_holds(case, sides["fixed"]) if "fixed" in sides else None
        if hb is False:
            out.update(outcome="inconclusive", notes=[f"the file before the fix does not have the bug's metadata "
                                                      f"({bh.get('key')} = {sides['broken']['bug_header']['value']!r})"])
            return out
        if hf is True:
            out.update(outcome="inconclusive", notes=[f"the file after the fix still has the bug's metadata "
                                                      f"({bh.get('key')} = {sides['fixed']['bug_header']['value']!r})"])
            return out
    if case["kind"] == "control":
        only = sides["only"]
        out["metrics"] = {d: [metric(only, d, strict=True)] for d in DETECTORS}
        out["fired"] = [d for d in DETECTORS if out["metrics"][d][0] > 0]
        out["outcome"] = "flagged" if out["fired"] else "clean"
        return out
    b, f = sides["broken"], sides.get("fixed")
    if f is not None and b.get("revision") == f.get("revision") and b.get("repo") == f.get("repo") \
            and b.get("engine_commit") == f.get("engine_commit"):
        out.update(outcome="inconclusive", notes=["broken and fixed sides resolved to the same file and llama.cpp build"])
        return out
    if f is not None and case["kind"] == "file" and b.get("tokenizer_key") == f.get("tokenizer_key") \
            and b.get("add_bos_key") == f.get("add_bos_key") and case.get("category") not in ("template", "metadata"):
        out["notes"].append("the tokenizer metadata is identical on both sides")
    causes = case.get("causes")
    for d in DETECTORS:
        if f is not None:
            out["metrics"][d] = compare_sides(b, f, d)
        else:
            out["metrics"][d] = [metric(b, d, strict=True, causes=causes), None]
    if f is not None:
        out["fired"] = [d for d in DETECTORS if out["metrics"][d][2] > 0]
        broke = [f"{d} {out['metrics'][d][3]}" for d in DETECTORS if out["metrics"][d][3] > 0]
        if broke:
            out["notes"].append("the fix also introduced new problems: " + ", ".join(broke))
    else:
        out["fired"] = [d for d in DETECTORS if out["metrics"][d][0] > 0]
    if case.get("detector") == "none":
        out["outcome"] = "not-covered"
    else:
        out["outcome"] = "caught" if out["fired"] else "missed"
    tb, tf = (b.get("trigger") or {}).get("total"), ((f or {}).get("trigger") or {}).get("total")
    if case.get("trigger") and tb is not None:
        out["trigger"] = {"broken": tb, "fixed": tf, "reproduces": tb > (tf or 0)}
        if not out["trigger"]["reproduces"]:
            out["notes"].append("the trigger from the bug report does not reproduce: check the case")
    return out


# ----------------------------------------------------------------------------- commands

def _load_sides(out: Path) -> dict[str, dict[str, dict]]:
    by: dict = {}
    p = out / "sides.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                r = json.loads(line)
                by.setdefault(r["case"], {})[r["side"]] = r
    return by


# wrong-platform costs nothing to retry and lets a run on the right platform fill the case in; a load-error is a
# result (that build cannot load that file), so it is kept
RETRY = {"build-error", "engine-error", "error", "header-error", "no-repo", "no-revision", "no-reference",
         "wrong-platform"}


def run(cases_path: Path, out: Path, llama_src: Path | None, only: list[str] | None, n_gen: int, seed: int,
        max_shrink: int, retry: bool = True) -> int:
    from .hub import engine_problem

    cases = load_cases(cases_path)
    errs = validate(cases)
    if errs:
        print("bench/cases.json has problems:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 2
    if only:
        cases = [c for c in cases if c["id"] in only]
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "hf-cache"
    resolved_p = out / "resolved.json"
    resolved = json.loads(resolved_p.read_text()) if resolved_p.exists() else {}
    builds = Builds(llama_src, out / "builds")
    if any(s["engine"] == "current" for c in cases for s in side_specs(c).values()):
        problem = engine_problem()
        if problem:
            print(problem, file=sys.stderr)
            return 2
    done = _load_sides(out)
    with open(out / "sides.jsonl", "a", encoding="utf-8") as fh:
        for i, c in enumerate(cases, 1):
            specs = side_specs(c)
            prevs = {s: done.get(c["id"], {}).get(s) for s in specs}
            todo = [s for s, p in prevs.items() if not p or (retry and p.get("status") in RETRY)]
            if todo:  # rerun the whole case from a fresh resolution: the file it picked last time may be the problem
                todo = list(specs)
                for s in specs:
                    resolved.pop(f"{c['id']}/{s}", None)
            for side, spec in specs.items():
                if side not in todo:
                    continue
                t0 = time.time()
                try:
                    rec = run_side(c, side, spec, resolved, builds, cache, n_gen, seed, max_shrink)
                except Exception as exc:
                    rec = {"case": c["id"], "side": side, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                resolved_p.write_text(json.dumps(resolved, indent=1))
                strict = side == "only"  # controls are scored on high-severity problems only
                summary = (" ".join(f"{d}={metric(rec, d, strict=strict)}" for d in DETECTORS) if rec["status"] == "ok"
                           else rec.get("error", "")[:200])
                print(f"[{i}/{len(cases)} {time.time() - t0:4.0f}s] {c['id']} {side}: {rec['status']} {summary}", flush=True)
    text = report(out, cases_path)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(f"\nreport: {out / 'report.md'}")
    return 0


def _fired_cell(d: str, m: list) -> str:
    """"encode 12→0" for a two-sided case, "encode 121→897 (40 repaired, 816 new)" when the fix also broke
    other strings, "encode 12" for a control (one side), "encode 12→-" with no fixed side."""
    if len(m) == 1:
        return f"{d} {m[0]}"
    if len(m) == 4:
        return f"{d} {m[0]}→{m[1]}" + (f" ({m[2]} repaired, {m[3]} new)" if m[3] else "")
    return f"{d} {m[0]}→{m[1] if m[1] is not None else '-'}"


def report(out: Path, cases_path: Path = CASES) -> str:
    cases = load_cases(cases_path)
    sides = _load_sides(out)
    scored = [(c, score_case(c, sides.get(c["id"], {}))) for c in cases if c["id"] in sides]
    by = lambda o: [(c, s) for c, s in scored if s["outcome"] == o]
    caught, missed, nc, inc = by("caught"), by("missed"), by("not-covered"), by("inconclusive")
    other = by("other-platform")
    controls = [(c, s) for c, s in scored if c["kind"] == "control" and s["outcome"] not in ("inconclusive", "other-platform")]
    flagged = [x for x in controls if x[1]["outcome"] == "flagged"]
    covered = len(caught) + len(missed)
    ran = [r for by_side in sides.values() for r in by_side.values()]
    platforms = sorted({r["platform"] for r in ran if r.get("platform")})
    impls = sorted({(r.get("special") or {}).get("ref_impl") for r in ran if r.get("status") == "ok"} - {None})

    lines = ["# InferCheck bug benchmark", "",
             "Real tokenizer and special-token bugs from llama.cpp's history and from published GGUF files, replayed: "
             "each case runs InferCheck on the version with the bug and the version with the fix. A check catches a bug "
             "when the fix repairs strings (or start/stop tokens) that it flagged with the bug.", ""]
    if platforms or impls:
        lines += [f"Run on {', '.join(platforms) or '?'}; reference for the tokens around a prompt: "
                  f"{', '.join(impls) or '?'}.", ""]
    if covered:
        lines.append(f"**InferCheck caught {len(caught)} of {covered} replayed bugs**"
                     f" ({sum(1 for c, _ in caught if c['kind'] == 'file')} of "
                     f"{sum(1 for c, _ in caught + missed if c['kind'] == 'file')} bugs in GGUF files, "
                     f"{sum(1 for c, _ in caught if c['kind'] == 'engine')} of "
                     f"{sum(1 for c, _ in caught + missed if c['kind'] == 'engine')} in llama.cpp), "
                     "from the GGUF metadata alone.")
        own = [c for c, _ in caught + missed if c.get("found_by_infercheck")]
        if own:  # catching what it found itself proves little: give the score without those too
            others = [(c, s) for c, s in caught + missed if not c.get("found_by_infercheck")]
            lines.append("")
            lines.append(f"{len(own)} of the {covered} were first found by InferCheck itself; on the "
                         f"{len(others)} bugs found by others it caught "
                         f"{sum(1 for _, s in others if s['outcome'] == 'caught')}.")
    lines += ["",
              f"- caught: {len(caught)}, missed: {len(missed)}",
              f"- not covered yet (chat templates, model metadata): {len(nc)}",
              f"- inconclusive (could not build, find or load a side): {len(inc)}",
              *([f"- not run here (the bug only shows on another platform): {len(other)}"] if other else []),
              f"- controls (no known bug): {len(controls) - len(flagged)} clean, {len(flagged)} flagged", ""]
    if caught:
        per = {d: sum(1 for _, s in caught if d in s["fired"]) for d in DETECTORS}
        only_new = sum(1 for _, s in caught if not set(s["fired"]) & {"encode", "decode"})
        lines += ["## Which checks caught them", "", "| check | what it compares | bugs caught |", "|---|---|---:|"]
        lines += [f"| {d} | {DETECTOR_HELP[d]} | {per[d]} |" for d in DETECTORS]
        lines += ["", f"{only_new} of the {len(caught)} were caught only by the refdecode, wrap, stop or stress checks.", ""]

    lines += ["## Cases", "",
              "Counts are problems found with the bug → with the fix, for each check that fired.", "",
              "| case | date | where | category | result | checks that fired | trigger reproduces |", "|---|---|---|---|---|---|---|"]
    for c, s in scored:
        fired = ", ".join(_fired_cell(d, s["metrics"][d]) for d in s["fired"]) or "-"
        trig = "-" if "trigger" not in s else ("yes" if s["trigger"]["reproduces"] else "**no**")
        exp = "" if s["outcome"] != "caught" or s.get("expected") in s["fired"] else f" (expected {s['expected']})"
        lines.append(f"| {c['id']} | {c.get('date', '')} | {c['kind']} | {c['category']} | {s['outcome']}{exp} | {fired} | {trig} |")
    lines += ["", "## Details", ""]
    for c, s in scored:
        lines.append(f"### {c['id']}: {c['title']}")
        lines.append("")
        if c.get("notes"):
            lines.append(c["notes"])
            lines.append("")
        for side, r in sorted(sides.get(c["id"], {}).items()):
            where = f"{r.get('repo')}@{(r.get('revision') or '')[:10]}" if r.get("repo") else "?"
            eng = r.get("engine_commit", "")
            bh = r.get("bug_header")
            lines.append(f"- {side}: {r.get('status')}, {where}, llama.cpp {eng[:12] if eng != 'current' else 'current build'}"
                         + (f", reference {r.get('reference')}" if r.get("reference") else "")
                         + (f", {bh['key']} = {bh['value']!r}" if bh else ""))
            for i in ((r.get("special") or {}).get("issues") or []):
                if i["severity"] == "high":
                    lines.append(f"  - {i['check']}: {i['detail']}")
            for x in (r.get("examples") or [])[:3]:
                lines.append(f"  - {x['kind']} ({x.get('cause')}): `{x['minimal']!r}`")
            st = r.get("stress") or {}
            for x in (st.get("failures") or [])[:3]:
                lines.append(f"  - stress {x['case']}: {x['kind']} {x.get('error', '')}")
        for n in s["notes"]:
            lines.append(f"- note: {n}")
        lines.append("- sources: " + ", ".join(c["sources"]))
        lines.append("")
    return "\n".join(lines) + "\n"


def list_cases(cases_path: Path = CASES) -> str:
    cases = load_cases(cases_path)
    rows = ["| id | date | where | category | expected check | title |", "|---|---|---|---|---|---|"]
    rows += [f"| {c['id']} | {c.get('date', '')} | {c['kind']} | {c['category']} | {c.get('detector', '-')} | {c['title']} |"
             for c in cases]
    errs = validate(cases)
    return "\n".join(rows) + ("\n\nproblems:\n  " + "\n  ".join(errs) if errs else "") + "\n"
