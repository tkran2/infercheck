"""Adversarial text for tokenizer differential testing.

Two kinds of input:

- **fixed cases**: hand-picked strings that probe one rule each (whitespace runs before a
  word, digit grouping, contractions, normalization, special-token boundaries, ...);
- **generated cases**: seeded random concatenations of fragments drawn from pools of
  characters that tokenizers treat differently, with random separators. The same seed
  always produces the same corpus, so a finding can be reproduced from (seed, index).

Special-token strings are added per tokenizer from its own added_tokens list.
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass

# -- character pools ---------------------------------------------------------------------

WHITESPACE = [" ", "\t", "\n", "\r", "\r\n", "\x0b", "\x0c", "\x85", " ", " ", " ",
              " ", " ", " ", " ", " ", " ", " ", " ", "　"]
INVISIBLE = ["​", "‌", "‍", "⁠", "﻿", "­", "᠎", "͏", "️",
             "︎", "؜", "‎", "‏", "⁦", "⁩"]
DIGITS = ["0", "7", "42", "123", "1234", "12345", "1234567", "0000", "3.14", "1,000", "1e10",
          "١٢٣٤",            # Arabic-Indic digits (Nd)
          "१२३४",            # Devanagari digits (Nd)
          "１２３４",            # fullwidth digits (Nd)
          "\U0001d7ce\U0001d7cf",                # mathematical bold digits (Nd)
          "²³¹",                  # superscripts (No)
          "ⅠⅡⅢ",                  # Roman numerals (Nl)
          "½⅓",                        # vulgar fractions (No)
          "①②",                        # circled digits (No)
          "〇一二"]                  # CJK zero and numerals
LETTERS = ["a", "Z", "hello", "Hello", "HELLO", "parseHTTPResponse", "snake_case_name", "café",
           "naïve", "straße", "STRASSE", "İstanbul", "ı", "ſ", "K", "Å",
           "ﬁle", "ǅ", "ẞ",       # fi ligature, titlecase Dž, capital sharp s
           "да", "αβγ", "שלום", "مرحبا",
           "हिन्दी", "สวัสดี",
           "中文", "日本語", "ひらがな", "カタカナ",
           "한국어", "각",  # Hangul syllables and conjoining jamo
           "ＡＢＣ",                 # fullwidth Latin
           "\U0001d400\U0001d401",               # mathematical bold letters
           "ʰʲ", "々"]             # modifier letters, iteration mark
MARKS = ["́", "̈", "́̈", "̧", "⃝", "ि", "्", "ั",
         "〪", "\U0001f3fb"]              # last: emoji skin-tone modifier (Sk)
PUNCT = ["!", "?", ".", ",", ";", ":", "...", "!?", "'", '"', "`", "-", "--", "—", "–", "_", "__", "(", ")",
         "[", "]", "{", "}", "<", ">", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "~",
         "’", "“", "”", "«", "»", "、", "。", "！", "？",
         "¿", "¡", "…", "·", "•", "§", "€", "£", "¥",
         "™", "©", "°", "→", "≤", "∞"]
CONTRACTIONS = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'RE", "'Ve", "'LL", "'D",
                "’s", "’ll", "'ſ", "n't", "N'T", "o'clock", "rock'n'roll", "'''", "'"]
EMOJI = ["\U0001f600", "\U0001f30d", "\U0001f468‍\U0001f469‍\U0001f467",
         "\U0001f44d\U0001f3fd", "\U0001f1fa\U0001f1f8", "❤️", "\U0001f3f3️‍\U0001f308",
         "1️⃣", "\U0001faf6"]
CONTROL = ["\x00", "\x01", "\x1b", "\x1c", "\x1f", "\x7f", "\x80", "\x9f", "", "\U000f0000",
           "\U0010ffff", "͸", "￾", "￿", "�", "\U000e0041"]  # last: tag character
CODE = ["def f(x):\n    return x\n", "if (a && b) {\n\treturn;\n}", "x = [1, 2, 3]  # comment",
        "<div class=\"a\">", "SELECT * FROM t;", "    \n    \n", "\t\t\tindented", "a\r\nb\r\n",
        "https://example.com/a?b=c&d=e", "foo.bar_baz(1)", "0x1F", "{\"k\": [1, 2]}", "#include <x.h>"]

POOLS = {"ws": WHITESPACE, "inv": INVISIBLE, "dig": DIGITS, "let": LETTERS, "mark": MARKS,
         "punct": PUNCT, "contr": CONTRACTIONS, "emoji": EMOJI, "ctrl": CONTROL, "code": CODE}
POOL_WEIGHTS = {"ws": 5, "inv": 1, "dig": 3, "let": 5, "mark": 2, "punct": 3, "contr": 2, "emoji": 1,
                "ctrl": 1, "code": 1}


@dataclass(frozen=True)
class Case:
    id: str
    text: str
    kind: str  # fixed | generated | special


def fixed_cases() -> list[Case]:
    """One rule per string; each pool element alone, and in the contexts rules care about."""
    out: list[str] = []
    for ws in WHITESPACE:
        for run in (1, 2, 3):
            s = ws * run
            out += [s, "a" + s + "b", s + "word", "word" + s, s + "1", s + "!", s + "́", "1" + s + "2",
                    s + "\n", "a" + s, "x" + s + "é"]
    for d in DIGITS:
        out += [d, "a" + d, d + "a", d + "_", "_" + d, " " + d, d + " ", d + "́", "x" + d + "y",
                d * 3, d + "." + d]
    for n in range(1, 13):
        out += ["9" * n, "x" + "8" * n, "8" * n + "x", "8" * n + "_", "8" * n + " ", "١" * n]
    for c in CONTRACTIONS:
        out += [c, "it" + c, "IT" + c, " " + c, c + "x", "we" + c + "go"]
    for m in MARKS:
        out += [m, "a" + m, " " + m, "  " + m, "1" + m, "!" + m, m + m, "é" + m]
    for p in PUNCT:
        out += [p, " " + p, "  " + p, p * 3, "a" + p + "b", p + "\n\n"]
    for e in EMOJI + CONTROL + INVISIBLE:
        out += [e, " " + e, "a" + e + "b", e * 2, e + " " + e]
    for w in LETTERS:
        out += [w, " " + w, "  " + w, w.upper(), w.lower(), w + "'s", w + "́"]
    for c in CODE:
        out += [c, c * 2]
    out += ["", " ", "a" * 200, " " * 100, "\n" * 50, "ab" * 100, "é" * 100, "中" * 100,
            "".join(chr(c) for c in range(0x20, 0x7f))]
    seen, cases = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            cases.append(Case(f"f{len(cases)}", s, "fixed"))
    return cases


def special_cases(special: list[str]) -> list[Case]:
    """Added-token strings in the positions where matching rules differ."""
    out: list[str] = []
    for tok in special[:60]:
        out += [tok, tok + tok, " " + tok, tok + " ", "a" + tok + "b", "a " + tok + " b", "a  " + tok + "b",
                "\n" + tok + "\n", tok[:-1], tok[1:], tok.upper() if tok.upper() != tok else tok + "x",
                tok + "́", "é" + tok, tok + "1"]
    seen, cases = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            cases.append(Case(f"s{len(cases)}", s, "special"))
    return cases


def generated_cases(n: int, seed: int = 0, special: list[str] | None = None) -> list[Case]:
    """n random strings: 1-8 fragments from weighted pools, joined by random separators."""
    rng = random.Random(seed)
    pools = list(POOLS)
    weights = [POOL_WEIGHTS[p] for p in pools]
    seps = ["", "", "", " ", " ", "  ", "\n", "\t", " ", "　"]
    cases = []
    for i in range(n):
        parts = []
        for _ in range(rng.randint(1, 8)):
            if special and rng.random() < 0.05:
                parts.append(rng.choice(special))
                continue
            pool = POOLS[rng.choices(pools, weights)[0]]
            frag = rng.choice(pool)
            if rng.random() < 0.15:
                frag = frag * rng.randint(2, 5)
            if rng.random() < 0.1:
                frag = unicodedata.normalize(rng.choice(["NFD", "NFKC", "NFKD"]), frag)
            parts.append(frag)
            parts.append(rng.choice(seps))
        cases.append(Case(f"g{seed}-{i}", "".join(parts), "generated"))
    return cases
