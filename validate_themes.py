#!/usr/bin/env python3
"""Validate a coloring-book themes file (default: themes/themes.json).

Usage:
    python3 validate_themes.py [path/to/themes.json] [--strict]

Prints one report block per theme.  Exit status: 0 when clean, 1 when there are
errors (or, with --strict, warnings), 2 for a bad command line.

The checks follow how coloring_book_generator.py consumes the file.  Its
load_themes() is a bare json.load() with no validation, so a bad entry only
shows up later as a KeyError / TypeError / ZeroDivisionError halfway through a
book run.  generate_book() uses exactly two fields per theme: "name" (the book
title, also used in the output directory name) and "prompts" (page i uses
prompts[i % len(prompts)] with a ", with ..." variation appended).  This script
mirrors that contract instead of importing the generator, which pulls in
cv2/PIL and does work at import time.  Standard library only.

ERRORS - the generator would fail, or the data is ambiguous
    file-not-found, file-unreadable, invalid-encoding, invalid-json (incl. a
    UTF-8 BOM, which the loader rejects), top-level-not-object, no-themes,
    duplicate-key (json.load silently keeps the last one), invalid-key (not a
    lowercase slug), theme-not-object, missing-field, wrong-type, empty-value
    (blank name/prompt, empty prompts list), unsafe-name (path separator in the
    name), length (hard limits), duplicate-prompt (within one theme)

WARNINGS - quality heuristics, fatal only with --strict
    unknown-field, whitespace, length (soft limits), few-prompts,
    trailing-punctuation, conflicting-wording, duplicate-prompt (across
    themes), duplicate-name
"""

from __future__ import annotations

import argparse
import codecs
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Same location coloring_book_generator.load_themes() uses by default.
DEFAULT_THEMES_PATH = Path(__file__).resolve().parent / "themes" / "themes.json"

ERROR = "error"
WARNING = "warning"

# Fields generate_book() reads from every theme.  Anything else is ignored.
REQUIRED_FIELDS = ("name", "prompts")

# Theme keys become `--theme` values and appear in output file names.
_SLUG_RE = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*")

# --- Length bounds ----------------------------------------------------------
# Derived from the 17 shipped themes (120 prompts):
#   name     12-19 chars
#   prompts  5-8 per theme (themes/README.md: "5-8 prompts per theme recommended")
#            81-165 chars each (11-23 words)
# Soft limits (warnings) sit at roughly half the shortest / twice the longest
# shipped value.  Hard limits (errors) are where things actually break:
#   * name -> directory name; 100 chars leaves room for the timestamp / cover
#     prefix inside a 255-byte file name.
#   * prompt -> the Pollinations backend sends it in a GET URL, percent-encoded
#     (about 1.35x) plus ~330 chars of appended style text.  1000 chars keeps
#     that URL under a conservative 2000-character limit.
NAME_WARN_CHARS = 40
NAME_MAX_CHARS = 100
MIN_PROMPTS = 5
PROMPT_WARN_MIN_CHARS = 40
PROMPT_WARN_MAX_CHARS = 330
PROMPT_MAX_CHARS = 1000

# generate_book() appends ", with extra fine details" etc. to each prompt, so a
# prompt that already ends in punctuation produces ",," or ".," in the request.
_TRAILING_PUNCTUATION = ".,;:!?-"

# Whitespace other than a single plain space, a run of spaces, or a control char.
_ODD_WHITESPACE_RE = re.compile(r"[^\S ]|  |[\x00-\x1f\x7f]")

# --- Wording that fights black-and-white line art ---------------------------
# themes/README.md DON'Ts: color names, shading/gradient keywords, photo-
# realistic prompts.  Matching is by whole word and skips negated uses such as
# "no shading" (which several shipped prompts contain on purpose).  This is a
# heuristic: color names that are part of a proper name ("blue whale") are
# flagged too, hence warning level.
_WORDING_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("color name", re.compile(
        r"\b(?:red|orange|yellow|green|blue|purple|pink|brown|magenta|cyan|"
        r"turquoise|teal|crimson|scarlet|maroon|gold|golden|silver|beige|indigo)\b",
        re.I)),
    ("color wording", re.compile(
        r"\b(?:colou?rful|colou?red|multi-?colou?r(?:ed)?|full[- ]colou?r|in colou?r|"
        r"vibrant|rainbow|pastel|neon|water-?colou?r)\b",
        re.I)),
    ("shading/gray tone", re.compile(
        r"\b(?:shading|shaded|shadows?|gradients?|gr[ae]y(?:scale)?|halftones?)\b",
        re.I)),
    ("photographic/painted style", re.compile(
        r"\b(?:photo-?realis(?:tic|m)|hyper-?realistic|photograph\w*|photos?|realistic|"
        r"realism|3d render(?:ing)?s?|oil painting|airbrush(?:ed)?|cgi)\b",
        re.I)),
    ("dark/filled background", re.compile(
        r"\b(?:black|dark) background\b|\bsolid black\b|\bblack fill(?:s|ed)?\b",
        re.I)),
)
_NEGATORS = frozenset({"no", "not", "non", "without", "avoid", "avoiding", "never"})


# ==================== Results ====================

@dataclass(frozen=True)
class Finding:
    """One problem.  `code` is a stable id; `theme` is None for file-level ones."""
    level: str
    code: str
    message: str
    theme: Optional[str] = None
    where: str = ""


@dataclass(frozen=True)
class ThemeInfo:
    """What the report header shows for a theme (None = unusable value)."""
    key: str
    name: Optional[str]
    prompt_count: Optional[int]


@dataclass
class Report:
    source: str
    themes: List[ThemeInfo] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.level == ERROR]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.level == WARNING]

    @property
    def prompt_total(self) -> int:
        return sum(t.prompt_count or 0 for t in self.themes)

    def for_theme(self, key: Optional[str]) -> List[Finding]:
        return [f for f in self.findings if f.theme == key]


# ==================== Loading ====================

class _JsonObject(dict):
    """A dict that remembers keys which appeared more than once in the source.

    json.load() keeps the last value for a repeated key without complaint, so
    a pasted-over theme would otherwise vanish silently.
    """
    duplicates: Tuple[str, ...] = ()


def _object_hook(pairs: List[Tuple[str, Any]]) -> _JsonObject:
    obj = _JsonObject()
    duplicates = []
    for key, value in pairs:
        if key in obj:
            duplicates.append(key)
        obj[key] = value  # last one wins, exactly like the real loader
    obj.duplicates = tuple(dict.fromkeys(duplicates))
    return obj


def validate_file(path: Any) -> Report:
    """Read `path` the way the generator does (UTF-8 JSON) and validate it."""
    path = Path(path)
    report = Report(str(path))

    def fail(code: str, message: str) -> Report:
        report.findings.append(Finding(ERROR, code, message))
        return report

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return fail("file-not-found", f"themes file not found: {path}")
    except OSError as exc:
        return fail("file-unreadable", f"cannot read {path}: {exc.strerror or exc}")

    if raw.startswith(codecs.BOM_UTF8):
        return fail("invalid-json",
                    "file starts with a UTF-8 byte-order mark, which json.load() "
                    "rejects; save it as UTF-8 without BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return fail("invalid-encoding",
                    f"not valid UTF-8 (byte offset {exc.start}: {exc.reason}); "
                    "themes/README.md specifies UTF-8")
    try:
        data = json.loads(text, object_pairs_hook=_object_hook)
    except json.JSONDecodeError as exc:
        return fail("invalid-json", f"line {exc.lineno}, column {exc.colno}: {exc.msg}")
    except RecursionError:
        return fail("invalid-json", "nesting is too deep")

    return validate_data(data, source=str(path))


# ==================== Checks ====================

def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _excerpt(text: str, limit: int = 50) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit - 3] + "..."


def _show_key(key: str) -> str:
    """A key as shown in the report; odd ones (spaces, newlines) are quoted so
    they cannot break the one-line-per-theme layout."""
    return key if key.isprintable() and key == key.strip() and " " not in key else repr(key)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _normalise(text: str) -> str:
    """Key for duplicate detection: ignores case, spacing and trailing punctuation."""
    return " ".join(text.split()).casefold().rstrip(" " + _TRAILING_PUNCTUATION)


def _negated(text: str, start: int) -> bool:
    """True if a negator ("no", "without", ...) precedes `start` within 3 words."""
    clause = re.split(r"[,;:.!?()]", text[:start])[-1]
    return any(w in _NEGATORS for w in re.findall(r"[a-z]+", clause.lower())[-3:])


def find_conflicting_wording(text: str) -> List[Tuple[str, str]]:
    """Return (category, matched word) pairs that clash with B&W line art."""
    hits: List[Tuple[str, str]] = []
    for label, pattern in _WORDING_RULES:
        for match in pattern.finditer(text):
            if not _negated(text, match.start()):
                hits.append((label, match.group(0).lower()))
    return list(dict.fromkeys(hits))


class _Validator:
    """Runs every check, keeping the state needed to spot cross-theme duplicates."""

    def __init__(self, source: str) -> None:
        self.report = Report(source)
        self._seen_prompts: Dict[str, Tuple[str, int]] = {}
        self._seen_names: Dict[str, str] = {}

    def add(self, level: str, code: str, message: str,
            theme: Optional[str] = None, where: str = "") -> None:
        self.report.findings.append(Finding(level, code, message, theme, where))

    # ---- file / theme level ----

    def run(self, data: Any) -> None:
        if not isinstance(data, dict):
            self.add(ERROR, "top-level-not-object",
                     "top level must be an object mapping theme keys to themes, "
                     f"got {_json_type(data)}")
            return
        for key in getattr(data, "duplicates", ()):
            self.add(ERROR, "duplicate-key",
                     "theme key appears more than once; json.load() silently keeps "
                     "only the last one", key)
        if not data:
            self.add(ERROR, "no-themes", "file defines no themes")
            return
        for key, value in data.items():
            self._theme(key, value)

    def _theme(self, key: Any, value: Any) -> None:
        if not (isinstance(key, str) and _SLUG_RE.fullmatch(key)):
            self.add(ERROR, "invalid-key",
                     f"key {key!r} is not a slug; use lowercase letters, digits, "
                     'and single "-" or "_" separators (e.g. "my_theme")', str(key), "key")
        key = str(key)

        if not isinstance(value, dict):
            self.report.themes.append(ThemeInfo(key, None, None))
            self.add(ERROR, "theme-not-object",
                     'theme must be an object with "name" and "prompts", '
                     f"got {_json_type(value)}", key)
            return

        name, prompts = value.get("name"), value.get("prompts")
        self.report.themes.append(ThemeInfo(
            key,
            name if isinstance(name, str) else None,
            len(prompts) if isinstance(prompts, list) else None))

        for dup in getattr(value, "duplicates", ()):
            self.add(ERROR, "duplicate-key",
                     f'field "{dup}" appears more than once; only the last value is used',
                     key, dup)
        for required in REQUIRED_FIELDS:
            if required not in value:
                self.add(ERROR, "missing-field", f'missing required field "{required}"',
                         key, required)
        for extra in value:
            if extra not in REQUIRED_FIELDS:
                self.add(WARNING, "unknown-field",
                         f'unknown field "{extra}" is ignored by the generator', key, extra)

        if "name" in value:
            self._name(key, name)
        if "prompts" in value:
            self._prompts(key, prompts)

    # ---- name ----

    def _name(self, key: str, name: Any) -> None:
        if not isinstance(name, str):
            self.add(ERROR, "wrong-type",
                     f'"name" must be a string, got {_json_type(name)}', key, "name")
            return
        text = name.strip()
        if not text:
            self.add(ERROR, "empty-value", '"name" is blank', key, "name")
            return
        if any(ch in name for ch in "/\\\x00"):
            self.add(ERROR, "unsafe-name",
                     "contains a path separator or NUL, but the generator uses the name "
                     "in the book's output directory name", key, "name")
        self._whitespace(name, key, "name")
        if len(text) > NAME_MAX_CHARS:
            self.add(ERROR, "length",
                     f"{len(text)} chars is too long for a title / directory name "
                     f"(limit {NAME_MAX_CHARS})", key, "name")
        elif len(text) > NAME_WARN_CHARS:
            self.add(WARNING, "length",
                     f"{len(text)} chars is unusually long for a name (shipped names are "
                     f"12-19; warns above {NAME_WARN_CHARS}) - is a prompt pasted here?",
                     key, "name")
        first = self._seen_names.setdefault(" ".join(text.split()).casefold(), key)
        if first != key:
            self.add(WARNING, "duplicate-name",
                     f"same name as theme {_show_key(first)}; the books would share a title",
                     key, "name")

    # ---- prompts ----

    def _prompts(self, key: str, prompts: Any) -> None:
        if not isinstance(prompts, list):
            self.add(ERROR, "wrong-type",
                     f'"prompts" must be an array of strings, got {_json_type(prompts)}',
                     key, "prompts")
            return
        if not prompts:
            self.add(ERROR, "empty-value",
                     '"prompts" is empty; the generator computes prompts[i % len(prompts)] '
                     "and would raise ZeroDivisionError", key, "prompts")
            return
        for index, prompt in enumerate(prompts):
            self._prompt(key, index, prompt)
        if len(prompts) < MIN_PROMPTS:
            self.add(WARNING, "few-prompts",
                     f"only {_count(len(prompts), 'prompt')}; {MIN_PROMPTS}-8 are recommended "
                     "because pages cycle through them", key, "prompts")

    def _prompt(self, key: str, index: int, prompt: Any) -> None:
        where = f"prompts[{index}]"
        if not isinstance(prompt, str):
            self.add(ERROR, "wrong-type",
                     f"prompt must be a string, got {_json_type(prompt)}", key, where)
            return
        text = prompt.strip()
        if not text:
            self.add(ERROR, "empty-value", "prompt is blank", key, where)
            return

        self._whitespace(prompt, key, where)
        if len(text) > PROMPT_MAX_CHARS:
            self.add(ERROR, "length",
                     f"{len(text)} chars is too long (limit {PROMPT_MAX_CHARS}; the "
                     "Pollinations backend sends prompts in a GET URL)", key, where)
        elif len(text) > PROMPT_WARN_MAX_CHARS:
            self.add(WARNING, "length",
                     f"{len(text)} chars is unusually long (shipped prompts are 81-165; "
                     f"warns above {PROMPT_WARN_MAX_CHARS}); the line-art instructions the "
                     "generator appends may be ignored", key, where)
        elif len(text) < PROMPT_WARN_MIN_CHARS:
            self.add(WARNING, "length",
                     f"{len(text)} chars is unusually short/vague (shipped prompts are "
                     f"81-165; warns below {PROMPT_WARN_MIN_CHARS})", key, where)
        if text[-1] in _TRAILING_PUNCTUATION:
            self.add(WARNING, "trailing-punctuation",
                     f'ends with "{text[-1]}", but the generator appends ", with ..." '
                     f'variations, so the request would contain "{text[-1]},"', key, where)
        hits = find_conflicting_wording(text)
        if hits:
            by_label: Dict[str, List[str]] = {}
            for label, word in hits:
                by_label.setdefault(label, []).append(word)
            found = "; ".join(f"{label}: {', '.join(words)}" for label, words in by_label.items())
            self.add(WARNING, "conflicting-wording",
                     f"wording at odds with black-and-white line art - {found}", key, where)

        first_key, first_index = self._seen_prompts.setdefault(_normalise(text), (key, index))
        if (first_key, first_index) != (key, index):
            if first_key == key:
                self.add(ERROR, "duplicate-prompt",
                         f"same prompt as prompts[{first_index}] (two pages would match): "
                         f'"{_excerpt(text)}"', key, where)
            else:
                self.add(WARNING, "duplicate-prompt",
                         f"same prompt as {_show_key(first_key)}.prompts[{first_index}]: "
                         f'"{_excerpt(text)}"', key, where)

    def _whitespace(self, text: str, key: str, where: str) -> None:
        if text != text.strip():
            self.add(WARNING, "whitespace", "has leading or trailing whitespace", key, where)
        if _ODD_WHITESPACE_RE.search(text.strip()):
            self.add(WARNING, "whitespace",
                     "contains repeated spaces, tabs, newlines or control characters",
                     key, where)


def validate_data(data: Any, source: str = "<data>") -> Report:
    """Validate already-parsed JSON (a plain dict works; duplicate keys are
    only detectable via validate_file, which parses the text itself)."""
    validator = _Validator(source)
    validator.run(data)
    return validator.report


# ==================== Output ====================

def _status(findings: Sequence[Finding]) -> str:
    if any(f.level == ERROR for f in findings):
        return "FAIL"
    return "WARN" if findings else "OK"


def _finding_lines(findings: Sequence[Finding]) -> List[str]:
    return [f"          {f.level:<7} {f.where or '-':<12} {f.message} [{f.code}]"
            for f in findings]


def format_report(report: Report, strict: bool = False) -> str:
    lines = [f"Validating {report.source}", ""]

    file_findings = report.for_theme(None)
    if file_findings:
        lines.append(f"  {_status(file_findings):<4}  (file)")
        lines.extend(_finding_lines(file_findings))

    width = min(max((len(_show_key(t.key)) for t in report.themes), default=0), 24)
    for info in report.themes:
        findings = report.for_theme(info.key)
        count = "?" if info.prompt_count is None else info.prompt_count
        label = "?" if info.name is None else _excerpt(info.name, 40)
        lines.append(f"  {_status(findings):<4}  {_show_key(info.key):<{width}}  {label} "
                     f"({count} prompt{'' if count == 1 else 's'})")
        lines.extend(_finding_lines(findings))

    errors, warnings = len(report.errors), len(report.warnings)
    lines += ["", f"{_count(len(report.themes), 'theme')}, "
                  f"{_count(report.prompt_total, 'prompt')} checked: "
                  f"{_count(errors, 'error')}, {_count(warnings, 'warning')}"]
    if errors:
        verdict = "FAIL"
    elif warnings and strict:
        verdict = f"FAIL (--strict: {_count(warnings, 'warning')} treated as errors)"
    elif warnings:
        verdict = "PASS (with warnings; --strict makes them fatal)"
    else:
        verdict = "PASS"
    lines.append(f"Result: {verdict}")
    return "\n".join(lines)


def exit_code(report: Report, strict: bool = False) -> int:
    return 1 if report.errors or (strict and report.warnings) else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_themes.py",
        description="Validate a coloring-book themes file against what the generator reads.",
        epilog="Exit status: 0 = valid, 1 = errors (or warnings with --strict), "
               "2 = bad command line.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_THEMES_PATH),
                        help="themes JSON file (default: the bundled themes/themes.json)")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as errors (exit 1)")
    args = parser.parse_args(argv)

    report = validate_file(args.path)
    print(format_report(report, strict=args.strict))
    return exit_code(report, args.strict)


def _make_output_safe() -> None:
    """Report text can echo any character from the file; never die printing it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")


if __name__ == "__main__":
    _make_output_safe()
    sys.exit(main())
