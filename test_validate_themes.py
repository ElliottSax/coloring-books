#!/usr/bin/env python3
"""Tests for validate_themes.py.

Run:  python3 test_validate_themes.py

Standard library only: no network, no prompts, no image libraries, and
coloring_book_generator is never imported.  Scratch files live in temporary
directories that are removed afterwards.
"""

import codecs
import contextlib
import copy
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import validate_themes as vt

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "validate_themes.py"
SHIPPED = HERE / "themes" / "themes.json"

# Built with chr() so this file stays pure ASCII.
E_ACUTE = chr(0xE9)
SNOWMAN = chr(0x2603)


# ---------------------------------------------------------------- fixtures

def prompts(tag, n=5):
    """n distinct prompts that trip no check."""
    return [f"{tag} study {i} with ornate detail, adult coloring book page, "
            "black line art on white background" for i in range(n)]


def theme(name, tag, n=5):
    return {"name": name, "prompts": prompts(tag, n)}


def clean():
    """A small themes file with no errors and no warnings."""
    return {"alpha": theme("Alpha Garden", "alpha"), "beta": theme("Beta Forest", "beta")}


def prompt_of_length(n):
    """A clean-looking prompt of exactly n characters (n >= 22)."""
    base = "ornate lotus pattern "
    return base + "x" * (n - len(base))


def shipped():
    with open(SHIPPED, encoding="utf-8") as fh:
        return json.load(fh)


class TempDirCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write(self, content, name="themes.json"):
        """Write str/bytes verbatim, or any other value as JSON."""
        path = self.dir / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        elif isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_text(json.dumps(content, indent=2), encoding="utf-8")
        return path

    def validate(self, content):
        return vt.validate_file(self.write(content))

    def run_main(self, *args):
        """Run vt.main in-process; returns (exit code, stdout)."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = vt.main([str(a) for a in args])
        return code, out.getvalue()

    # -- assertions --

    def matches(self, report, code, level=None, theme=None, where=None):
        return [f for f in report.findings
                if f.code == code
                and (level is None or f.level == level)
                and (theme is None or f.theme == theme)
                and (where is None or f.where == where)]

    def assertFinds(self, report, code, level, theme=None, where=None):
        if not self.matches(report, code, level, theme, where):
            self.fail(f"expected {level} '{code}' (theme={theme}, where={where}); got:\n"
                      + vt.format_report(report))

    def assertNoFinding(self, report, code, theme=None):
        found = self.matches(report, code, theme=theme)
        if found:
            self.fail(f"unexpected '{code}': {found}\n" + vt.format_report(report))

    def assertClean(self, report):
        if report.findings:
            self.fail("expected no findings; got:\n" + vt.format_report(report))

    def assertOnlyAffects(self, report, key):
        """Every finding belongs to theme `key` (others stay clean)."""
        stray = [f for f in report.findings if f.theme != key]
        if stray:
            self.fail(f"findings outside theme {key!r}: {stray}\n" + vt.format_report(report))


# ---------------------------------------------------------- shipped themes

class ShippedThemesTest(TempDirCase):
    def test_default_path_is_the_bundled_file(self):
        self.assertEqual(vt.DEFAULT_THEMES_PATH, SHIPPED)
        self.assertTrue(SHIPPED.is_file())

    def test_shipped_themes_have_no_errors(self):
        report = vt.validate_file(SHIPPED)
        self.assertEqual(report.errors, [], vt.format_report(report))

    def test_shipped_themes_are_warning_free(self):
        # Ratchet: the bundled data is currently clean even in --strict mode.  If
        # a new prompt trips a wording heuristic, reword it (or fix the heuristic).
        report = vt.validate_file(SHIPPED)
        self.assertEqual(report.warnings, [], vt.format_report(report))

    def test_every_theme_is_reported(self):
        data, report = shipped(), vt.validate_file(SHIPPED)
        self.assertGreater(len(data), 0)
        self.assertEqual([t.key for t in report.themes], list(data))
        for info in report.themes:
            self.assertEqual(info.name, data[info.key]["name"])
            self.assertEqual(info.prompt_count, len(data[info.key]["prompts"]))
        self.assertEqual(report.prompt_total, sum(len(t["prompts"]) for t in data.values()))

    def test_validator_agrees_with_the_generator_contract(self):
        # What generate_book() needs, asserted without the validator.
        for key, entry in shipped().items():
            with self.subTest(theme=key):
                self.assertRegex(key, r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")
                self.assertEqual(set(entry), {"name", "prompts"})
                self.assertIsInstance(entry["name"], str)
                self.assertTrue(entry["name"].strip())
                self.assertIsInstance(entry["prompts"], list)
                self.assertTrue(entry["prompts"])
                self.assertTrue(all(isinstance(p, str) and p.strip() for p in entry["prompts"]))

    def test_cli_default_run_passes(self):
        code, out = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertIn("Result: PASS", out)
        for key in shipped():
            self.assertRegex(out, rf"(?m)^  OK\s+{key}\s")


class LimitsTest(unittest.TestCase):
    def test_limits_are_the_documented_ones(self):
        # Derived from the shipped data (see validate_themes.py); changing one is a
        # policy change, so it should be a deliberate edit here too.
        self.assertEqual((vt.MIN_PROMPTS, vt.NAME_WARN_CHARS, vt.NAME_MAX_CHARS,
                          vt.PROMPT_WARN_MIN_CHARS, vt.PROMPT_WARN_MAX_CHARS,
                          vt.PROMPT_MAX_CHARS),
                         (5, 40, 100, 40, 330, 1000))


# ------------------------------------------------------ file-level errors

class FileLevelErrorsTest(TempDirCase):
    def test_missing_file(self):
        report = vt.validate_file(self.dir / "nope.json")
        self.assertFinds(report, "file-not-found", vt.ERROR)

    def test_directory_instead_of_file(self):
        self.assertFinds(vt.validate_file(self.dir), "file-unreadable", vt.ERROR)

    def test_invalid_json_reports_position(self):
        for label, text in [("trailing comma", '{"a": 1,}'),
                            ("unquoted key", "{a: 1}"),
                            ("truncated", '{"a": {"name": "A"'),
                            ("empty file", ""),
                            ("not json", "themes go here")]:
            with self.subTest(label):
                report = self.validate(text)
                self.assertFinds(report, "invalid-json", vt.ERROR)
                self.assertIn("line", report.errors[0].message)

    def test_utf8_bom_is_rejected_like_the_real_loader(self):
        body = '{"a": {"name": "A", "prompts": []}}'
        with self.assertRaises(json.JSONDecodeError):  # what load_themes() would hit
            json.loads(codecs.BOM_UTF8.decode("utf-8") + body)
        report = self.validate(codecs.BOM_UTF8 + body.encode("utf-8"))
        self.assertFinds(report, "invalid-json", vt.ERROR)
        self.assertIn("byte-order mark", report.errors[0].message)

    def test_invalid_utf8(self):
        raw = json.dumps({"a": {"name": "Caf" + E_ACUTE, "prompts": []}},
                         ensure_ascii=False).encode("latin-1")
        self.assertFinds(self.validate(raw), "invalid-encoding", vt.ERROR)

    def test_top_level_must_be_an_object(self):
        for value in ([], [clean()], "text", 5, None, True):
            with self.subTest(value=value):
                self.assertFinds(self.validate(json.dumps(value)),
                                 "top-level-not-object", vt.ERROR)

    def test_empty_object_defines_no_themes(self):
        self.assertFinds(self.validate("{}"), "no-themes", vt.ERROR)

    def test_duplicate_theme_key(self):
        text = ('{"alpha": {"name": "First", "prompts": ["one"]},'
                ' "alpha": {"name": "Second", "prompts": ["two"]}}')
        self.assertEqual(json.loads(text)["alpha"]["name"], "Second")  # first is lost
        report = self.validate(text)
        self.assertFinds(report, "duplicate-key", vt.ERROR, theme="alpha")
        self.assertEqual([t.key for t in report.themes], ["alpha"])

    def test_duplicate_field_inside_a_theme(self):
        text = '{"alpha": {"name": "A", "name": "B", "prompts": ["one"]}}'
        self.assertFinds(self.validate(text), "duplicate-key", vt.ERROR,
                         theme="alpha", where="name")

    def test_pathologically_nested_json_is_reported_not_raised(self):
        for text in ("[" * 200000, '{"a":' * 100000):
            self.assertFinds(self.validate(text), "invalid-json", vt.ERROR)


# ----------------------------------------------- theme structure & types

class ThemeStructureErrorsTest(TempDirCase):
    def broken(self, **overrides):
        """clean() with theme alpha's fields replaced (None deletes a field)."""
        data = clean()
        for field_name, value in overrides.items():
            if value is None and field_name.startswith("del_"):
                del data["alpha"][field_name[4:]]
            else:
                data["alpha"][field_name] = value
        return data

    def test_clean_file_has_no_findings(self):
        self.assertClean(self.validate(clean()))

    def test_keys_must_be_slugs(self):
        for key in ["Alpha", "my theme", "", " alpha", "alpha ", "alpha/beta", "_alpha",
                    "alpha_", "alpha__beta", "-alpha", "caf" + E_ACUTE, "alpha\n", "alpha.v2"]:
            with self.subTest(key=key):
                data = clean()
                data[key] = data.pop("alpha")
                self.assertFinds(self.validate(data), "invalid-key", vt.ERROR, theme=key)

    def test_slug_keys_are_accepted(self):
        for key in ["a", "my_theme", "my-theme", "theme2", "2fast", "a-b_c"]:
            with self.subTest(key=key):
                data = clean()
                data[key] = data.pop("alpha")
                self.assertNoFinding(self.validate(data), "invalid-key")

    def test_non_string_keys_given_through_the_python_api(self):
        # JSON object keys are always strings, but validate_data() accepts any dict.
        report = vt.validate_data({5: theme("Five", "five")})
        self.assertFinds(report, "invalid-key", vt.ERROR, theme="5")

    def test_theme_must_be_an_object(self):
        for value in ([], "text", 5, None, True, [theme("A", "a")]):
            with self.subTest(value=value):
                data = clean()
                data["alpha"] = value
                report = self.validate(data)
                self.assertFinds(report, "theme-not-object", vt.ERROR, theme="alpha")
                self.assertOnlyAffects(report, "alpha")

    def test_missing_required_fields(self):
        report = self.validate(self.broken(del_name=None))
        self.assertFinds(report, "missing-field", vt.ERROR, theme="alpha", where="name")
        report = self.validate(self.broken(del_prompts=None))
        self.assertFinds(report, "missing-field", vt.ERROR, theme="alpha", where="prompts")
        data = clean()
        data["alpha"] = {}
        report = self.validate(data)
        self.assertEqual(len(self.matches(report, "missing-field", vt.ERROR)), 2)

    def test_name_must_be_a_string(self):
        for value in (5, 1.5, None, True, ["Alpha"], {"en": "Alpha"}):
            with self.subTest(value=value):
                report = self.validate(self.broken(name=value))
                self.assertFinds(report, "wrong-type", vt.ERROR, theme="alpha", where="name")
                self.assertOnlyAffects(report, "alpha")

    def test_prompts_must_be_a_list(self):
        # A bare string would be indexed character by character by the generator.
        for value in ("one long prompt string", None, 5, True, {"0": "prompt"}):
            with self.subTest(value=value):
                report = self.validate(self.broken(prompts=value))
                self.assertFinds(report, "wrong-type", vt.ERROR, theme="alpha", where="prompts")
                self.assertOnlyAffects(report, "alpha")

    def test_blank_name(self):
        for value in ("", "   ", "\t\n"):
            with self.subTest(value=value):
                self.assertFinds(self.validate(self.broken(name=value)),
                                 "empty-value", vt.ERROR, theme="alpha", where="name")

    def test_empty_prompts_list(self):
        # generate_book() does prompts[i % len(prompts)] -> ZeroDivisionError.
        report = self.validate(self.broken(prompts=[]))
        self.assertFinds(report, "empty-value", vt.ERROR, theme="alpha", where="prompts")
        self.assertIn("ZeroDivisionError", report.errors[0].message)

    def test_prompt_items_must_be_strings(self):
        for value in (5, 1.5, None, True, ["nested"], {"text": "x"}):
            with self.subTest(value=value):
                items = prompts("alpha")
                items[2] = value
                report = self.validate(self.broken(prompts=items))
                self.assertFinds(report, "wrong-type", vt.ERROR, theme="alpha",
                                 where="prompts[2]")
                self.assertOnlyAffects(report, "alpha")

    def test_blank_prompts(self):
        for value in ("", "  ", "\n\t "):
            with self.subTest(value=value):
                items = prompts("alpha")
                items[1] = value
                self.assertFinds(self.validate(self.broken(prompts=items)),
                                 "empty-value", vt.ERROR, theme="alpha", where="prompts[1]")

    def test_names_that_would_break_the_output_directory(self):
        # generate_book() builds the book directory from the name.
        for value in ("Cats/Dogs", "..\\up", "NUL\x00name"):
            with self.subTest(value=value):
                self.assertFinds(self.validate(self.broken(name=value)),
                                 "unsafe-name", vt.ERROR, theme="alpha", where="name")

    def test_ordinary_punctuation_in_names_is_fine(self):
        for value in ("Zen & Meditation", "Cats: A Study", "Mom's Garden",
                      "Caf" + E_ACUTE + " Dreams"):
            with self.subTest(value=value):
                self.assertClean(self.validate(self.broken(name=value)))

    def test_hard_length_limits(self):
        report = self.validate(self.broken(name="N" * (vt.NAME_MAX_CHARS + 1)))
        self.assertFinds(report, "length", vt.ERROR, theme="alpha", where="name")

        items = prompts("alpha")
        items[0] = prompt_of_length(vt.PROMPT_MAX_CHARS + 1)
        report = self.validate(self.broken(prompts=items))
        self.assertFinds(report, "length", vt.ERROR, theme="alpha", where="prompts[0]")

    def test_duplicate_prompt_within_a_theme_is_an_error(self):
        base = prompts("alpha")[1]
        variants = {"exact": base,
                    "case": base.upper(),
                    "spacing": base.replace(" with ", "   with  "),
                    "trailing period": base + "."}
        for label, dup in variants.items():
            with self.subTest(label):
                items = prompts("alpha")
                items[3] = dup
                report = self.validate(self.broken(prompts=items))
                self.assertFinds(report, "duplicate-prompt", vt.ERROR,
                                 theme="alpha", where="prompts[3]")
                self.assertIn("prompts[1]", report.errors[0].message)

    def test_a_broken_theme_does_not_taint_the_others(self):
        report = self.validate(self.broken(prompts=[]))
        self.assertOnlyAffects(report, "alpha")
        self.assertEqual(report.for_theme("beta"), [])
        self.assertEqual([t.key for t in report.themes], ["alpha", "beta"])


# ------------------------------------------------------------- warnings

class WarningsTest(TempDirCase):
    def with_prompt(self, text, index=0):
        data = clean()
        data["alpha"]["prompts"][index] = text
        return data

    def only_warnings(self, report, code, where=None):
        self.assertFinds(report, code, vt.WARNING, theme="alpha", where=where)
        self.assertEqual(report.errors, [], vt.format_report(report))

    def test_unknown_field(self):
        data = clean()
        data["alpha"]["description"] = "not used by the generator"
        report = self.validate(data)
        self.only_warnings(report, "unknown-field", where="description")

    def test_leading_and_trailing_whitespace(self):
        base = prompts("alpha")[0]
        cases = {"prompt leading": (self.with_prompt(" " + base), "prompts[0]"),
                 "prompt trailing newline": (self.with_prompt(base + "\n"), "prompts[0]")}
        data = clean()
        data["alpha"]["name"] = "  Alpha Garden "
        cases["name"] = (data, "name")
        for label, (data, where) in cases.items():
            with self.subTest(label):
                report = self.validate(data)
                self.only_warnings(report, "whitespace", where=where)
                self.assertIn("leading or trailing", report.warnings[0].message)

    def test_repeated_spaces_tabs_newlines_and_control_chars(self):
        base = prompts("alpha")[0]
        for label, text in [("double space", base.replace(" with ", "  with ")),
                            ("tab", base.replace(" with ", "\twith ")),
                            ("newline", base.replace("detail, ", "detail,\n")),
                            ("control char", base.replace("detail", "de\x07tail"))]:
            with self.subTest(label):
                report = self.validate(self.with_prompt(text))
                self.only_warnings(report, "whitespace", where="prompts[0]")

    def test_soft_length_limits_and_their_boundaries(self):
        for n, expect_warning in [(vt.PROMPT_WARN_MIN_CHARS - 1, True),
                                  (vt.PROMPT_WARN_MIN_CHARS, False),
                                  (vt.PROMPT_WARN_MAX_CHARS, False),
                                  (vt.PROMPT_WARN_MAX_CHARS + 1, True),
                                  (vt.PROMPT_MAX_CHARS, True)]:
            with self.subTest(chars=n):
                report = self.validate(self.with_prompt(prompt_of_length(n)))
                self.assertEqual(report.errors, [], vt.format_report(report))
                self.assertEqual(bool(self.matches(report, "length", vt.WARNING)),
                                 expect_warning, vt.format_report(report))

    def test_name_length_boundary(self):
        for n, expect_warning in [(vt.NAME_WARN_CHARS, False), (vt.NAME_WARN_CHARS + 1, True),
                                  (vt.NAME_MAX_CHARS, True)]:
            with self.subTest(chars=n):
                data = clean()
                data["alpha"]["name"] = "N" * n
                report = self.validate(data)
                self.assertEqual(report.errors, [], vt.format_report(report))
                self.assertEqual(bool(self.matches(report, "length", vt.WARNING, where="name")),
                                 expect_warning)

    def test_few_prompts(self):
        # themes/README.md: "5-8 prompts per theme recommended"; shipped minimum is 5.
        for n in (1, 4):
            with self.subTest(prompts=n):
                data = clean()
                data["alpha"] = theme("Alpha Garden", "alpha", n)
                self.only_warnings(self.validate(data), "few-prompts", where="prompts")
        data = clean()
        data["alpha"] = theme("Alpha Garden", "alpha", 5)
        self.assertNoFinding(self.validate(data), "few-prompts")

    def test_trailing_punctuation(self):
        # The generator appends ", with extra fine details" etc.
        for suffix in (".", ",", ";", ":", "!", "?", "-", "..."):
            with self.subTest(suffix=suffix):
                text = prompts("alpha")[0] + suffix
                self.only_warnings(self.validate(self.with_prompt(text)),
                                   "trailing-punctuation", where="prompts[0]")

    def test_duplicate_prompt_across_themes_is_only_a_warning(self):
        data = clean()
        data["beta"]["prompts"][2] = data["alpha"]["prompts"][4].swapcase()
        report = self.validate(data)
        self.assertFinds(report, "duplicate-prompt", vt.WARNING, theme="beta", where="prompts[2]")
        self.assertIn("alpha.prompts[4]", report.warnings[0].message)
        self.assertEqual(report.errors, [], vt.format_report(report))

    def test_duplicate_theme_names(self):
        data = clean()
        data["beta"]["name"] = "  ALPHA   garden "
        report = self.validate(data)
        self.assertFinds(report, "duplicate-name", vt.WARNING, theme="beta", where="name")
        self.assertEqual(report.errors, [], vt.format_report(report))

    def test_conflicting_wording_is_a_warning_with_location(self):
        text = "a vivid blue dragon with soft gradient shading, adult coloring book page"
        report = self.validate(self.with_prompt(text, index=2))
        self.only_warnings(report, "conflicting-wording", where="prompts[2]")
        message = report.warnings[0].message
        for word in ("blue", "gradient", "shading"):
            self.assertIn(word, message)

    def test_negated_wording_is_not_a_conflict(self):
        for text in ("owl, adult coloring book page, black line art, no shading",
                     "owl, coloring page, clean lines, without gradients or gray tones",
                     "owl, coloring page, drawn in a way that is not realistic"):
            with self.subTest(text):
                self.assertClean(self.validate(self.with_prompt(text)))

    def test_conflicting_wording_table(self):
        flagged = {
            "a blue dragon, coloring page": "blue",
            "colorful mandala, adult coloring book": "colorful",
            "mandala with soft gradient shading": "gradient",
            "gray wolf, coloring page": "gray",
            "grayscale owl": "grayscale",
            "photorealistic lion, coloring page": "photorealistic",
            "photo-realistic lion": "photo-realistic",
            "realistic horse portrait": "realistic",
            "3D render of a castle": "3d render",
            "owl on a black background": "black background",
            "castle on dark background": "dark background",
            "full color rose, coloring page": "full color",
            "rainbow over a garden": "rainbow",
            "watercolor koi": "watercolor",
            "golden dragon, line art": "golden",
            "owl with shadows beneath": "shadows",
        }
        for text, word in flagged.items():
            with self.subTest(text):
                self.assertIn(word, [w for _, w in vt.find_conflicting_wording(text)])
        fine = [
            "mandala, black line art on white, coloring page style, no shading",
            "owl, clean lines, without gradients",
            "owl, no shading or gradients",
            "owl, no shading, no gradients, no gray",
            "non-shaded owl",
            "avoid color, coloring page",
            "adult coloring book page, easy to color",
            "black cat sitting with ornate fur patterns",
            "underwater coral reef with fish",
            "rose garden with lilies and cherry blossom",
            "goldfish in a bowl",
            "redwood forest, greenhouse and a bluebird",
            "3D geometric tessellation pattern",
            "pure black lines on pure white background",
        ]
        for text in fine:
            with self.subTest(text):
                self.assertEqual(vt.find_conflicting_wording(text), [])


# --------------------------------------------------- strict + exit codes

class StrictAndExitCodesTest(TempDirCase):
    def warn_only(self):
        data = clean()
        data["alpha"]["description"] = "extra field -> warning only"
        return data

    def error_only(self):
        data = clean()
        data["alpha"]["prompts"] = []
        return data

    def test_clean_file_exits_zero_with_or_without_strict(self):
        path = self.write(clean())
        for extra in ([], ["--strict"]):
            with self.subTest(extra=extra):
                code, out = self.run_main(path, *extra)
                self.assertEqual(code, 0, out)
                self.assertIn("Result: PASS", out)
                self.assertIn("0 errors, 0 warnings", out)

    def test_warnings_pass_normally_and_fail_under_strict(self):
        path = self.write(self.warn_only())
        code, out = self.run_main(path)
        self.assertEqual(code, 0, out)
        self.assertIn("PASS (with warnings", out)
        code, out = self.run_main(path, "--strict")
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL (--strict", out)

    def test_errors_fail_with_or_without_strict(self):
        path = self.write(self.error_only())
        for extra in ([], ["--strict"]):
            with self.subTest(extra=extra):
                code, out = self.run_main(path, *extra)
                self.assertEqual(code, 1, out)
                self.assertIn("Result: FAIL", out)

    def test_strict_flag_may_precede_the_path(self):
        code, _ = self.run_main("--strict", self.write(self.warn_only()))
        self.assertEqual(code, 1)

    def test_exit_code_function(self):
        warn = vt.validate_data(self.warn_only())
        err = vt.validate_data(self.error_only())
        ok = vt.validate_data(clean())
        self.assertEqual([vt.exit_code(r, s) for r in (ok, warn, err) for s in (False, True)],
                         [0, 0, 0, 1, 1, 1])

    def test_file_level_failures_exit_one(self):
        code, out = self.run_main(self.dir / "missing.json")
        self.assertEqual(code, 1)
        self.assertIn("file-not-found", out)

    def test_report_lists_every_theme_with_its_status(self):
        data = clean()
        data["gamma"] = theme("Gamma Sea", "gamma")
        data["gamma"]["extra"] = 1
        data["delta"] = {"name": "Delta Sky", "prompts": []}
        code, out = self.run_main(self.write(data))
        self.assertEqual(code, 1)
        self.assertRegex(out, r"(?m)^  OK\s+alpha\s+Alpha Garden \(5 prompts\)")
        self.assertRegex(out, r"(?m)^  OK\s+beta\s+Beta Forest \(5 prompts\)")
        self.assertRegex(out, r"(?m)^  WARN\s+gamma\s+Gamma Sea \(5 prompts\)")
        self.assertRegex(out, r"(?m)^  FAIL\s+delta\s+Delta Sky \(0 prompts\)")
        self.assertIn("4 themes, 15 prompts checked: 1 error, 1 warning", out)
        # findings are printed under their theme and tagged with the check id
        self.assertRegex(out, r"(?s)gamma.*\[unknown-field\].*delta.*\[empty-value\]")

    def test_odd_keys_and_names_cannot_break_the_one_line_per_theme_layout(self):
        data = {"has\nnewline": theme("Line\nBreak", "odd")}
        code, out = self.run_main(self.write(data))
        lines = out.splitlines()
        self.assertEqual(code, 1)
        # key shown quoted/escaped, name collapsed onto the same single line
        header = [ln for ln in lines
                  if re.match(r"  FAIL\s+'has\\nnewline'\s+Line Break \(5 prompts\)$", ln)]
        self.assertEqual(len(header), 1, out)
        self.assertEqual([ln for ln in lines if ln.startswith(("Break", "newline"))], [], out)

    def test_help_and_bad_usage(self):
        for argv, expected in ((["--help"], 0), (["--bogus"], 2), (["a.json", "b.json"], 2)):
            with self.subTest(argv=argv):
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        vt.main(argv)
                self.assertEqual(caught.exception.code, expected)


# ------------------------------------------- real process, real exit codes

class SubprocessTest(TempDirCase):
    def run_script(self, *args, cwd=None, env_extra=None):
        env = dict(os.environ, **(env_extra or {}))
        return subprocess.run([sys.executable, str(SCRIPT)] + [str(a) for a in args],
                              capture_output=True, cwd=cwd or self.dir, env=env,
                              encoding="utf-8", errors="replace", timeout=60)

    def test_exit_codes(self):
        warn = clean()
        warn["alpha"]["note"] = "extra"
        bad = clean()
        bad["beta"]["name"] = 7
        for label, data, extra, expected in [("clean", clean(), [], 0),
                                             ("clean strict", clean(), ["--strict"], 0),
                                             ("warnings", warn, [], 0),
                                             ("warnings strict", warn, ["--strict"], 1),
                                             ("errors", bad, [], 1),
                                             ("errors strict", bad, ["--strict"], 1)]:
            with self.subTest(label):
                result = self.run_script(self.write(data), *extra)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                self.assertEqual(result.stderr, "")

    def test_default_path_does_not_depend_on_cwd(self):
        result = self.run_script(cwd=self.dir)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(str(SHIPPED), result.stdout)
        self.assertIn("Result: PASS", result.stdout)

    def test_missing_file_is_a_clean_failure_not_a_traceback(self):
        result = self.run_script(self.dir / "absent.json")
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stdout + result.stderr)
        self.assertIn("file-not-found", result.stdout)

    def test_unprintable_characters_never_crash_the_report(self):
        data = {"cafe": {"name": f"Caf{E_ACUTE} {SNOWMAN}", "prompts": [
            f"caf{E_ACUTE} scene with a snowman {SNOWMAN}, adult coloring book page, "
            "black line art"]}}
        result = self.run_script(self.write(data), env_extra={"PYTHONIOENCODING": "ascii"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertTrue(result.stdout.isascii())

    def test_script_imports_only_the_standard_library(self):
        probe = (
            "import sys; sys.path.insert(0, sys.argv[1]); import validate_themes; "
            "bad = {'cv2', 'PIL', 'numpy', 'requests', 'reportlab', 'coloring_book_generator'}"
            " & set(sys.modules); sys.exit(1 if bad else 0)")
        result = subprocess.run([sys.executable, "-c", probe, str(HERE)],
                                capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


# --------------------------------------- each error class on the real data

class MutatedShippedDataTest(TempDirCase):
    """Break one thing in a copy of the real themes.json and expect exactly that."""

    def mutated(self, change):
        data = copy.deepcopy(shipped())
        change(data)
        return self.validate(data)

    def test_baseline_is_clean(self):
        self.assertClean(self.mutated(lambda d: None))

    def test_missing_prompts(self):
        report = self.mutated(lambda d: d["zen"].pop("prompts"))
        self.assertFinds(report, "missing-field", vt.ERROR, theme="zen", where="prompts")
        self.assertOnlyAffects(report, "zen")

    def test_wrong_type_name(self):
        def change(d):
            d["food"]["name"] = 5
        report = self.mutated(change)
        self.assertFinds(report, "wrong-type", vt.ERROR, theme="food", where="name")
        self.assertOnlyAffects(report, "food")

    def test_blank_prompt(self):
        def change(d):
            d["ocean"]["prompts"][1] = "   "
        report = self.mutated(change)
        self.assertFinds(report, "empty-value", vt.ERROR, theme="ocean", where="prompts[1]")
        self.assertOnlyAffects(report, "ocean")

    def test_duplicate_prompt_in_one_theme(self):
        def change(d):
            d["space"]["prompts"].append(d["space"]["prompts"][0])
        report = self.mutated(change)
        self.assertFinds(report, "duplicate-prompt", vt.ERROR, theme="space", where="prompts[8]")
        self.assertOnlyAffects(report, "space")

    def test_prompt_copied_between_themes(self):
        def change(d):
            d["celtic"]["prompts"][0] = d["mandalas"]["prompts"][0]
        report = self.mutated(change)
        self.assertFinds(report, "duplicate-prompt", vt.WARNING, theme="celtic", where="prompts[0]")
        self.assertEqual(report.errors, [])

    def test_key_that_is_not_a_slug(self):
        def change(d):
            d["Zen Garden"] = d.pop("zen")
        report = self.mutated(change)
        self.assertFinds(report, "invalid-key", vt.ERROR, theme="Zen Garden")
        self.assertOnlyAffects(report, "Zen Garden")

    def test_conflicting_wording_introduced_into_a_prompt(self):
        def change(d):
            d["flowers"]["prompts"][0] += ", vibrant colors with soft gradients"
        report = self.mutated(change)
        self.assertFinds(report, "conflicting-wording", vt.WARNING, theme="flowers",
                         where="prompts[0]")
        self.assertEqual(report.errors, [])
        self.assertOnlyAffects(report, "flowers")

    def test_whitespace_introduced_into_a_prompt(self):
        def change(d):
            d["japanese"]["prompts"][3] = d["japanese"]["prompts"][3] + " "
        report = self.mutated(change)
        self.assertFinds(report, "whitespace", vt.WARNING, theme="japanese", where="prompts[3]")
        self.assertOnlyAffects(report, "japanese")


if __name__ == "__main__":
    unittest.main()
