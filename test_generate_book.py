#!/usr/bin/env python3
"""
Characterization tests for ColoringBookGenerator.generate_book()

These tests pin down what generate_book() does - book directory layout,
metadata.json contents, page numbering, prompt selection, the order of calls
and the log messages - so that restructuring the method can be shown to
preserve its behaviour. They were first run against the unrefactored
implementation.

Nothing slow or external is used. The image backend, post-processing,
pre-flight checks, (normally) image validation, the rate-limit sleep and the
clock are replaced with fakes, `requests` is blocked, and every book is
written to a temporary directory.

Run with: python3 test_generate_book.py
"""

import inspect
import io
import json
import logging
import random
import tempfile
import unittest
import warnings
from datetime import datetime
from pathlib import Path
from unittest import mock

import coloring_book_generator as cbg

try:
    import numpy  # noqa: F401  (needed by the real validator when check_binary is on)
    from PIL import Image, ImageDraw
    HAVE_IMAGING = True
except ImportError:
    HAVE_IMAGING = False


SEED = 7  # chosen so every one of the first four pages gets a different non-empty variation
THEME_KEY = "test_theme"
THEME_NAME = "Test Theme: Demo"
THEME_PROMPTS = ["alpha prompt", "beta prompt", "gamma prompt"]

FROZEN_NOW = datetime(2026, 1, 2, 3, 4, 5)
TIMESTAMP = "20260102_030405"
# The title becomes a directory name: spaces turn into "_" and ":" is dropped
DEFAULT_BOOK_DIRNAME = f"Test_Theme_Demo_{TIMESTAMP}"

_VALIDATE_SIGNATURE = inspect.signature(cbg.validate_image_quality)


class _FrozenDatetime(datetime):
    """datetime whose now() never moves, so book directory names are predictable."""

    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW


class _ListHandler(logging.Handler):
    """Collects log records instead of printing them."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def raw(page):
    """Bytes the fake backend returns for its page-th call."""
    return f"RAW-{page}".encode()


def processed(page):
    """Bytes the default fake post-processing turns raw(page) into."""
    return b"PROCESSED:" + raw(page)


def page_file(page):
    return f"page_{page:03d}.png"


def valid_result(name):
    """A passing validate_image_quality() result whose metrics identify the page."""
    return {
        "valid": True,
        "warnings": [],
        "issues": [],
        "metrics": {"width": 2550, "height": 3300, "source": name},
    }


def expected_prompts(num_pages, backend_rng_draws=0):
    """Replay the global RNG the way generate_book() is expected to consume it.

    Page i uses the theme prompts in rotation plus one random.choice() from
    PROMPT_VARIATIONS, drawn just before that page is generated. Any draws the
    (fake) backend makes while generating the page come before the next page's
    variation is chosen.
    """
    rng = random.Random(SEED)
    prompts = []
    for i in range(num_pages):
        prompts.append(THEME_PROMPTS[i % len(THEME_PROMPTS)] + rng.choice(cbg.PROMPT_VARIATIONS))
        for _ in range(backend_rng_draws):
            rng.random()
    return prompts


def make_png(width, height, dpi=(300, 300)):
    """A 1-bit line-art style PNG with a single black diagonal."""
    img = Image.new("1", (width, height), 1)
    ImageDraw.Draw(img).line([(0, 0), (width - 1, height - 1)], fill=0, width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG", dpi=dpi)
    return buf.getvalue()


class _BookTestCase(unittest.TestCase):
    """Fakes everything slow or external around generate_book()."""

    use_fake_validator = True

    def setUp(self):
        # Knobs the individual tests turn
        self.backend_overrides = {}      # 1-based backend call number -> value returned instead of bytes
        self.validation_overrides = {}   # page file name -> validation result to return
        self.preflight_result = True
        self.backend_rng_draws = 0       # global-RNG draws the fake backend makes per call
        self.post_process = lambda image_bytes: b"PROCESSED:" + image_bytes

        # Every call made to a fake, in order
        self.events = []
        self._backend_calls = 0

        tmp = tempfile.TemporaryDirectory(prefix="test_generate_book_")
        self.addCleanup(tmp.cleanup)
        self.output_dir = Path(tmp.name)

        rng_state = random.getstate()
        self.addCleanup(random.setstate, rng_state)

        self._capture_logs()
        self._install_fakes()

    def tearDown(self):
        self.assertFalse(self.network_guard.called, "test attempted network access")

    # ---- fixtures ----

    def _capture_logs(self):
        handler = _ListHandler()
        old_level, old_propagate = cbg.logger.level, cbg.logger.propagate
        cbg.logger.addHandler(handler)
        cbg.logger.setLevel(logging.INFO)
        cbg.logger.propagate = False  # keep the test output quiet

        def restore():
            cbg.logger.removeHandler(handler)
            cbg.logger.setLevel(old_level)
            cbg.logger.propagate = old_propagate

        self.addCleanup(restore)
        self._log_handler = handler

    def _install_fakes(self):
        test = self

        def fake_preflight_checks(force_lineart=False):
            test.events.append(("preflight", force_lineart))
            return test.preflight_result

        def fake_generate_image_bytes(gen, prompt, size=(1536, 1536)):
            test._backend_calls += 1
            test.events.append(("generate", prompt, size))
            for _ in range(test.backend_rng_draws):
                random.random()
            if test._backend_calls in test.backend_overrides:
                return test.backend_overrides[test._backend_calls]
            return raw(test._backend_calls)

        def fake_post_process_image(gen, image_bytes):
            test.events.append(("post_process", image_bytes))
            return test.post_process(image_bytes)

        def fake_validate_image_quality(*args, **kwargs):
            bound = _VALIDATE_SIGNATURE.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            # generate_book() may only choose check_binary; the thresholds stay at their defaults
            test.assertEqual(arguments["min_width"], cbg.MIN_PRINT_WIDTH)
            test.assertEqual(arguments["min_height"], cbg.MIN_PRINT_HEIGHT)
            test.assertEqual(arguments["required_dpi"], cbg.REQUIRED_DPI)
            path = arguments["image_path"]
            # the image must already be on disk, holding the post-processed bytes
            test.events.append(("validate", path.name, arguments["check_binary"], path.read_bytes()))
            if path.name in test.validation_overrides:
                return test.validation_overrides[path.name]
            return valid_result(path.name)

        def fake_sleep(seconds):
            test.events.append(("sleep", seconds))

        patches = [
            mock.patch.object(cbg, "preflight_checks", fake_preflight_checks),
            mock.patch.object(cbg.ColoringBookGenerator, "_generate_image_bytes", fake_generate_image_bytes),
            mock.patch.object(cbg.ColoringBookGenerator, "_post_process_image", fake_post_process_image),
            mock.patch.object(cbg.time, "sleep", fake_sleep),
            mock.patch.object(cbg, "datetime", _FrozenDatetime),
            mock.patch.dict(cbg.THEMES, {THEME_KEY: {"name": THEME_NAME, "prompts": THEME_PROMPTS}}, clear=True),
            mock.patch("builtins.input", side_effect=AssertionError("generate_book() must never prompt for input")),
        ]
        if self.use_fake_validator:
            patches.append(mock.patch.object(cbg, "validate_image_quality", fake_validate_image_quality))
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        # No test may reach the network (checked again in tearDown, since callers may swallow the error)
        guard = mock.patch("requests.sessions.Session.request",
                           side_effect=AssertionError("network access attempted"))
        self.network_guard = guard.start()
        self.addCleanup(guard.stop)

    # ---- helpers ----

    @property
    def logs(self):
        return [(r.levelname, r.getMessage()) for r in self._log_handler.records]

    def events_of(self, kind):
        return [e for e in self.events if e[0] == kind]

    def make_generator(self, **kwargs):
        return cbg.ColoringBookGenerator(output_dir=str(self.output_dir), **kwargs)

    def run_book(self, num_pages=3, book_title=None, gen=None, theme=THEME_KEY):
        """Seed the RNG and generate a book of the test theme."""
        gen = gen or self.make_generator()
        random.seed(SEED)
        return gen.generate_book(theme=theme, num_pages=num_pages, book_title=book_title)

    def read_metadata(self, book_dir):
        text = (book_dir / "metadata.json").read_text()
        return text, json.loads(text)

    @staticmethod
    def names_in(path):
        return sorted(p.name for p in path.iterdir())


class GenerateBookCharacterizationTests(_BookTestCase):

    def test_book_layout_metadata_and_call_sequence(self):
        """4 pages from a 3-prompt theme (so the prompts wrap around)."""
        prompts = expected_prompts(4)
        # Guard against a degenerate seed: variations must really show up in the prompts
        self.assertTrue(any(p not in THEME_PROMPTS for p in prompts))

        book_dir = self.run_book(num_pages=4)

        # directory layout
        self.assertIsInstance(book_dir, Path)
        self.assertEqual(book_dir, self.output_dir / DEFAULT_BOOK_DIRNAME)
        self.assertEqual(self.names_in(self.output_dir), [DEFAULT_BOOK_DIRNAME])
        self.assertEqual(self.names_in(book_dir), ["images", "metadata.json"])
        images_dir = book_dir / "images"
        self.assertEqual(self.names_in(images_dir), [page_file(i) for i in range(1, 5)])
        for i in range(1, 5):
            self.assertEqual((images_dir / page_file(i)).read_bytes(), processed(i))

        # metadata.json
        expected = {
            "title": THEME_NAME,
            "theme": THEME_KEY,
            "pages": 4,
            "generated": 4,
            "timestamp": TIMESTAMP,
            "pages_data": [
                {
                    "page": i,
                    "prompt": prompts[i - 1],
                    "file": page_file(i),
                    "quality_metrics": valid_result(page_file(i))["metrics"],
                }
                for i in range(1, 5)
            ],
        }
        text, metadata = self.read_metadata(book_dir)
        self.assertEqual(metadata, expected)
        self.assertEqual(text, json.dumps(expected, indent=2))  # also pins key order and layout
        self.assertEqual([p.split(",")[0] for p in prompts],
                         ["alpha prompt", "beta prompt", "gamma prompt", "alpha prompt"])

        # pre-flight once, then for every page: generate -> post-process -> validate -> sleep
        expected_events = [("preflight", False)]
        for i in range(1, 5):
            expected_events += [
                ("generate", prompts[i - 1], (1536, 1536)),
                ("post_process", raw(i)),
                ("validate", page_file(i), False, processed(i)),
                ("sleep", 2),
            ]
        self.assertEqual(self.events, expected_events)

    def test_prompt_variation_is_drawn_per_page_just_before_generation(self):
        """A backend that uses the global RNG must see the same stream as before.

        The variation for page N is drawn after page N-1 was generated, not
        up front for the whole book.
        """
        prompts = expected_prompts(4, backend_rng_draws=1)
        self.assertNotEqual(prompts, expected_prompts(4))  # the seed must tell the two orders apart
        self.backend_rng_draws = 1

        book_dir = self.run_book(num_pages=4)

        _, metadata = self.read_metadata(book_dir)
        self.assertEqual([p["prompt"] for p in metadata["pages_data"]], prompts)
        self.assertEqual([e[1] for e in self.events_of("generate")], prompts)

    def test_custom_title_names_directory_and_metadata(self):
        book_dir = self.run_book(num_pages=1, book_title="My Book: Vol 2")
        self.assertEqual(book_dir.name, f"My_Book_Vol_2_{TIMESTAMP}")
        _, metadata = self.read_metadata(book_dir)
        self.assertEqual(metadata["title"], "My Book: Vol 2")
        self.assertEqual(metadata["theme"], THEME_KEY)

    def test_empty_title_falls_back_to_theme_name(self):
        book_dir = self.run_book(num_pages=1, book_title="")
        self.assertEqual(book_dir.name, DEFAULT_BOOK_DIRNAME)
        _, metadata = self.read_metadata(book_dir)
        self.assertEqual(metadata["title"], THEME_NAME)

    def test_positional_arguments(self):
        random.seed(SEED)
        book_dir = self.make_generator().generate_book(THEME_KEY, 2, "Positional Title")
        self.assertEqual(book_dir.name, f"Positional_Title_{TIMESTAMP}")
        _, metadata = self.read_metadata(book_dir)
        self.assertEqual((metadata["pages"], metadata["generated"]), (2, 2))

    def test_num_pages_defaults_to_thirty(self):
        random.seed(SEED)
        book_dir = self.make_generator().generate_book(THEME_KEY)
        _, metadata = self.read_metadata(book_dir)
        self.assertEqual((metadata["pages"], metadata["generated"]), (30, 30))
        self.assertEqual(len(self.names_in(book_dir / "images")), 30)
        self.assertEqual(self.names_in(book_dir / "images")[-1], "page_030.png")

    def test_zero_pages_writes_an_empty_book(self):
        book_dir = self.run_book(num_pages=0)
        self.assertEqual(self.names_in(book_dir), ["images", "metadata.json"])
        self.assertEqual(self.names_in(book_dir / "images"), [])
        _, metadata = self.read_metadata(book_dir)
        self.assertEqual((metadata["pages"], metadata["generated"], metadata["pages_data"]), (0, 0, []))
        self.assertEqual(self.events, [("preflight", False)])  # no backend call, no sleep
        self.assertEqual(self.logs[-2:], [("INFO", "Generated 0/0 pages"),
                                          ("INFO", f"Book saved to: {book_dir}")])

    def test_backend_failure_skips_only_that_page(self):
        self.backend_overrides = {2: None}

        book_dir = self.run_book(num_pages=3)

        self.assertEqual(self.names_in(book_dir / "images"), [page_file(1), page_file(3)])
        _, metadata = self.read_metadata(book_dir)
        self.assertEqual((metadata["pages"], metadata["generated"]), (3, 2))
        self.assertEqual([p["page"] for p in metadata["pages_data"]], [1, 3])  # numbering keeps its slots
        # nothing is post-processed or validated for the failed page, but the rate-limit sleep still happens
        self.assertEqual([e[1] for e in self.events_of("post_process")], [raw(1), raw(3)])
        self.assertEqual([e[1] for e in self.events_of("validate")], [page_file(1), page_file(3)])
        self.assertEqual(self.events_of("sleep"), [("sleep", 2)] * 3)

    def test_empty_backend_response_counts_as_a_failure(self):
        self.backend_overrides = {2: b""}

        book_dir = self.run_book(num_pages=3)

        self.assertEqual(self.names_in(book_dir / "images"), [page_file(1), page_file(3)])
        self.assertIn(("WARNING", "  ✗ Failed to generate page 2"), self.logs)
        self.assertEqual([e[1] for e in self.events_of("post_process")], [raw(1), raw(3)])
        _, metadata = self.read_metadata(book_dir)
        self.assertEqual([p["page"] for p in metadata["pages_data"]], [1, 3])

    def test_failed_validation_excludes_page_from_metadata(self):
        self.validation_overrides[page_file(2)] = {
            "valid": False,
            "warnings": [],
            "issues": ["CRITICAL: Image too small (10x10), needs 2550x3300 for print"],
            "metrics": {"width": 10, "height": 10},
        }

        book_dir = self.run_book(num_pages=3)

        _, metadata = self.read_metadata(book_dir)
        self.assertEqual((metadata["pages"], metadata["generated"]), (3, 2))
        self.assertEqual([p["page"] for p in metadata["pages_data"]], [1, 3])
        # Current behaviour: the rejected image stays on disk (create_pdf would still pick it up)
        self.assertEqual(self.names_in(book_dir / "images"), [page_file(i) for i in (1, 2, 3)])
        self.assertEqual(self.events_of("sleep"), [("sleep", 2)] * 3)

    def test_validation_warnings_keep_the_page_and_its_metrics(self):
        self.validation_overrides[page_file(2)] = {
            "valid": True,
            "warnings": ["first warning", "second warning"],
            "issues": [],
            "metrics": {"width": 2550, "marker": "warned"},
        }

        book_dir = self.run_book(num_pages=2)

        _, metadata = self.read_metadata(book_dir)
        self.assertEqual(metadata["generated"], 2)
        self.assertEqual(metadata["pages_data"][1]["quality_metrics"], {"width": 2550, "marker": "warned"})

    def test_validation_result_without_metrics_gives_empty_quality_metrics(self):
        self.validation_overrides[page_file(1)] = {"valid": True, "warnings": []}

        book_dir = self.run_book(num_pages=1)

        _, metadata = self.read_metadata(book_dir)
        self.assertEqual(metadata["pages_data"][0]["quality_metrics"], {})

    def test_mixed_outcomes_and_log_messages(self):
        """One book with every kind of page outcome, checking metadata and the exact log lines."""
        self.backend_overrides = {2: None}                            # page 2: backend gives nothing
        self.validation_overrides[page_file(3)] = {                   # page 3: rejected, with issues
            "valid": False,
            "warnings": [],
            "issues": ["CRITICAL: Image too small (100x100), needs 2550x3300 for print", "second issue"],
            "metrics": {"width": 100, "height": 100},
        }
        self.validation_overrides[page_file(4)] = {                   # page 4: accepted, with warnings
            "valid": True,
            "warnings": ["DPI metadata is (72, 72), should be 300 for print", "File size very small (2.1KB)"],
            "issues": [],
            "metrics": {"width": 2550, "marker": "warned"},
        }
        self.validation_overrides[page_file(5)] = {                   # page 5: rejected, no issue list
            "valid": False,
            "error": "Validation failed: boom",
            "warnings": [],
        }
        self.validation_overrides[page_file(6)] = {"valid": True, "warnings": []}   # page 6: no metrics
        prompts = expected_prompts(6)
        book_dir = self.run_book(num_pages=6)

        _, metadata = self.read_metadata(book_dir)
        self.assertEqual(metadata, {
            "title": THEME_NAME,
            "theme": THEME_KEY,
            "pages": 6,
            "generated": 3,
            "timestamp": TIMESTAMP,
            "pages_data": [
                {"page": 1, "prompt": prompts[0], "file": page_file(1),
                 "quality_metrics": valid_result(page_file(1))["metrics"]},
                {"page": 4, "prompt": prompts[3], "file": page_file(4),
                 "quality_metrics": {"width": 2550, "marker": "warned"}},
                {"page": 6, "prompt": prompts[5], "file": page_file(6), "quality_metrics": {}},
            ],
        })
        self.assertEqual(self.names_in(book_dir / "images"), [page_file(i) for i in (1, 3, 4, 5, 6)])

        self.assertEqual(self.logs, [
            ("INFO", f"Generating coloring book: {THEME_NAME}"),
            ("INFO", f"Theme: {THEME_KEY}, Pages: 6"),
            ("INFO", f"Output: {book_dir}"),
            ("INFO", "Generating page 1/6..."),
            ("INFO", "  ✓ Saved and validated: page_001.png"),
            ("INFO", "Generating page 2/6..."),
            ("WARNING", "  ✗ Failed to generate page 2"),
            ("INFO", "Generating page 3/6..."),
            ("ERROR", "  ✗ Quality validation FAILED for page_003.png"),
            ("ERROR", "    - CRITICAL: Image too small (100x100), needs 2550x3300 for print"),
            ("ERROR", "    - second issue"),
            ("ERROR", "  This is a CRITICAL error - image will not print correctly!"),
            ("INFO", "Generating page 4/6..."),
            ("WARNING", "  ⚠ Quality warnings for page_004.png:"),
            ("WARNING", "    - DPI metadata is (72, 72), should be 300 for print"),
            ("WARNING", "    - File size very small (2.1KB)"),
            ("INFO", "  ✓ Saved and validated: page_004.png"),
            ("INFO", "Generating page 5/6..."),
            ("ERROR", "  ✗ Quality validation FAILED for page_005.png"),
            ("ERROR", "  This is a CRITICAL error - image will not print correctly!"),
            ("INFO", "Generating page 6/6..."),
            ("INFO", "  ✓ Saved and validated: page_006.png"),
            ("INFO", "Generated 3/6 pages"),
            ("INFO", f"Book saved to: {book_dir}"),
        ])

    def test_force_lineart_reaches_preflight_and_validation(self):
        self.run_book(num_pages=2, gen=self.make_generator(force_lineart=True))
        self.assertEqual(self.events_of("preflight"), [("preflight", True)])
        self.assertEqual([e[2] for e in self.events_of("validate")], [True, True])

    def test_preflight_failure_aborts_before_touching_disk(self):
        self.preflight_result = False

        with self.assertRaises(RuntimeError) as ctx:
            self.run_book(num_pages=2)

        self.assertEqual(str(ctx.exception), "Pre-flight checks failed - cannot generate book safely")
        self.assertEqual(self.names_in(self.output_dir), [])
        self.assertEqual(self.events, [("preflight", False)])
        self.assertEqual(self.logs, [])

    def test_unknown_theme_raises_before_touching_disk(self):
        with self.assertRaises(ValueError) as ctx:
            self.run_book(theme="no_such_theme")

        self.assertEqual(str(ctx.exception), f"Unknown theme: no_such_theme. Available: ['{THEME_KEY}']")
        self.assertEqual(self.names_in(self.output_dir), [])
        self.assertEqual(self.events, [("preflight", False)])  # pre-flight ran first
        self.assertEqual(self.logs, [])

    def test_preflight_is_checked_before_the_theme(self):
        self.preflight_result = False
        with self.assertRaises(RuntimeError):
            self.run_book(theme="no_such_theme")

    def test_post_processing_error_propagates_and_no_metadata_is_written(self):
        """Current behaviour: a page that raises aborts the whole book (the parallel generator is gentler)."""
        def explode(image_bytes):
            raise RuntimeError("Line art conversion failed: boom")
        self.post_process = explode

        with self.assertRaises(RuntimeError) as ctx:
            self.run_book(num_pages=3)

        self.assertIn("Line art conversion failed", str(ctx.exception))
        self.assertEqual(len(self.events_of("generate")), 1)  # stopped at the first page
        (book_dir,) = self.output_dir.iterdir()
        self.assertEqual(self.names_in(book_dir), ["images"])  # no metadata.json
        self.assertEqual(self.names_in(book_dir / "images"), [])

    def test_missing_output_directory_is_recreated(self):
        out = self.output_dir / "nested" / "out"
        gen = self.make_generator()
        gen.output_dir = out  # as if the directory vanished after the generator was created
        self.assertFalse(out.exists())

        book_dir = self.run_book(num_pages=1, gen=gen)

        self.assertEqual(book_dir, out / DEFAULT_BOOK_DIRNAME)
        self.assertEqual(self.names_in(book_dir), ["images", "metadata.json"])

    def test_second_run_in_the_same_second_reuses_the_directory(self):
        first = self.run_book(num_pages=3)

        second = self.run_book(num_pages=2)  # same title + frozen clock => same directory name

        self.assertEqual(second, first)
        _, metadata = self.read_metadata(second)
        self.assertEqual((metadata["pages"], metadata["generated"]), (2, 2))


@unittest.skipUnless(HAVE_IMAGING, "Pillow/NumPy not installed")
class GenerateBookWithRealValidatorTests(_BookTestCase):
    """Same pipeline, but validate_image_quality() is the real one."""

    use_fake_validator = False

    def test_quality_metrics_from_the_real_validator_end_up_in_metadata(self):
        png = make_png(cbg.MIN_PRINT_WIDTH, cbg.MIN_PRINT_HEIGHT)
        self.post_process = lambda image_bytes: png

        book_dir = self.run_book(num_pages=2, gen=self.make_generator(force_lineart=True))

        text, metadata = self.read_metadata(book_dir)
        self.assertEqual((metadata["pages"], metadata["generated"]), (2, 2))
        for entry in metadata["pages_data"]:
            metrics = entry["quality_metrics"]
            self.assertEqual((metrics["width"], metrics["height"]), (2550, 3300))
            self.assertEqual((metrics["physical_width"], metrics["physical_height"]), (8.5, 11.0))
            self.assertEqual(metrics["mode"], "1")
            self.assertEqual(metrics["unique_colors"], 2)  # only measured because force_lineart is on
            self.assertAlmostEqual(metrics["dpi"][0], 300, delta=0.01)
            self.assertIn("file_size_kb", metrics)
        self.assertEqual((book_dir / "images" / page_file(1)).read_bytes(), png)

    def test_real_validator_rejects_an_undersized_page(self):
        self.post_process = lambda image_bytes: make_png(100, 100)

        with warnings.catch_warnings():
            # When it rejects an image before loading the pixels, validate_image_quality() leaves
            # PIL's file handle to the garbage collector. That is unrelated to generate_book().
            warnings.simplefilter("ignore", ResourceWarning)
            book_dir = self.run_book(num_pages=1)

        _, metadata = self.read_metadata(book_dir)
        self.assertEqual((metadata["pages"], metadata["generated"], metadata["pages_data"]), (1, 0, []))
        messages = [message for _, message in self.logs]
        self.assertIn("  ✗ Quality validation FAILED for page_001.png", messages)
        self.assertIn("    - CRITICAL: Image too small (100x100), needs 2550x3300 for print", messages)
        self.assertIn("Generated 0/1 pages", messages)


class ParallelGeneratorContractTests(unittest.TestCase):
    """What ParallelColoringBookGenerator (a subclass in another module) relies on."""

    def test_methods_and_signatures_used_by_the_subclass(self):
        cls = cbg.ColoringBookGenerator

        self.assertEqual(list(inspect.signature(cls._save_book_metadata).parameters),
                         ["self", "book_dir", "title", "theme", "num_pages", "generated_pages", "timestamp"])

        params = inspect.signature(cls._generate_image_bytes).parameters
        self.assertEqual(list(params), ["self", "prompt", "size"])
        self.assertEqual(params["size"].default, (1536, 1536))

        self.assertEqual(list(inspect.signature(cls._post_process_image).parameters), ["self", "image_bytes"])

        params = inspect.signature(cls.generate_book).parameters
        self.assertEqual(list(params), ["self", "theme", "num_pages", "book_title"])
        self.assertEqual(params["num_pages"].default, 30)
        self.assertIsNone(params["book_title"].default)

    def test_module_level_names_used_by_the_subclass(self):
        self.assertIsInstance(cbg.THEMES, dict)
        self.assertIsInstance(cbg.PROMPT_VARIATIONS, list)
        self.assertIsInstance(cbg.logger, logging.Logger)
        self.assertTrue(callable(cbg.preflight_checks))
        self.assertTrue(callable(cbg.validate_image_quality))

    def test_save_book_metadata_output_format(self):
        with tempfile.TemporaryDirectory(prefix="test_generate_book_") as tmp:
            gen = cbg.ColoringBookGenerator(output_dir=tmp)
            pages = [{"page": 1, "prompt": "p", "file": "page_001.png", "quality_metrics": {"width": 1}}]

            gen._save_book_metadata(Path(tmp), "T", "th", 5, pages, TIMESTAMP)

            text = (Path(tmp) / "metadata.json").read_text()
        self.assertEqual(text, json.dumps({
            "title": "T",
            "theme": "th",
            "pages": 5,
            "generated": 1,
            "timestamp": TIMESTAMP,
            "pages_data": pages,
        }, indent=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
