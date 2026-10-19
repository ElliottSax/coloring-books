#!/usr/bin/env python3
"""
Tests for ParallelColoringBookGenerator (coloring_book_generator_parallel.py)

Covers:
- constructor validation (max_workers, max_retries, the >10 workers warning)
- per-page retry with exponential backoff (2s, 4s, 8s, ...)
- the progress_callback contract (one call per page, ordering, calling thread,
  exceptions swallowed)
- batching math, the batch barrier and the effective concurrency
- result ordering, failed pages and worker exceptions
- the threading guarantees documented in the parallel module's docstrings

Everything external is faked.  The image backend, post-processing, quality
validation and pre-flight checks are patched, ``time.sleep`` is patched (backoff
delays are recorded, never slept), any network call fails loudly, and all
output goes to temporary directories.  No OpenCV/PIL work is done, so the whole
suite runs in about a second.

Concurrency is forced with barriers/conditions (never with sleeps), and every
blocking wait has a timeout, so a regression fails a test instead of hanging it.

Run with: python3 test_parallel_generator.py
"""

import collections
import concurrent.futures
import json
import logging
import random
import re
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import requests

import coloring_book_generator_parallel as par
from coloring_book_generator_parallel import ParallelColoringBookGenerator

THEME = "unit-test-theme"
# 100 distinct prompts: page N (1-based) always receives "page prompt NNN" because
# the generator uses prompts[i % len(prompts)].  The fake backend recovers the page
# number from the prompt (prompt variations only ever *append* ", ...").
THEME_PROMPTS = [f"page prompt {n:03d}" for n in range(1, 101)]
FAKE_THEMES = {THEME: {"name": "Unit Test Theme", "prompts": THEME_PROMPTS}}

# Longest any test blocks on a barrier/condition before giving up.  Healthy runs
# never wait; a bug fails within this bound instead of hanging the suite.
WAIT = 5

# Scripted per-attempt outcomes (see FakeBackend.plan)
OK = "ok"                    # every stage succeeds
NONE = "none"                # backend returns None (what the real backends do on HTTP errors)
RAISE = "raise"              # backend raises
POST_FAIL = "post_fail"      # backend ok, post-processing raises
BAD_QUALITY = "bad_quality"  # backend + post-processing ok, quality validation reports invalid


def page_of(prompt):
    """Page number encoded in one of THEME_PROMPTS (plus any ', variation' suffix)."""
    return int(prompt.split(",")[0].rsplit(" ", 1)[1])


class FakeBackend:
    """Thread-safe fake of the three per-page stages that run on worker threads.

    ``generate``      replaces ColoringBookGenerator._generate_image_bytes
    ``post_process``  replaces ColoringBookGenerator._post_process_image
    ``validate``      replaces validate_image_quality

    The stages talk to each other through the bytes they pass along (the outcome
    is embedded in the image bytes, and ``validate`` reads the file the worker
    just wrote), so one scripted outcome per attempt drives all three stages.
    Everything is guarded by a lock: unittest.mock call counters are not atomic
    across threads, so they are deliberately not used for concurrent calls.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._plans = {}
            self.attempts = collections.Counter()           # page -> attempts started
            self.prompts = collections.defaultdict(list)    # page -> prompt of each attempt
            self.worker_idents = set()                      # threads that ran the backend
            self.events = []                                # ("start"|"end", page), global order
            self._in_flight = 0
            self.peak_in_flight = 0
            self.rendezvous = None      # optional Barrier every in-flight page meets at
            self.before_finish = None   # optional hook(page) run just before a page finishes

    def plan(self, page, *outcomes, then=OK):
        """Script the first attempts of ``page``; later attempts get ``then``."""
        with self._lock:
            self._plans[page] = (list(outcomes), then)

    def _outcome(self, page, attempt):
        scripted, then = self._plans.get(page, ((), OK))
        return scripted[attempt] if attempt < len(scripted) else then

    # -- stage 1: the backend ------------------------------------------------
    def generate(self, prompt, size=(1536, 1536)):
        page = page_of(prompt)
        with self._lock:
            attempt = self.attempts[page]
            self.attempts[page] += 1
            outcome = self._outcome(page, attempt)
            self.prompts[page].append(prompt)
            self.worker_idents.add(threading.get_ident())
            self.events.append(("start", page))
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            if self.rendezvous is not None:
                self.rendezvous.wait(WAIT)   # BrokenBarrierError if the peers never arrive
            if outcome == NONE:
                return None
            if outcome == RAISE:
                raise RuntimeError(f"simulated backend crash (page {page}, attempt {attempt + 1})")
            return f"IMG page={page} attempt={attempt + 1} outcome={outcome}".encode()
        finally:
            with self._lock:
                self._in_flight -= 1

    # -- stage 2: post-processing (installed as a staticmethod: no ``self``) --
    @staticmethod
    def post_process(image_bytes):
        if b"outcome=post_fail" in image_bytes:
            raise RuntimeError("simulated post-processing crash")
        return b"PROCESSED " + image_bytes

    # -- stage 3: quality validation -----------------------------------------
    def validate(self, image_path, check_binary=True, **kwargs):
        content = Path(image_path).read_bytes()
        page = int(re.search(rb"page=(\d+)", content).group(1))
        if self.before_finish is not None:
            self.before_finish(page)
        with self._lock:
            self.events.append(("end", page))
        if b"outcome=bad_quality" in content:
            return {"valid": False, "issues": ["simulated quality failure"],
                    "warnings": [], "metrics": {}}
        return {"valid": True, "issues": [], "warnings": [],
                "metrics": {"fake": True, "file": Path(image_path).name}}


class FirstPageOfBatchFinishesLast:
    """Make the first page of every batch finish *last*, deterministically.

    Wire ``before_finish`` into FakeBackend and ``on_progress`` in as the
    progress_callback.  Page ``first`` of a batch may only complete once the
    calling thread has seen a callback for every other page of that batch, so
    results always arrive out of page order.  Needs max_workers >= batch size
    (otherwise the first page would hog a worker its peers need).
    """

    def __init__(self, num_pages, batch_size):
        self.num_pages = num_pages
        self.batch_size = batch_size
        self.timed_out = False
        self._cond = threading.Condition()
        self._callbacks = 0

    def on_progress(self, completed, total, success):       # calling thread
        with self._cond:
            self._callbacks += 1
            self._cond.notify_all()

    def before_finish(self, page):                           # worker thread
        first = ((page - 1) // self.batch_size) * self.batch_size + 1
        if page != first:
            return
        in_batch = min(self.batch_size, self.num_pages - first + 1)
        needed = (first - 1) + (in_batch - 1)   # earlier batches + the rest of this batch
        with self._cond:
            if not self._cond.wait_for(lambda: self._callbacks >= needed, timeout=WAIT):
                self.timed_out = True
                raise RuntimeError("gate timed out waiting for the other pages of the batch")


class _ListHandler(logging.Handler):
    """Collects the records emitted on the generator's logger."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def messages(self, min_level=logging.WARNING):
        return [r.getMessage() for r in list(self.records) if r.levelno >= min_level]


class ParallelTestCase(unittest.TestCase):
    """Fixture: temp output dir, fake pipeline, instant time.sleep, captured logs."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

        self.backend = FakeBackend()
        self.sleeps = []                      # every duration passed to time.sleep()
        sleeps_lock = threading.Lock()

        def fake_sleep(seconds):
            with sleeps_lock:
                self.sleeps.append(seconds)

        # time.sleep is a process-global: nothing in these tests may really sleep.
        self._patch(time, "sleep", fake_sleep)
        self._patch(par, "THEMES", FAKE_THEMES)
        self.preflight = self._patch(par, "preflight_checks", return_value=True)
        self._patch(par, "validate_image_quality", self.backend.validate)
        self._patch(ParallelColoringBookGenerator, "_generate_image_bytes", self.backend.generate)
        self._patch(ParallelColoringBookGenerator, "_post_process_image",
                    staticmethod(FakeBackend.post_process))
        # The fakes above bypass every real backend; make any stray network call loud.
        for name in ("get", "post"):
            self._patch(requests, name, side_effect=AssertionError("network access in a unit test"))

        # Keep the console quiet; everything is still available in self.logs.
        self.logs = _ListHandler()
        par.logger.addHandler(self.logs)
        propagate = par.logger.propagate
        par.logger.propagate = False
        self.addCleanup(setattr, par.logger, "propagate", propagate)
        self.addCleanup(par.logger.removeHandler, self.logs)

    def _patch(self, target, attribute, new=mock.DEFAULT, **kwargs):
        patcher = mock.patch.object(target, attribute, new, **kwargs)
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    # -- helpers -------------------------------------------------------------
    def reset_fakes(self):
        """Forget everything recorded so far (for tests that loop with subTest)."""
        self.backend.reset()
        self.sleeps.clear()
        self.logs.records.clear()
        shutil.rmtree(self.tmp / "images", ignore_errors=True)

    def make_generator(self, **kwargs):
        kwargs.setdefault("output_dir", str(self.tmp / "out"))
        return ParallelColoringBookGenerator(**kwargs)

    def run_book(self, gen, num_pages, **kwargs):
        """Generate a book; return (book_dir, parsed metadata.json)."""
        book_dir = gen.generate_book_parallel(THEME, num_pages=num_pages, **kwargs)
        return book_dir, json.loads((book_dir / "metadata.json").read_text())

    def run_page(self, gen, page=1):
        """Run one page through _generate_page_with_retry on the current thread."""
        images_dir = self.tmp / "images"
        images_dir.mkdir(exist_ok=True)
        result = gen._generate_page_with_retry(page, THEME_PROMPTS[page - 1], images_dir)
        return result, images_dir

    @staticmethod
    def pages_in(metadata):
        return [entry["page"] for entry in metadata["pages_data"]]


# ===========================================================================
# Constructor validation
# ===========================================================================

class ConstructorTests(ParallelTestCase):

    def test_defaults(self):
        gen = self.make_generator()
        self.assertEqual(gen.max_workers, 3)
        self.assertEqual(gen.max_retries, 2)
        self.assertIsNone(gen.progress_callback)

    def test_progress_callback_is_stored(self):
        def callback(completed, total, success):
            pass

        self.assertIs(self.make_generator(progress_callback=callback).progress_callback, callback)

    def test_max_workers_below_one_is_rejected(self):
        for bad in (0, -1, -10):
            with self.subTest(max_workers=bad):
                with self.assertRaisesRegex(ValueError, f"max_workers must be >= 1, got {bad}"):
                    self.make_generator(max_workers=bad)

    def test_up_to_ten_workers_is_accepted_silently(self):
        for workers in (1, 2, 10):
            with self.subTest(max_workers=workers):
                self.assertEqual(self.make_generator(max_workers=workers).max_workers, workers)
        self.assertEqual(self.logs.messages(logging.WARNING), [])

    def test_more_than_ten_workers_warns_but_is_accepted(self):
        gen = self.make_generator(max_workers=11)
        self.assertEqual(gen.max_workers, 11)
        warnings = self.logs.messages(logging.WARNING)
        self.assertTrue(any("max_workers=11" in m and "rate limiting" in m for m in warnings), warnings)
        self.assertTrue(any("2-4 workers" in m for m in warnings), warnings)

    def test_negative_max_retries_is_rejected(self):
        for bad in (-1, -5):
            with self.subTest(max_retries=bad):
                with self.assertRaisesRegex(ValueError, f"max_retries must be >= 0, got {bad}"):
                    self.make_generator(max_retries=bad)

    def test_zero_max_retries_is_accepted(self):
        self.assertEqual(self.make_generator(max_retries=0).max_retries, 0)


# ===========================================================================
# Per-page retry with exponential backoff
# ===========================================================================

class RetryTests(ParallelTestCase):

    def test_first_attempt_success_makes_one_attempt_and_never_sleeps(self):
        gen = self.make_generator(max_retries=2)
        (page, ok, data), images_dir = self.run_page(gen, 1)
        self.assertEqual((page, ok), (1, True))
        self.assertEqual(self.backend.attempts[1], 1)
        self.assertEqual(self.sleeps, [])
        self.assertEqual(data, {"page": 1, "prompt": THEME_PROMPTS[0], "file": "page_001.png",
                                "quality_metrics": {"fake": True, "file": "page_001.png"}})
        self.assertTrue((images_dir / "page_001.png").exists())

    def test_page_succeeds_after_n_failures(self):
        # Every kind of per-attempt failure is retried, not just "backend returned None".
        for failure in (NONE, RAISE, POST_FAIL, BAD_QUALITY):
            for failures in (1, 2):    # max_retries=2: even two failures are recoverable
                with self.subTest(failure=failure, failures=failures):
                    self.reset_fakes()
                    gen = self.make_generator(max_retries=2)
                    self.backend.plan(1, *[failure] * failures)

                    (page, ok, data), images_dir = self.run_page(gen, 1)

                    self.assertEqual((page, ok), (1, True))
                    self.assertEqual(data["page"], 1)
                    self.assertEqual(self.backend.attempts[1], failures + 1)
                    self.assertEqual(self.sleeps, [2, 4][:failures])
                    # Retries rewrite the same file; what is left is the successful attempt.
                    self.assertEqual([p.name for p in images_dir.iterdir()], ["page_001.png"])
                    self.assertEqual((images_dir / "page_001.png").read_bytes(),
                                     f"PROCESSED IMG page=1 attempt={failures + 1} outcome=ok".encode())

    def test_gives_up_after_exactly_max_retries_plus_one_attempts(self):
        for max_retries in (1, 2, 3, 5):
            for failure in (NONE, RAISE):
                with self.subTest(max_retries=max_retries, failure=failure):
                    self.reset_fakes()
                    gen = self.make_generator(max_retries=max_retries)
                    self.backend.plan(1, then=failure)

                    result, _ = self.run_page(gen, 1)

                    self.assertEqual(result, (1, False, None))
                    self.assertEqual(self.backend.attempts[1], max_retries + 1)
                    # one backoff between consecutive attempts, none after the last one
                    self.assertEqual(len(self.sleeps), max_retries)

    def test_one_failure_too_many_is_not_recovered(self):
        # max_retries=2 allows 3 attempts: 2 failures recover, 3 do not.
        gen = self.make_generator(max_retries=2)
        self.backend.plan(1, NONE, NONE, NONE)
        result, _ = self.run_page(gen, 1)
        self.assertEqual(result, (1, False, None))
        self.assertEqual(self.backend.attempts[1], 3)

    def test_backoff_delays_double_starting_at_two_seconds(self):
        gen = self.make_generator(max_retries=5)
        self.backend.plan(1, then=NONE)
        self.run_page(gen, 1)
        self.assertEqual(self.sleeps, [2, 4, 8, 16, 32])

    def test_default_retry_budget_uses_two_and_four_second_backoff(self):
        gen = self.make_generator()    # max_retries defaults to 2
        self.backend.plan(1, then=NONE)
        self.run_page(gen, 1)
        self.assertEqual(self.backend.attempts[1], 3)
        self.assertEqual(self.sleeps, [2, 4])

    def test_max_retries_zero_means_a_single_attempt(self):
        gen = self.make_generator(max_retries=0)
        self.backend.plan(1, then=NONE)
        self.backend.plan(2, then=RAISE)

        self.assertEqual(self.run_page(gen, 1)[0], (1, False, None))
        self.assertEqual(self.run_page(gen, 2)[0], (2, False, None))
        self.assertEqual(self.run_page(gen, 3)[0][:2], (3, True))

        self.assertEqual(dict(self.backend.attempts), {1: 1, 2: 1, 3: 1})
        self.assertEqual(self.sleeps, [])

    def test_logs_each_retry_with_its_delay(self):
        gen = self.make_generator(max_retries=2)
        self.backend.plan(1, then=NONE)
        self.run_page(gen, 1)
        warnings = self.logs.messages(logging.WARNING)
        self.assertTrue(any("[1] Retry 1/2 in 2s" in m for m in warnings), warnings)
        self.assertTrue(any("[1] Retry 2/2 in 4s" in m for m in warnings), warnings)

    def test_worker_exception_is_converted_to_a_failed_page(self):
        # _generate_single_page is what runs on the worker thread: it must never raise.
        gen = self.make_generator(max_retries=0)
        images_dir = self.tmp / "images"
        images_dir.mkdir()
        for failure, message in ((RAISE, "simulated backend crash"),
                                 (POST_FAIL, "simulated post-processing crash")):
            with self.subTest(failure=failure):
                self.logs.records.clear()
                self.backend.plan(1, then=failure)

                result = gen._generate_single_page(1, THEME_PROMPTS[0], images_dir)

                self.assertEqual(result, (1, False, None))
                errors = self.logs.messages(logging.ERROR)
                self.assertTrue(
                    any("[1]" in m and "Failed:" in m and message in m for m in errors), errors)


# ===========================================================================
# progress_callback
# ===========================================================================

class ProgressCallbackTests(ParallelTestCase):

    def recorder(self):
        calls = []

        def callback(completed, total, success):
            calls.append((completed, total, success, threading.get_ident()))

        return calls, callback

    def test_called_once_per_page_including_failures(self):
        calls, callback = self.recorder()
        gen = self.make_generator(max_workers=3, max_retries=1, progress_callback=callback)
        self.backend.plan(1, NONE)                  # fails once, recovers on the retry
        self.backend.plan(2, then=NONE)             # never recovers
        self.backend.plan(3, then=RAISE)            # crashes on every attempt
        self.backend.plan(4, BAD_QUALITY)           # invalid once, recovers on the retry
        self.backend.plan(5, then=BAD_QUALITY)      # invalid on every attempt

        _, metadata = self.run_book(gen, 6)

        self.assertEqual(len(calls), 6)                                  # one per page...
        self.assertEqual(dict(self.backend.attempts), {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1})
        self.assertTrue(sum(self.backend.attempts.values()) > len(calls))   # ...not one per attempt
        self.assertEqual([c[0] for c in calls], [1, 2, 3, 4, 5, 6])      # failures count as completed too
        self.assertEqual([c[1] for c in calls], [6] * 6)                 # total = pages requested
        successes = [c[2] for c in calls]
        self.assertTrue(all(isinstance(s, bool) for s in successes), successes)
        self.assertEqual(successes.count(True), 3)                       # pages 1, 4 (after retry) and 6
        self.assertEqual(successes.count(False), 3)                      # pages 2, 3, 5
        self.assertEqual(self.pages_in(metadata), [1, 4, 6])             # flags agree with metadata

    def test_completed_increases_by_one_per_page_even_when_pages_finish_out_of_order(self):
        num_pages, batch = 7, 3
        gate = FirstPageOfBatchFinishesLast(num_pages, batch)
        self.backend.before_finish = gate.before_finish
        calls, record = self.recorder()

        def callback(completed, total, success):
            record(completed, total, success)
            gate.on_progress(completed, total, success)

        gen = self.make_generator(max_workers=batch, progress_callback=callback)
        self.run_book(gen, num_pages)

        self.assertFalse(gate.timed_out)
        self.assertEqual([c[0] for c in calls], list(range(1, num_pages + 1)))   # strictly +1 each time
        self.assertEqual({c[1] for c in calls}, {num_pages})
        self.assertTrue(all(c[2] is True for c in calls))
        # sanity: the scenario really did complete out of page order
        finished = [page for kind, page in self.backend.events if kind == "end"]
        self.assertEqual((finished[2], finished[5], finished[6]), (1, 4, 7))

    def test_runs_on_the_thread_that_called_generate_book_parallel(self):
        # Never on a worker - and "the caller", not necessarily the process's main thread.
        for from_helper_thread in (False, True):
            with self.subTest(from_helper_thread=from_helper_thread):
                self.reset_fakes()
                seen, callers, errors = [], [], []
                gen = self.make_generator(
                    output_dir=str(self.tmp / f"out-{from_helper_thread}"), max_workers=3,
                    progress_callback=lambda completed, total, success: seen.append(threading.get_ident()))

                def run():
                    callers.append(threading.get_ident())
                    try:
                        gen.generate_book_parallel(THEME, num_pages=6)
                    except BaseException as exc:    # surface failures from the helper thread
                        errors.append(exc)

                if from_helper_thread:
                    helper = threading.Thread(target=run)
                    helper.start()
                    helper.join(WAIT * 4)
                    self.assertFalse(helper.is_alive())
                    self.assertNotEqual(callers[0], threading.get_ident())
                else:
                    run()

                self.assertEqual(errors, [])
                self.assertEqual(seen, callers * 6)                  # all six calls, on the caller's thread
                self.assertTrue(self.backend.worker_idents)          # workers did run...
                self.assertNotIn(callers[0], self.backend.worker_idents)   # ...on other threads

    def test_exception_in_callback_is_swallowed_and_does_not_abort_the_book(self):
        for raising_calls in ("every call", "second call only"):
            with self.subTest(raising_calls=raising_calls):
                self.reset_fakes()
                calls = []

                def callback(completed, total, success):
                    calls.append(completed)
                    if raising_calls == "every call" or completed == 2:
                        raise ValueError("boom")

                gen = self.make_generator(output_dir=str(self.tmp / f"out-{raising_calls}"),
                                          max_workers=2, progress_callback=callback)

                _, metadata = self.run_book(gen, 5)          # must not raise

                self.assertEqual(calls, [1, 2, 3, 4, 5])     # still called for every page afterwards
                self.assertEqual(self.pages_in(metadata), [1, 2, 3, 4, 5])
                self.assertEqual(metadata["generated"], 5)
                swallowed = [m for m in self.logs.messages(logging.WARNING)
                             if "progress_callback raised: boom" in m]
                self.assertEqual(len(swallowed), 5 if raising_calls == "every call" else 1)

    def test_base_exceptions_are_not_swallowed(self):
        # Only Exception is guarded; an abort (think KeyboardInterrupt) must get out.
        class Abort(BaseException):
            pass

        def callback(completed, total, success):
            raise Abort()

        gen = self.make_generator(max_workers=3, progress_callback=callback)
        with self.assertRaises(Abort):
            gen.generate_book_parallel(THEME, num_pages=6)      # batches: pages 1-3, then 4-6

        # The executor of the batch in flight is joined before the exception escapes...
        self.assertEqual(sorted(p for kind, p in self.backend.events if kind == "end"), [1, 2, 3])
        # ...later batches never start, and no metadata is written for an aborted run.
        self.assertEqual(sorted(self.backend.attempts), [1, 2, 3])
        (book_dir,) = gen.output_dir.iterdir()
        self.assertFalse((book_dir / "metadata.json").exists())


# ===========================================================================
# Batching
# ===========================================================================

class BatchingTests(ParallelTestCase):

    # (num_pages, max_workers, batch_size, expected batches)
    CASES = [
        (7, 3, None, [[1, 2, 3], [4, 5, 6], [7]]),    # not divisible; batch defaults to max_workers
        (6, 3, None, [[1, 2, 3], [4, 5, 6]]),         # exact multiple
        (7, 3, 2, [[1, 2], [3, 4], [5, 6], [7]]),     # override smaller than max_workers
        (7, 2, 5, [[1, 2, 3, 4, 5], [6, 7]]),         # override larger than max_workers
        (4, 3, 10, [[1, 2, 3, 4]]),                   # override larger than num_pages: one batch
        (1, 3, None, [[1]]),                          # a single page
    ]

    def test_batch_math_and_barrier_between_batches(self):
        for num_pages, max_workers, batch_size, batches in self.CASES:
            with self.subTest(num_pages=num_pages, max_workers=max_workers, batch_size=batch_size):
                self.reset_fakes()
                gen = self.make_generator(output_dir=str(self.tmp / f"out-{num_pages}-{batch_size}"),
                                          max_workers=max_workers)
                threads_before = threading.active_count()

                _, metadata = self.run_book(gen, num_pages, batch_size=batch_size)

                # every page ran exactly once, and the book is complete and in order
                starts = [p for kind, p in self.backend.events if kind == "start"]
                self.assertEqual(sorted(starts), list(range(1, num_pages + 1)))
                self.assertEqual(self.pages_in(metadata), list(range(1, num_pages + 1)))

                # batch k+1 starts only after every page of batch k has finished
                first_start = {p: i for i, (kind, p) in enumerate(self.backend.events) if kind == "start"}
                last_end = {p: i for i, (kind, p) in enumerate(self.backend.events) if kind == "end"}
                for current, following in zip(batches, batches[1:]):
                    self.assertLess(max(last_end[p] for p in current),
                                    min(first_start[p] for p in following))

                # a 2 s pause between batches (and nowhere else: no retries happen here)
                self.assertEqual(self.sleeps, [2] * (len(batches) - 1))
                # each batch gets its own executor, shut down before the run returns
                self.assertEqual(threading.active_count(), threads_before)

    def test_concurrency_is_the_smaller_of_batch_size_and_max_workers(self):
        # Lower bound: the barrier only releases once `expected` pages are inside the
        # backend together, so the run can only complete if that many ran concurrently.
        # Upper bound: a batch submits `batch_size` tasks to a fresh pool of exactly
        # `max_workers` threads (absence of extra threads cannot be observed without
        # waiting on a timer, so the pool size itself is what gets checked).
        real_executor = concurrent.futures.ThreadPoolExecutor
        pool_sizes = []

        class SpyExecutor(real_executor):
            def __init__(self, *args, **kwargs):
                pool_sizes.append(kwargs.get("max_workers", args[0] if args else None))
                super().__init__(*args, **kwargs)

        for num_pages, max_workers, batch_size, expected in (
                (8, 2, 8, 2),       # max_workers is the limit
                (6, 6, 2, 2),       # batch_size is the limit
                (4, 4, None, 4),    # default batch size == max_workers
        ):
            with self.subTest(num_pages=num_pages, max_workers=max_workers, batch_size=batch_size):
                self.reset_fakes()
                pool_sizes.clear()
                self.backend.rendezvous = threading.Barrier(expected)
                gen = self.make_generator(output_dir=str(self.tmp / f"out-{max_workers}-{batch_size}"),
                                          max_workers=max_workers, max_retries=0)

                with mock.patch.object(concurrent.futures, "ThreadPoolExecutor", SpyExecutor):
                    _, metadata = self.run_book(gen, num_pages, batch_size=batch_size)

                self.assertEqual(self.backend.peak_in_flight, expected)
                self.assertEqual(self.pages_in(metadata), list(range(1, num_pages + 1)))
                batches = -(-num_pages // (batch_size or max_workers))       # ceil division
                self.assertEqual(pool_sizes, [max_workers] * batches)        # one fresh pool per batch


# ===========================================================================
# Whole-book behaviour
# ===========================================================================

class BookTests(ParallelTestCase):

    def test_metadata_pages_are_sorted_even_when_pages_finish_out_of_order(self):
        num_pages, batch = 7, 3
        gate = FirstPageOfBatchFinishesLast(num_pages, batch)
        self.backend.before_finish = gate.before_finish
        gen = self.make_generator(max_workers=batch, progress_callback=gate.on_progress)

        _, metadata = self.run_book(gen, num_pages)

        self.assertFalse(gate.timed_out)
        finished = [page for kind, page in self.backend.events if kind == "end"]
        self.assertEqual((finished[2], finished[5]), (1, 4))    # first page of a batch finished last
        self.assertEqual(self.pages_in(metadata), [1, 2, 3, 4, 5, 6, 7])

    def test_failed_pages_are_left_out_of_metadata(self):
        gen = self.make_generator(max_workers=3, max_retries=0)
        self.backend.plan(2, then=NONE)
        self.backend.plan(5, then=BAD_QUALITY)

        _, metadata = self.run_book(gen, 6)

        self.assertEqual(self.pages_in(metadata), [1, 3, 4, 6])
        self.assertEqual(metadata["pages"], 6)          # pages requested
        self.assertEqual(metadata["generated"], 4)      # pages that made it
        self.assertEqual(metadata["theme"], THEME)
        self.assertEqual(metadata["title"], "Unit Test Theme")

    def test_worker_exception_is_a_failed_page_not_a_crash(self):
        gen = self.make_generator(max_workers=3, max_retries=1)
        self.backend.plan(2, then=RAISE)         # backend crashes on every attempt
        self.backend.plan(4, then=POST_FAIL)     # post-processing crashes on every attempt

        _, metadata = self.run_book(gen, 5)      # must not raise

        self.assertEqual(self.pages_in(metadata), [1, 3, 5])
        self.assertEqual(metadata["generated"], 3)
        # crashes are retried like any other failure...
        self.assertEqual((self.backend.attempts[2], self.backend.attempts[4]), (2, 2))
        # ...and reported, attributed to their page
        errors = self.logs.messages(logging.ERROR)
        self.assertTrue(any("[2]" in m and "simulated backend crash" in m for m in errors), errors)
        self.assertTrue(any("[4]" in m and "simulated post-processing crash" in m for m in errors), errors)

    def test_unknown_theme_raises_value_error_before_doing_any_work(self):
        gen = self.make_generator()
        with self.assertRaises(ValueError) as caught:
            gen.generate_book_parallel("no-such-theme", num_pages=3)
        self.assertIn("Unknown theme: no-such-theme", str(caught.exception))
        self.assertIn(THEME, str(caught.exception))              # lists what is available
        self.assertEqual(dict(self.backend.attempts), {})        # nothing was generated...
        self.assertEqual(list(gen.output_dir.iterdir()), [])     # ...and no book directory exists

    def test_negative_batch_size_is_rejected_instead_of_silently_generating_nothing(self):
        # A negative batch size makes the batch count negative, so without the guard
        # the run would "succeed" with an empty book.
        for bad in (-1, -3):
            with self.subTest(batch_size=bad):
                gen = self.make_generator(output_dir=str(self.tmp / f"out{bad}"))
                with self.assertRaisesRegex(ValueError, f"batch_size must not be negative, got {bad}"):
                    gen.generate_book_parallel(THEME, num_pages=5, batch_size=bad)
                self.assertEqual(dict(self.backend.attempts), {})
                self.assertEqual(list(gen.output_dir.iterdir()), [])

    def test_preflight_failure_raises_runtime_error_before_doing_any_work(self):
        self.preflight.return_value = False
        for force_lineart in (False, True):
            with self.subTest(force_lineart=force_lineart):
                self.preflight.reset_mock()
                gen = self.make_generator(force_lineart=force_lineart)
                with self.assertRaisesRegex(RuntimeError, "Pre-flight checks failed"):
                    gen.generate_book_parallel(THEME, num_pages=3)
                self.preflight.assert_called_once_with(force_lineart=force_lineart)
                self.assertEqual(dict(self.backend.attempts), {})
                self.assertEqual(list(gen.output_dir.iterdir()), [])


# ===========================================================================
# Guarantees promised by the thread-safety documentation
# ===========================================================================

class ThreadingGuaranteeTests(ParallelTestCase):

    def test_pages_in_flight_together_write_distinct_files_without_cross_talk(self):
        self.backend.rendezvous = threading.Barrier(4)    # all four pages inside the backend at once
        gen = self.make_generator(max_workers=4, max_retries=0)

        book_dir, metadata = self.run_book(gen, 4)

        self.assertEqual(self.backend.peak_in_flight, 4)
        self.assertEqual(len(self.backend.worker_idents), 4)
        images = book_dir / "images"
        self.assertEqual(sorted(p.name for p in images.iterdir()),
                         [f"page_{n:03d}.png" for n in range(1, 5)])
        for entry in metadata["pages_data"]:
            n = entry["page"]
            self.assertEqual(entry["file"], f"page_{n:03d}.png")
            self.assertEqual(entry["prompt"], self.backend.prompts[n][0])     # this page's own prompt
            self.assertEqual((images / entry["file"]).read_bytes(),           # ...and this page's own bytes
                             f"PROCESSED IMG page={n} attempt=1 outcome=ok".encode())

    def test_a_run_does_not_mutate_the_generator(self):
        gen = self.make_generator(max_workers=3, max_retries=1,
                                  progress_callback=lambda completed, total, success: None)
        self.backend.plan(2, NONE)               # exercise the retry path...
        self.backend.plan(3, then=RAISE)         # ...and the failure path
        before = dict(vars(gen))

        self.run_book(gen, 5)

        self.assertEqual(vars(gen), before)

    def test_prompt_variations_are_chosen_on_the_calling_thread(self):
        real_choice = random.choice
        chosen_on = []

        def spy(sequence):
            chosen_on.append(threading.get_ident())
            return real_choice(sequence)

        gen = self.make_generator(max_workers=3)
        with mock.patch.object(random, "choice", spy):
            self.run_book(gen, 6)

        self.assertEqual(chosen_on, [threading.get_ident()] * 6)      # once per page, never on a worker

    def test_seeding_random_makes_the_prompts_reproducible(self):
        self.addCleanup(random.setstate, random.getstate())
        prompts = []
        for run in range(2):
            self.reset_fakes()
            gen = self.make_generator(output_dir=str(self.tmp / f"run-{run}"), max_workers=3)
            random.seed(1234)
            self.run_book(gen, 6)
            prompts.append([self.backend.prompts[n][0] for n in range(1, 7)])
        self.assertEqual(prompts[0], prompts[1])

    def test_metadata_is_written_once_by_the_calling_thread_after_every_callback(self):
        events = []
        original = ParallelColoringBookGenerator._save_book_metadata

        def spy(gen_self, *args, **kwargs):
            events.append(("metadata", threading.get_ident()))
            return original(gen_self, *args, **kwargs)

        gen = self.make_generator(
            max_workers=3, progress_callback=lambda completed, total, success: events.append("callback"))
        with mock.patch.object(ParallelColoringBookGenerator, "_save_book_metadata", spy):
            self.run_book(gen, 5)

        self.assertEqual(events, ["callback"] * 5 + [("metadata", threading.get_ident())])


if __name__ == "__main__":
    unittest.main(verbosity=2)
