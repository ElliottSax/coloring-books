#!/usr/bin/env python3
"""
Parallel version of ColoringBookGenerator

Adds concurrent generation capabilities to speed up book creation.
Can reduce 30-page book from ~10 minutes to ~4-5 minutes.

Usage:
    from coloring_book_generator_parallel import ParallelColoringBookGenerator

    gen = ParallelColoringBookGenerator(
        backend='pollinations',
        force_lineart=True,
        max_workers=3  # Generate 3 pages concurrently
    )

    book_dir = gen.generate_book_parallel(
        theme='mandalas',
        num_pages=30
    )

Thread safety
-------------
Threading model
    generate_book_parallel() runs on the calling thread (the main thread in a
    normal script) and creates a fresh ThreadPoolExecutor for each batch of pages.
    Leaving the pool's ``with`` block joins every worker, so a batch is a barrier:
    batch k+1 starts only after every page of batch k, including its retries and
    backoff sleeps, has finished.  Only _generate_page_with_retry() and what it
    calls (_generate_single_page(), the backend, post-processing and quality
    validation) run on worker threads.  Everything else stays on the calling
    thread: choosing prompts, creating directories, collecting results, calling
    progress_callback and writing metadata.json.  The ``completed`` counter and
    the list of generated pages are therefore touched by one thread only and need
    no locks.

State shared between worker threads
    Workers share no mutable state: what they have in common is read-only or is
    partitioned per page.

    * Generator attributes (backend, force_lineart, lineart_method, api_token,
      hf_token, max_retries): assigned in __init__ and only read afterwards.
      Nothing reachable from a worker assigns to ``self``.
    * Filesystem paths: every page writes only its own images/page_NNN.png, a name
      derived from the page number, which is unique within a run.  The retries of
      one page run one after another on one worker and rewrite that same file.
      The book directory and images/ are created before any worker starts.  The
      Replicate backend stages downloads in a uuid4-named temp file, so those
      cannot collide either.
    * Logger: logging handlers serialize their output, so records are never torn,
      but lines from different pages interleave.  Most messages from this module
      carry a "[page]" prefix; the backend and image-processing messages from
      coloring_book_generator (and the indented validation-issue lines) do not, so
      a log line cannot always be attributed to a page.
    * random: used only to pick each page's prompt variation, on the calling
      thread before any worker is submitted.  Workers never touch it, so
      random.seed() makes the prompts reproducible however workers are scheduled.
    * Backend HTTP calls: the backends use the module-level requests.get/post,
      each of which builds and closes its own Session, so no connection pool,
      cookies or credentials are shared between workers.
    * OpenCV / NumPy / Pillow post-processing: each call builds its own arrays and
      PIL images from the bytes it is given and keeps no module-level state.
      OpenCV releases the GIL, so workers really do run in parallel here, and it
      has an internal thread pool of its own, so the number of busy CPU threads can
      exceed max_workers.  Every page in flight holds several full-resolution
      (2550x3300) buffers, so peak memory grows roughly with max_workers.

Guarantees (exercised by test_parallel_generator.py)
    * Every page writes a distinct file, so pages never overwrite one another.
    * No Exception escapes a page's worker: one raised by the backend,
      post-processing or validation becomes a failed page (after its retries)
      instead of aborting the book or the other pages.
    * Results are collected on the calling thread and sorted by page number within
      each batch.  Batches run in order, so metadata.json lists pages in ascending
      order whatever order they finished in.
    * progress_callback runs on the calling thread, never on a worker.  It is
      called exactly once per page (not once per attempt), ``completed`` rises by
      one per call in completion order (not page order), and an exception it
      raises is logged and swallowed.
    * metadata.json is written once, by the calling thread, after every worker has
      been joined.
    * A run never modifies the generator it runs on.

Callers must not
    * Modify the generator (backend, force_lineart, lineart_method, tokens,
      max_retries, max_workers, progress_callback) while a run is in flight,
      whether from another thread or from the progress callback.  Workers read
      most of these without locking, and the calling thread re-reads max_workers
      for every batch and progress_callback for every page, so a change would
      apply to some pages but not others.
    * Run several books at once on one instance.  Use one generator per concurrent
      run; runs on a shared instance share its callback, which would then receive
      interleaved counts from different books.
    * Start two runs with the same title and output_dir within the same second.
      The book directory is "<title>_<YYYYmmdd_HHMMSS>", so they would share one
      directory and overwrite each other's pages and metadata.
    * Call os.chdir() while a run is in flight if output_dir is relative.  The
      working directory is process-wide and every worker resolves its page path
      against it; pass an absolute output_dir if your application changes directory.
    * Read the book (create_pdf, a file watcher, ...) before generate_book_parallel
      returns.  Page files are written in place, so a reader can see a half-written
      PNG, and images/ can hold a PNG for a page that failed quality validation and
      is therefore missing from metadata.json.
    * Do slow or blocking work in progress_callback.  It runs on the collecting
      thread, so it delays result collection and with it the start of the next
      batch (pages already submitted keep running).  It cannot cancel a run, since
      ordinary exceptions are logged and ignored, and it must not call back into
      the generator.
    * Expect Ctrl-C to stop work that is already running.  KeyboardInterrupt (or any
      other BaseException) on the calling thread leaves the pool's ``with`` block,
      which waits for the batch in flight, retries and backoff sleeps included.
      Later batches never start and no metadata.json is written.
"""

import concurrent.futures
from typing import Callable, Tuple, Optional
from pathlib import Path
from datetime import datetime
import time
import random

from coloring_book_generator import (
    ColoringBookGenerator,
    THEMES,
    PROMPT_VARIATIONS,
    logger,
    preflight_checks,
    validate_image_quality
)


class ParallelColoringBookGenerator(ColoringBookGenerator):
    """Coloring book generator with parallel processing support.

    Threading summary (the full contract is "Thread safety" in the module docstring):

    * Only per-page generation runs on worker threads.  Prompt selection, result
      collection, ``progress_callback`` and ``metadata.json`` stay on the thread
      that calls :meth:`generate_book_parallel`.
    * Workers only read generator attributes and each writes only its own
      ``page_NNN.png``, so no locks are needed, and none are provided: do not
      modify the generator while a run is in flight.
    * Use one generator per concurrent run.
    """

    def __init__(self, output_dir: str = "output", backend: str = "pollinations",
                 force_lineart: bool = False, lineart_method: str = "enhanced",
                 max_workers: int = 3, max_retries: int = 2,
                 progress_callback: Optional[Callable[[int, int, bool], None]] = None):
        """Initialize parallel generator.

        Args:
            output_dir: Output directory
            backend: AI backend to use
            force_lineart: Whether to force line art conversion
            lineart_method: Line art processing method
            max_workers: Maximum concurrent workers (default: 3)
                        Recommended: 2-4 to avoid rate limiting
            max_retries: Extra attempts for a page that fails (default: 2).
                        0 means every page gets exactly one attempt.
            progress_callback: Optional callable(completed, total, success), called
                        once per page after it has finished for good (retries
                        included).  It runs on the thread that called
                        generate_book_parallel, never on a worker, so state that
                        only the callback touches needs no locking.
                        ``completed`` counts the pages finished so far (1..total,
                        in completion order, not page order), ``total`` is the
                        number of pages requested and ``success`` is False for a
                        page that failed for good.  Keep it quick: it blocks result
                        collection, so a slow callback can delay the next batch.
                        Exceptions it raises are logged and swallowed, so it cannot
                        be used to cancel a run.

        Raises:
            ValueError: If max_workers is less than 1 or max_retries is negative
        """
        super().__init__(output_dir, backend, force_lineart, lineart_method)

        # Validate max_workers
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        if max_workers > 10:
            logger.warning(f"max_workers={max_workers} is very high, may cause rate limiting")
            logger.warning("Recommended: 2-4 workers for most APIs")

        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        # Not reassigned after construction.  Worker threads read max_retries (and the
        # base-class settings) without locking, and the calling thread re-reads
        # max_workers per batch and progress_callback per page, so none of these may be
        # changed while a run is in flight (see "Thread safety" in the module docstring).
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.progress_callback = progress_callback

    def _generate_single_page(self, page_num: int, prompt: str,
                              images_dir: Path) -> Tuple[int, bool, dict]:
        """Generate a single page (designed for parallel execution).

        Runs on a worker thread, alongside other pages.  It reads generator
        attributes but never assigns to ``self``.  Its only side effects are
        log records and writing ``images_dir/page_NNN.png``, a file that no other
        page touches.  A file written for a page that then fails quality validation
        is not deleted: the page is just reported as failed (and left out of
        metadata.json), and a retry overwrites the file if it gets as far as
        writing.

        No ``Exception`` escapes: one raised by the backend, post-processing or
        validation is logged and reported as a failed page.

        Args:
            page_num: Page number (1-indexed)
            prompt: Image generation prompt
            images_dir: Directory to save images

        Returns:
            Tuple of (page_num, success, page_data); page_data is None on failure
        """
        # Unique per page number, so no two workers ever write the same file.
        image_path = images_dir / f"page_{page_num:03d}.png"
        success = False
        page_data = None

        try:
            logger.info(f"  [{page_num}] Generating...")

            # Generate image using centralized backend logic (shared with main generator).
            # Network I/O through module-level requests calls, which share no Session.
            image_bytes = self._generate_image_bytes(prompt, size=(1536, 1536))

            if image_bytes:
                # Post-process (upscale and/or line art).  OpenCV/PIL work on arrays
                # and images local to this call; nothing is shared with other pages.
                image_bytes = self._post_process_image(image_bytes)
                image_path.write_bytes(image_bytes)
                success = True

            if success:
                # Validate (reads back the file this worker has just written)
                validation = validate_image_quality(
                    image_path,
                    check_binary=self.force_lineart
                )

                if not validation['valid']:
                    logger.error(f"  [{page_num}] ✗ Quality validation FAILED")
                    for issue in validation.get('issues', []):
                        logger.error(f"    - {issue}")
                    success = False
                elif validation.get('warnings'):
                    logger.warning(f"  [{page_num}] ⚠ Quality warnings")

                if success:
                    page_data = {
                        "page": page_num,
                        "prompt": prompt,
                        "file": str(image_path.name),
                        "quality_metrics": validation.get('metrics', {})
                    }
                    logger.info(f"  [{page_num}] ✓ Complete")

        except Exception as e:
            logger.error(f"  [{page_num}] ✗ Failed: {e}")
            success = False

        return (page_num, success, page_data)

    def _generate_page_with_retry(self, page_num: int, prompt: str,
                                  images_dir: Path) -> Tuple[int, bool, dict]:
        """Run _generate_single_page, retrying failures with backoff.

        This is the function that runs on a worker thread.  The attempts for one
        page are strictly sequential, so retries rewrite the same ``page_NNN.png``
        with no overlap.  After failed attempt n (counting from 1) the worker
        sleeps ``2 ** n`` seconds (2s, 4s, 8s, ...); no sleep follows the final
        attempt, and ``max_retries=0`` means exactly one attempt.  The delay is
        fixed, without jitter, so workers that fail together retry together.

        The sleep happens inside the worker: a page that is backing off keeps its
        worker slot, and the batch barrier in generate_book_parallel waits for it,
        so one stubborn page can hold up the next batch for its attempts plus up to
        ``2 + 4 + ... + 2 ** max_retries`` seconds of backoff.

        Returns:
            The last attempt's (page_num, success, page_data).  No ``Exception``
            escapes (see _generate_single_page).
        """
        for attempt in range(self.max_retries + 1):
            result = self._generate_single_page(page_num, prompt, images_dir)
            if result[1] or attempt == self.max_retries:
                return result
            delay = 2 ** (attempt + 1)
            logger.warning(f"  [{page_num}] Retry {attempt + 1}/{self.max_retries} in {delay}s")
            time.sleep(delay)
        return result

    def generate_book_parallel(self, theme: str, num_pages: int = 30,
                              book_title: str = None,
                              batch_size: Optional[int] = None) -> Path:
        """Generate a coloring book with parallel processing.

        Runs on the calling thread.  Pages are generated in batches; each batch gets
        a fresh pool of ``max_workers`` threads, and the calling thread collects the
        results, calls ``progress_callback`` and writes ``metadata.json``.  The
        full thread-safety contract, including what callers must not do while a
        run is in flight, is in the module docstring.

        Args:
            theme: Theme name
            num_pages: Number of pages to generate
            book_title: Optional custom title
            batch_size: Pages submitted per batch (default: max_workers).  A batch
                        is a barrier: the next one starts only when every page of
                        this one is done.  Its pages run on a pool of
                        ``max_workers`` threads, so at most
                        ``min(batch_size, max_workers)`` pages are in flight at
                        once.  Must not be negative.

        Returns:
            Path to generated book directory.  Pages that failed (after their
            retries) are missing from its metadata.json, but a PNG may remain in
            images/ for a page that failed quality validation.

        Raises:
            ValueError: If batch_size is negative or the theme is unknown
                        (no book directory is created)
            RuntimeError: If the pre-flight checks fail (no book directory is
                        created)
        """
        # Reject a bad argument before any work or side effects.  None and 0 both mean
        # "use max_workers"; a negative size would make total_batches negative below
        # and silently generate nothing.
        if batch_size is not None and batch_size < 0:
            raise ValueError(f"batch_size must not be negative, got {batch_size}")

        # PRE-FLIGHT CHECKS
        if not preflight_checks(force_lineart=self.force_lineart):
            raise RuntimeError("Pre-flight checks failed - cannot generate book safely")

        if theme not in THEMES:
            raise ValueError(f"Unknown theme: {theme}. Available: {list(THEMES.keys())}")

        theme_data = THEMES[theme]
        title = book_title or theme_data["name"]
        safe_title = title.replace(" ", "_").replace(":", "")

        # Create book directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        book_dir = self.output_dir / f"{safe_title}_{timestamp}"
        book_dir.mkdir(parents=True, exist_ok=True)
        images_dir = book_dir / "images"
        images_dir.mkdir(exist_ok=True)

        logger.info(f"Generating coloring book: {title}")
        logger.info(f"Theme: {theme}, Pages: {num_pages}")
        logger.info(f"Parallel workers: {self.max_workers}")
        logger.info(f"Output: {book_dir}")

        # Prepare prompts with shared variations.  This (the only use of ``random``)
        # happens here on the calling thread, before any worker exists.
        prompts = theme_data["prompts"]

        page_prompts = []
        for i in range(num_pages):
            prompt = prompts[i % len(prompts)]
            prompt += random.choice(PROMPT_VARIATIONS)
            page_prompts.append((i + 1, prompt))

        # Generate pages in parallel batches
        batch_size = batch_size or self.max_workers
        generated = []
        completed = 0
        total_batches = (num_pages + batch_size - 1) // batch_size

        logger.info(f"Processing in {total_batches} batch(es) of {batch_size} pages")

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, num_pages)
            batch_items = page_prompts[start_idx:end_idx]

            logger.info(f"\nBatch {batch_num + 1}/{total_batches}: Pages {start_idx + 1}-{end_idx}")

            # Process batch in parallel.  A fresh pool per batch: leaving this ``with``
            # block joins every worker, which is what makes a batch a barrier (and why
            # a KeyboardInterrupt here still waits for the batch in flight).
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(self._generate_page_with_retry, page_num, prompt, images_dir): page_num
                    for page_num, prompt in batch_items
                }

                # Collect results as they complete.  This loop runs on the calling
                # thread only: workers never touch ``completed``, ``batch_results``,
                # ``generated`` or the callback, so none of them needs a lock.
                batch_results = []
                for future in concurrent.futures.as_completed(futures):
                    page_num, success, page_data = future.result()
                    completed += 1
                    if self.progress_callback:
                        # Calling thread, once per page, in completion order.  A slow
                        # callback delays collection (and so the next batch); only
                        # Exception is swallowed, so it cannot cancel the run.
                        try:
                            self.progress_callback(completed, num_pages, bool(success and page_data))
                        except Exception as e:
                            logger.warning(f"progress_callback raised: {e}")
                    if success and page_data:
                        batch_results.append(page_data)

            # Pages finish in any order: sort by page number and add to generated list
            batch_results.sort(key=lambda x: x['page'])
            generated.extend(batch_results)

            # Rate limiting between batches (not needed within batch)
            if batch_num < total_batches - 1:
                time.sleep(2)

        # Save metadata (calling thread, once every worker of every batch has been joined)
        self._save_book_metadata(book_dir, title, theme, num_pages, generated, timestamp)

        logger.info(f"\n{'='*70}")
        logger.info(f"Generated {len(generated)}/{num_pages} pages")
        logger.info(f"Book saved to: {book_dir}")
        logger.info(f"{'='*70}")

        return book_dir


def main():
    """Example usage of parallel generator."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate coloring books in parallel')
    parser.add_argument('--theme', required=True, choices=list(THEMES.keys()),
                       help='Theme to generate')
    parser.add_argument('--pages', type=int, default=30,
                       help='Number of pages to generate')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of parallel workers (2-4 recommended)')
    parser.add_argument('--retries', type=int, default=2,
                       help='Extra attempts per failed page')
    parser.add_argument('--title', help='Custom book title')
    parser.add_argument('--backend', default='pollinations',
                       choices=['pollinations', 'huggingface', 'replicate'],
                       help='AI backend to use')
    parser.add_argument('--lineart-method', default='enhanced',
                       choices=['enhanced', 'standard', 'detailed'],
                       help='Line art processing method')

    args = parser.parse_args()

    gen = ParallelColoringBookGenerator(
        backend=args.backend,
        force_lineart=True,
        lineart_method=args.lineart_method,
        max_workers=args.workers,
        max_retries=args.retries
    )

    book_dir = gen.generate_book_parallel(
        theme=args.theme,
        num_pages=args.pages,
        book_title=args.title
    )

    print(f"\n✓ Book generated: {book_dir}")


if __name__ == "__main__":
    main()
