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
    """Coloring book generator with parallel processing support."""

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
            max_retries: Extra attempts for a page that fails (default: 2)
            progress_callback: Optional callable(completed, total, success) invoked
                        from the main thread after each page finishes

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

        self.max_workers = max_workers
        self.max_retries = max_retries
        self.progress_callback = progress_callback

    def _generate_single_page(self, page_num: int, prompt: str,
                              images_dir: Path) -> Tuple[int, bool, dict]:
        """Generate a single page (designed for parallel execution).

        Args:
            page_num: Page number (1-indexed)
            prompt: Image generation prompt
            images_dir: Directory to save images

        Returns:
            Tuple of (page_num, success, page_data)
        """
        image_path = images_dir / f"page_{page_num:03d}.png"
        success = False
        page_data = None

        try:
            logger.info(f"  [{page_num}] Generating...")

            # Generate image using centralized backend logic (shared with main generator)
            image_bytes = self._generate_image_bytes(prompt, size=(1536, 1536))

            if image_bytes:
                # Post-process (upscale and/or line art)
                image_bytes = self._post_process_image(image_bytes)
                image_path.write_bytes(image_bytes)
                success = True

            if success:
                # Validate
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
        """Run _generate_single_page, retrying failures with backoff."""
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

        Args:
            theme: Theme name
            num_pages: Number of pages to generate
            book_title: Optional custom title
            batch_size: Pages to generate concurrently (default: max_workers)

        Returns:
            Path to generated book directory
        """
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

        # Prepare prompts with shared variations
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

            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(self._generate_page_with_retry, page_num, prompt, images_dir): page_num
                    for page_num, prompt in batch_items
                }

                # Collect results as they complete
                batch_results = []
                for future in concurrent.futures.as_completed(futures):
                    page_num, success, page_data = future.result()
                    completed += 1
                    if self.progress_callback:
                        try:
                            self.progress_callback(completed, num_pages, bool(success and page_data))
                        except Exception as e:
                            logger.warning(f"progress_callback raised: {e}")
                    if success and page_data:
                        batch_results.append(page_data)

            # Sort by page number and add to generated list
            batch_results.sort(key=lambda x: x['page'])
            generated.extend(batch_results)

            # Rate limiting between batches (not needed within batch)
            if batch_num < total_batches - 1:
                time.sleep(2)

        # Save metadata
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
