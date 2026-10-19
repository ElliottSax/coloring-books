#!/usr/bin/env python3
"""
Adult Coloring Book Generator for Amazon KDP
Generates intricate line art designs and compiles into print-ready PDFs.

Version 2.1.0 - Enhanced Quality Assurance
- Automatic quality validation on every generated image
- Pre-flight dependency checks
- Post-generation verification
- Comprehensive error handling
- Prevents recurrence of known issues
"""

import os
import json
import time
import random
import requests
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Quality Assurance Constants
MIN_PRINT_WIDTH = 2550  # 8.5" at 300 DPI
MIN_PRINT_HEIGHT = 3300  # 11" at 300 DPI
REQUIRED_DPI = 300
MAX_ACCEPTABLE_COLORS = 2  # For line art (black & white only)

# Image Quality Constants
MIN_FILE_SIZE_BYTES = 10_000  # 10KB - smaller indicates generation issue
MAX_FILE_SIZE_BYTES = 10_000_000  # 10MB - larger may cause upload issues
RATE_LIMIT_DELAY_SECONDS = 2  # Delay between API calls to avoid rate limiting

# Prompt Variations (shared between generators)
PROMPT_VARIATIONS = [
    "",
    ", with extra fine details",
    ", with bold thick lines",
    ", with intricate background patterns",
    ", centered composition",
    ", full page design",
]

# API Keys (set via environment variable)
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

# KDP Print Dimensions (in pixels at 300 DPI)
KDP_SIZES = {
    "8.5x11": (2550, 3300),      # Standard US Letter
    "8x10": (2400, 3000),         # Popular coloring book size
    "6x9": (1800, 2700),          # Trade paperback
    "8.25x8.25": (2475, 2475),    # Square format
}


# ==================== QUALITY ASSURANCE FUNCTIONS ====================

def validate_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are installed.

    Returns:
        Dict with dependency name and availability status
    """
    deps = {
        'opencv': False,
        'numpy': False,
        'PIL': False,
        'reportlab': False,
        'requests': True  # Already imported above
    }

    try:
        import cv2
        deps['opencv'] = True
        logger.debug(f"✓ OpenCV {cv2.__version__} available")
    except ImportError:
        logger.warning("✗ OpenCV not available - line art processing will be disabled")

    try:
        import numpy
        deps['numpy'] = True
        logger.debug(f"✓ NumPy {numpy.__version__} available")
    except ImportError:
        logger.warning("✗ NumPy not available - line art processing will be disabled")

    try:
        from PIL import Image
        deps['PIL'] = True
        logger.debug("✓ Pillow available")
    except ImportError:
        logger.warning("✗ Pillow not available - PDF generation may fail")

    try:
        from reportlab.pdfgen import canvas
        deps['reportlab'] = True
        logger.debug("✓ ReportLab available")
    except ImportError:
        logger.warning("✗ ReportLab not available - PDF generation will be disabled")

    return deps


def validate_image_quality(image_path: Path,
                          min_width: int = MIN_PRINT_WIDTH,
                          min_height: int = MIN_PRINT_HEIGHT,
                          check_binary: bool = True,
                          required_dpi: int = REQUIRED_DPI) -> Dict[str, Any]:
    """Validate that a generated image meets print quality standards.

    Args:
        image_path: Path to the image file
        min_width: Minimum acceptable width in pixels
        min_height: Minimum acceptable height in pixels
        check_binary: Whether to check for pure black & white
        required_dpi: Required DPI for print quality

    Returns:
        Dict with validation results and any issues found
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return {
            'valid': False,
            'error': 'PIL/NumPy not available for validation',
            'warnings': []
        }

    results = {
        'valid': True,
        'warnings': [],
        'issues': [],
        'metrics': {}
    }

    try:
        img = Image.open(image_path)
        width, height = img.size

        # Check dimensions
        results['metrics']['width'] = width
        results['metrics']['height'] = height
        results['metrics']['physical_width'] = width / required_dpi
        results['metrics']['physical_height'] = height / required_dpi

        if width < min_width or height < min_height:
            results['valid'] = False
            results['issues'].append(
                f"CRITICAL: Image too small ({width}x{height}), needs {min_width}x{min_height} for print"
            )

        # Check DPI metadata
        dpi = img.info.get('dpi', (72, 72))
        results['metrics']['dpi'] = dpi

        if dpi[0] < required_dpi or dpi[1] < required_dpi:
            results['warnings'].append(
                f"DPI metadata is {dpi}, should be {required_dpi} for print"
            )

        # Check color mode
        results['metrics']['mode'] = img.mode

        # Check binary (pure black & white) if requested
        if check_binary:
            arr = np.array(img)
            unique_colors = len(np.unique(arr))
            results['metrics']['unique_colors'] = unique_colors

            if unique_colors > MAX_ACCEPTABLE_COLORS:
                results['warnings'].append(
                    f"Image has {unique_colors} colors, expected {MAX_ACCEPTABLE_COLORS} for line art"
                )

        # Check file size (should be reasonable, not too small or huge)
        file_size = image_path.stat().st_size
        results['metrics']['file_size_kb'] = file_size / 1024

        if file_size < MIN_FILE_SIZE_BYTES:
            results['warnings'].append(
                f"File size very small ({file_size/1024:.1f}KB), may indicate generation issue"
            )
        elif file_size > MAX_FILE_SIZE_BYTES:
            results['warnings'].append(
                f"File size very large ({file_size/1024/1024:.1f}MB), may cause upload issues"
            )

        # Log validation summary
        if results['valid'] and not results['warnings']:
            logger.debug(f"✓ Quality validation passed: {image_path.name}")
        elif results['warnings']:
            logger.warning(f"⚠ Quality warnings for {image_path.name}: {', '.join(results['warnings'])}")

    except Exception as e:
        results['valid'] = False
        results['error'] = f"Validation failed: {str(e)}"
        logger.error(f"Validation error for {image_path}: {e}")

    return results


def check_system_resources() -> Dict[str, Any]:
    """Check available system resources before generation.

    Returns:
        Dict with resource availability information
    """
    import shutil

    resources = {
        'disk_space_available': True,
        'memory_available': True,
        'warnings': []
    }

    # Check disk space in output directory
    try:
        stat = shutil.disk_usage(Path.cwd())
        free_gb = stat.free / (1024**3)
        resources['free_disk_gb'] = free_gb

        if free_gb < 1:
            resources['disk_space_available'] = False
            resources['warnings'].append(f"Low disk space: {free_gb:.2f} GB free")
        elif free_gb < 5:
            resources['warnings'].append(f"Limited disk space: {free_gb:.2f} GB free")
    except Exception as e:
        resources['warnings'].append(f"Could not check disk space: {e}")

    return resources


def preflight_checks(force_lineart: bool = False) -> bool:
    """Run all pre-flight checks before generation.

    Args:
        force_lineart: Whether line art processing is required

    Returns:
        True if all critical checks pass, False otherwise
    """
    logger.info("Running pre-flight checks...")

    all_ok = True

    # Check dependencies
    deps = validate_dependencies()

    if force_lineart and not (deps['opencv'] and deps['numpy']):
        logger.error("CRITICAL: Line art processing requires OpenCV and NumPy")
        logger.error("Install with: pip install opencv-python numpy")
        all_ok = False

    # Check system resources
    resources = check_system_resources()

    if not resources['disk_space_available']:
        logger.error("CRITICAL: Insufficient disk space")
        all_ok = False

    for warning in resources.get('warnings', []):
        logger.warning(warning)

    if all_ok:
        logger.info("✓ All pre-flight checks passed")
    else:
        logger.error("✗ Pre-flight checks failed - cannot proceed")

    return all_ok


# ==================== END QUALITY ASSURANCE FUNCTIONS ====================

# ==================== THEME LOADING ====================

def load_themes(themes_file: Path = None) -> Dict[str, Dict]:
    """Load coloring book themes from JSON file.

    Args:
        themes_file: Path to themes JSON file (default: themes/themes.json)

    Returns:
        Dictionary of themes

    Raises:
        FileNotFoundError: If themes file not found
        json.JSONDecodeError: If themes file is invalid
    """
    if themes_file is None:
        # Default to themes/themes.json relative to this file
        script_dir = Path(__file__).parent
        themes_file = script_dir / "themes" / "themes.json"

    if not themes_file.exists():
        logger.error(f"Themes file not found: {themes_file}")
        raise FileNotFoundError(f"Themes file not found: {themes_file}")

    try:
        with open(themes_file, 'r') as f:
            themes = json.load(f)
        logger.debug(f"Loaded {len(themes)} themes from {themes_file}")
        return themes
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in themes file: {e}")
        raise


# Load themes at module level
try:
    THEMES = load_themes()
except FileNotFoundError:
    logger.warning("Themes file not found, using empty themes dict")
    THEMES = {}
except Exception as e:
    logger.warning(f"Error loading themes: {e}, using empty themes dict")
    THEMES = {}

# ==================== END THEME LOADING ====================

# Legacy theme data replaced with JSON configuration
# THEMES dictionary moved to themes/themes.json for easier maintenance



class ColoringBookGenerator:
    """Generate adult coloring book pages using AI."""

    def __init__(self, output_dir: str = "output", backend: str = "huggingface", force_lineart: bool = False, lineart_method: str = "enhanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_token = REPLICATE_API_TOKEN
        self.hf_token = HF_TOKEN
        self.backend = backend  # "huggingface", "pollinations", or "replicate"
        self.force_lineart = force_lineart  # Post-process to pure B&W line art
        self.lineart_method = lineart_method  # "enhanced", "standard", or "detailed"

    def generate_image_replicate(self, prompt: str, size: tuple = (2550, 3300)) -> Optional[str]:
        """Generate a coloring page using Replicate API."""

        # Enhanced prompt for line art
        full_prompt = f"{prompt}, black and white line art, no shading, no gradients, no gray tones, pure black lines on pure white background, high contrast, suitable for coloring, clean crisp lines, 300 DPI print quality"

        negative_prompt = "color, colored, shading, gradient, gray, grayscale, shadows, blur, low quality, photograph, realistic, 3d render"

        try:
            # Using SDXL for high quality line art
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Token {self.api_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",  # SDXL
                    "input": {
                        "prompt": full_prompt,
                        "negative_prompt": negative_prompt,
                        "width": min(size[0], 1024),  # SDXL max
                        "height": min(size[1], 1024),
                        "num_inference_steps": 30,
                        "guidance_scale": 7.5,
                    }
                },
                timeout=30
            )

            if response.status_code != 201:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None

            prediction = response.json()
            prediction_id = prediction["id"]

            # Poll for completion
            for _ in range(60):  # 5 minute timeout
                time.sleep(5)

                status_response = requests.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Token {self.api_token}"},
                    timeout=30
                )

                if status_response.status_code == 200:
                    result = status_response.json()
                    if result["status"] == "succeeded":
                        output = result.get("output")
                        if output:
                            return output[0] if isinstance(output, list) else output
                    elif result["status"] == "failed":
                        logger.error(f"Generation failed: {result.get('error')}")
                        return None

            logger.error("Generation timed out")
            return None

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None

    def generate_image_pollinations(self, prompt: str, size: tuple = (1024, 1024)) -> Optional[bytes]:
        """Generate a coloring page using Pollinations.ai (completely free, no API key)."""
        import urllib.parse

        # IMPROVED: Generate high-contrast images that convert well to line art
        # Instead of trying to force the AI to make line art, we generate clear images
        # and convert them with our enhanced edge detection
        if self.force_lineart:
            # Optimized for edge detection conversion: high contrast, clear shapes
            full_prompt = f"{prompt}, high contrast illustration, clear defined edges, bold shapes, simple clean design, flat colors, cartoon style, clear outlines, no gradients, no blur, sharp edges, simple composition"
        else:
            # Traditional approach for direct line art generation
            full_prompt = f"{prompt}, BOLD AND EASY coloring book style, EXTRA THICK black outlines, LARGE simple shapes, MINIMAL detail, pure black lines on white background, NO shading NO gradients NO gray tones, simple chunky shapes easy to color, thick bold lineart, cartoon style simplicity, big areas to fill with color, no intricate patterns, no fine details"

        # URL-encode the prompt
        encoded_prompt = urllib.parse.quote(full_prompt)

        # Pollinations.ai API - completely free, no auth needed
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={size[0]}&height={size[1]}&nologo=true&model=flux"

        try:
            logger.info(f"  Requesting from Pollinations.ai...")
            response = requests.get(url, timeout=120)

            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type or len(response.content) > 10000:
                    logger.info(f"  Success! Got {len(response.content)} bytes")
                    return response.content
                else:
                    logger.warning(f"  Unexpected response: {content_type}")
            else:
                logger.warning(f"  Pollinations error: {response.status_code}")

        except requests.exceptions.Timeout:
            logger.warning("  Pollinations timed out, retrying...")
            try:
                response = requests.get(url, timeout=180)
                if response.status_code == 200:
                    return response.content
            except Exception as e:
                logger.warning(f"  Retry failed: {e}")
        except Exception as e:
            logger.warning(f"  Pollinations error: {e}")

        return None

    def generate_image_huggingface(self, prompt: str, size: tuple = (1024, 1024)) -> Optional[bytes]:
        """Generate a coloring page using HuggingFace free inference API."""

        # Enhanced prompt for line art - STRONG enforcement
        full_prompt = f"{prompt}, pure black ink lines on pure white paper, adult coloring book page, clean vector line art, no fills, no shading, no gradients, no gray, no halftones, only black outlines on white, high contrast linework, professional coloring book illustration, thick clean outlines ready for coloring"

        negative_prompt = "color, colored, red, blue, green, yellow, orange, purple, pink, shading, gradient, gray, grayscale, shadows, blur, photograph, realistic, 3d render, painted, watercolor, text, words, letters, writing, caption, label, signature, logo"

        # Try the new router API with different models
        models = [
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-2-1",
        ]

        for model in models:
            try:
                logger.info(f"  Trying model: {model}")

                # Use new router.huggingface.co endpoint
                response = requests.post(
                    f"https://router.huggingface.co/hf-inference/models/{model}",
                    headers={
                        "Authorization": f"Bearer {self.hf_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": full_prompt,
                        "parameters": {
                            "negative_prompt": negative_prompt,
                            "num_inference_steps": 25,
                            "guidance_scale": 7.5,
                        }
                    },
                    timeout=180
                )

                if response.status_code == 200:
                    # Response is the image bytes directly
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type:
                        logger.info(f"  Success with {model}")
                        return response.content
                    else:
                        logger.warning(f"  Unexpected response type: {content_type}")
                elif response.status_code == 503:
                    # Model loading, wait and retry
                    logger.info(f"  Model loading, waiting...")
                    time.sleep(30)
                    # Retry same model
                    response = requests.post(
                        f"https://router.huggingface.co/hf-inference/models/{model}",
                        headers={
                            "Authorization": f"Bearer {self.hf_token}",
                            "Content-Type": "application/json"
                        },
                        json={"inputs": full_prompt},
                        timeout=180
                    )
                    if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
                        return response.content
                else:
                    logger.warning(f"  Model {model} failed: {response.status_code} - {response.text[:300]}")

            except requests.exceptions.Timeout:
                logger.warning(f"  Model {model} timed out")
            except Exception as e:
                logger.warning(f"  Model {model} error: {e}")

        return None

    def upscale_to_print_quality(self, image_bytes: bytes, target_size: tuple = (2550, 3300)) -> bytes:
        """Upscale image to print quality resolution if needed.

        Args:
            image_bytes: Input image as bytes
            target_size: Target size for print (default: 8.5x11" at 300 DPI)

        Returns:
            Upscaled image as bytes

        Raises:
            RuntimeError: If upscaling fails critically (prevents bad outputs)
        """
        try:
            import cv2
            import numpy as np
            from PIL import Image
            from io import BytesIO

            # Load image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("CRITICAL: Failed to decode image for upscaling")
                raise RuntimeError("Image decode failed - cannot ensure print quality")

            original_size = (img.shape[1], img.shape[0])

            # Check if upscaling needed
            if img.shape[0] < target_size[1] or img.shape[1] < target_size[0]:
                logger.info(f"  Upscaling from {img.shape[1]}x{img.shape[0]} to {target_size[0]}x{target_size[1]} for print quality")
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

                # QUALITY ASSURANCE: Verify upscaling actually worked
                if img.shape[1] != target_size[0] or img.shape[0] != target_size[1]:
                    logger.error(f"CRITICAL: Upscaling failed - got {img.shape[1]}x{img.shape[0]}, expected {target_size}")
                    raise RuntimeError("Upscaling verification failed")

                # Convert to PIL and save with DPI
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                pil_img.info['dpi'] = (300, 300)

                output = BytesIO()
                pil_img.save(output, format='PNG', dpi=(300, 300))

                # QUALITY ASSURANCE: Verify DPI was set
                output.seek(0)
                verify_img = Image.open(output)
                actual_dpi = verify_img.info.get('dpi', (0, 0))
                if actual_dpi[0] < 300 or actual_dpi[1] < 300:
                    logger.warning(f"DPI metadata issue: got {actual_dpi}, expected (300, 300)")

                output.seek(0)
                return output.getvalue()
            else:
                # Image already large enough, just set DPI metadata
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                output = BytesIO()
                pil_img.save(output, format='PNG', dpi=(300, 300))
                return output.getvalue()

        except ImportError as e:
            logger.error(f"CRITICAL: Missing dependencies for upscaling: {e}")
            logger.error("Cannot guarantee print quality without OpenCV/NumPy/Pillow")
            raise RuntimeError("Missing required dependencies - install opencv-python numpy Pillow")
        except Exception as e:
            logger.error(f"CRITICAL: Upscaling failed: {e}")
            logger.error("This means the image WILL NOT meet print quality standards!")
            raise RuntimeError(f"Upscaling failed critically: {e}")

    def convert_to_coloring_page(self, image_bytes: bytes, method: str = "enhanced",
                                 target_size: tuple = (2550, 3300)) -> bytes:
        """Post-process image to enforce pure black & white line art.

        Args:
            image_bytes: Input image as bytes
            method: 'enhanced' (thick bold lines), 'standard' (normal), or 'detailed' (fine lines)
            target_size: Target size for print (default: 8.5x11" at 300 DPI)

        Returns:
            Processed image as bytes

        Raises:
            RuntimeError: If processing fails critically
        """
        try:
            import cv2
            import numpy as np
            from io import BytesIO
            from PIL import Image

            # Load image from bytes
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("CRITICAL: Failed to decode image for line art conversion")
                raise RuntimeError("Image decode failed during line art conversion")

            # Upscale if needed for print quality (before edge detection)
            if img.shape[0] < target_size[1] or img.shape[1] < target_size[0]:
                logger.info(f"  Upscaling from {img.shape[1]}x{img.shape[0]} to {target_size[0]}x{target_size[1]} for print quality")
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

                # QUALITY ASSURANCE: Verify upscaling
                if img.shape[1] != target_size[0] or img.shape[0] != target_size[1]:
                    logger.error(f"CRITICAL: Upscaling in line art conversion failed")
                    raise RuntimeError("Upscaling verification failed during line art processing")

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if method == "enhanced":
                # ENHANCED METHOD: Thick bold lines perfect for coloring
                # 1. Bilateral filter: smooth while preserving edges
                bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

                # 2. Adaptive thresholding to find dark areas
                thresh = cv2.adaptiveThreshold(
                    bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )

                # 3. Canny edge detection
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 30, 100)

                # 4. Combine edges with thresholded image
                combined = cv2.bitwise_and(edges, cv2.bitwise_not(thresh))

                # 5. Dilate for THICK outlines (perfect for coloring)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                thick_edges = cv2.dilate(combined, kernel, iterations=2)

                # 6. Clean up small noise
                kernel_clean = np.ones((2, 2), np.uint8)
                cleaned = cv2.morphologyEx(thick_edges, cv2.MORPH_CLOSE, kernel_clean)

                # 7. Invert: black lines on white background
                result = cv2.bitwise_not(cleaned)

            elif method == "detailed":
                # DETAILED METHOD: Fine lines for intricate designs
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)

                # Sensitive edge detection for details
                edges = cv2.Canny(blurred, 20, 60)

                # Minimal dilation
                kernel = np.ones((1, 1), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)

                # Invert
                result = cv2.bitwise_not(edges)

            else:  # standard
                # STANDARD METHOD: Balanced approach
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 30, 100)

                # Medium thickness
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                edges = cv2.dilate(edges, kernel, iterations=1)

                result = cv2.bitwise_not(edges)

            # Final cleanup: ensure pure white background
            # Anything not pure black becomes white
            _, result = cv2.threshold(result, 250, 255, cv2.THRESH_BINARY)

            # QUALITY ASSURANCE: Verify pure binary output
            unique_values = len(np.unique(result))
            if unique_values > MAX_ACCEPTABLE_COLORS:
                logger.warning(f"Line art has {unique_values} colors instead of 2 - applying stronger threshold")
                # Apply more aggressive binarization
                _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
                unique_values = len(np.unique(result))
                if unique_values > MAX_ACCEPTABLE_COLORS:
                    logger.error(f"CRITICAL: Could not achieve pure binary output ({unique_values} colors)")

            # Convert to PIL to set DPI metadata
            pil_img = Image.fromarray(result)
            pil_img.info['dpi'] = (300, 300)  # Set 300 DPI for print quality

            # Save to bytes with DPI metadata
            output = BytesIO()
            pil_img.save(output, format='PNG', dpi=(300, 300))

            # QUALITY ASSURANCE: Verify final output
            output.seek(0)
            verify_img = Image.open(output)
            final_size = verify_img.size
            final_dpi = verify_img.info.get('dpi', (0, 0))

            if final_size[0] != target_size[0] or final_size[1] != target_size[1]:
                logger.error(f"CRITICAL: Final image size wrong: {final_size} vs {target_size}")
                raise RuntimeError("Line art conversion produced wrong size output")

            if final_dpi[0] < 300 or final_dpi[1] < 300:
                logger.warning(f"DPI metadata may not be set correctly: {final_dpi}")

            output.seek(0)
            return output.getvalue()

        except ImportError:
            logger.error("CRITICAL: OpenCV not installed - cannot do line art conversion")
            logger.error("Install with: pip install opencv-python numpy")
            raise RuntimeError("Missing dependencies for line art conversion")
        except RuntimeError:
            # Re-raise runtime errors (our own quality checks)
            raise
        except Exception as e:
            logger.error(f"CRITICAL: Line art conversion failed: {e}")
            raise RuntimeError(f"Line art conversion failed: {e}")

    def download_image(self, url: str, filepath: Path) -> bool:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                filepath.write_bytes(response.content)
                return True
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
        return False

    def _generate_image_bytes(self, prompt: str, size: tuple = (1536, 1536)) -> Optional[bytes]:
        """Generate image bytes using configured backend.

        Centralized backend selection logic shared between generators.

        Args:
            prompt: Image generation prompt
            size: Image size tuple (width, height)

        Returns:
            Image bytes on success, None on failure
        """
        if self.backend == "pollinations":
            return self.generate_image_pollinations(prompt, size=size)
        elif self.backend == "huggingface":
            return self.generate_image_huggingface(prompt)
        else:  # replicate
            import tempfile
            import uuid
            image_url = self.generate_image_replicate(prompt)
            if image_url:
                # Download to temp file then read bytes
                temp_path = Path(tempfile.gettempdir()) / f"temp_{uuid.uuid4()}.png"
                try:
                    if self.download_image(image_url, temp_path):
                        image_bytes = temp_path.read_bytes()
                        temp_path.unlink()  # Clean up
                        return image_bytes
                except Exception as e:
                    logger.error(f"Error reading temp image: {e}")
                    if temp_path.exists():
                        temp_path.unlink()
            return None

    def _post_process_image(self, image_bytes: bytes) -> bytes:
        """Post-process image (upscaling and/or line art).

        Extracted helper method to reduce code duplication.

        Args:
            image_bytes: Raw image bytes from generation

        Returns:
            Processed image bytes
        """
        if self.force_lineart:
            return self.convert_to_coloring_page(image_bytes, method=self.lineart_method)
        else:
            return self.upscale_to_print_quality(image_bytes)

    def _save_book_metadata(self, book_dir: Path, title: str, theme: str,
                           num_pages: int, generated_pages: list, timestamp: str):
        """Save book metadata to JSON file.

        Extracted helper method for better organization.

        Args:
            book_dir: Book directory path
            title: Book title
            theme: Theme name
            num_pages: Total pages requested
            generated_pages: List of successfully generated pages
            timestamp: Generation timestamp
        """
        metadata = {
            "title": title,
            "theme": theme,
            "pages": num_pages,
            "generated": len(generated_pages),
            "timestamp": timestamp,
            "pages_data": generated_pages
        }

        metadata_file = book_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

    def _create_book_dirs(self, title: str, timestamp: str) -> Tuple[Path, Path]:
        """Create the directory tree for a new book.

        Args:
            title: Book title (spaces become "_" and ":" is dropped in the directory name)
            timestamp: Generation timestamp, appended to the directory name

        Returns:
            Tuple of (book_dir, images_dir)
        """
        safe_title = title.replace(" ", "_").replace(":", "")
        book_dir = self.output_dir / f"{safe_title}_{timestamp}"
        book_dir.mkdir(parents=True, exist_ok=True)
        images_dir = book_dir / "images"
        images_dir.mkdir(exist_ok=True)
        return book_dir, images_dir

    def _build_page_prompt(self, prompts: List[str], index: int) -> str:
        """Build the prompt for one page.

        Cycles through the theme prompts and adds variety using the shared
        variations. Draws from the global random state once per call, so call
        it page by page, just before generating the page.

        Args:
            prompts: Theme prompts, used in rotation
            index: Page index (0-indexed)

        Returns:
            Prompt with a randomly chosen variation appended
        """
        return prompts[index % len(prompts)] + random.choice(PROMPT_VARIATIONS)

    def _validate_page_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Run quality validation on a saved page image and log the outcome.

        Args:
            image_path: Path to the saved page image

        Returns:
            The validation result, or None if the image failed validation
        """
        validation = validate_image_quality(
            image_path,
            check_binary=self.force_lineart
        )

        if not validation['valid']:
            logger.error(f"  ✗ Quality validation FAILED for {image_path.name}")
            for issue in validation.get('issues', []):
                logger.error(f"    - {issue}")
            logger.error("  This is a CRITICAL error - image will not print correctly!")
            return None

        if validation.get('warnings'):
            logger.warning(f"  ⚠ Quality warnings for {image_path.name}:")
            for warning in validation['warnings']:
                logger.warning(f"    - {warning}")

        return validation

    def _generate_and_validate_page(self, page_num: int, prompt: str,
                                    images_dir: Path) -> Optional[Dict[str, Any]]:
        """Generate, save and quality-check a single page.

        Args:
            page_num: Page number (1-indexed)
            prompt: Image generation prompt
            images_dir: Directory to save the page image in

        Returns:
            Page metadata dict if the page was generated and passed validation,
            None otherwise
        """
        image_path = images_dir / f"page_{page_num:03d}.png"

        # Generate image using centralized backend logic
        image_bytes = self._generate_image_bytes(prompt, size=(1536, 1536))
        if not image_bytes:
            logger.warning(f"  ✗ Failed to generate page {page_num}")
            return None

        # Post-process (upscale and/or convert to line art)
        image_path.write_bytes(self._post_process_image(image_bytes))

        # QUALITY ASSURANCE: Validate the generated image
        validation = self._validate_page_image(image_path)
        if validation is None:
            # Don't add to generated list if validation fails critically
            return None

        logger.info(f"  ✓ Saved and validated: {image_path.name}")
        # Add metrics to metadata
        return {
            "page": page_num,
            "prompt": prompt,
            "file": str(image_path.name),
            "quality_metrics": validation.get('metrics', {})
        }

    def _generate_all_pages(self, prompts: List[str], num_pages: int,
                            images_dir: Path) -> List[Dict[str, Any]]:
        """Generate every page of a book, one after another.

        Args:
            prompts: Theme prompts, used in rotation
            num_pages: Number of pages to generate
            images_dir: Directory to save the page images in

        Returns:
            Metadata dicts for the pages that were generated and passed validation
        """
        generated = []

        for i in range(num_pages):
            prompt = self._build_page_prompt(prompts, i)

            logger.info(f"Generating page {i+1}/{num_pages}...")

            page_data = self._generate_and_validate_page(i + 1, prompt, images_dir)
            if page_data:
                generated.append(page_data)

            # Rate limiting
            time.sleep(2)

        return generated

    def generate_book(self, theme: str, num_pages: int = 30, book_title: str = None) -> Path:
        """Generate a complete coloring book."""

        # PRE-FLIGHT CHECKS: Ensure system is ready
        if not preflight_checks(force_lineart=self.force_lineart):
            raise RuntimeError("Pre-flight checks failed - cannot generate book safely")

        if theme not in THEMES:
            raise ValueError(f"Unknown theme: {theme}. Available: {list(THEMES.keys())}")

        theme_data = THEMES[theme]
        title = book_title or theme_data["name"]

        # Create book directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        book_dir, images_dir = self._create_book_dirs(title, timestamp)

        logger.info(f"Generating coloring book: {title}")
        logger.info(f"Theme: {theme}, Pages: {num_pages}")
        logger.info(f"Output: {book_dir}")

        # Generate pages
        generated = self._generate_all_pages(theme_data["prompts"], num_pages, images_dir)

        # Save metadata using helper method
        self._save_book_metadata(book_dir, title, theme, num_pages, generated, timestamp)

        logger.info(f"Generated {len(generated)}/{num_pages} pages")
        logger.info(f"Book saved to: {book_dir}")

        return book_dir

    def create_pdf(self, book_dir: Path, size: str = "8.5x11") -> Optional[Path]:
        """Compile images into a print-ready PDF."""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
            from PIL import Image
        except ImportError:
            logger.error("Install reportlab and Pillow: pip install reportlab Pillow")
            return None

        images_dir = book_dir / "images"
        if not images_dir.exists():
            logger.error(f"Images directory not found: {images_dir}")
            return None

        # Get all images
        images = sorted(images_dir.glob("*.png"))
        if not images:
            logger.error("No images found")
            return None

        # Load metadata
        metadata_file = book_dir / "metadata.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            title = metadata.get("title", "Coloring Book")
        else:
            title = "Coloring Book"

        # Create PDF
        pdf_path = book_dir / f"{title.replace(' ', '_')}.pdf"

        # Use letter size for KDP
        page_width, page_height = letter

        c = canvas.Canvas(str(pdf_path), pagesize=letter)

        # Add title page
        c.setFont("Helvetica-Bold", 36)
        c.drawCentredString(page_width/2, page_height/2 + 50, title)
        c.setFont("Helvetica", 18)
        c.drawCentredString(page_width/2, page_height/2 - 20, "Adult Coloring Book")
        c.showPage()

        # Add coloring pages
        margin = 0.5 * inch
        content_width = page_width - 2 * margin
        content_height = page_height - 2 * margin

        for img_path in images:
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size

                # Calculate scaling to fit page
                scale = min(content_width / img_width, content_height / img_height)
                new_width = img_width * scale
                new_height = img_height * scale

                # Center on page
                x = (page_width - new_width) / 2
                y = (page_height - new_height) / 2

                c.drawImage(str(img_path), x, y, new_width, new_height)
                c.showPage()

            except Exception as e:
                logger.error(f"Error adding {img_path}: {e}")

        c.save()
        logger.info(f"PDF created: {pdf_path}")

        return pdf_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Adult Coloring Books for KDP")
    parser.add_argument("--theme", choices=list(THEMES.keys()), default="mandalas",
                       help="Coloring book theme")
    parser.add_argument("--pages", type=int, default=30,
                       help="Number of pages (default: 30)")
    parser.add_argument("--title", type=str, default=None,
                       help="Custom book title")
    parser.add_argument("--output", type=str, default="output",
                       help="Output directory")
    parser.add_argument("--pdf", action="store_true",
                       help="Also create PDF")
    parser.add_argument("--list-themes", action="store_true",
                       help="List available themes")
    parser.add_argument("--backend", choices=["pollinations", "huggingface", "replicate"], default="pollinations",
                       help="Image generation backend (default: pollinations - free, unlimited)")
    parser.add_argument("--force-lineart", action="store_true",
                       help="Post-process images to pure black & white line art using edge detection")
    parser.add_argument("--lineart-method", choices=["enhanced", "standard", "detailed"], default="enhanced",
                       help="Line art extraction method: enhanced (thick bold lines), standard (balanced), detailed (fine lines)")

    args = parser.parse_args()

    if args.list_themes:
        print("\nAvailable Themes:")
        print("-" * 40)
        for key, data in THEMES.items():
            print(f"  {key}: {data['name']}")
            print(f"    Prompts: {len(data['prompts'])}")
        return

    generator = ColoringBookGenerator(
        output_dir=args.output,
        backend=args.backend,
        force_lineart=args.force_lineart,
        lineart_method=args.lineart_method
    )

    book_dir = generator.generate_book(
        theme=args.theme,
        num_pages=args.pages,
        book_title=args.title
    )

    if args.pdf and book_dir:
        generator.create_pdf(book_dir)


if __name__ == "__main__":
    main()
