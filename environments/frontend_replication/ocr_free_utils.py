"""
OCR-free text block detection via color injection.

Implements Design2Code's approach (Si et al., NAACL 2025):
1. Inject unique colors into each text element in the HTML
2. Render two versions with offset colors (0 and +50)
3. Diff pixels between the two renders to isolate text regions
4. Map pixel clusters back to text elements via their injected colors
5. Extract bounding boxes, text content, and original colors

This avoids OCR entirely by leveraging HTML structure and color tracking.
"""

from __future__ import annotations

import asyncio
import logging
import re
from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np
from bs4 import BeautifulSoup, NavigableString

logger = logging.getLogger(__name__)

# Text-bearing HTML elements to inject colors into
TEXT_TAGS = frozenset(
    [
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "div",
        "span",
        "a",
        "b",
        "strong",
        "em",
        "i",
        "li",
        "td",
        "th",
        "button",
        "label",
        "header",
        "footer",
        "figcaption",
        "blockquote",
        "code",
        "pre",
        "nav",
        "section",
        "article",
        "main",
        "aside",
    ]
)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert (R, G, B) tuple to #RRGGBB hex string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert #RRGGBB hex string to (R, G, B) tuple."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


class ColorPool:
    """
    Pool of unique colors for injection into HTML text elements.

    Generates 15^3 = 3,375 unique colors by varying RGB values
    from 10-250 in steps of 16. Supports an offset parameter
    for creating color-shifted versions (used for pixel diffing).
    """

    def __init__(self, offset: int = 0):
        color_values = list(range(10, 251, 16))  # [10, 26, 42, ..., 250]
        self._colors = [
            rgb_to_hex(
                (
                    (r + offset) % 256,
                    (g + offset) % 256,
                    (b + offset) % 256,
                )
            )
            for r in color_values
            for g in color_values
            for b in color_values
        ]
        self._index = 0

    def pop(self) -> str:
        """Return next unique color hex string."""
        if self._index >= len(self._colors):
            # Wrap around if we exceed the pool (unlikely: 3375 colors)
            self._index = 0
            logger.warning("ColorPool exhausted, wrapping around")
        color = self._colors[self._index]
        self._index += 1
        return color

    @property
    def used(self) -> int:
        return self._index


def _get_direct_text(element) -> str:
    """Get direct text content of an element (not from children)."""
    texts = []
    for child in element.children:
        if isinstance(child, NavigableString) and not isinstance(
            child, (type(None),)
        ):
            text = str(child).strip()
            if text:
                texts.append(text)
    return " ".join(texts)


def _get_all_text(element) -> str:
    """Get all text content from an element and its descendants."""
    return element.get_text(separator=" ", strip=True)


def process_html(
    html: str, offset: int = 0
) -> tuple[str, list[tuple[str, str]]]:
    """
    Inject unique colors into text elements in the HTML.

    Args:
        html: Raw HTML string
        offset: Color offset (0 for base, 50 for shifted version)

    Returns:
        (modified_html, text_color_map) where text_color_map is a list of
        (text_content, hex_color) tuples for each injected element.
    """
    soup = BeautifulSoup(html, "html.parser")
    pool = ColorPool(offset=offset)
    text_color_map: list[tuple[str, str]] = []

    # First: clear all background colors to transparent
    for tag in soup.find_all(True):
        existing_style = tag.get("style", "")
        # Remove existing color and background-color
        existing_style = re.sub(
            r"(?:^|;)\s*(?:color|background-color)\s*:[^;]*",
            "",
            existing_style,
        )
        # Set background to transparent
        tag["style"] = (
            f"{existing_style}; background-color: rgba(255,255,255,0.0) !important;"
        )

    # Second: inject unique colors into text-bearing elements
    for tag in soup.find_all(TEXT_TAGS):
        text = _get_direct_text(tag)
        if not text:
            # Try getting all text if no direct text
            text = _get_all_text(tag)
        if not text:
            continue

        color = pool.pop()
        existing_style = tag.get("style", "")
        tag["style"] = (
            f"{existing_style}; color: {color} !important; opacity: 1.0 !important;"
        )
        text_color_map.append((text.lower(), color))

    logger.debug(
        f"Injected {pool.used} colors (offset={offset}), "
        f"{len(text_color_map)} text elements"
    )

    return str(soup), text_color_map


def find_different_pixels(
    img1: np.ndarray, img2: np.ndarray, tolerance: int = 8
) -> np.ndarray:
    """
    Find pixels that differ between two color-shifted renders.

    Design2Code approach: detect pixels where the color shifted by
    approximately 50 (the offset) with ±tolerance on each channel.

    Args:
        img1: First render (offset=0), BGR format from OpenCV
        img2: Second render (offset=50), BGR format from OpenCV
        tolerance: Per-channel tolerance for detecting shifted pixels

    Returns:
        Nx2 array of (row, col) coordinates of changed pixels
    """
    if img1.shape != img2.shape:
        # Resize to match if needed
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]

    # Convert to int16 to handle the modular arithmetic
    i1 = img1.astype(np.int16)
    i2 = img2.astype(np.int16)

    # Check if (pixel1 + 50) % 256 ≈ pixel2
    # Equivalent: abs((pixel2 - pixel1) % 256 - 50) <= tolerance
    diff = (i2 - i1) % 256
    shifted = np.abs(diff - 50) <= tolerance

    # All 3 channels must match the shift
    all_channels_shifted = np.all(shifted, axis=2)

    # Get coordinates
    coords = np.argwhere(all_channels_shifted)  # Nx2: (row, col)
    return coords


def _intersect_coords(
    coords1: np.ndarray, coords2: np.ndarray
) -> np.ndarray:
    """Find intersection of two Nx2 coordinate arrays."""
    if coords1.size == 0 or coords2.size == 0:
        return np.array([], dtype=np.int64).reshape(0, 2)

    # Use set intersection for efficiency
    set1 = set(map(tuple, coords1))
    set2 = set(map(tuple, coords2))
    common = set1 & set2

    if not common:
        return np.array([], dtype=np.int64).reshape(0, 2)

    return np.array(list(common), dtype=np.int64)


@dataclass
class TextBlock:
    """A detected text block from color injection."""

    text: str
    bbox: tuple[float, float, float, float]  # (x, y, w, h) normalized [0,1]
    color_rgb: tuple[int, int, int]  # Average original color (RGB)


async def get_blocks_ocr_free(
    html: str,
    width: int = 1280,
    height: int = 800,
    render_fn=None,
) -> list[TextBlock]:
    """
    Extract text blocks from HTML using Design2Code's color-injection method.

    Args:
        html: HTML source to analyze
        width: Viewport width for rendering
        height: Viewport height for rendering
        render_fn: Async function(html, width, height) -> bytes (PNG).
                   Defaults to render_html_playwright.

    Returns:
        List of TextBlock with text content, normalized bounding boxes, and colors.
    """
    if render_fn is None:
        from frontend_replication import render_html_playwright

        render_fn = render_html_playwright

    # Step 1: Create two color-injected versions
    html_v1, text_color_map = process_html(html, offset=0)
    html_v2, _ = process_html(html, offset=50)

    if not text_color_map:
        logger.warning("No text elements found in HTML")
        return []

    # Step 2: Render both versions
    png_v1 = await render_fn(html_v1, width=width, height=height)
    png_v2 = await render_fn(html_v2, width=width, height=height)

    # Also render the original (unmodified) HTML for true color sampling
    png_orig = await render_fn(html, width=width, height=height)

    # Decode PNGs to numpy arrays
    img_v1 = cv2.imdecode(
        np.frombuffer(png_v1, np.uint8), cv2.IMREAD_COLOR
    )
    img_v2 = cv2.imdecode(
        np.frombuffer(png_v2, np.uint8), cv2.IMREAD_COLOR
    )
    img_orig = cv2.imdecode(
        np.frombuffer(png_orig, np.uint8), cv2.IMREAD_COLOR
    )

    if img_v1 is None or img_v2 is None or img_orig is None:
        logger.error("Failed to decode rendered images")
        return []

    h_img, w_img = img_v1.shape[:2]

    # Step 3: Find pixels that changed between the two renders
    diff_pixels = find_different_pixels(img_v1, img_v2)
    if diff_pixels.size == 0:
        logger.warning("No pixel differences found between color-shifted renders")
        return []

    logger.debug(f"Found {len(diff_pixels)} different pixels")

    # Step 4: For each text element, find its bounding box
    blocks: list[TextBlock] = []
    diff_set = set(map(tuple, diff_pixels))

    for text, color_hex in text_color_map:
        color_rgb = hex_to_rgb(color_hex)
        # Convert to BGR for OpenCV
        color_bgr = np.array(
            [color_rgb[2], color_rgb[1], color_rgb[0]], dtype=np.uint8
        )

        # Create mask: pixels matching this injected color (±4 tolerance)
        lower = np.clip(color_bgr.astype(np.int16) - 4, 0, 255).astype(
            np.uint8
        )
        upper = np.clip(color_bgr.astype(np.int16) + 4, 0, 255).astype(
            np.uint8
        )
        mask = cv2.inRange(img_v1, lower, upper)

        # Get coordinates of matching pixels
        color_coords = np.argwhere(mask > 0)  # Nx2: (row, col)

        if color_coords.size == 0:
            continue

        # Intersect with diff pixels (only keep pixels that actually changed)
        filtered = []
        for r, c in color_coords:
            if (r, c) in diff_set:
                filtered.append((r, c))

        if not filtered:
            continue

        filtered = np.array(filtered)

        # Compute bounding box
        y_min, x_min = filtered.min(axis=0)
        y_max, x_max = filtered.max(axis=0)

        # Skip degenerate boxes
        box_w = x_max - x_min + 1
        box_h = y_max - y_min + 1
        if box_w < 3 or box_h < 3:
            continue

        # Normalized coordinates [0, 1]
        bbox = (
            x_min / w_img,
            y_min / h_img,
            box_w / w_img,
            box_h / h_img,
        )

        # Sample original color from the unmodified render
        region = img_orig[y_min : y_max + 1, x_min : x_max + 1]
        if region.size > 0:
            avg_bgr = cv2.mean(region)[:3]
            avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
        else:
            avg_rgb = (0, 0, 0)

        blocks.append(
            TextBlock(
                text=text,
                bbox=bbox,
                color_rgb=avg_rgb,
            )
        )

    logger.info(
        f"Detected {len(blocks)} text blocks from {len(text_color_map)} elements"
    )
    return blocks
