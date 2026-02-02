"""
Design2Code-style visual scoring pipeline for frontend replication benchmark.

Implements five equally-weighted metrics (each 0-1):
1. Block-Match (Size): Hungarian matching of visual blocks by bounding box area
2. Text: SequenceMatcher on matched block text content
3. Position: Chebyshev distance between matched block positions
4. Color: CIEDE2000 delta-E between matched block colors
5. CLIP: Cosine similarity of CLIP embeddings on inpainted screenshots

Reference: Design2Code (Si et al., NAACL 2025)
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@dataclass
class VisualBlock:
    """A detected visual block (text element) from a rendered webpage."""

    x: float
    y: float
    width: float
    height: float
    text: str = ""
    color: tuple[int, int, int] = (0, 0, 0)  # RGB

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class ScoringResult:
    """Result of scoring a generated page against a reference."""

    size_score: float = 0.0
    text_score: float = 0.0
    position_score: float = 0.0
    color_score: float = 0.0
    clip_score: float = 0.0
    final_score: float = 0.0
    num_ref_blocks: int = 0
    num_gen_blocks: int = 0
    num_matched: int = 0


async def detect_blocks_from_html(
    html: str,
    width: int = 1280,
    height: int = 800,
    render_fn=None,
) -> list[VisualBlock]:
    """
    Detect text blocks from HTML using Design2Code's color-injection method.

    This is the preferred method — it yields exact text content and accurate
    bounding boxes by injecting unique colors into DOM elements and diffing
    two offset renders.

    Returns blocks with normalized [0,1] coordinates.
    """
    from ocr_free_utils import TextBlock, get_blocks_ocr_free

    raw_blocks: list[TextBlock] = await get_blocks_ocr_free(
        html, width=width, height=height, render_fn=render_fn,
    )

    visual_blocks = [
        VisualBlock(
            x=tb.bbox[0],
            y=tb.bbox[1],
            width=tb.bbox[2],
            height=tb.bbox[3],
            text=tb.text,
            color=tb.color_rgb,
        )
        for tb in raw_blocks
    ]
    return visual_blocks


def find_possible_merge(
    blocks: list[VisualBlock],
    cost_threshold: float = 0.05,
) -> list[VisualBlock]:
    """
    Iteratively merge consecutive blocks if merging improves matching cost.

    Design2Code merges adjacent text blocks that likely belong to the same
    visual element (e.g., a heading split across <span> tags). Two consecutive
    blocks are merged if they are horizontally adjacent (same y-band) and
    the merged block's area is close to the sum of the individual areas.
    """
    if len(blocks) <= 1:
        return blocks

    merged = list(blocks)
    changed = True
    while changed:
        changed = False
        new_merged = []
        i = 0
        while i < len(merged):
            if i + 1 < len(merged):
                b1 = merged[i]
                b2 = merged[i + 1]

                # Check if blocks are vertically overlapping (same row)
                y1_min, y1_max = b1.y, b1.y + b1.height
                y2_min, y2_max = b2.y, b2.y + b2.height
                overlap = min(y1_max, y2_max) - max(y1_min, y2_min)
                min_h = min(b1.height, b2.height)

                if min_h > 0 and overlap / min_h > 0.5:
                    # Horizontally close?
                    gap = abs((b2.x) - (b1.x + b1.width))
                    if gap < 0.05:  # 5% of page width
                        # Merge
                        x_min = min(b1.x, b2.x)
                        y_min = min(b1.y, b2.y)
                        x_max = max(b1.x + b1.width, b2.x + b2.width)
                        y_max = max(b1.y + b1.height, b2.y + b2.height)
                        merged_text = (b1.text + " " + b2.text).strip()
                        # Average color
                        avg_color = tuple(
                            (c1 + c2) // 2 for c1, c2 in zip(b1.color, b2.color)
                        )
                        new_merged.append(
                            VisualBlock(
                                x=x_min,
                                y=y_min,
                                width=x_max - x_min,
                                height=y_max - y_min,
                                text=merged_text,
                                color=avg_color,
                            )
                        )
                        i += 2
                        changed = True
                        continue
            new_merged.append(merged[i])
            i += 1
        merged = new_merged

    return merged


def detect_blocks_from_screenshot(image: np.ndarray) -> list[VisualBlock]:
    """
    Detect visual blocks from a rendered webpage screenshot using contour detection.

    This is a simplified version of Design2Code's OCR-free block detection.
    It detects distinct visual regions (text blocks, buttons, cards, etc.)
    by finding contours in an edge-detected version of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection to find block boundaries
    edges = cv2.Canny(gray, 30, 100)

    # Dilate to connect nearby edges into blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = image.shape[:2]
    min_area = (h_img * w_img) * 0.0005  # min 0.05% of image area
    max_area = (h_img * w_img) * 0.9  # max 90% of image area

    blocks = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < min_area or area > max_area:
            continue
        if w < 10 or h < 5:
            continue

        # Sample dominant color from the block region (convert BGR -> RGB)
        region = image[y : y + h, x : x + w]
        avg_bgr = cv2.mean(region)[:3]
        avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))

        blocks.append(
            VisualBlock(
                x=float(x),
                y=float(y),
                width=float(w),
                height=float(h),
                text="",  # text extraction done separately if needed
                color=avg_rgb,
            )
        )

    return blocks



def inpaint_image_regions(
    image: np.ndarray,
    blocks: list[VisualBlock],
    image_aspect_threshold: float = 0.5,
) -> np.ndarray:
    """
    Inpaint detected image/logo regions using TELEA algorithm.

    Following Design2Code, we detect likely image regions (large blocks with
    low text content and photo-like properties) and inpaint them before
    computing CLIP similarity.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h_img, w_img = image.shape[:2]

    for block in blocks:
        x, y, w, h = int(block.x), int(block.y), int(block.width), int(block.height)
        region = image[y : y + h, x : x + w]

        if region.size == 0:
            continue

        # Heuristic: detect image-like regions
        # - Large relative area
        # - High color variance (photos have diverse colors)
        # - Aspect ratio typical of images
        area_ratio = (w * h) / (h_img * w_img)

        if area_ratio < 0.01:
            continue

        # Check color variance — images tend to have high variance
        std = np.std(region.astype(float))
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edge_density = np.mean(cv2.Canny(gray_region, 50, 150) > 0)

        # High color variance + moderate edges suggests a photo/image
        is_likely_image = std > 40 and edge_density > 0.05 and area_ratio > 0.02

        if is_likely_image:
            mask[y : y + h, x : x + w] = 255

    if np.any(mask):
        inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return inpainted
    return image.copy()


def compute_block_matching(
    ref_blocks: list[VisualBlock],
    gen_blocks: list[VisualBlock],
    img_width: float,
    img_height: float,
    text_sim_threshold: float = 0.3,
) -> list[tuple[int, int, float]]:
    """
    Match reference blocks to generated blocks using the Hungarian algorithm.

    Cost matrix based on spatial distance + text similarity.
    Returns list of (ref_idx, gen_idx, cost) tuples for matched pairs.
    """
    if not ref_blocks or not gen_blocks:
        return []

    n_ref = len(ref_blocks)
    n_gen = len(gen_blocks)
    cost_matrix = np.full((n_ref, n_gen), fill_value=1e6)

    for i, ref in enumerate(ref_blocks):
        for j, gen in enumerate(gen_blocks):
            # Design2Code: matching cost based on text similarity
            text_sim = 0.0
            if ref.text and gen.text:
                text_sim = difflib.SequenceMatcher(None, ref.text, gen.text).ratio()
            cost_matrix[i, j] = -text_sim  # negative because we minimize

    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        # cost is negative text_sim, so cost < -0.0 means some text match
        if cost_matrix[r, c] < -0.01:
            matches.append((r, c, cost_matrix[r, c]))

    return matches


def compute_size_score(
    ref_blocks: list[VisualBlock],
    gen_blocks: list[VisualBlock],
    matches: list[tuple[int, int, float]],
) -> float:
    """
    Size score: coverage × size-fidelity of matched pairs.

    Two components:
    1. Coverage: fraction of reference area that was matched (penalizes missing content)
    2. Fidelity: for matched pairs, how similar their sizes are (IoU of width×height)

    This avoids penalizing models that structure HTML differently but produce
    visually equivalent output (fewer/larger blocks vs many small blocks).
    """
    if not ref_blocks and not gen_blocks:
        return 1.0
    if not ref_blocks or not gen_blocks:
        return 0.0

    # Coverage: what fraction of reference area did we match?
    ref_total_area = sum(b.area for b in ref_blocks)
    matched_ref_area = sum(ref_blocks[r].area for r, c, _ in matches)
    coverage = matched_ref_area / ref_total_area if ref_total_area > 0 else 0.0

    # Fidelity: for matched pairs, how close are their sizes?
    if not matches:
        return 0.0
    fidelity_sum = 0.0
    weight_sum = 0.0
    for r, c, _ in matches:
        ra = ref_blocks[r].area
        ga = gen_blocks[c].area
        if ra + ga > 0:
            # IoU-style: min/max gives 1.0 for identical sizes, <1 for mismatches
            fidelity_sum += min(ra, ga) / max(ra, ga) * ra
            weight_sum += ra
    fidelity = fidelity_sum / weight_sum if weight_sum > 0 else 0.0

    return coverage * fidelity


def compute_text_score(
    ref_blocks: list[VisualBlock],
    gen_blocks: list[VisualBlock],
    matches: list[tuple[int, int, float]],
) -> float:
    """
    Text score: average SequenceMatcher ratio across matched block pairs.

    Design2Code uses SequenceMatcher for text similarity.
    """
    if not matches:
        return 0.0

    scores = []
    for r, c, _ in matches:
        ref_text = ref_blocks[r].text
        gen_text = gen_blocks[c].text
        if ref_text or gen_text:
            sim = difflib.SequenceMatcher(None, ref_text, gen_text).ratio()
            scores.append(sim)

    return sum(scores) / len(ref_blocks) if ref_blocks else 0.0


def compute_position_score(
    ref_blocks: list[VisualBlock],
    gen_blocks: list[VisualBlock],
    matches: list[tuple[int, int, float]],
    img_width: float,
    img_height: float,
) -> float:
    """
    Position score: 1 - average normalized Chebyshev distance.
    """
    if not matches:
        return 0.0

    scores = []
    for r, c, _ in matches:
        cx_ref, cy_ref = ref_blocks[r].center
        cx_gen, cy_gen = gen_blocks[c].center
        dx = abs(cx_ref - cx_gen) / img_width if img_width > 0 else 0
        dy = abs(cy_ref - cy_gen) / img_height if img_height > 0 else 0
        dist = max(dx, dy)
        scores.append(max(0.0, 1.0 - dist))

    return sum(scores) / len(ref_blocks) if ref_blocks else 0.0


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert RGB to CIELAB color space via XYZ."""
    # Normalize RGB to [0, 1]
    r, g, b = [c / 255.0 for c in rgb]

    # sRGB to linear RGB
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = linearize(r), linearize(g), linearize(b)

    # Linear RGB to XYZ (D65 illuminant)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Normalize by D65 reference white
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    lightness = (116 * f(y)) - 16
    a = 500 * (f(x) - f(y))
    b_val = 200 * (f(y) - f(z))

    return (lightness, a, b_val)


def ciede2000(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    """
    Compute CIEDE2000 color difference.
    Simplified implementation for scoring purposes.
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Step 1: Calculate C'ab, h'ab
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2

    G = 0.5 * (1 - math.sqrt(C_avg**7 / (C_avg**7 + 25**7)))
    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)

    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)

    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    # Step 2: Calculate dL', dC', dH'
    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360

    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))

    # Step 3: Calculate CIEDE2000
    Lp_avg = (L1 + L2) / 2
    Cp_avg = (C1p + C2p) / 2

    if C1p * C2p == 0:
        hp_avg = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hp_avg = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hp_avg = (h1p + h2p + 360) / 2
    else:
        hp_avg = (h1p + h2p - 360) / 2

    T = (
        1
        - 0.17 * math.cos(math.radians(hp_avg - 30))
        + 0.24 * math.cos(math.radians(2 * hp_avg))
        + 0.32 * math.cos(math.radians(3 * hp_avg + 6))
        - 0.20 * math.cos(math.radians(4 * hp_avg - 63))
    )

    SL = 1 + (0.015 * (Lp_avg - 50) ** 2) / math.sqrt(20 + (Lp_avg - 50) ** 2)
    SC = 1 + 0.045 * Cp_avg
    SH = 1 + 0.015 * Cp_avg * T

    RT_term = (
        -2
        * math.sqrt(Cp_avg**7 / (Cp_avg**7 + 25**7))
        * math.sin(math.radians(60 * math.exp(-(((hp_avg - 275) / 25) ** 2))))
    )

    dE = math.sqrt(
        (dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT_term * (dCp / SC) * (dHp / SH)
    )

    return dE


def compute_color_score(
    ref_blocks: list[VisualBlock],
    gen_blocks: list[VisualBlock],
    matches: list[tuple[int, int, float]],
) -> float:
    """
    Color score: average CIEDE2000 similarity across matched block pairs.
    Design2Code: similarity = max(0, 1 - (delta_e / 100))
    """
    if not matches:
        return 0.0

    scores = []
    for r, c, _ in matches:
        ref_rgb = ref_blocks[r].color
        gen_rgb = gen_blocks[c].color

        ref_lab = rgb_to_lab(ref_rgb)
        gen_lab = rgb_to_lab(gen_rgb)

        delta_e = ciede2000(ref_lab, gen_lab)
        similarity = max(0.0, 1.0 - (delta_e / 100.0))
        scores.append(similarity)

    return sum(scores) / len(ref_blocks) if ref_blocks else 0.0


_clip_model = None
_clip_processor = None


def _get_clip_model():
    """Lazy-load and cache CLIP model as module-level singleton."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model_name = "openai/clip-vit-base-patch32"
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        _clip_model = CLIPModel.from_pretrained(model_name)
        _clip_model.eval()
    return _clip_model, _clip_processor


def compute_clip_score(
    ref_image: np.ndarray,
    gen_image: np.ndarray,
    ref_blocks: list[VisualBlock],
    gen_blocks: list[VisualBlock],
) -> float:
    """
    CLIP score: cosine similarity of CLIP embeddings on inpainted screenshots.
    Images are inpainted to remove photo-like regions before comparison.
    """
    try:
        import torch
    except ImportError:
        logger.warning("transformers/torch not available, returning CLIP score 0.0")
        return 0.0

    # Inpaint both images to remove photo regions
    ref_inpainted = inpaint_image_regions(ref_image, ref_blocks)
    gen_inpainted = inpaint_image_regions(gen_image, gen_blocks)

    # Convert to PIL
    ref_pil = Image.fromarray(cv2.cvtColor(ref_inpainted, cv2.COLOR_BGR2RGB))
    gen_pil = Image.fromarray(cv2.cvtColor(gen_inpainted, cv2.COLOR_BGR2RGB))

    model, processor = _get_clip_model()

    with torch.no_grad():
        ref_inputs = processor(images=ref_pil, return_tensors="pt")
        gen_inputs = processor(images=gen_pil, return_tensors="pt")

        ref_out = model.get_image_features(pixel_values=ref_inputs["pixel_values"])
        gen_out = model.get_image_features(pixel_values=gen_inputs["pixel_values"])

        # Handle both tensor and BaseModelOutputWithPooling returns
        ref_features = ref_out if isinstance(ref_out, torch.Tensor) else ref_out.pooler_output
        gen_features = gen_out if isinstance(gen_out, torch.Tensor) else gen_out.pooler_output

        # Normalize
        ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
        gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (ref_features @ gen_features.T).item()

    # CLIP image-image cosine similarity is typically in [0.5, 1.0].
    # Clamp to [0, 1] rather than rescaling from [-1, 1] which would
    # compress the useful range and lose discriminative power.
    return max(0.0, min(1.0, similarity))


def _normalize_blocks(
    blocks: list[VisualBlock],
    img_width: float,
    img_height: float,
) -> list[VisualBlock]:
    """Normalize block coordinates to [0, 1] range relative to image dimensions."""
    return [
        VisualBlock(
            x=b.x / img_width,
            y=b.y / img_height,
            width=b.width / img_width,
            height=b.height / img_height,
            text=b.text,
            color=b.color,
        )
        for b in blocks
    ]


def score_pages(
    ref_image: np.ndarray,
    gen_image: np.ndarray,
    ref_blocks: Optional[list[VisualBlock]] = None,
    gen_blocks: Optional[list[VisualBlock]] = None,
    use_clip: bool = True,
) -> ScoringResult:
    """
    Score a generated page against a reference page using Design2Code metrics.

    For HTML-based scoring (Design2Code color-injection), use score_pages_async
    which supports ref_html/gen_html parameters.

    Args:
        ref_image: Reference webpage screenshot (BGR, OpenCV format)
        gen_image: Generated webpage screenshot (BGR, OpenCV format)
        ref_blocks: Pre-detected reference blocks (auto-detected if None)
        gen_blocks: Pre-detected generated blocks (auto-detected if None)
        use_clip: Whether to compute CLIP score (requires transformers+torch)

    Returns:
        ScoringResult with all five component scores and final weighted average.
    """
    h_ref, w_ref = ref_image.shape[:2]
    h_gen, w_gen = gen_image.shape[:2]

    # Detect blocks on each image at its native resolution (no distortion)
    if ref_blocks is None:
        ref_blocks = detect_blocks_from_screenshot(ref_image)
    if gen_blocks is None:
        gen_blocks = detect_blocks_from_screenshot(gen_image)

    # Normalize block coordinates to [0, 1] so images of different sizes
    # can be compared without aspect-ratio-destroying resizes
    ref_norm = _normalize_blocks(ref_blocks, float(w_ref), float(h_ref))
    gen_norm = _normalize_blocks(gen_blocks, float(w_gen), float(h_gen))

    return _score_from_blocks(
        ref_image, gen_image, ref_blocks, gen_blocks,
        ref_norm, gen_norm, use_clip,
    )


def _score_from_blocks(
    ref_image: np.ndarray,
    gen_image: np.ndarray,
    ref_blocks: list[VisualBlock],
    gen_blocks: list[VisualBlock],
    ref_norm: list[VisualBlock],
    gen_norm: list[VisualBlock],
    use_clip: bool,
) -> ScoringResult:
    """Core scoring logic shared by sync and async paths."""
    h_ref, w_ref = ref_image.shape[:2]

    # Merge blocks that belong together
    ref_norm = find_possible_merge(ref_norm)
    gen_norm = find_possible_merge(gen_norm)

    # Match in normalized [0,1] x [0,1] space
    matches = compute_block_matching(ref_norm, gen_norm, 1.0, 1.0)

    # Compute component scores using normalized blocks
    size = compute_size_score(ref_norm, gen_norm, matches)
    text = compute_text_score(ref_norm, gen_norm, matches)
    position = compute_position_score(ref_norm, gen_norm, matches, 1.0, 1.0)
    color = compute_color_score(ref_norm, gen_norm, matches)

    clip = 0.0
    if use_clip:
        if gen_image.shape[:2] != ref_image.shape[:2]:
            gen_resized = cv2.resize(gen_image, (w_ref, h_ref))
        else:
            gen_resized = gen_image
        clip = compute_clip_score(ref_image, gen_resized, ref_blocks, gen_blocks)

    # Design2Code: equal 0.2 weight for all 5 metrics
    final = 0.2 * (size + text + position + color + clip)

    return ScoringResult(
        size_score=size,
        text_score=text,
        position_score=position,
        color_score=color,
        clip_score=clip,
        final_score=final,
        num_ref_blocks=len(ref_norm),
        num_gen_blocks=len(gen_norm),
        num_matched=len(matches),
    )


async def score_pages_async(
    ref_image: np.ndarray,
    gen_image: np.ndarray,
    ref_html: Optional[str] = None,
    gen_html: Optional[str] = None,
    ref_blocks: Optional[list[VisualBlock]] = None,
    gen_blocks: Optional[list[VisualBlock]] = None,
    use_clip: bool = True,
    render_fn=None,
) -> ScoringResult:
    """
    Async scoring with optional HTML-based Design2Code block detection.

    If ref_html/gen_html are provided, uses color-injection for accurate
    text + bounding box extraction. Otherwise falls back to contour detection.
    """
    h_ref, w_ref = ref_image.shape[:2]
    h_gen, w_gen = gen_image.shape[:2]

    # Prefer HTML-based detection (Design2Code color-injection),
    # falling back to contour-based detection on Playwright errors.
    if ref_blocks is not None:
        # Pre-computed blocks (already normalized [0,1] from detect_blocks_from_html)
        ref_norm = ref_blocks
    elif ref_html is not None:
        try:
            ref_blocks = await detect_blocks_from_html(
                ref_html, width=w_ref, height=h_ref, render_fn=render_fn,
            )
            ref_norm = ref_blocks  # already normalized [0,1]
        except Exception as e:
            logger.warning(f"HTML-based ref block detection failed, falling back to contours: {e}")
            ref_blocks = detect_blocks_from_screenshot(ref_image)
            ref_norm = _normalize_blocks(ref_blocks, float(w_ref), float(h_ref))
    else:
        ref_blocks = detect_blocks_from_screenshot(ref_image)
        ref_norm = _normalize_blocks(ref_blocks, float(w_ref), float(h_ref))

    if gen_html is not None and gen_blocks is None:
        try:
            gen_blocks = await detect_blocks_from_html(
                gen_html, width=w_gen, height=h_gen, render_fn=render_fn,
            )
            gen_norm = gen_blocks  # already normalized [0,1]
        except Exception as e:
            logger.warning(f"HTML-based gen block detection failed, falling back to contours: {e}")
            gen_blocks = detect_blocks_from_screenshot(gen_image)
            gen_norm = _normalize_blocks(gen_blocks, float(w_gen), float(h_gen))
    else:
        if gen_blocks is None:
            gen_blocks = detect_blocks_from_screenshot(gen_image)
        gen_norm = _normalize_blocks(gen_blocks, float(w_gen), float(h_gen))

    return _score_from_blocks(
        ref_image, gen_image, ref_blocks, gen_blocks,
        ref_norm, gen_norm, use_clip,
    )
