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

        # Sample dominant color from the block region
        region = image[y : y + h, x : x + w]
        avg_color = tuple(int(c) for c in cv2.mean(region)[:3])

        blocks.append(
            VisualBlock(
                x=float(x),
                y=float(y),
                width=float(w),
                height=float(h),
                text="",  # text extraction done separately if needed
                color=avg_color,  # BGR from OpenCV
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
            # Spatial distance (normalized Chebyshev)
            cx_ref, cy_ref = ref.center
            cx_gen, cy_gen = gen.center
            dx = abs(cx_ref - cx_gen) / img_width
            dy = abs(cy_ref - cy_gen) / img_height
            spatial_dist = max(dx, dy)

            # Size similarity
            size_ratio = min(ref.area, gen.area) / max(ref.area, gen.area) if max(ref.area, gen.area) > 0 else 0
            size_dist = 1.0 - size_ratio

            # Text similarity (if available)
            text_sim = 0.0
            if ref.text and gen.text:
                text_sim = difflib.SequenceMatcher(None, ref.text, gen.text).ratio()

            # Combined cost: spatial + size, bonus for text match
            cost = 0.5 * spatial_dist + 0.3 * size_dist + 0.2 * (1.0 - text_sim)
            cost_matrix[i, j] = cost

    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 2.0:  # filter out very poor matches
            matches.append((r, c, cost_matrix[r, c]))

    return matches


def compute_size_score(
    ref_blocks: list[VisualBlock],
    gen_blocks: list[VisualBlock],
    matches: list[tuple[int, int, float]],
) -> float:
    """
    Size score: ratio of matched block area to total block area.
    Design2Code: final_size_score = sum(matched_areas) / sum(all_areas)
    """
    if not ref_blocks and not gen_blocks:
        return 1.0
    if not matches:
        return 0.0

    matched_area = 0.0
    for r, c, _ in matches:
        matched_area += min(ref_blocks[r].area, gen_blocks[c].area)

    total_ref_area = sum(b.area for b in ref_blocks)
    total_gen_area = sum(b.area for b in gen_blocks)
    total_area = total_ref_area + total_gen_area

    if total_area == 0:
        return 1.0

    return min(1.0, (2.0 * matched_area) / total_area)


def compute_text_score(
    ref_blocks: list[VisualBlock],
    gen_blocks: list[VisualBlock],
    matches: list[tuple[int, int, float]],
) -> float:
    """
    Text score: average SequenceMatcher ratio across matched block pairs.
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

    return sum(scores) / len(scores) if scores else 0.0


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

    return sum(scores) / len(scores)


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

    return sum(scores) / len(scores)


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
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        logger.warning("transformers/torch not available, returning CLIP score 0.0")
        return 0.0

    # Inpaint both images to remove photo regions
    ref_inpainted = inpaint_image_regions(ref_image, ref_blocks)
    gen_inpainted = inpaint_image_regions(gen_image, gen_blocks)

    # Convert to PIL
    ref_pil = Image.fromarray(cv2.cvtColor(ref_inpainted, cv2.COLOR_BGR2RGB))
    gen_pil = Image.fromarray(cv2.cvtColor(gen_inpainted, cv2.COLOR_BGR2RGB))

    # Load CLIP model
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    with torch.no_grad():
        ref_inputs = processor(images=ref_pil, return_tensors="pt")
        gen_inputs = processor(images=gen_pil, return_tensors="pt")

        ref_features = model.get_image_features(**ref_inputs)
        gen_features = model.get_image_features(**gen_inputs)

        # Normalize
        ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
        gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (ref_features @ gen_features.T).item()

    # CLIP cosine similarity is in [-1, 1], map to [0, 1]
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def score_pages(
    ref_image: np.ndarray,
    gen_image: np.ndarray,
    ref_blocks: Optional[list[VisualBlock]] = None,
    gen_blocks: Optional[list[VisualBlock]] = None,
    use_clip: bool = True,
) -> ScoringResult:
    """
    Score a generated page against a reference page using Design2Code metrics.

    Args:
        ref_image: Reference webpage screenshot (BGR, OpenCV format)
        gen_image: Generated webpage screenshot (BGR, OpenCV format)
        ref_blocks: Pre-detected reference blocks (auto-detected if None)
        gen_blocks: Pre-detected generated blocks (auto-detected if None)
        use_clip: Whether to compute CLIP score (requires transformers+torch)

    Returns:
        ScoringResult with all five component scores and final weighted average.
    """
    h_img, w_img = ref_image.shape[:2]

    # Resize generated image to match reference dimensions
    if gen_image.shape[:2] != ref_image.shape[:2]:
        gen_image = cv2.resize(gen_image, (w_img, h_img))

    # Detect blocks if not provided
    if ref_blocks is None:
        ref_blocks = detect_blocks_from_screenshot(ref_image)
    if gen_blocks is None:
        gen_blocks = detect_blocks_from_screenshot(gen_image)

    # Match blocks
    matches = compute_block_matching(
        ref_blocks, gen_blocks, float(w_img), float(h_img)
    )

    # Compute component scores
    size = compute_size_score(ref_blocks, gen_blocks, matches)
    text = compute_text_score(ref_blocks, gen_blocks, matches)
    position = compute_position_score(
        ref_blocks, gen_blocks, matches, float(w_img), float(h_img)
    )
    color = compute_color_score(ref_blocks, gen_blocks, matches)

    clip = 0.0
    if use_clip:
        clip = compute_clip_score(ref_image, gen_image, ref_blocks, gen_blocks)

    # Equal weighting (20% each) following Design2Code
    final = (size + text + position + color + clip) / 5.0

    return ScoringResult(
        size_score=size,
        text_score=text,
        position_score=position,
        color_score=color,
        clip_score=clip,
        final_score=final,
        num_ref_blocks=len(ref_blocks),
        num_gen_blocks=len(gen_blocks),
        num_matched=len(matches),
    )


async def score_pages_async(
    ref_image: np.ndarray,
    gen_image: np.ndarray,
    ref_blocks: Optional[list[VisualBlock]] = None,
    gen_blocks: Optional[list[VisualBlock]] = None,
    use_clip: bool = True,
) -> ScoringResult:
    """Async wrapper for score_pages (runs in executor to avoid blocking)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, score_pages, ref_image, gen_image, ref_blocks, gen_blocks, use_clip
    )
