"""
Standalone benchmark script: test SOTA models on brex.com replication.

Usage:
    uv run python environments/frontend_replication/run_brex_benchmark.py \
        --screenshot /path/to/brex_reference.png \
        --models sonnet opus gpt-4.1 gpt-5 gemini-2.5-pro gemini-3-pro \
        --max-turns 15 \
        --save-results
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from openai import AsyncOpenAI

from frontend_replication import (
    SYSTEM_PROMPT,
    decode_screenshot,
    extract_html,
    extract_image_urls,
    image_to_base64,
    render_html_playwright,
)
from scoring import score_pages, score_pages_async

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Load endpoints from configs
ENDPOINTS_PATH = Path(__file__).parent.parent.parent / "configs" / "endpoints.py"


def load_endpoints() -> dict:
    """Load endpoints from configs/endpoints.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("endpoints", ENDPOINTS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ENDPOINTS


def load_reference_screenshot(path: str, max_width: int = 1280, max_height: int = 7500) -> str:
    """Load a screenshot file, resize to fit within max dimensions, return as base64 data URI."""
    from PIL import Image
    from io import BytesIO

    img = Image.open(path)
    # Convert RGBA to RGB (white background)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    # Resize to fit within max_width x max_height maintaining aspect ratio
    ratio = min(max_width / img.width, max_height / img.height, 1.0)
    if ratio < 1.0:
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"Resized screenshot to {new_w}x{new_h}")

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return image_to_base64(buf.getvalue())


TASK_DESCRIPTION = ""  # Set from args or default

# Import descriptions from the main env module
from frontend_replication import PILOT_PAGES

# Build lookup tables from PILOT_PAGES
DEFAULT_DESCRIPTIONS = {name.split(".")[0].replace("_", "-"): desc for name, _url, desc in PILOT_PAGES}
# Also add exact name keys
DEFAULT_DESCRIPTIONS.update({name: desc for name, _url, desc in PILOT_PAGES})

TASK_URLS = {name.split(".")[0].replace("_", "-"): url for name, url, _desc in PILOT_PAGES}
TASK_URLS.update({name: url for name, url, _desc in PILOT_PAGES})


async def run_single_model(
    model_alias: str,
    endpoint: dict,
    ref_screenshot_b64: str,
    max_turns: int,
    save_dir: Path | None,
    task_url: str = "",
    task_description: str = "",
    ref_html: str | None = None,
    ref_blocks: list | None = None,
    image_urls: list[str] | None = None,
) -> dict:
    """Run the multi-turn benchmark for a single model on brex.com."""
    logger.info(f"=== Starting {model_alias} ({endpoint['model']}) ===")

    api_key = os.environ.get(endpoint["key"], "")
    if not api_key:
        logger.error(f"Missing API key: {endpoint['key']} for {model_alias}")
        return {"model": model_alias, "error": f"Missing {endpoint['key']}"}

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=endpoint["url"],
    )

    prompt_text = (
        "Replicate the website shown in the screenshot above.\n\n"
        f"Website URL (for context only): {task_url}\n\n"
        f"Description:\n{task_description}\n\n"
    )
    if image_urls:
        prompt_text += "Image URLs from the original website (use these in <img> tags):\n"
        for img in image_urls:
            ctx = f" — {img['context']}" if img.get("context") else ""
            prompt_text += f"  {img['url']}{ctx}\n"
        prompt_text += "\n"
    prompt_text += (
        "Write the complete HTML in a ```html code block."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": ref_screenshot_b64}},
            {"type": "text", "text": prompt_text},
        ]},
    ]

    turn_scores = []
    final_html = ""
    final_screenshot_b64 = ""
    total_start = time.time()

    for turn in range(1, max_turns + 1):
        logger.info(f"  [{model_alias}] Turn {turn}/{max_turns}")
        turn_start = time.time()

        try:
            response = await client.chat.completions.create(
                model=endpoint["model"],
                messages=messages,
                temperature=0.7,
            )
        except Exception as e:
            logger.error(f"  [{model_alias}] API error on turn {turn}: {e}")
            break

        assistant_msg = response.choices[0].message.content
        if not assistant_msg:
            logger.warning(f"  [{model_alias}] Empty response on turn {turn}")
            break

        # Extract HTML
        html = extract_html(assistant_msg)
        if html is None:
            logger.warning(f"  [{model_alias}] No HTML found on turn {turn}")
            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": (
                "I couldn't find any HTML in your response. "
                "Please provide your HTML code inside a ```html code block."
            )})
            continue

        final_html = html

        # Render the HTML
        try:
            screenshot_bytes = await render_html_playwright(html, width=1280, height=800)
            screenshot_b64 = image_to_base64(screenshot_bytes)
            final_screenshot_b64 = screenshot_b64
        except Exception as e:
            logger.error(f"  [{model_alias}] Render error on turn {turn}: {e}")
            break

        # Score this turn — prefer HTML-based Design2Code scoring
        ref_img = decode_screenshot(ref_screenshot_b64)
        gen_img = decode_screenshot(screenshot_b64)
        result = await score_pages_async(
            ref_img, gen_img,
            ref_html=ref_html, gen_html=html,
            ref_blocks=ref_blocks,
            use_clip=True,
            render_fn=render_html_playwright,
        )

        turn_time = time.time() - turn_start
        turn_data = {
            "turn": turn,
            "score": result.final_score,
            "size": result.size_score,
            "text": result.text_score,
            "position": result.position_score,
            "color": result.color_score,
            "clip": result.clip_score,
            "ref_blocks": result.num_ref_blocks,
            "gen_blocks": result.num_gen_blocks,
            "matched": result.num_matched,
            "time_s": round(turn_time, 1),
        }
        turn_scores.append(turn_data)
        logger.info(
            f"  [{model_alias}] Turn {turn}: score={result.final_score:.3f} "
            f"(size={result.size_score:.3f} text={result.text_score:.3f} "
            f"pos={result.position_score:.3f} color={result.color_score:.3f}) "
            f"blocks={result.num_ref_blocks}/{result.num_gen_blocks}/{result.num_matched} "
            f"time={turn_time:.1f}s"
        )

        # Save intermediate screenshots
        if save_dir:
            turn_dir = save_dir / model_alias
            turn_dir.mkdir(parents=True, exist_ok=True)
            with open(turn_dir / f"turn_{turn}.html", "w") as f:
                f.write(html)
            with open(turn_dir / f"turn_{turn}.png", "wb") as f:
                f.write(screenshot_bytes)

        # MT-GRPO: mandatory turns, only visual feedback — no scores shown to model
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": [
            {"type": "text", "text": (
                f"Turn {turn}/{max_turns}. Here is your current rendering (first image) "
                f"and the target (second image). Refine your HTML to better match the target."
            )},
            {"type": "image_url", "image_url": {"url": screenshot_b64}},
            {"type": "image_url", "image_url": {"url": ref_screenshot_b64}},
        ]})

    total_time = time.time() - total_start
    final_score = turn_scores[-1]["score"] if turn_scores else 0.0
    best_score = max((t["score"] for t in turn_scores), default=0.0)

    result = {
        "model": model_alias,
        "model_id": endpoint["model"],
        "final_score": final_score,
        "best_score": best_score,
        "turns_used": len(turn_scores),
        "total_time_s": round(total_time, 1),
        "turn_scores": turn_scores,
    }

    logger.info(
        f"=== {model_alias} DONE: score={final_score:.3f}, "
        f"turns={len(turn_scores)}, time={total_time:.1f}s ==="
    )

    return result


async def main():
    parser = argparse.ArgumentParser(description="Run frontend replication benchmark")
    parser.add_argument("--screenshot", required=True, help="Path to reference screenshot PNG")
    parser.add_argument("--task", default="linear", help="Task name (for description lookup and URL)")
    parser.add_argument("--url", default=None, help="Website URL (overrides task default)")
    parser.add_argument("--description", default=None, help="Custom description (overrides task default)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sonnet", "gemini-2.5-pro", "gemini-3-pro", "gpt-5.2"],
        help="Model aliases from configs/endpoints.py",
    )
    parser.add_argument("--max-turns", type=int, default=5, help="Max turns per model")
    parser.add_argument("--save-results", action="store_true", help="Save HTML + screenshots per turn")
    parser.add_argument("--save-dir", default="linear_benchmark_outputs", help="Directory to save per-turn outputs")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON path")
    parser.add_argument("--ref-html", default=None, help="Path to reference HTML file for Design2Code scoring")
    args = parser.parse_args()

    task_description = args.description or DEFAULT_DESCRIPTIONS.get(args.task, "")
    task_url = args.url or TASK_URLS.get(args.task, "")
    if not task_description:
        logger.error(f"No description for task '{args.task}'. Use --description or add to DEFAULT_DESCRIPTIONS.")
        sys.exit(1)

    # Load reference screenshot
    ref_b64 = load_reference_screenshot(args.screenshot)
    logger.info(f"Loaded reference screenshot from {args.screenshot}")

    # Load reference HTML if provided (for Design2Code scoring)
    ref_html = None
    if args.ref_html:
        ref_html = Path(args.ref_html).read_text()
        logger.info(f"Loaded reference HTML from {args.ref_html} ({len(ref_html)} chars)")
    else:
        # Try auto-discover: look for .html next to screenshot
        html_path = Path(args.screenshot).with_suffix(".html")
        if html_path.exists():
            ref_html = html_path.read_text()
            logger.info(f"Auto-loaded reference HTML from {html_path} ({len(ref_html)} chars)")

    # Load endpoints
    endpoints = load_endpoints()

    # Validate models
    for model in args.models:
        if model not in endpoints:
            logger.error(f"Unknown model alias: {model}. Available: {', '.join(endpoints.keys())}")
            sys.exit(1)

    save_dir = Path(args.save_dir) if args.save_results else None
    if save_dir:
        save_dir.mkdir(exist_ok=True)

    # Pre-compute reference blocks once (saves 3 Playwright renders per turn)
    ref_blocks = None
    if ref_html:
        from scoring import detect_blocks_from_html
        from PIL import Image as _Image
        from io import BytesIO as _BytesIO

        img = _Image.open(args.screenshot)
        if img.mode == "RGBA":
            bg = _Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        if img.width > 1280:
            ratio = 1280 / img.width
            img = img.resize((1280, int(img.height * ratio)), _Image.LANCZOS)
        ref_blocks = await detect_blocks_from_html(
            ref_html, width=img.width, height=img.height,
            render_fn=render_html_playwright,
        )
        logger.info(f"Pre-computed {len(ref_blocks)} reference blocks (cached for all models)")

    # Extract image URLs from reference HTML to pass to models
    image_urls = extract_image_urls(ref_html) if ref_html else []
    if image_urls:
        logger.info(f"Extracted {len(image_urls)} image URLs from reference HTML")

    # Run models sequentially (to avoid rate limits and for clean logging)
    results = []
    for model_alias in args.models:
        endpoint = endpoints[model_alias]
        result = await run_single_model(
            model_alias=model_alias,
            endpoint=endpoint,
            ref_screenshot_b64=ref_b64,
            max_turns=args.max_turns,
            save_dir=save_dir,
            task_url=task_url,
            task_description=task_description,
            ref_html=ref_html,
            ref_blocks=ref_blocks,
            image_urls=image_urls,
        )
        results.append(result)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"FRONTEND REPLICATION BENCHMARK: {args.task.upper()}")
    print("=" * 80)
    print(f"{'Model':<20} {'Best':>8} {'Final':>8} {'Turns':>6} {'Time':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x.get("best_score", 0), reverse=True):
        print(
            f"{r['model']:<20} "
            f"{r.get('best_score', 0):>8.3f} "
            f"{r.get('final_score', 0):>8.3f} "
            f"{r.get('turns_used', 0):>6} "
            f"{r.get('total_time_s', 0):>7.1f}s"
        )
    print("=" * 80)

    # Print per-turn progression for each model
    for r in results:
        if "turn_scores" in r and r["turn_scores"]:
            print(f"\n{r['model']} — score progression:")
            for t in r["turn_scores"]:
                bar = "█" * int(t["score"] * 40)
                print(f"  Turn {t['turn']:>2}: {t['score']:.3f} {bar}")

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
