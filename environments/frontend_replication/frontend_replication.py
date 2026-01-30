"""
Frontend Replication Benchmark Environment

A multi-turn environment where models replicate real websites from screenshots
and natural language descriptions using raw HTML/CSS/JS. The model iteratively
refines its output over up to 20 turns, receiving rendered screenshots as feedback.

Scoring uses Design2Code metrics: block-match, text, position, color, CLIP.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any

import cv2
import numpy as np
from datasets import Dataset

import verifiers as vf
from verifiers.types import Messages, State

from scoring import score_pages_async

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_VIEWPORT_WIDTH = 1280
DEFAULT_VIEWPORT_HEIGHT = 800
MAX_TURNS = 5

SYSTEM_PROMPT = """You are an expert frontend engineer. Your task is to replicate a target website as closely as possible using only HTML, CSS, and JavaScript in a single index.html file.

You will receive:
1. A screenshot of the target website
2. A natural language description of the website

On each turn, write or update your HTML code. You will then see a screenshot of your current rendering alongside the target for comparison. Refine iteratively until you are satisfied.

Rules:
- Write everything in a single index.html file (inline CSS and JS are fine)
- Do NOT use external resources (CDNs, Google Fonts via URL, external images, SVGs from URLs)
- Use system fonts: Arial, Helvetica, sans-serif, serif, monospace
- Match the layout, typography, colors, spacing, and overall structure as closely as possible
- When you are satisfied with your replication, include the exact string "DONE" at the end of your message

Handling images, logos, and complex visuals:
- The description marks non-codeable elements with [IMAGE: ...]. These are images, graphics, or complex UI mockups that CANNOT be replicated with HTML/CSS alone.
- For ANY element marked [IMAGE]: use a simple colored <div> placeholder with approximate dimensions and a matching background color. Add a short text label inside (e.g., "Product Screenshot"). Do NOT attempt to recreate these visually — just use a placeholder box.
- For logos: use a placeholder div with the brand name as text, styled to approximate size and color.
- For icons: use simple Unicode characters (e.g., "→", "✓", "⚡") — do NOT draw SVG icons.
- Focus your effort on layout, text content, colors, and spacing — not on recreating images or complex graphics.

Output your HTML inside a code block:
```html
<!DOCTYPE html>
<html>
...
</html>
```"""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def extract_html(message_content: str) -> str | None:
    """Extract HTML from a code block in the model's response."""
    # Try ```html ... ``` first
    pattern = r"```html\s*\n(.*?)```"
    match = re.search(pattern, message_content, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code block
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, message_content, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if "<html" in content.lower() or "<!doctype" in content.lower():
            return content

    # Try raw HTML (no code block)
    if "<!doctype" in message_content.lower() or "<html" in message_content.lower():
        start = message_content.lower().find("<!doctype")
        if start == -1:
            start = message_content.lower().find("<html")
        if start >= 0:
            end = message_content.lower().rfind("</html>")
            if end >= 0:
                return message_content[start : end + 7]

    return None


def image_to_base64(image_bytes: bytes) -> str:
    """Convert raw PNG bytes to base64 data URI."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def decode_screenshot(screenshot_b64: str) -> np.ndarray:
    """Decode a base64 PNG screenshot to OpenCV BGR array."""
    if screenshot_b64.startswith("data:"):
        screenshot_b64 = screenshot_b64.split(",", 1)[1]
    img_bytes = base64.b64decode(screenshot_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


async def render_html_playwright(
    html: str,
    width: int = DEFAULT_VIEWPORT_WIDTH,
    height: int = DEFAULT_VIEWPORT_HEIGHT,
) -> bytes:
    """Render HTML string to PNG bytes using Playwright."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.set_content(html, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(500)
        screenshot_bytes = await page.screenshot(full_page=True, animations="disabled")
        await browser.close()
        return screenshot_bytes


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class FrontendReplicationEnv(vf.MultiTurnEnv):
    """
    Multi-turn environment for frontend website replication.

    Each rollout:
    1. Model receives target screenshot + NL description
    2. Model writes HTML
    3. Environment renders HTML in headless Chrome, screenshots it
    4. Model sees its rendering vs target, refines
    5. Repeat until model says DONE or max_turns reached

    Turn-level rewards:
        Each trajectory step receives a Design2Code score as its reward,
        enabling MT-GRPO-style per-turn credit assignment. The RL trainer
        uses these per-step rewards to compute fine-grained advantages
        rather than relying solely on the sparse final-turn score.
    """

    def __init__(self, allow_early_stop: bool = True, **kwargs):
        kwargs.setdefault("max_turns", MAX_TURNS)
        super().__init__(**kwargs)
        self.allow_early_stop = allow_early_stop
        self.add_rubric(FrontendReplicationMonitorRubric())
        self._browser = None
        self._playwright = None

    async def _get_browser(self):
        """Get or create a persistent browser instance."""
        if self._browser is None or not self._browser.is_connected():
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
            )
            logger.info("Launched persistent Playwright browser")
        return self._browser

    @vf.teardown
    async def teardown_browser(self):
        """Close the persistent browser on environment shutdown."""
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception as e:
                logger.warning(f"Browser close error: {e}")
            self._browser = None
        if hasattr(self, "_playwright") and self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.warning(f"Playwright stop error: {e}")
            self._playwright = None
        logger.info("Tore down persistent Playwright browser")

    async def render_html(
        self,
        html: str,
        width: int = DEFAULT_VIEWPORT_WIDTH,
        height: int = DEFAULT_VIEWPORT_HEIGHT,
    ) -> bytes:
        """Render HTML to PNG bytes using the persistent browser."""
        browser = await self._get_browser()
        page = await browser.new_page(viewport={"width": width, "height": height})
        try:
            await page.set_content(html, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(500)
            screenshot_bytes = await page.screenshot(full_page=True, animations="disabled")
            return screenshot_bytes
        finally:
            await page.close()

    @vf.stop
    async def model_signaled_done(self, state: State) -> bool:
        """Stop when the model includes 'DONE' in its response.

        Disabled when allow_early_stop=False (e.g. during RL training with
        MT-GRPO, which requires fixed-length rollouts for dense per-turn rewards).
        """
        if not self.allow_early_stop:
            return False
        if not state["trajectory"]:
            return False
        last_completion = state["trajectory"][-1]["completion"]
        if not last_completion:
            return False
        last_content = last_completion[-1].get("content", "")
        return "DONE" in last_content

    async def render_completion(self, state: State):
        """After rollout ends, render + score the final turn's HTML and assign its step reward.

        env_response is only called when building the *next* turn's prompt, so the
        final turn (stopped by DONE or max_turns) is never rendered/scored. We do
        that here to ensure every trajectory step has a turn-level reward for
        MT-GRPO training.
        """
        await super().render_completion(state)

        if not state["trajectory"]:
            return
        last_step = state["trajectory"][-1]
        if last_step["reward"] is not None:
            return  # Already scored

        # Extract and render the final turn's HTML (env_response was never called)
        last_content = last_step["completion"][-1].get("content", "") if last_step["completion"] else ""
        if isinstance(last_content, list):
            last_content = " ".join(
                part.get("text", "") for part in last_content if part.get("type") == "text"
            )
        html = extract_html(last_content)
        if html is None:
            last_step["reward"] = 0.0
            return

        state["current_html"] = html
        try:
            screenshot_bytes = await self.render_html(
                html, width=state["viewport_width"], height=state["viewport_height"],
            )
            state["current_screenshot_b64"] = image_to_base64(screenshot_bytes)
        except Exception as e:
            logger.warning(f"Final turn render failed: {e}")
            last_step["reward"] = 0.0
            return

        ref_b64 = state.get("reference_screenshot_b64", "")
        if not ref_b64:
            last_step["reward"] = 0.0
            return

        ref_img = decode_screenshot(ref_b64)
        gen_img = decode_screenshot(state["current_screenshot_b64"])
        try:
            result = await score_pages_async(
                ref_img, gen_img,
                ref_html=state.get("reference_html") or None,
                gen_html=html,
                use_clip=False,
                render_fn=self.render_html,
            )
            last_step["reward"] = result.final_score
            # Also store in turn_scores for consistency
            turn_scores = state.get("turn_scores", [])
            turn_scores.append({
                "size_score": result.size_score,
                "text_score": result.text_score,
                "position_score": result.position_score,
                "color_score": result.color_score,
                "final_score": result.final_score,
                "num_ref_blocks": result.num_ref_blocks,
                "num_gen_blocks": result.num_gen_blocks,
                "num_matched": result.num_matched,
            })
            state["turn_scores"] = turn_scores
            logger.info(f"Final turn score: {result.final_score:.3f}")
        except Exception as e:
            logger.warning(f"Final turn scoring failed: {e}")
            last_step["reward"] = 0.0

    async def setup_state(self, state: State) -> State:
        """Initialize per-rollout state."""
        info = state.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)

        state["viewport_width"] = info.get("viewport", {}).get("width", DEFAULT_VIEWPORT_WIDTH)
        state["viewport_height"] = info.get("viewport", {}).get("height", DEFAULT_VIEWPORT_HEIGHT)
        state["reference_screenshot_b64"] = info.get("reference_screenshot", "")
        state["description"] = info.get("description", "")
        state["url"] = info.get("url", "")
        state["current_html"] = ""
        state["current_screenshot_b64"] = ""
        state["turn"] = 0
        state["render_times"] = []
        state["turn_scores"] = []  # list of ScoringResult dicts per turn
        state["reference_html"] = info.get("reference_html", "")

        return await super().setup_state(state)

    async def get_prompt_messages(self, state: State) -> Messages:
        """Build the initial or continuation prompt."""
        if len(state["trajectory"]) == 0:
            # First turn: show target screenshot + description
            content_parts: list[dict[str, Any]] = []

            # Add reference screenshot
            ref_b64 = state["reference_screenshot_b64"]
            if ref_b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": ref_b64 if ref_b64.startswith("data:") else f"data:image/png;base64,{ref_b64}"},
                })

            # Add description
            description = state["description"]
            url = state["url"]
            prompt_text = "Replicate the website shown in the screenshot above.\n\n"
            if url:
                prompt_text += f"Website URL (for context only): {url}\n\n"
            if description:
                prompt_text += f"Description:\n{description}\n\n"
            prompt_text += (
                "Write the complete HTML in a ```html code block. "
                "Use placeholder divs for images/logos with approximate correct dimensions and colors. "
                "When you are satisfied with your result, end your message with DONE."
            )
            content_parts.append({"type": "text", "text": prompt_text})

            return state["prompt"] + [{"role": "user", "content": content_parts}]
        else:
            return await super().get_prompt_messages(state)

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        """
        After each model turn:
        1. Extract HTML from the model's response
        2. Render it in headless Chrome
        3. Return screenshots for comparison
        """
        last_msg = messages[-1]
        content = last_msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )

        html = extract_html(content)
        if html is None:
            return [{
                "role": "user",
                "content": "I couldn't find any HTML in your response. Please provide your HTML code inside a ```html code block.",
            }]

        state["current_html"] = html
        state["turn"] += 1

        # Render the HTML using persistent browser
        s = time.time()
        screenshot_bytes = await self.render_html(
            html,
            width=state["viewport_width"],
            height=state["viewport_height"],
        )
        render_time = time.time() - s
        state["render_times"].append(render_time)
        screenshot_b64 = image_to_base64(screenshot_bytes)
        state["current_screenshot_b64"] = screenshot_b64

        # --- MT-GRPO: Score this turn internally (training signal only) ---
        # Per MT-GRPO (arXiv:2505.11821): scores are used as RL training
        # signals, NOT shown to the model. The model learns to self-assess
        # from visual feedback (screenshots) alone.
        ref_b64 = state["reference_screenshot_b64"]
        score_result = None
        if ref_b64:
            ref_img = decode_screenshot(ref_b64)
            gen_img = decode_screenshot(screenshot_b64)
            ref_html = state.get("reference_html") or None
            gen_html = html
            try:
                score_result = await score_pages_async(
                    ref_img, gen_img,
                    ref_html=ref_html, gen_html=gen_html,
                    use_clip=False,
                    render_fn=self.render_html,
                )
            except Exception as e:
                logger.warning(f"Turn scoring failed: {e}")

        # Store scores internally for RL training signals — never shown to model
        turn_score = 0.0
        if score_result is not None:
            turn_score = score_result.final_score
            turn_scores = state.get("turn_scores", [])
            turn_scores.append({
                "size_score": score_result.size_score,
                "text_score": score_result.text_score,
                "position_score": score_result.position_score,
                "color_score": score_result.color_score,
                "final_score": score_result.final_score,
                "num_ref_blocks": score_result.num_ref_blocks,
                "num_gen_blocks": score_result.num_gen_blocks,
                "num_matched": score_result.num_matched,
            })
            state["turn_scores"] = turn_scores
            logger.info(
                f"Turn {state['turn']} score: {score_result.final_score:.3f} "
                f"(size={score_result.size_score:.3f} text={score_result.text_score:.3f} "
                f"pos={score_result.position_score:.3f} color={score_result.color_score:.3f})"
            )

        # MT-GRPO: assign turn-level reward to the trajectory step.
        # env_response is called when building the *next* turn's prompt,
        # so the current turn's step is the last one in the trajectory.
        if state["trajectory"]:
            state["trajectory"][-1]["reward"] = turn_score

        # Build feedback: only visual (screenshots) + turn counter
        feedback_text = (
            f"Turn {state['turn']}/{self.max_turns}. "
            f"Here is your current rendering (first image) and the target (second image). "
            f"Refine your HTML to better match the target. When satisfied, end with DONE."
        )
        content_parts: list[dict[str, Any]] = []
        content_parts.append({"type": "text", "text": feedback_text})
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": screenshot_b64},
        })
        if ref_b64:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": ref_b64 if ref_b64.startswith("data:") else f"data:image/png;base64,{ref_b64}"},
            })

        return [{"role": "user", "content": content_parts}]


# ---------------------------------------------------------------------------
# Monitor rubric for tracking frontend-specific metrics
# ---------------------------------------------------------------------------
class FrontendReplicationMonitorRubric(vf.Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.turns_used)
        self.add_metric(self.avg_render_time)
        self.add_metric(self.did_signal_done)

    async def turns_used(self, state: State) -> int:
        return state.get("turn", 0)

    async def avg_render_time(self, state: State) -> float:
        times = state.get("render_times", [])
        return sum(times) / len(times) if times else 0.0

    async def did_signal_done(self, state: State) -> float:
        if not state.get("trajectory"):
            return 0.0
        last = state["trajectory"][-1]["completion"]
        if not last:
            return 0.0
        return 1.0 if "DONE" in last[-1].get("content", "") else 0.0


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
async def frontend_visual_reward(completion: list, state: State) -> float:
    """
    Main reward: Design2Code visual similarity score.
    Compares the final rendered screenshot against the reference.
    """
    ref_b64 = state.get("reference_screenshot_b64", "")
    gen_b64 = state.get("current_screenshot_b64", "")

    if not ref_b64 or not gen_b64:
        return 0.0

    ref_img = decode_screenshot(ref_b64)
    gen_img = decode_screenshot(gen_b64)

    result = await score_pages_async(ref_img, gen_img, use_clip=True)

    # Cache component scores in state for metric tracking
    state["scoring_result"] = {
        "size_score": result.size_score,
        "text_score": result.text_score,
        "position_score": result.position_score,
        "color_score": result.color_score,
        "clip_score": result.clip_score,
        "final_score": result.final_score,
        "num_ref_blocks": result.num_ref_blocks,
        "num_gen_blocks": result.num_gen_blocks,
        "num_matched": result.num_matched,
    }

    return result.final_score


async def size_score_metric(state: State) -> float:
    return state.get("scoring_result", {}).get("size_score", 0.0)


async def text_score_metric(state: State) -> float:
    return state.get("scoring_result", {}).get("text_score", 0.0)


async def position_score_metric(state: State) -> float:
    return state.get("scoring_result", {}).get("position_score", 0.0)


async def color_score_metric(state: State) -> float:
    return state.get("scoring_result", {}).get("color_score", 0.0)


async def clip_score_metric(state: State) -> float:
    return state.get("scoring_result", {}).get("clip_score", 0.0)


# ---------------------------------------------------------------------------
# Environment loader (entry point)
# ---------------------------------------------------------------------------
def load_environment(
    dataset_path: str | None = None,
    num_eval_examples: int = -1,
    max_turns: int = MAX_TURNS,
    **kwargs,
) -> vf.Environment:
    """
    Load the Frontend Replication benchmark environment.

    Args:
        dataset_path: Path to HuggingFace dataset or local JSON with tasks.
                     If None, loads the built-in pilot dataset.
        num_eval_examples: Number of eval examples to use (-1 for all).
        max_turns: Maximum turns per rollout (default 20).
    """
    if dataset_path is not None:
        from datasets import load_dataset as hf_load_dataset

        dataset = hf_load_dataset(dataset_path)
        if isinstance(dataset, dict):
            dataset = dataset.get("test", dataset.get("eval", list(dataset.values())[0]))
    else:
        dataset = load_pilot_dataset()

    if num_eval_examples > 0:
        dataset = dataset.select(range(min(num_eval_examples, len(dataset))))

    rubric = vf.Rubric(funcs=[frontend_visual_reward], weights=[1.0])
    rubric.add_metric(size_score_metric)
    rubric.add_metric(text_score_metric)
    rubric.add_metric(position_score_metric)
    rubric.add_metric(color_score_metric)
    rubric.add_metric(clip_score_metric)

    env = FrontendReplicationEnv(
        dataset=dataset,
        eval_dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        max_turns=max_turns,
        **kwargs,
    )

    return env


def load_pilot_dataset() -> Dataset:
    """Load the built-in pilot dataset with 5 example tasks."""
    tasks = [
        {
            "question": "Replicate the following website.",
            "answer": "",
            "info": json.dumps({
                "url": "https://www.brex.com/",
                "description": (
                    "Brex homepage: Modern fintech SaaS landing page. "
                    "Top navigation bar with Brex logo (left), menu items (Products, Solutions, Resources, Pricing), "
                    "and CTA buttons (Sign in, Get started). "
                    "Hero section with large headline 'The AI-powered spend platform', "
                    "subheadline about controlling spend, and two CTA buttons. "
                    "Clean white background, dark text, green accent color for primary buttons. "
                    "Below the fold: logos of partner companies, feature cards in a grid layout."
                ),
                "viewport": {"width": 1280, "height": 800},
                "sections": ["nav", "hero", "logos", "features"],
                "reference_screenshot": "",
            }),
        },
        {
            "question": "Replicate the following website.",
            "answer": "",
            "info": json.dumps({
                "url": "https://stripe.com/",
                "description": (
                    "Stripe homepage: Developer-focused payments platform. "
                    "Dark gradient header/hero with animated gradient mesh background. "
                    "Navigation: Stripe logo, Products, Solutions, Developers, Resources, Pricing. "
                    "Hero: 'Financial infrastructure for the internet' headline in white, "
                    "descriptive paragraph, 'Start now' and 'Contact sales' buttons. "
                    "Right side has a code snippet preview showing API integration. "
                    "Below: client logos (Amazon, Google, etc.), feature sections with icons."
                ),
                "viewport": {"width": 1280, "height": 800},
                "sections": ["nav", "hero", "code-preview", "logos", "features"],
                "reference_screenshot": "",
            }),
        },
        {
            "question": "Replicate the following website.",
            "answer": "",
            "info": json.dumps({
                "url": "https://linear.app/",
                "description": (
                    "Linear homepage: Project management tool for software teams. "
                    "Dark theme (near-black background, white/gray text). "
                    "Minimal nav: Linear logo, Features, Method, Customers, Changelog, Pricing, Sign up/Log in. "
                    "Hero: Bold headline 'Linear is a purpose-built tool for planning and building products', "
                    "with subtle animated gradient behind text. "
                    "CTA: 'Get started' button with subtle glow effect. "
                    "Below: product screenshot showing the Linear interface (use placeholder). "
                    "Feature sections with icons and descriptions in a grid."
                ),
                "viewport": {"width": 1280, "height": 800},
                "sections": ["nav", "hero", "product-screenshot", "features"],
                "reference_screenshot": "",
            }),
        },
        {
            "question": "Replicate the following website.",
            "answer": "",
            "info": json.dumps({
                "url": "https://vercel.com/",
                "description": (
                    "Vercel homepage: Frontend cloud platform. "
                    "Dark theme with black background. "
                    "Nav: Vercel triangle logo, Features, Customers, Enterprise, Docs, Pricing, Blog. "
                    "Hero: 'Your complete platform for the web' in large white text. "
                    "Subtitle about frontend experience. Two buttons: 'Start Deploying' (white) and 'Get a Demo' (outline). "
                    "Below hero: animated deploy visualization (use placeholder). "
                    "Framework logos section: Next.js, React, Svelte, Nuxt, etc. "
                    "Feature grid with dark cards, subtle borders."
                ),
                "viewport": {"width": 1280, "height": 800},
                "sections": ["nav", "hero", "deploy-viz", "frameworks", "features"],
                "reference_screenshot": "",
            }),
        },
        {
            "question": "Replicate the following website.",
            "answer": "",
            "info": json.dumps({
                "url": "https://notion.so/",
                "description": (
                    "Notion homepage: All-in-one workspace. "
                    "Light/white background with clean design. "
                    "Nav: Notion logo (icon + 'Notion' text), Product (dropdown), Download, Solutions, Resources, Pricing, "
                    "Request a demo, Get Notion free. "
                    "Hero: 'Write, plan, share. With AI at your side.' large centered text. "
                    "Subtitle about getting work done. 'Get Notion free' primary button. "
                    "Below: large product screenshot showing the Notion interface (use placeholder). "
                    "Trusted by logos: Figma, Amazon, Toyota, etc."
                ),
                "viewport": {"width": 1280, "height": 800},
                "sections": ["nav", "hero", "product-screenshot", "logos"],
                "reference_screenshot": "",
            }),
        },
    ]

    return Dataset.from_list(tasks)
