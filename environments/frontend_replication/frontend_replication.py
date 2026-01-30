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
- You may use <img> tags referencing the original image URLs from the website. The page will be rendered with full internet access, so external images will load.
- Use system fonts: Arial, Helvetica, sans-serif, serif, monospace
- Match the layout, typography, colors, spacing, and overall structure as closely as possible
- When you are satisfied with your replication, include the exact string "DONE" at the end of your message
- Focus your effort on layout, text content, colors, spacing, and correct image placement.
- For icons: use simple Unicode characters (e.g., "→", "✓", "⚡") or inline SVG.

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


TRACKING_DOMAINS = {"facebook.com", "twitter.com", "t.co", "bat.bing.com", "google-analytics.com", "doubleclick.net", "googletagmanager.com"}


def extract_image_urls(html: str) -> list[dict[str, str]]:
    """Extract real image URLs with context from HTML (filters out tracking pixels).

    Returns list of dicts with 'url' and 'context' (alt text or nearby text).
    """
    results = []
    seen = set()
    for match in re.finditer(r'<img([^>]+)>', html):
        attrs = match.group(1)
        src_m = re.search(r'src=["\']?(https?://[^"\'\s>]+)', attrs)
        if not src_m:
            continue
        url = src_m.group(1)
        if url in seen or any(d in url for d in TRACKING_DOMAINS):
            continue
        seen.add(url)

        # Extract context: alt text, aria-label, or title
        context = ""
        for attr in ("alt", "aria-label", "title"):
            m = re.search(rf'{attr}=["\']([^"\']+)["\']', attrs)
            if m and m.group(1).strip():
                context = m.group(1).strip()
                break

        # If no alt, look for nearby heading/text before the <img>
        if not context:
            start = max(0, match.start() - 500)
            before = html[start:match.start()]
            # Find last heading or paragraph text
            heading = re.findall(r'<(?:h[1-6]|p|span|figcaption)[^>]*>([^<]{3,80})</', before)
            if heading:
                context = heading[-1].strip()

        results.append({"url": url, "context": context})
    return results


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
        state["image_urls"] = extract_image_urls(state["reference_html"]) if state["reference_html"] else []

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
            image_urls = state.get("image_urls", [])
            if image_urls:
                prompt_text += "Image URLs from the original website (use these in <img> tags):\n"
                for img in image_urls:
                    ctx = f" — {img['context']}" if img.get("context") else ""
                    prompt_text += f"  {img['url']}{ctx}\n"
                prompt_text += "\n"

            prompt_text += (
                "Write the complete HTML in a ```html code block. "
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


PILOT_PAGES = [
    (
        "linear.app_ai",
        "https://linear.app/ai",
        "Linear AI page: Dark theme (#0A0A0B background, white/light gray text). "
        "Top nav bar: Linear logo (left), Products, Resources, Pricing, Customers, Max, Contact links, "
        "then Log in and Sign up buttons (right). "
        "Hero section: Large bold white headline 'AI workflows for modern product teams', "
        "gray subtext about streamlining product development. "
        "Below hero: large dark product screenshot/UI mockup (~1100x600px). "
        "Next section: Two-column layout — 'Triage Intelligence' label, large heading 'Self-driving product operations', "
        "body text about automating overhead. Right column: dark UI card (~500x400px). "
        "Next row: Two feature blocks — 'Find duplicates before they slow you down' (left) and "
        "'Unlock the value of your backlog' (right), each with dark UI cards. "
        "Logos section: 'Leading AI companies plan and build with Linear' with company logos (Scale, Replit, Runway, ElevenLabs, Cohere). "
        "Next section: 'Linear for Agents' — heading 'Delegate and automate work with AI agents', body text, 'Learn more' link. "
        "Right side: UI card showing 'Assign to...' dropdown. Row of 5 circular dark agent avatar icons with '+' button. "
        "Two columns: 'Agents for every use case, ready to deploy' and 'Create your own agents' with dark UI cards. "
        "Next section: 'Other features' — heading 'AI that works where you work'. Two dark cards: "
        "Linear MCP integration mockup and AI-powered search mockup with descriptions. "
        "Next row: 'Stay in sync with Pulse updates' (left, Daily Pulse audio player UI) and "
        "'Enterprise-grade security' (right, shield icon). "
        "Bottom CTA: 'Plan and build with a little help from AI' with 'Contact sales' and 'Get started' buttons. "
        "Footer: Linear logo, columns for Features, Product, Company, Resources, Contact. "
        "Overall: very dark, minimal, polished SaaS aesthetic with subtle dark gray cards on near-black background."
    ),
    (
        "brex.com",
        "https://www.brex.com/",
        "Brex homepage: Modern fintech SaaS landing page with clean white background. "
        "Top nav: Brex logo (left, black text), Products, Solutions, Resources, Pricing menu items, "
        "Sign in and Get started (green) buttons on right. "
        "Hero section: Large bold headline 'The AI-powered spend platform' in dark text, "
        "subheadline about controlling spend with two CTA buttons — 'Get started' (green) and 'Contact sales' (outlined). "
        "Below hero: A large product UI screenshot showing the Brex dashboard with spend analytics, cards, and transactions. "
        "Partner logos section: row of company logos (Y Combinator, Coinbase, etc.) on light gray background. "
        "Feature sections: Cards in a grid layout highlighting key features — corporate cards, expense management, "
        "travel, bill pay — each with an icon, heading, and short description. "
        "Testimonials section with customer quotes. "
        "Footer: dark background with Brex logo, organized link columns (Products, Solutions, Resources, Company), "
        "legal links, and social icons at bottom."
    ),
    (
        "stripe.com",
        "https://stripe.com/",
        "Stripe homepage: Developer-focused payments platform with a bold gradient hero. "
        "Top nav: Stripe logo (white, left), Products, Solutions, Developers, Resources, Pricing links, "
        "Contact sales and Sign in buttons (right). All nav text is white on the dark gradient. "
        "Hero: Deep purple-to-blue gradient background with subtle mesh animation. "
        "Large white headline 'Financial infrastructure for the internet', descriptive paragraph below, "
        "two buttons: 'Start now' (white with dark text) and 'Contact sales' (outlined white). "
        "Right side of hero: code snippet preview showing a Stripe API integration example in a dark code editor. "
        "Below hero: 'Trusted by millions of companies' with client logos in a grid (Amazon, Google, Shopify, etc.). "
        "Feature sections: Multiple product cards (Payments, Billing, Connect, Radar) — each with "
        "an icon, title, description, and a product screenshot or illustration. "
        "Cards have subtle shadows and rounded corners on a light background. "
        "Bottom CTA: 'Ready to get started?' with prominent action buttons. "
        "Footer: light gray with Stripe logo, organized link columns, region selector."
    ),
    (
        "vercel.com",
        "https://vercel.com/",
        "Vercel homepage: Frontend cloud platform with dark theme. "
        "Black background throughout. "
        "Top nav: Vercel triangle logo (white, left), Products, Solutions, Resources, Enterprise, Docs, Pricing links, "
        "Contact and Sign Up buttons (right). "
        "Hero: Very large white text 'Your Web. Your Way.' centered, "
        "subtitle below about building and deploying the best web experiences. "
        "Two buttons: 'Start Deploying' (white fill, dark text) and 'Get a Demo' (white outline). "
        "Below hero: A large visual showing a deploy/build interface or animation — dark card with subtle glow. "
        "Framework logos section: horizontal row of logos — Next.js, React, Svelte, Nuxt, Astro, etc. — "
        "in muted white/gray on dark background. "
        "Feature grid: Dark cards with subtle border, each highlighting a feature (Edge Network, Previews, Analytics, etc.) "
        "with icons and short descriptions. Some cards contain product screenshots. "
        "Customer logos section: 'Trusted by the best frontend teams' with company logos. "
        "Bottom CTA: 'Start your frontend journey' with action buttons. "
        "Footer: dark with Vercel logo, link columns, social icons."
    ),
    (
        "notion.so",
        "https://notion.so/",
        "Notion homepage: All-in-one workspace with a light, clean design. "
        "White/cream background. "
        "Top nav: Notion logo (small icon + 'Notion' text, left), Product, Teams, Individuals, Download links, "
        "Request a demo, Get Notion free (blue) buttons on right. "
        "Hero: Large centered bold text 'The happiest satisfying all-in-one workspace' (or similar tagline), "
        "subtitle about writing, planning, and organizing. "
        "'Get Notion free' blue CTA button centered below. "
        "Below hero: Large product screenshot showing the Notion interface — a page with text, toggles, databases. "
        "Trusted-by section: Row of company logos (Figma, Amazon, Toyota, General Electric, etc.) on light background. "
        "Feature sections: Alternating layout — left text + right illustration, then reversed. "
        "Topics: Docs, Wikis, Projects, AI features — each with heading, description, and a product mockup image. "
        "Illustrations use Notion's signature flat/hand-drawn style with warm colors. "
        "Bottom CTA: 'Get started for free' with action button. "
        "Footer: light with Notion logo, organized link columns (Product, Download, Resources, Company), legal links."
    ),
]


def load_pilot_dataset() -> Dataset:
    """Load the built-in pilot dataset from crawled landing pages."""
    from pathlib import Path as _Path

    pages_dir = _Path(__file__).parent.parent.parent / "landing pages"
    tasks = []

    for name, url, description in PILOT_PAGES:
        html_path = pages_dir / f"{name}.html"
        png_path = pages_dir / f"{name}.png"

        ref_html = ""
        ref_screenshot_b64 = ""

        if html_path.exists():
            ref_html = html_path.read_text()
        else:
            logger.warning(f"Missing HTML for {name}: {html_path}")

        if png_path.exists():
            ref_screenshot_b64 = image_to_base64(png_path.read_bytes())
        else:
            logger.warning(f"Missing screenshot for {name}: {png_path}")

        tasks.append({
            "question": "Replicate the following website.",
            "answer": "",
            "info": json.dumps({
                "url": url,
                "description": description,
                "reference_screenshot": ref_screenshot_b64,
                "reference_html": ref_html,
                "viewport": {"width": 1280, "height": 800},
            }),
        })

    return Dataset.from_list(tasks)
