"""Crawl webpages and save full HTML + screenshots for the Design2Code dataset."""

import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAGES = [
    ("https://linear.app/ai", "linear.app_ai"),
    ("https://www.brex.com/", "brex.com"),
    ("https://stripe.com/", "stripe.com"),
    ("https://vercel.com/", "vercel.com"),
    ("https://notion.so/", "notion.so"),
]

OUTPUT_DIR = Path(__file__).parent.parent.parent / "landing pages"


async def crawl_page(url: str, name: str, output_dir: Path) -> None:
    """Crawl a single page: save rendered HTML and full-page screenshot."""
    from playwright.async_api import async_playwright

    html_path = output_dir / f"{name}.html"
    png_path = output_dir / f"{name}.png"

    logger.info(f"Crawling {url} -> {name}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        page = await browser.new_page(viewport={"width": 1280, "height": 800})

        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
        except Exception:
            # Some sites never reach networkidle due to analytics/websockets
            logger.info(f"  networkidle timeout, retrying with domcontentloaded")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        # Extra wait for lazy-loaded content
        await page.wait_for_timeout(5000)

        # Scroll down to trigger lazy loading, then back up
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(1000)

        # Save rendered DOM HTML
        html = await page.content()
        html_path.write_text(html, encoding="utf-8")
        logger.info(f"  Saved HTML: {len(html)} chars -> {html_path}")

        # Save full-page screenshot
        screenshot = await page.screenshot(full_page=True, animations="disabled")
        png_path.write_bytes(screenshot)
        logger.info(f"  Saved screenshot: {len(screenshot)} bytes -> {png_path}")

        await browser.close()


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for url, name in PAGES:
        html_path = OUTPUT_DIR / f"{name}.html"
        png_path = OUTPUT_DIR / f"{name}.png"

        # Skip if already crawled
        if html_path.exists() and png_path.exists():
            logger.info(f"Skipping {name} (already exists)")
            continue

        try:
            await crawl_page(url, name, OUTPUT_DIR)
        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
