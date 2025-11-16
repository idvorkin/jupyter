#!/usr/bin/env python3
"""
Capture screenshots of marimo notebook using Playwright.

Usage:
    1. Start marimo notebook: uv run marimo edit <notebook>.py --host 127.0.0.1 --port 8765 --headless --no-token &
    2. Run this script: uv run python scripts/capture_marimo_screenshots.py
    3. Screenshots will be saved to ~/tmp/marimo-screenshots/

Note: Marimo notebooks may require manual "Run" button click even in headless mode.
      For automated testing, consider exporting to HTML or running as Python script.
"""

import asyncio
from playwright.async_api import async_playwright
import sys
from pathlib import Path


async def capture_marimo_screenshots(
    url="http://localhost:8765", output_dir="~/tmp/marimo-screenshots"
):
    """Capture screenshots of marimo notebook at various scroll positions."""

    # Expand home directory
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1200})

        print(f"Navigating to {url}...")
        await page.goto(url, wait_until="networkidle", timeout=60000)

        print("Waiting for content to load...")
        await asyncio.sleep(10)

        # Get page dimensions
        page_height = await page.evaluate("document.documentElement.scrollHeight")
        viewport_height = await page.evaluate("window.innerHeight")
        max_scroll = max(0, page_height - viewport_height)

        print(f"Page height: {page_height}px, viewport: {viewport_height}px")
        print(f"Max scroll: {max_scroll}px")

        # Capture full page screenshot
        print("\n📸 Capturing full page...")
        await page.screenshot(
            path=str(output_path / "00-full-page.png"), full_page=True
        )

        # Capture screenshots at different scroll positions
        scroll_positions = [
            (0.0, "01-top"),
            (0.25, "02-upper-quarter"),
            (0.45, "03-mid-upper"),
            (0.60, "04-mid-section"),
            (0.70, "05-mid-lower"),
            (0.85, "06-lower-quarter"),
            (1.0, "07-bottom"),
        ]

        for ratio, name in scroll_positions:
            if max_scroll > 0:
                scroll_y = int(max_scroll * ratio)
                print(f"📸 Scrolling to {int(ratio * 100)}% ({scroll_y}px) - {name}...")
                await page.evaluate(f"window.scrollTo(0, {scroll_y})")
                await asyncio.sleep(2)
            else:
                print(f"📸 Capturing {name} (no scroll needed)...")

            await page.screenshot(path=str(output_path / f"{name}.png"))

        print("\n✅ All screenshots captured successfully!")
        print(f"   Location: {output_path}")
        print(
            f"   Files: 00-full-page.png, 01-top.png through 07-bottom.png ({len(scroll_positions) + 1} total)"
        )

        await browser.close()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "~/tmp/marimo-screenshots"

    print("📷 Marimo Screenshot Capture Tool")
    print(f"   URL: {url}")
    print(f"   Output: {output_dir}\n")

    asyncio.run(capture_marimo_screenshots(url, output_dir))
