"""Screenshots of the graphical interface for the documentation (Playwright + Chromium).

    pydemi gui --no-browser --port 8611 &
    python docs/tools/make_screenshots.py /path/to/CHGCAR http://localhost:8611

Opens the interface, loads the CHGCAR and saves one PNG per view in docs/_static/screenshots.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "_static" / "screenshots"


def main(chgcar: str, url: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1500, "height": 950}, device_scale_factor=1.5)
        pg.goto(url, wait_until="networkidle")
        pg.wait_for_selector("input[type=file]", state="attached", timeout=60_000)
        pg.wait_for_timeout(2000)
        pg.screenshot(path=OUT / "gui_start.png")
        pg.locator("input[type=file]").first.set_input_files(chgcar)
        pg.wait_for_timeout(4000)
        pg.wait_for_function("!document.querySelector('[data-testid=stStatusWidget]')", timeout=180_000)
        pg.wait_for_timeout(3000)
        tabs = pg.get_by_role("tab")
        names = [tabs.nth(i).inner_text().strip() for i in range(tabs.count())]
        print("tabs:", names)
        for i, n in enumerate(names):
            if n in ("CHGCAR", "Structure (ML)"):
                continue
            tabs.nth(i).click()
            pg.wait_for_timeout(1500)
            pg.wait_for_function("!document.querySelector('[data-testid=stStatusWidget]')", timeout=600_000)
            pg.wait_for_timeout(3500)
            pg.screenshot(path=OUT / f"gui_{n.split()[-1].lower()}.png")
            print("saved", n)
        b.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
