import json
import sys
import time
from pathlib import Path

from playwright.sync_api import Locator, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

VIEWPORT = {'width': 1600, 'height': 1000}
DEVICE_SCALE_FACTOR = 2
COLLAPSED_SUBTREES = ('plan', 'run_judgement_checks')
HIDDEN_ROWS = ('shared_for_demo',)
CHEVRON_OFFSET_FROM_PANEL_RIGHT = 16
TRACE_READY_ATTEMPTS = 12
TRACE_READY_WAIT_MILLISECONDS = 15_000
PARSING_POLL_MILLISECONDS = 250
PARSING_POLLS_LIMIT = 80
SETTLED_POLLS = 4
HIDE_ROW_SCRIPT = """
name => {
  const navigation = document.getElementById('trace-layout-panel-navigation')
  for (const element of navigation.querySelectorAll('span')) {
    if (element.textContent.trim() === name) {
      element.closest('div.group').parentElement.style.visibility = 'hidden'
    }
  }
}
"""
TEXT_BOX_SCRIPT = """
([panelId, text]) => {
  const panel = document.getElementById(panelId)
  const walker = document.createTreeWalker(panel, NodeFilter.SHOW_TEXT)
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const start = node.textContent.indexOf(text)
    if (start < 0) continue
    const range = document.createRange()
    range.setStart(node, start)
    range.setEnd(node, start + text.length)
    const rects = [...range.getClientRects()].filter(rect => rect.width > 0)
    if (!rects.length) continue
    const left = Math.min(...rects.map(rect => rect.left))
    const top = Math.min(...rects.map(rect => rect.top))
    const right = Math.max(...rects.map(rect => rect.right))
    const bottom = Math.max(...rects.map(rect => rect.bottom))
    return {x: left, y: top, width: right - left, height: bottom - top}
  }
  return null
}
"""
PREVIEW_PANEL_ID = 'trace-layout-panel-preview'


def tree_row(page: Page, name: str) -> Locator:
    return page.get_by_test_id('trace-layout-panel-navigation').get_by_text(name, exact=True).first


def collapse_subtree(page: Page, name: str) -> None:
    panel = page.get_by_test_id('trace-layout-panel-navigation').bounding_box()
    row = tree_row(page, name).bounding_box()
    assert panel is not None
    assert row is not None
    page.mouse.click(
        panel['x'] + panel['width'] - CHEVRON_OFFSET_FROM_PANEL_RIGHT, row['y'] + row['height'] / 2
    )


def show_output_raw(page: Page) -> None:
    raw = page.get_by_text('Raw', exact=True)
    if raw.count():
        raw.first.click()
    input_header = page.get_by_text('Input', exact=True)
    output_header = page.get_by_text('Output', exact=True)
    if input_header.count() and output_header.count():
        input_box = input_header.first.bounding_box()
        output_box = output_header.first.bounding_box()
        if input_box and output_box and output_box['y'] - input_box['y'] > 80:
            input_header.first.click()


def wait_until_parsed(page: Page) -> None:
    parsing = page.get_by_text('Parsing in background')
    settled = 0
    for _ in range(PARSING_POLLS_LIMIT):
        page.wait_for_timeout(PARSING_POLL_MILLISECONDS)
        settled = 0 if parsing.count() else settled + 1
        if settled == SETTLED_POLLS:
            return
    raise SystemExit('Langfuse kept parsing the selected observation')


def open_when_ingested(page: Page, url: str) -> None:
    for _ in range(TRACE_READY_ATTEMPTS):
        page.goto(url, wait_until='networkidle', timeout=60_000)
        try:
            tree_row(page, 'human_review').wait_for(timeout=TRACE_READY_WAIT_MILLISECONDS)
        except PlaywrightTimeoutError:
            continue
        return
    raise SystemExit(f'{url} did not show a finished public trace')


def text_box(
    page: Page, panel_id: str, text: str, widen_by_characters: int = 0
) -> dict[str, float]:
    box = page.evaluate(TEXT_BOX_SCRIPT, [panel_id, text])
    if box is None:
        raise SystemExit(f'{text!r} is not on the page')
    widening = box['width'] / len(text) * widen_by_characters
    return {**box, 'x': box['x'] - widening, 'width': box['width'] + widening * 2}


def select(page: Page, name: str, raw: bool) -> None:
    tree_row(page, name).click()
    wait_until_parsed(page)
    if raw:
        show_output_raw(page)
        wait_until_parsed(page)


def shoot(
    page: Page, directory: Path, name: str, boxes: dict[str, dict[str, float]]
) -> dict[str, object]:
    image = directory / f'trace-{name}.png'
    page.screenshot(path=image)
    return {'image': image.name, 'boxes': boxes}


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit('usage: trace_shots.py <public trace URL> <output directory> <injected text>')
    url, directory, injected = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    directory.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=DEVICE_SCALE_FACTOR)
        open_when_ingested(page, url)
        for name in COLLAPSED_SUBTREES:
            collapse_subtree(page, name)
        for name in HIDDEN_ROWS:
            page.evaluate(HIDE_ROW_SCRIPT, name)
        shots: dict[str, object] = {}
        select(page, 'verify', raw=True)
        shots['verify'] = shoot(
            page,
            directory,
            'verify',
            {
                'finding': text_box(page, PREVIEW_PANEL_ID, 'PURPOSE_UNCLEAR@'),
                'quote': text_box(page, PREVIEW_PANEL_ID, injected),
            },
        )
        select(page, 'route', raw=True)
        shots['route'] = shoot(
            page,
            directory,
            'route',
            {'route': text_box(page, PREVIEW_PANEL_ID, 'human_review', widen_by_characters=1)},
        )
        browser.close()
    print(
        json.dumps(
            {
                'viewport': VIEWPORT,
                'deviceScaleFactor': DEVICE_SCALE_FACTOR,
                'shots': shots,
                'capturedAt': time.time(),
            }
        )
    )


if __name__ == '__main__':
    main()
