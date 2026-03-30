"""
crawler.py

Why this exists:
    The builder layer needs to know how LangGraph, FastAPI, MCP etc. work
    so it can generate valid agent configs. We feed it that knowledge via RAG.
    This crawler is the data collection step — it turns documentation websites
    into clean markdown text that the chunker can split.

How it works:
    Playwright controls a headless Chromium browser. Unlike requests/httpx,
    Playwright executes JavaScript — critical for docs sites like LangGraph
    that render content client-side (React/Next.js).
    
    Flow per URL:
        1. Launch headless browser
        2. Navigate to page, wait for content to render
        3. Extract main content area (strip nav, footer, ads)
        4. Convert HTML → clean Markdown
        5. Follow internal links up to max_depth
        6. Return list of {url, title, markdown, metadata}

Where to use it:
    Called by ingester.py. Run once to build the corpus, then re-run
    periodically (weekly cron) to keep docs fresh.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from playwright.async_api import async_playwright, Page, Browser

logger = logging.getLogger(__name__)


# ─── Data contract ────────────────────────────────────────────────────────────
@dataclass
class CrawledPage:
    url      : str
    title    : str
    markdown : str                      # cleaned markdown content
    source   : str                      # e.g. "langchain", "fastapi", "mcp"
    metadata : dict = field(default_factory=dict)


# ─── Seed URLs — the docs we want ARCA's builder to know about ────────────────
# Add more as you expand tool support.
SEED_URLS = {
    "langchain": [
        "https://python.langchain.com/docs/introduction/",
        "https://python.langchain.com/docs/concepts/",
    ],
    "langgraph": [
        "https://langchain-ai.github.io/langgraph/concepts/",
        "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    ],
    "fastapi": [
        "https://fastapi.tiangolo.com/tutorial/",
        "https://fastapi.tiangolo.com/advanced/",
    ],
    "mcp": [
        "https://modelcontextprotocol.io/introduction",
        "https://modelcontextprotocol.io/docs/concepts/architecture",
    ],
    "openai": [
        "https://platform.openai.com/docs/guides/function-calling",
        "https://platform.openai.com/docs/guides/text-generation",
    ],
}

# CSS selectors for main content — strip nav/footer/sidebar noise
# Order matters: first match wins per domain
CONTENT_SELECTORS = [
    "article",
    "main",
    ".md-content",          # MkDocs (LangChain, LangGraph)
    ".content",
    "#content",
    "[role='main']",
    "body",                 # fallback
]

# Elements to strip before extracting text
NOISE_SELECTORS = [
    "nav", "header", "footer", "aside",
    ".sidebar", ".navigation", ".toc",
    ".admonition",                          # warning boxes (optional to keep)
    "script", "style", "noscript",
    ".edit-this-page", ".feedback",
]


# ─── Crawler ──────────────────────────────────────────────────────────────────
class DocsCrawler:
    def __init__(
        self,
        max_pages_per_source : int = 30,    # don't crawl entire internet
        max_depth            : int = 2,     # how many link-hops from seed URL
        delay_ms             : int = 500,   # polite delay between requests
        timeout_ms           : int = 15000, # page load timeout
    ):
        self.max_pages_per_source = max_pages_per_source
        self.max_depth            = max_depth
        self.delay_ms             = delay_ms
        self.timeout_ms           = timeout_ms

    async def crawl_all(self, seed_urls: dict = SEED_URLS) -> list[CrawledPage]:
        """
        Crawl all sources. Returns flat list of CrawledPage objects.
        Each source is crawled sequentially to be polite to servers.
        """
        all_pages = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)

            for source, urls in seed_urls.items():
                logger.info(f"[crawler] Starting source: {source} ({len(urls)} seed URLs)")
                pages = await self._crawl_source(browser, source, urls)
                all_pages.extend(pages)
                logger.info(f"[crawler] {source}: collected {len(pages)} pages")

            await browser.close()

        logger.info(f"[crawler] Total pages collected: {len(all_pages)}")
        return all_pages

    async def crawl_source(self, source: str) -> list[CrawledPage]:
        """Crawl a single source by name. Useful for incremental updates."""
        if source not in SEED_URLS:
            raise ValueError(f"Unknown source '{source}'. Add it to SEED_URLS.")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            pages = await self._crawl_source(browser, source, SEED_URLS[source])
            await browser.close()

        return pages

    async def _crawl_source(
        self,
        browser : Browser,
        source  : str,
        seed_urls : list[str],
    ) -> list[CrawledPage]:
        visited   = set()
        to_visit  = [(url, 0) for url in seed_urls]   # (url, depth)
        results   = []

        page = await browser.new_page()

        # Block images, fonts, media — we only need HTML text
        await page.route(
            "**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf,mp4,mp3}",
            lambda route: route.abort()
        )

        while to_visit and len(results) < self.max_pages_per_source:
            url, depth = to_visit.pop(0)

            if url in visited:
                continue
            visited.add(url)

            try:
                crawled = await self._fetch_page(page, url, source)
                if crawled:
                    results.append(crawled)
                    logger.debug(f"[crawler]   ✓ {url} ({len(crawled.markdown)} chars)")

                    # Follow internal links if we haven't hit max depth
                    if depth < self.max_depth:
                        links = await self._extract_links(page, url)
                        for link in links:
                            if link not in visited:
                                to_visit.append((link, depth + 1))

                await asyncio.sleep(self.delay_ms / 1000)

            except Exception as e:
                logger.warning(f"[crawler]   ✗ {url}: {e}")
                continue

        await page.close()
        return results

    async def _fetch_page(
        self,
        page    : Page,
        url     : str,
        source  : str,
    ) -> Optional[CrawledPage]:

        await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)

        # Wait for main content to render (handles React/Next.js hydration)
        try:
            await page.wait_for_selector(
                ", ".join(CONTENT_SELECTORS[:4]),
                timeout=5000
            )
        except Exception:
            pass   # proceed anyway with whatever rendered

        title = await page.title()
        html  = await page.content()

        markdown = self._html_to_markdown(html, url)

        if len(markdown.strip()) < 100:
            logger.debug(f"[crawler] Skipping {url} — too little content")
            return None

        return CrawledPage(
            url      = url,
            title    = title.strip(),
            markdown = markdown,
            source   = source,
            metadata = {
                "domain"    : urlparse(url).netloc,
                "path"      : urlparse(url).path,
                "char_count": len(markdown),
            }
        )

    def _html_to_markdown(self, html: str, base_url: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements first
        for selector in NOISE_SELECTORS:
            for el in soup.select(selector):
                el.decompose()

        # Extract main content
        content_el = None
        for selector in CONTENT_SELECTORS:
            content_el = soup.select_one(selector)
            if content_el:
                break

        if not content_el:
            content_el = soup.body or soup

        # Convert to markdown
        raw_md = md(
            str(content_el),
            heading_style     = "ATX",          # ## style headers
            bullets           = "-",
            strip             = ["a", "img"],   # remove link/image noise
        )

        # Clean up excessive whitespace
        raw_md = re.sub(r'\n{3,}', '\n\n', raw_md)
        raw_md = re.sub(r'[ \t]+\n', '\n', raw_md)

        return raw_md.strip()

    async def _extract_links(self, page: Page, base_url: str) -> list[str]:
        """Extract all internal links from the current page."""
        base_domain = urlparse(base_url).netloc

        links = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(el => el.href)"
        )

        internal_links = []
        for link in links:
            try:
                parsed = urlparse(link)
                # Keep only same-domain links, skip anchors and non-http
                if (
                    parsed.netloc == base_domain
                    and parsed.scheme in ("http", "https")
                    and not parsed.fragment                 # skip #anchor links
                    and not link.endswith((".pdf", ".zip", ".tar.gz"))
                ):
                    # Normalize: strip query params and trailing slash
                    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    clean = clean.rstrip("/")
                    internal_links.append(clean)
            except Exception:
                continue

        return list(set(internal_links))   # deduplicate