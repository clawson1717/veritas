"""Playwright browser session wrapper for web automation.

Provides a clean interface for web interactions used by the
research agent, with built-in retry logic and error handling.
"""

import asyncio
import re
from typing import Optional
from urllib.parse import quote_plus

from playwright.async_api import async_playwright, Page, Browser, BrowserContext, Playwright
from tenacity import retry, stop_after_attempt, wait_exponential


class BrowserError(Exception):
    """Raised when browser operations fail."""
    pass


class BrowserSession:
    """Managed Playwright browser session.
    
    Wraps Playwright to provide:
    - Automatic browser lifecycle management
    - Simplified navigation and extraction APIs
    - Built-in retry logic for flaky operations
    - Session state persistence
    
    Example:
        async with BrowserSession() as browser:
            await browser.navigate("https://example.com")
            content = await browser.extract_text()
    """
    
    def __init__(
        self,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ):
        self.headless = headless
        self.viewport = {"width": viewport_width, "height": viewport_height}
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
    
    async def __aenter__(self) -> "BrowserSession":
        """Async context manager entry - initialize Playwright and browser."""
        self._playwright = await async_playwright().start()
        
        # Launch browser with common args for stability
        launch_args = [
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ]
        
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=launch_args
        )
        
        self._context = await self._browser.new_context(
            viewport=self.viewport,
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        # Add stealth scripts to avoid detection
        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
        """)
        
        self._page = await self._context.new_page()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup resources."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            # Log but don't raise during cleanup
            print(f"Warning: Error during browser cleanup: {e}")
    
    async def navigate(self, url: str, wait_until: str = "networkidle") -> None:
        """Navigate to a URL with retry logic.
        
        Args:
            url: The URL to navigate to
            wait_until: When to consider navigation complete
                       (load, domcontentloaded, networkidle, commit)
        
        Raises:
            BrowserError: If navigation fails after all retries
        """
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True
        )
        async def _do_navigate():
            if not self._page:
                raise BrowserError("Browser not initialized. Use async context manager.")
            
            try:
                await self._page.goto(
                    url,
                    wait_until=wait_until,  # type: ignore
                    timeout=30000
                )
            except Exception as e:
                raise BrowserError(f"Navigation failed: {e}")
        
        await _do_navigate()
    
    async def extract_text(self, selector: Optional[str] = None) -> str:
        """Extract text content from the current page.
        
        Args:
            selector: Optional CSS selector to limit extraction.
                     If None, extracts all visible text from body.
        
        Returns:
            Extracted text content, cleaned and normalized.
        
        Raises:
            BrowserError: If extraction fails
        """
        if not self._page:
            raise BrowserError("Browser not initialized. Use async context manager.")
        
        try:
            if selector:
                # Wait for element to exist
                await self._page.wait_for_selector(selector, timeout=5000)
                element = await self._page.query_selector(selector)
                if not element:
                    return ""
                text = await element.text_content()
            else:
                # Extract from body, filtering out script/style content
                text = await self._page.evaluate("""
                    () => {
                        const scripts = document.querySelectorAll('script, style, nav, footer, header');
                        scripts.forEach(el => el.remove());
                        return document.body.innerText;
                    }
                """)
            
            # Clean up the text
            if text:
                # Normalize whitespace
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
            
            return text or ""
            
        except Exception as e:
            raise BrowserError(f"Text extraction failed: {e}")
    
    async def search(
        self,
        query: str,
        engine: str = "duckduckgo"
    ) -> list[dict]:
        """Perform a web search and return results.
        
        Uses DuckDuckGo HTML interface (no API key required).
        
        Args:
            query: The search query
            engine: Search engine to use (currently only "duckduckgo" supported)
        
        Returns:
            List of search result dicts with:
            - title: Result title
            - url: Result URL
            - snippet: Brief description/snippet
        
        Raises:
            BrowserError: If search fails
            ValueError: If unsupported engine specified
        """
        if engine != "duckduckgo":
            raise ValueError(f"Unsupported search engine: {engine}. Use 'duckduckgo'.")
        
        if not self._page:
            raise BrowserError("Browser not initialized. Use async context manager.")
        
        try:
            # Build DuckDuckGo search URL
            encoded_query = quote_plus(query)
            search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            await self.navigate(search_url, wait_until="networkidle")
            
            # Wait for results to load
            await self._page.wait_for_selector(".result", timeout=10000)
            
            # Extract results
            results = await self._page.evaluate("""
                () => {
                    const items = [];
                    const resultElements = document.querySelectorAll('.result');
                    
                    resultElements.forEach((el, index) => {
                        if (index >= 10) return; // Limit to 10 results
                        
                        const titleEl = el.querySelector('.result__a');
                        const snippetEl = el.querySelector('.result__snippet');
                        const urlEl = el.querySelector('.result__url');
                        
                        if (titleEl) {
                            items.push({
                                title: titleEl.innerText.trim(),
                                url: titleEl.href || (urlEl ? urlEl.innerText.trim() : ''),
                                snippet: snippetEl ? snippetEl.innerText.trim() : ''
                            });
                        }
                    });
                    
                    return items;
                }
            """)
            
            return results if results else []
            
        except Exception as e:
            raise BrowserError(f"Search failed: {e}")
    
    async def screenshot(self, path: str, full_page: bool = False) -> None:
        """Take a screenshot of the current page.
        
        Args:
            path: File path to save the screenshot
            full_page: Whether to capture the full page or just viewport
        
        Raises:
            BrowserError: If screenshot fails
        """
        if not self._page:
            raise BrowserError("Browser not initialized. Use async context manager.")
        
        try:
            await self._page.screenshot(path=path, full_page=full_page)
        except Exception as e:
            raise BrowserError(f"Screenshot failed: {e}")
    
    @property
    def current_url(self) -> str:
        """Get the current page URL."""
        if not self._page:
            return ""
        return self._page.url
    
    async def get_page_title(self) -> str:
        """Get the current page title."""
        if not self._page:
            raise BrowserError("Browser not initialized. Use async context manager.")
        return await self._page.title()
