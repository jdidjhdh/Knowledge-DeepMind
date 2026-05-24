import logging
import re
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebSearchService:
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        results = await self._search_bing(query, max_results)
        if not results:
            results = await self._search_baidu(query, max_results)
        if not results:
            results = await self._search_ddg(query, max_results)
        logger.info(f"联网搜索完成: '{query[:50]}...' -> {len(results)} 条结果")
        return results

    async def _search_bing(self, query: str, max_results: int) -> list[dict]:
        results = []
        try:
            encoded_query = query.replace(" ", "+")
            url = f"https://www.bing.com/search?q={encoded_query}&setlang=zh-cn&count=20"
            response = await self.client.get(url)
            if response.status_code != 200:
                logger.warning(f"Bing搜索失败: HTTP {response.status_code}")
                return results

            soup = BeautifulSoup(response.text, "lxml")
            for item in soup.select("li.b_algo")[:max_results]:
                title_el = item.select_one("h2 a")
                snippet_el = item.select_one(".b_caption p, .b_lineclamp2, .b_algoSlug")
                title = title_el.get_text(strip=True) if title_el else ""
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""

                clean_title = re.sub(r"\s+", " ", title)
                clean_snippet = re.sub(r"\s+", " ", snippet)[:300]

                if clean_title and clean_snippet:
                    results.append({
                        "title": clean_title,
                        "snippet": clean_snippet,
                        "url": f"https://www.bing.com/search?q={encoded_query}",
                    })

            logger.info(f"Bing搜索: {len(results)} 条结果")
        except Exception as e:
            logger.warning(f"Bing搜索异常: {e}")

        return results

    async def _search_baidu(self, query: str, max_results: int) -> list[dict]:
        results = []
        try:
            encoded_query = query.replace(" ", "+")
            url = f"https://www.baidu.com/s?wd={encoded_query}&rn=20"
            response = await self.client.get(url)
            if response.status_code != 200:
                return results

            soup = BeautifulSoup(response.text, "lxml")
            for item in soup.select(".result, .c-container")[:max_results]:
                title_el = item.select_one("h3 a")
                snippet_el = item.select_one(".c-abstract, .content-right_8Zs40, .c-span-last")
                title = title_el.get_text(strip=True) if title_el else ""
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""

                clean_title = re.sub(r"\s+", " ", title)
                clean_snippet = re.sub(r"\s+", " ", snippet)[:300]

                if clean_title and clean_snippet:
                    results.append({
                        "title": clean_title,
                        "snippet": clean_snippet,
                        "url": f"https://www.baidu.com/s?wd={encoded_query}",
                    })

            logger.info(f"百度搜索: {len(results)} 条结果")
        except Exception as e:
            logger.warning(f"百度搜索异常: {e}")

        return results

    async def _search_ddg(self, query: str, max_results: int) -> list[dict]:
        results = []
        try:
            encoded_query = query.replace(" ", "+")
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            response = await self.client.get(url)
            if response.status_code != 200:
                return results

            soup = BeautifulSoup(response.text, "lxml")
            result_blocks = soup.select(".result")
            for block in result_blocks[:max_results]:
                title_el = block.select_one(".result__title a, .result__a")
                snippet_el = block.select_one(".result__snippet")
                link_el = block.select_one(".result__url")

                title = title_el.get_text(strip=True) if title_el else ""
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                link = link_el.get_text(strip=True) if link_el else ""

                clean_title = re.sub(r"\s+", " ", title)
                clean_snippet = re.sub(r"\s+|\n", " ", snippet)

                if clean_title and clean_snippet:
                    results.append({
                        "title": clean_title,
                        "snippet": clean_snippet,
                        "url": link,
                    })

            logger.info(f"DuckDuckGo搜索: {len(results)} 条结果")
        except Exception as e:
            logger.warning(f"DuckDuckGo搜索异常: {e}")

        return results

    async def fetch_page(self, url: str, max_chars: int = 3000) -> str:
        try:
            response = await self.client.get(url)
            if response.status_code != 200:
                return ""
            soup = BeautifulSoup(response.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 30]
            content = "\n".join(lines)
            return content[:max_chars]
        except Exception as e:
            logger.warning(f"抓取页面失败 ({url[:50]}): {e}")
            return ""

    async def close(self):
        await self.client.aclose()