import trafilatura
import httpx
from typing import Optional, Dict, Any
from app.models.article import Article
import logging

logger = logging.getLogger(__name__)


class ContentExtractor:
    """
    Extracts and processes content from articles using a fallback strategy:
    1. Fetch and parse content from source_url using trafilatura
    2. Use content field if available
    3. Use summary field if no other content is accessible
    """

    def __init__(self, timeout: int = 10):
        """
        Initialize the content extractor with a timeout for HTTP requests.
        
        Args:
            timeout: Maximum time in seconds to wait for HTTP response
        """
        self.timeout = timeout

    async def extract_content(self, article: Article) -> Optional[str]:
        """
        Extract content from an article using the fallback strategy.
        
        Args:
            article: The article to extract content from
            
        Returns:
            Extracted content as string or None if no content could be extracted
        """
        # Strategy 1: Fetch from source_url using trafilatura
        if article.source_url:
            try:
                content = await self._fetch_from_url(article.source_url)
                if content:
                    logger.info(f"Content extracted from source_url for article {article.id}")
                    return content
            except Exception as e:
                logger.warning(f"Failed to extract content from source_url for article {article.id}: {str(e)}")
        
        # Strategy 2: Use content field if available
        if article.content:
            logger.info(f"Using content field for article {article.id}")
            return article.content
        
        # Strategy 3: Use summary field as fallback
        if article.summary:
            logger.info(f"Using summary field for article {article.id}")
            return article.summary
        
        logger.warning(f"No content could be extracted for article {article.id}")
        return None

    async def _fetch_from_url(self, url: str) -> Optional[str]:
        """
        Fetch and extract content from a URL using trafilatura.
        
        Args:
            url: The URL to fetch content from
            
        Returns:
            Extracted content as string or None if extraction failed
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Use trafilatura to extract the main content
                content = trafilatura.extract(response.text)
                return content
        except httpx.TimeoutException:
            logger.warning(f"Timeout while fetching content from {url}")
            return None
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error while fetching content from {url}: {e.response.status_code}")
            return None
        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {str(e)}")
            return None 