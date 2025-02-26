import os
import json
import httpx
import re
from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models.article import Article
from app.core.config import get_database, get_settings
import logging
from pydantic import BaseModel
import asyncio
from datetime import datetime, timezone
from bs4 import BeautifulSoup  # Add this for better HTML parsing

router = APIRouter(prefix="/llm-scoring", tags=["llm-scoring"])
logger = logging.getLogger(__name__)


# Pydantic model for requesting LLM evaluation
class LLMScoringRequest(BaseModel):
    article_id: str
    use_cached: bool = True


# Pydantic model for LLM evaluation response
class LLMScoringResponse(BaseModel):
    article_id: str
    article_title: str
    llm_score: float
    component_scores: Dict[str, float]
    analysis: str
    read_recommendation: str


class LLMScoringService:
    """
    Service for scoring articles using LLM capabilities to evaluate content quality,
    information density, and other factors that require semantic understanding.
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        settings = get_settings()

        self.anthropic_api_key = settings.ANTHROPIC_API_KEY

        # Which LLM service to use ("openai" or "anthropic")
        self.llm_service = "anthropic"

        # Set default LLM model based on service
        if self.llm_service == "openai":
            self.default_model = "gpt-4o"
        else:
            self.default_model = "claude-3-7-sonnet-latest"

    async def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Get article by ID from the database.

        Args:
            article_id: Article ID

        Returns:
            Article document or None if not found
        """
        article = await self.db.later.find_one({"id": article_id})
        if not article:
            # Try with MongoDB _id
            article = await self.db.later.find_one({"_id": article_id})

        return article

    async def check_cached_llm_score(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if we already have a cached LLM score for this article.

        Args:
            article_id: Article ID

        Returns:
            Cached LLM score or None if not found
        """
        cache = await self.db.llm_scores.find_one({"article_id": article_id})
        return cache

    async def save_llm_score(self, article_id: str, score_data: Dict[str, Any]) -> None:
        """
        Save LLM score to cache collection.

        Args:
            article_id: Article ID
            score_data: Score data to save
        """
        # Add timestamp
        score_data["created_at"] = datetime.now(timezone.utc)
        score_data["article_id"] = article_id

        # Save to database
        await self.db.llm_scores.update_one(
            {"article_id": article_id}, {"$set": score_data}, upsert=True
        )

    async def fetch_article_from_readwise(self, article_id: str) -> Dict[str, Any]:
        """
        Fetch article content directly from Readwise API

        Args:
            article_id: The Readwise article ID

        Returns:
            Article data with full HTML content
        """
        settings = get_settings()
        readwise_token = settings.READWISE_TOKEN

        if not readwise_token:
            raise ValueError("Readwise API token not configured")

        # First get the article metadata from our database
        article = await self.get_article_by_id(article_id)
        if not article:
            raise HTTPException(
                status_code=404, detail=f"Article {article_id} not found"
            )

        # Then fetch the full content from Readwise API
        headers = {
            "Authorization": f"Token {readwise_token}",
            "Content-Type": "application/json",
        }

        # Use the Readwise API to get the full article with HTML content
        params = {
            "id": article_id,
            "withHtmlContent": "true",  # Request the HTML content
        }

        logger.info(f"Fetching article {article_id} from Readwise API")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://readwise.io/api/v3/list/", headers=headers, params=params
            )

            if response.status_code != 200:
                logger.error(f"Readwise API error: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Readwise API error: {response.text}",
                )

            data = response.json()

            if not data.get("results") or len(data["results"]) == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Article {article_id} not found in Readwise",
                )

            # Get the first (and should be only) result
            readwise_article = data["results"][0]

            # Merge with our database article
            merged_article = {**article}

            # Update with Readwise data
            if "html_content" in readwise_article:
                merged_article["content"] = readwise_article["html_content"]

            # Update other fields if needed
            for field in ["title", "author", "word_count", "summary"]:
                if field in readwise_article and readwise_article[field]:
                    merged_article[field] = readwise_article[field]

            return merged_article

    # Function to estimate token count (rough approximation)
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        This is a rough approximation - about 4 characters per token for English text.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // 4  # Rough approximation

    async def generate_llm_score_prompt(self, article: Dict[str, Any]) -> str:
        """
        Generate prompt for LLM to score article.

        Args:
            article: Article document

        Returns:
            Prompt for LLM
        """
        # Get content to analyze - try to use HTML content if available
        content = (
            article.get("content")
            or article.get("extracted_content")
            or article.get("summary", "")
        )

        # If content is HTML, strip the HTML tags to get plain text
        if content and (
            "<html" in content.lower()
            or "<body" in content.lower()
            or "<div" in content.lower()
        ):
            try:
                # Use BeautifulSoup for better HTML parsing
                soup = BeautifulSoup(content, "html.parser")

                # Remove script and style elements
                for script_or_style in soup(
                    ["script", "style", "nav", "footer", "header", "aside"]
                ):
                    script_or_style.decompose()

                # Get text and clean it up
                content = soup.get_text(separator=" ")
                content = re.sub(r"\s+", " ", content).strip()

                # Log the content length before and after processing
                logger.info(
                    f"HTML content processed: {len(article.get('content', ''))} chars -> {len(content)} chars"
                )
            except Exception as e:
                logger.warning(
                    f"Error parsing HTML with BeautifulSoup: {str(e)}. Falling back to regex."
                )
                # Fallback to regex if BeautifulSoup fails
                content = re.sub(r"<[^>]+>", " ", content)
                content = re.sub(r"\s+", " ", content).strip()

        if not content:
            raise ValueError("No content available for article")

        # Truncate content if needed (models have context limits)
        max_content_length = 15000  # Characters
        original_length = len(content)
        if original_length > max_content_length:
            truncated_content = content[:max_content_length] + "... [Content truncated]"
            logger.info(
                f"Content truncated: {original_length} chars -> {len(truncated_content)} chars"
            )
        else:
            truncated_content = content

        # Estimate token count
        estimated_tokens = self.estimate_token_count(truncated_content)
        logger.info(f"Estimated token count for prompt: {estimated_tokens} tokens")

        # Build prompt
        prompt = f"""You are an expert article evaluator assessing the quality and value of saved articles.

ARTICLE TITLE: {article.get("title", "No title")}
ARTICLE SOURCE: {article.get("site_name", "Unknown source")}
ARTICLE AUTHOR: {article.get("author", "Unknown author")}
WORD COUNT: {article.get("word_count", 0)}

I need you to evaluate this article for its reading priority using the following specific criteria:

1. INFORMATION DENSITY (1-10): How much unique, valuable information is provided per paragraph. High scores have substantial insights in each paragraph.

2. PRACTICAL VALUE (1-10): How actionable or applicable the content is. Can readers apply these insights in their work/life?

3. DEPTH OF ANALYSIS (1-10): How deeply the article explores its subject. Does it provide surface coverage or deep insights?

4. UNIQUENESS (1-10): How original or novel are the ideas presented? Is this information available elsewhere?

5. LONGEVITY (1-10): How long will this information remain valuable? Timeless content scores higher than time-sensitive content.

ARTICLE CONTENT:
{truncated_content}

Based on your evaluation, provide:
1. A score for each criterion (1-10)
2. An overall priority score (1-100)
3. A brief analysis of the article's strengths and weaknesses (max 150 words)
4. A clear read/skip recommendation

Format your response as a JSON object like this:
{{
  "component_scores": {{
    "information_density": X,
    "practical_value": X,
    "depth_of_analysis": X,
    "uniqueness": X,
    "longevity": X
  }},
  "overall_score": X,
  "analysis": "Your analysis here",
  "recommendation": "READ: This article is worth your time because..." OR "SKIP: This article is not worth your time because..."
}}
"""
        return prompt

    async def call_llm_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM API to get article evaluation.

        Args:
            prompt: Prompt for LLM

        Returns:
            LLM response parsed as JSON
        """
        try:
            if self.llm_service == "openai":
                return await self.call_openai_api(prompt)
            else:
                return await self.call_anthropic_api(prompt)
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error calling LLM API: {str(e)}"
            )

    async def call_anthropic_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call Anthropic Claude API to get article evaluation.

        Args:
            prompt: Prompt for Claude

        Returns:
            Claude response parsed as JSON
        """
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")

        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Estimate token count for logging
        estimated_tokens = self.estimate_token_count(prompt)
        logger.info(f"Calling Anthropic API with estimated {estimated_tokens} tokens")

        data = {
            "model": self.default_model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,  # Low temperature for more consistent evaluations
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=data
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Anthropic API error: {response.text}",
                )

            response_data = response.json()

            # Log token usage if available in the response
            if "usage" in response_data:
                input_tokens = response_data["usage"].get("input_tokens", 0)
                output_tokens = response_data["usage"].get("output_tokens", 0)
                total_tokens = input_tokens + output_tokens
                logger.info(
                    f"Anthropic API token usage: {input_tokens} input + {output_tokens} output = {total_tokens} total"
                )

            response_text = response_data["content"][0]["text"]

            # Extract JSON from response
            try:
                # Find JSON in response (might be wrapped in markdown code blocks)
                json_match = re.search(
                    r"```(?:json)?\s*({.*?})\s*```", response_text, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON without code blocks
                    json_str = re.search(r"{.*}", response_text, re.DOTALL).group(0)

                return json.loads(json_str)
            except Exception as e:
                logger.error(f"Error parsing Claude response as JSON: {str(e)}")
                raise ValueError(f"Failed to parse Claude response as JSON: {str(e)}")

    async def call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call OpenAI API to get article evaluation.

        Args:
            prompt: Prompt for OpenAI

        Returns:
            OpenAI response parsed as JSON
        """
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured")

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.default_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert article evaluator who provides detailed analysis of article quality.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,  # Low temperature for more consistent evaluations
            "response_format": {"type": "json_object"},  # Request JSON format
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=data
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"OpenAI API error: {response.text}",
                )

            response_data = response.json()
            response_text = response_data["choices"][0]["message"]["content"]

            # Parse JSON response
            try:
                return json.loads(response_text)
            except Exception as e:
                logger.error(f"Error parsing OpenAI response as JSON: {str(e)}")
                raise ValueError(f"Failed to parse OpenAI response as JSON: {str(e)}")

    async def score_article(
        self, article_id: str, use_cached: bool = True
    ) -> Dict[str, Any]:
        """
        Score article using LLM.

        Args:
            article_id: Article ID
            use_cached: Whether to use cached score if available

        Returns:
            Article score and analysis
        """
        # Check for cached score
        if use_cached:
            cached_score = await self.check_cached_llm_score(article_id)
            if cached_score:
                logger.info(f"Using cached LLM score for article {article_id}")
                return cached_score

        # Get article from Readwise API with full HTML content
        try:
            article = await self.fetch_article_from_readwise(article_id)
        except HTTPException as e:
            # If Readwise API fails, fall back to database
            logger.warning(
                f"Failed to fetch from Readwise API: {str(e)}, falling back to database"
            )
            article = await self.get_article_by_id(article_id)

        if not article:
            raise HTTPException(
                status_code=404, detail=f"Article {article_id} not found"
            )

        # Generate prompt
        try:
            prompt = await self.generate_llm_score_prompt(article)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Call LLM API
        llm_response = await self.call_llm_api(prompt)

        # Format result
        result = {
            "article_id": article_id,
            "article_title": article.get("title", ""),
            "llm_score": llm_response.get("overall_score", 0),
            "component_scores": llm_response.get("component_scores", {}),
            "analysis": llm_response.get("analysis", ""),
            "read_recommendation": llm_response.get("recommendation", ""),
        }

        # Save to cache
        await self.save_llm_score(article_id, result)

        return result


@router.post("/score", response_model=LLMScoringResponse)
async def score_article(
    request: LLMScoringRequest = Body(...),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Score an article using LLM.
    """
    try:
        service = LLMScoringService(db)
        result = await service.score_article(request.article_id, request.use_cached)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in score_article: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch-score/{limit}")
async def batch_score_articles(
    limit: int = 10,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Score a batch of unscored articles using LLM.
    """
    try:
        # Get articles that haven't been scored by LLM yet
        pipeline = [
            {
                "$match": {
                    "$expr": {
                        "$not": {
                            "$in": [
                                "$id",
                                {
                                    "$map": {
                                        "input": {
                                            "$ifNull": [
                                                {"$objectToArray": "$llm_scores"},
                                                [],
                                            ]
                                        },
                                        "as": "score",
                                        "in": "$$score.k",
                                    }
                                },
                            ]
                        }
                    }
                }
            },
            {"$sample": {"size": limit}},
        ]

        cursor = db.later.aggregate(pipeline)
        articles = await cursor.to_list(length=None)

        if not articles:
            return {"message": "No unscored articles found"}

        # Score each article
        service = LLMScoringService(db)
        results = []

        for article in articles:
            try:
                result = await service.score_article(
                    article.get("id"), use_cached=False
                )
                results.append(result)
                # Sleep briefly to avoid rate limits
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error scoring article {article.get('id')}: {str(e)}")

        return {"articles_scored": len(results), "results": results}
    except Exception as e:
        logger.error(f"Error in batch_score_articles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare/{article_id}")
async def compare_llm_with_algorithm(
    article_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Compare LLM score with algorithmic score for an article.
    """
    try:
        # Get article
        article = await db.later.find_one({"id": article_id})
        if not article:
            article = await db.later.find_one({"_id": article_id})

        if not article:
            raise HTTPException(
                status_code=404, detail=f"Article {article_id} not found"
            )

        # Get LLM score
        llm_service = LLMScoringService(db)
        llm_score = await llm_service.score_article(article_id, use_cached=True)

        # Get algorithmic score
        from app.services.prioritization import PrioritizationService

        priority_service = PrioritizationService(db)

        # Process article if it doesn't have priority score
        if "priority_score" not in article or article["priority_score"] is None:
            # Use your existing scoring pipeline for this article
            processed_article = await priority_service.process_articles([article])
            article = processed_article[0]

        # Format comparison result
        result = {
            "article_id": article_id,
            "article_title": article.get("title", ""),
            "llm_score": {
                "overall_score": llm_score.get("llm_score", 0),
                "component_scores": llm_score.get("component_scores", {}),
                "analysis": llm_score.get("analysis", ""),
                "recommendation": llm_score.get("read_recommendation", ""),
            },
            "algorithmic_score": {
                "priority_score": article.get("priority_score", 0),
                "component_scores": article.get("component_scores", {}),
            },
            "score_difference": abs(
                llm_score.get("llm_score", 0) - article.get("priority_score", 0)
            ),
        }

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in compare_llm_with_algorithm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_llm_scoring_stats(
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Get statistics on LLM scoring.
    """
    try:
        # Count total scored articles
        total_scored = await db.llm_scores.count_documents({})

        # Get score distribution
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_score": {"$avg": "$llm_score"},
                    "min_score": {"$min": "$llm_score"},
                    "max_score": {"$max": "$llm_score"},
                    "scores": {"$push": "$llm_score"},
                }
            }
        ]

        result = await db.llm_scores.aggregate(pipeline).to_list(length=1)

        if not result:
            return {"total_scored": 0}

        stats = result[0]
        scores = stats.pop("scores", [])

        # Calculate score distribution
        score_distribution = {
            "0-20": len([s for s in scores if s < 20]),
            "21-40": len([s for s in scores if 20 <= s < 40]),
            "41-60": len([s for s in scores if 40 <= s < 60]),
            "61-80": len([s for s in scores if 60 <= s < 80]),
            "81-100": len([s for s in scores if s >= 80]),
        }

        # Calculate median score
        median_score = sorted(scores)[len(scores) // 2] if scores else 0

        stats["median_score"] = median_score
        stats["score_distribution"] = score_distribution
        stats["total_scored"] = total_scored

        # Get component score averages
        component_pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_information_density": {
                        "$avg": "$component_scores.information_density"
                    },
                    "avg_practical_value": {
                        "$avg": "$component_scores.practical_value"
                    },
                    "avg_depth_of_analysis": {
                        "$avg": "$component_scores.depth_of_analysis"
                    },
                    "avg_uniqueness": {"$avg": "$component_scores.uniqueness"},
                    "avg_longevity": {"$avg": "$component_scores.longevity"},
                }
            }
        ]

        component_result = await db.llm_scores.aggregate(component_pipeline).to_list(
            length=1
        )

        if component_result:
            stats["component_averages"] = {
                "information_density": component_result[0].get(
                    "avg_information_density", 0
                ),
                "practical_value": component_result[0].get("avg_practical_value", 0),
                "depth_of_analysis": component_result[0].get(
                    "avg_depth_of_analysis", 0
                ),
                "uniqueness": component_result[0].get("avg_uniqueness", 0),
                "longevity": component_result[0].get("avg_longevity", 0),
            }

        return stats
    except Exception as e:
        logger.error(f"Error in get_llm_scoring_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
