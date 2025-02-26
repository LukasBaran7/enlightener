import requests
import json
from typing import Dict, Any, List


def score_article(article_id: str, use_cached: bool = True) -> Dict[str, Any]:
    """
    Score an article using the LLM scoring service.

    Args:
        article_id: The ID of the article to score
        use_cached: Whether to use cached scores (default: True)

    Returns:
        Dict containing the scoring results
    """
    # Your API endpoint
    api_url = "http://127.0.0.1:8000/llm-scoring/score"

    # Request body
    payload = {"article_id": article_id, "use_cached": use_cached}

    # Make the request
    response = requests.post(api_url, json=payload)

    # Check for errors
    response.raise_for_status()

    return response.json()


def batch_score(limit: int = 10) -> Dict[str, Any]:
    """
    Use the batch scoring endpoint to score multiple articles.

    Args:
        limit: Maximum number of articles to score

    Returns:
        Dict containing the batch scoring results
    """
    # Batch scoring endpoint
    api_url = f"http://127.0.0.1:8000/llm-scoring/batch-score/{limit}"

    # Make the request
    response = requests.get(api_url)

    # Check for errors
    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    # Example usage for a single article
    article_id = "01hf2hteqbyjbjawng2kfz7y99"
    # try:
    # result = score_article(article_id)
    # print(json.dumps(result, indent=2))
    # except requests.exceptions.RequestException as e:
    # print(f"Error: {e}")

    # Example usage for batch scoring
    try:
        batch_results = batch_score(limit=5)
        articles_scored = batch_results.get(
            "articles_scored", 0
        )  # This is an integer, not a list
        print(f"Batch processed {articles_scored} articles")

        # Print each result
        results = batch_results.get("results", [])
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  Article: {result.get('article_title', 'Unknown')}")
            print(f"  Score: {result.get('llm_score', 'N/A')}")

    except requests.exceptions.RequestException as e:
        print(f"Batch error: {e}")
