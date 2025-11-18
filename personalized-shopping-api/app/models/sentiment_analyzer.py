"""
Sentiment analysis for product reviews
"""

import re
import asyncio
from typing import List, Dict, Literal
import logging
from langchain_core.messages import HumanMessage

from .llm_factory import get_llm, LLMType
from app.core.config import settings as config

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyze sentiment of product reviews"""

    def __init__(self, method: Literal["llm", "rule_based"] = "llm"):
        """
        Initialize sentiment analyzer

        Args:
            method: Analysis method - "llm" or "rule_based"
        """
        self.method = method
        if method == "llm":
            self.llm = get_llm(LLMType.SENTIMENT)

    def analyze_review(self, review_text: str) -> float:
        """
        Analyze sentiment of a single review

        Args:
            review_text: Review text

        Returns:
            Sentiment score (0.0 to 1.0, where 1.0 is most positive)
        """
        if self.method == "llm":
            return self._analyze_with_llm(review_text)
        else:
            return self._analyze_with_rules(review_text)

    def analyze_reviews_batch(self, reviews: List[str]) -> List[float]:
        """
        Analyze sentiment of multiple reviews

        Args:
            reviews: List of review texts

        Returns:
            List of sentiment scores
        """
        scores = []
        for review in reviews:
            try:
                score = self.analyze_review(review)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Failed to analyze review: {e}")
                scores.append(0.5)  # Neutral fallback

        return scores

    def _analyze_with_llm(self, review_text: str) -> float:
        """
        Analyze sentiment using LLM

        Args:
            review_text: Review text

        Returns:
            Sentiment score (0.0 to 1.0)
        """
        prompt = f"""Analyze the sentiment of this product review and provide a score.

Review: "{review_text}"

Provide ONLY a numerical sentiment score from 0.0 to 1.0 where:
- 0.0 = Very negative (1 star)
- 0.25 = Negative (2 stars)
- 0.5 = Neutral (3 stars)
- 0.75 = Positive (4 stars)
- 1.0 = Very positive (5 stars)

Output only the number, nothing else."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])

            # Extract score from response
            score_text = response.content.strip()

            # Try to extract float
            score_match = re.search(r'(\d+\.?\d*)', score_text)
            if score_match:
                score = float(score_match.group(1))
                # Clamp to valid range
                score = max(0.0, min(1.0, score))
                return score

            logger.warning(f"Could not parse sentiment score from: {score_text}")
            return 0.5  # Neutral fallback

        except Exception as e:
            logger.error(f"LLM sentiment analysis failed: {e}")
            return 0.5  # Neutral fallback

    def _analyze_with_rules(self, review_text: str) -> float:
        """
        Analyze sentiment using rule-based approach

        Args:
            review_text: Review text

        Returns:
            Sentiment score (0.0 to 1.0)
        """
        # Simple rule-based sentiment
        text_lower = review_text.lower()

        # Positive indicators
        positive_words = [
            'excellent', 'great', 'amazing', 'love', 'perfect', 'best',
            'wonderful', 'fantastic', 'awesome', 'good', 'nice', 'quality',
            'recommend', 'happy', 'satisfied', 'pleased'
        ]

        # Negative indicators
        negative_words = [
            'terrible', 'awful', 'bad', 'poor', 'worst', 'hate', 'disappointed',
            'waste', 'broken', 'defective', 'useless', 'horrible', 'never',
            'return', 'refund', 'complaint'
        ]

        # Count occurrences
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Calculate score
        total_signals = positive_count + negative_count

        if total_signals == 0:
            return 0.5  # Neutral if no signals

        sentiment_ratio = positive_count / total_signals

        # Map to 0-1 scale with adjustment for intensity
        if sentiment_ratio >= 0.8:
            return 0.9  # Very positive
        elif sentiment_ratio >= 0.6:
            return 0.75  # Positive
        elif sentiment_ratio >= 0.4:
            return 0.5  # Neutral
        elif sentiment_ratio >= 0.2:
            return 0.25  # Negative
        else:
            return 0.1  # Very negative

    def calculate_average_sentiment(
        self,
        review_texts: List[str],
        min_reviews: int = None
    ) -> Dict:
        """
        Calculate average sentiment with metadata

        Args:
            review_texts: List of review texts
            min_reviews: Minimum reviews required for confidence

        Returns:
            Dictionary with avg_sentiment, confidence, and review_count
        """
        if min_reviews is None:
            min_reviews = config.min_reviews_for_inclusion

        if not review_texts:
            return {
                'avg_sentiment': 0.5,
                'confidence': 0.0,
                'review_count': 0
            }

        scores = self.analyze_reviews_batch(review_texts)
        avg_score = sum(scores) / len(scores) if scores else 0.5

        # Confidence based on review count
        confidence = min(len(review_texts) / min_reviews, 1.0)

        return {
            'avg_sentiment': avg_score,
            'confidence': confidence,
            'review_count': len(review_texts)
        }

    async def analyze_reviews_batch_async(
        self,
        reviews: List[str],
        batch_size: int = 5,
        max_concurrent: int = 3
    ) -> List[float]:
        """
        Analyze sentiment of multiple reviews with async batching for performance

        This method groups reviews into batches and processes them concurrently,
        significantly improving performance for large numbers of reviews.

        Args:
            reviews: List of review texts
            batch_size: Number of reviews to analyze per batch (default: 5)
            max_concurrent: Maximum concurrent batch requests (default: 3)

        Returns:
            List of sentiment scores (0.0 to 1.0)

        Performance:
            - Sequential (old): 100 reviews × 20s = ~33 minutes
            - Batched (this): (100 reviews / 5 per batch / 3 concurrent) × 25s = ~7 minutes
            - 5x speedup with batching + concurrency
        """
        if not reviews:
            return []

        # Rule-based method doesn't benefit from batching
        if self.method == "rule_based":
            return self.analyze_reviews_batch(reviews)

        # Create batches
        batches = [
            reviews[i:i + batch_size]
            for i in range(0, len(reviews), batch_size)
        ]

        logger.info(
            f"Processing {len(reviews)} reviews in {len(batches)} batches "
            f"(batch_size={batch_size}, max_concurrent={max_concurrent})"
        )

        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch: List[str]) -> List[float]:
            """Process a single batch with concurrency control"""
            async with semaphore:
                return await self._analyze_batch_with_llm_async(batch)

        # Process all batches concurrently (with limit)
        batch_results = await asyncio.gather(
            *[process_batch(batch) for batch in batches],
            return_exceptions=True
        )

        # Flatten results and handle errors
        scores = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.warning(f"Batch {i} failed: {result}, using neutral scores")
                scores.extend([0.5] * len(batches[i]))
            else:
                scores.extend(result)

        return scores

    async def _analyze_batch_with_llm_async(self, reviews: List[str]) -> List[float]:
        """
        Analyze a batch of reviews with a single LLM call

        Args:
            reviews: List of review texts (typically 5-10)

        Returns:
            List of sentiment scores
        """
        if not reviews:
            return []

        # Create a prompt that analyzes multiple reviews at once
        reviews_text = "\n".join([
            f"{i+1}. \"{review}\""
            for i, review in enumerate(reviews)
        ])

        prompt = f"""Analyze the sentiment of these {len(reviews)} product reviews.
Provide ONLY a comma-separated list of sentiment scores from 0.0 to 1.0.

Scale:
- 0.0 = Very negative (1 star)
- 0.25 = Negative (2 stars)
- 0.5 = Neutral (3 stars)
- 0.75 = Positive (4 stars)
- 1.0 = Very positive (5 stars)

Reviews:
{reviews_text}

Output format: score1,score2,score3,...
Example: 0.75,0.9,0.25

Output ONLY the comma-separated scores, nothing else:"""

        try:
            # Use async invoke
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            scores_text = response.content.strip()

            # Parse comma-separated scores
            scores = []
            for score_str in scores_text.split(','):
                try:
                    score = float(score_str.strip())
                    # Clamp to valid range
                    score = max(0.0, min(1.0, score))
                    scores.append(score)
                except ValueError:
                    logger.warning(f"Could not parse score: {score_str}")
                    scores.append(0.5)

            # Ensure we have the right number of scores
            while len(scores) < len(reviews):
                scores.append(0.5)
            scores = scores[:len(reviews)]

            return scores

        except Exception as e:
            logger.error(f"Batch LLM sentiment analysis failed: {e}")
            # Return neutral scores for all reviews in batch
            return [0.5] * len(reviews)
