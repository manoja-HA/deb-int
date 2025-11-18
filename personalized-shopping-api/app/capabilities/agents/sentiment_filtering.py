"""
Sentiment Filtering Agent

Filters product candidates based on review sentiment analysis.
This agent analyzes customer reviews and removes products with poor sentiment.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

from app.capabilities.base import BaseAgent, AgentMetadata, AgentContext
from app.repositories.review_repository import ReviewRepository
from app.models.sentiment_analyzer import SentimentAnalyzer
from app.core.config import settings

logger = logging.getLogger(__name__)


class ProductCandidate(BaseModel):
    """Product candidate with metadata"""

    product_id: str
    product_name: str
    product_category: str
    avg_price: float = Field(ge=0)
    purchase_count: int = Field(default=0, ge=0)


class FilteredProduct(ProductCandidate):
    """Product candidate with sentiment data"""

    avg_sentiment: float = Field(ge=0, le=1, description="Average sentiment score")
    review_count: int = Field(default=0, ge=0, description="Number of reviews")


class SentimentFilteringInput(BaseModel):
    """Input for sentiment filtering agent"""

    candidate_products: List[ProductCandidate] = Field(
        description="List of product candidates to filter"
    )
    sentiment_threshold: float = Field(
        default=settings.SENTIMENT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum sentiment score to pass filter"
    )
    min_reviews: int = Field(
        default=settings.MIN_REVIEWS_FOR_INCLUSION,
        ge=0,
        description="Minimum number of reviews required for including product in sentiment calculation"
    )


class SentimentFilteringOutput(BaseModel):
    """Output from sentiment filtering agent"""

    filtered_products: List[FilteredProduct] = Field(
        description="Products that passed the sentiment filter"
    )
    products_considered: int = Field(
        ge=0,
        description="Total number of products considered"
    )
    products_filtered_out: int = Field(
        ge=0,
        description="Number of products removed due to poor sentiment"
    )


class SentimentFilteringAgent(BaseAgent[SentimentFilteringInput, SentimentFilteringOutput]):
    """
    Agent that filters products by review sentiment

    Responsibilities:
    - Fetch reviews for each product candidate
    - Calculate average sentiment using SentimentAnalyzer
    - Filter out products below sentiment threshold
    - Handle products with no reviews (assign neutral sentiment)
    - Track filtering metrics

    This agent encapsulates the review-based filtering logic that was
    previously embedded in RecommendationService (Agent 3).
    """

    def __init__(
        self,
        review_repository: ReviewRepository,
        sentiment_analyzer: SentimentAnalyzer,
    ):
        """
        Initialize the sentiment filtering agent

        Args:
            review_repository: Repository for accessing product reviews
            sentiment_analyzer: Analyzer for computing sentiment scores
        """
        metadata = AgentMetadata(
            id="sentiment_filtering",
            name="Sentiment Filtering Agent",
            description="Filters products based on review sentiment analysis",
            version="1.0.0",
            input_schema=SentimentFilteringInput,
            output_schema=SentimentFilteringOutput,
            tags=["filtering", "sentiment", "reviews", "quality"],
        )
        super().__init__(metadata)
        self.review_repo = review_repository
        self.sentiment_analyzer = sentiment_analyzer

    async def _execute(
        self,
        input_data: SentimentFilteringInput,
        context: AgentContext,
    ) -> SentimentFilteringOutput:
        """
        Filter products by sentiment threshold

        Args:
            input_data: Contains candidate products and filter parameters
            context: Execution context

        Returns:
            Filtered products with sentiment scores
        """
        candidate_products = input_data.candidate_products
        threshold = input_data.sentiment_threshold
        min_reviews = input_data.min_reviews

        filtered_products = []
        filtered_out = 0

        for product in candidate_products:
            product_id = product.product_id

            # Fetch reviews for this product
            reviews = self.review_repo.get_by_product_id(product_id)

            # Calculate sentiment
            if reviews:
                review_texts = [r["review_text"] for r in reviews]
                sentiment_data = self.sentiment_analyzer.calculate_average_sentiment(
                    review_texts,
                    min_reviews=min_reviews,
                )
                avg_sentiment = sentiment_data["avg_sentiment"]
                review_count = len(reviews)
            else:
                # No reviews: assign neutral sentiment (0.5)
                # This allows products without reviews to pass through
                avg_sentiment = 0.5
                review_count = 0

            # Apply sentiment filter
            if avg_sentiment >= threshold:
                # Product passes filter
                filtered_product = FilteredProduct(
                    product_id=product.product_id,
                    product_name=product.product_name,
                    product_category=product.product_category,
                    avg_price=product.avg_price,
                    purchase_count=product.purchase_count,
                    avg_sentiment=avg_sentiment,
                    review_count=review_count,
                )
                filtered_products.append(filtered_product)
            else:
                # Product filtered out due to poor sentiment
                filtered_out += 1
                self._logger.debug(
                    f"Filtered out product {product_id} ({product.product_name}) "
                    f"with sentiment {avg_sentiment:.2f} < {threshold}"
                )

        self._logger.info(
            f"Sentiment filtering: {len(filtered_products)}/{len(candidate_products)} "
            f"products passed (filtered out {filtered_out})"
        )

        return SentimentFilteringOutput(
            filtered_products=filtered_products,
            products_considered=len(candidate_products),
            products_filtered_out=filtered_out,
        )
