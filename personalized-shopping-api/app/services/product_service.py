"""Product service - Business logic for product operations"""

from typing import List
import logging

from app.domain.schemas.product import ProductReview
from app.repositories.product_repository import ProductRepository
from app.repositories.review_repository import ReviewRepository
from app.models.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

class ProductService:
    """Product business logic service"""

    def __init__(
        self,
        product_repository: ProductRepository,
        review_repository: ReviewRepository,
    ):
        self.product_repo = product_repository
        self.review_repo = review_repository
        self.sentiment_analyzer = SentimentAnalyzer(method="rule_based")  # Use rule-based for speed

    async def get_product_reviews_with_sentiment(
        self,
        product_id: str,
    ) -> List[ProductReview]:
        """Get product reviews with computed sentiment"""
        reviews = self.review_repo.get_by_product_id(product_id)

        # Analyze sentiment
        result_reviews = []
        for review in reviews:
            sentiment_score = self.sentiment_analyzer.analyze_review(review["review_text"])

            # Determine label
            if sentiment_score >= 0.7:
                label = "positive"
            elif sentiment_score >= 0.4:
                label = "neutral"
            else:
                label = "negative"

            result_reviews.append(
                ProductReview(
                    review_id=review["review_id"],
                    product_id=review["product_id"],
                    review_text=review["review_text"],
                    review_date=review["review_date"],
                    sentiment_score=sentiment_score,
                    sentiment_label=label,
                )
            )

        return result_reviews
