"""
Product Scoring and Ranking Agent

Scores and ranks product candidates using collaborative filtering, category affinity,
and sentiment scores. Applies diversity constraints and selects top recommendations.
"""

from typing import List, Dict, Any
from collections import Counter
from pydantic import BaseModel, Field
import logging

from app.capabilities.base import BaseAgent, AgentMetadata, AgentContext
from app.domain.schemas.customer import CustomerProfile
from app.domain.schemas.recommendation import ProductRecommendation
from app.core.config import settings

logger = logging.getLogger(__name__)


class ScoredProduct(BaseModel):
    """Product with calculated scores"""

    product_id: str
    product_name: str
    product_category: str
    avg_price: float = Field(ge=0)
    purchase_count: int = Field(default=0, ge=0)
    avg_sentiment: float = Field(ge=0, le=1)
    review_count: int = Field(default=0, ge=0)


class ProductScoringInput(BaseModel):
    """Input for product scoring agent"""

    customer_profile: CustomerProfile = Field(
        description="Customer profile with favorite categories"
    )
    products: List[ScoredProduct] = Field(
        description="Products to score and rank"
    )
    purchase_counts: Dict[str, int] = Field(
        description="Map of product_id to number of similar customers who purchased it"
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations to return"
    )
    max_per_category: int = Field(
        default=2,
        ge=1,
        description="Maximum products per category (for diversity)"
    )


class ProductScoringOutput(BaseModel):
    """Output from product scoring agent"""

    recommendations: List[ProductRecommendation] = Field(
        description="Ranked product recommendations with scores and reasons"
    )
    total_scored: int = Field(
        ge=0,
        description="Total number of products scored"
    )


class ProductScoringAgent(BaseAgent[ProductScoringInput, ProductScoringOutput]):
    """
    Agent that scores, ranks, and selects product recommendations

    Responsibilities:
    - Calculate collaborative filtering score (how many similar customers bought it)
    - Calculate category affinity score (how well it matches customer preferences)
    - Combine scores using weighted formula
    - Apply diversity constraints (max per category)
    - Select top N recommendations
    - Generate recommendation reasons

    Scoring Formula:
        final_score = 0.6 * collab_score + 0.4 * category_score + 0.2 * sentiment_score

    This agent encapsulates the scoring and ranking logic (Agent 4) that was
    previously embedded in RecommendationService.
    """

    def __init__(self):
        """Initialize the product scoring agent"""
        metadata = AgentMetadata(
            id="product_scoring",
            name="Product Scoring and Ranking Agent",
            description="Scores and ranks products using collaborative filtering and category affinity",
            version="1.0.0",
            input_schema=ProductScoringInput,
            output_schema=ProductScoringOutput,
            tags=["scoring", "ranking", "recommendation", "collaborative-filtering"],
        )
        super().__init__(metadata)

    async def _execute(
        self,
        input_data: ProductScoringInput,
        context: AgentContext,
    ) -> ProductScoringOutput:
        """
        Score, rank, and select top product recommendations

        Args:
            input_data: Contains products to score, customer profile, and parameters
            context: Execution context

        Returns:
            Ranked product recommendations with scores and reasons
        """
        profile = input_data.customer_profile
        products = input_data.products
        purchase_counts = input_data.purchase_counts
        top_n = input_data.top_n
        max_per_category = input_data.max_per_category

        # Calculate max purchase count for normalization
        max_count = max(purchase_counts.values()) if purchase_counts else 1

        # Score each product
        scored_products = []
        for product in products:
            product_id = product.product_id
            purchase_count = purchase_counts.get(product_id, 0)

            # Collaborative filtering score (normalized)
            collab_score = purchase_count / max_count

            # Category affinity score
            category_score = self._calculate_category_affinity(
                profile.favorite_categories,
                product.product_category
            )

            # Weighted final score
            final_score = (
                settings.COLLABORATIVE_WEIGHT * collab_score +
                settings.CATEGORY_AFFINITY_WEIGHT * category_score +
                0.2 * product.avg_sentiment
            )

            scored_products.append({
                "product_id": product_id,
                "product_name": product.product_name,
                "product_category": product.product_category,
                "avg_price": product.avg_price,
                "avg_sentiment": product.avg_sentiment,
                "review_count": product.review_count,
                "similar_customer_count": purchase_count,
                "collab_score": collab_score,
                "category_score": category_score,
                "final_score": final_score,
            })

        # Sort by final score (descending)
        sorted_products = sorted(
            scored_products,
            key=lambda x: x["final_score"],
            reverse=True
        )

        # Apply diversity constraint (max per category)
        category_counts = Counter()
        final_products = []
        for product in sorted_products:
            category = product["product_category"]
            if category_counts[category] < max_per_category:
                final_products.append(product)
                category_counts[category] += 1
            if len(final_products) >= top_n:
                break

        # Build ProductRecommendation objects with reasons
        recommendations = []
        for product in final_products:
            # Determine source and generate reason
            source, reason = self._generate_reason(product)

            recommendation = ProductRecommendation(
                product_id=product["product_id"],
                product_name=product["product_name"],
                product_category=product["product_category"],
                avg_price=product["avg_price"],
                recommendation_score=product["final_score"],
                reason=reason,
                similar_customer_count=product["similar_customer_count"],
                avg_sentiment=product["avg_sentiment"],
                source=source,
            )
            recommendations.append(recommendation)

        self._logger.info(
            f"Scored {len(products)} products, selected top {len(recommendations)} "
            f"with diversity constraints"
        )

        return ProductScoringOutput(
            recommendations=recommendations,
            total_scored=len(products),
        )

    def _calculate_category_affinity(
        self,
        target_categories: List[str],
        product_category: str
    ) -> float:
        """
        Calculate how well a product's category matches customer preferences

        Args:
            target_categories: Customer's favorite categories (ordered by preference)
            product_category: Product's category

        Returns:
            Affinity score between 0.0 and 1.0:
            - 1.0: Product is in customer's #1 favorite category
            - 0.8: Product is in customer's #2 favorite category
            - 0.6: Product is in customer's #3 favorite category
            - 0.2: Product is not in any favorite category
        """
        if not target_categories:
            return 0.5  # Neutral score if no preferences

        if product_category in target_categories:
            # Position-based score (earlier = higher affinity)
            position = target_categories.index(product_category)
            return 1.0 - (position * 0.2)

        # Not in favorites
        return 0.2

    def _generate_reason(self, product: Dict[str, Any]) -> tuple[str, str]:
        """
        Generate recommendation reason based on scores

        Args:
            product: Product dict with scores

        Returns:
            Tuple of (source, reason) where source is one of:
            - "collaborative": Driven by similar customer purchases
            - "category_affinity": Driven by category match
            - "trending": Balanced across factors
        """
        collab_score = product["collab_score"]
        category_score = product["category_score"]
        avg_sentiment = product["avg_sentiment"]
        similar_count = product["similar_customer_count"]

        # Determine primary source
        if collab_score > 0.7:
            source = "collaborative"
            reason = f"Highly popular with {similar_count} similar customers"
        elif category_score > 0.8:
            source = "category_affinity"
            reason = f"Matches your preference for {product['product_category']}"
        else:
            source = "trending"
            reason = "Good balance of popularity and category fit"

        # Add sentiment qualifier if excellent
        if avg_sentiment > 0.8:
            reason += f" with excellent reviews ({avg_sentiment:.0%} positive)"

        return source, reason
