"""
Product domain schemas
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

class ProductRecommendation(BaseModel):
    """Product recommendation with scoring"""

    product_id: str
    product_name: str
    product_category: str
    avg_price: float = Field(ge=0)
    recommendation_score: float = Field(ge=0, le=1, description="Recommendation strength")
    reason: str = Field(description="Why this product is recommended")
    similar_customer_count: int = Field(ge=0, description="Number of similar customers who bought this")
    avg_sentiment: float = Field(ge=0, le=1, description="Average review sentiment")
    source: Literal["collaborative", "category_affinity", "trending"] = Field(
        default="collaborative"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "product_id": "291",
                    "product_name": "Laptop",
                    "product_category": "Electronics",
                    "avg_price": 520.30,
                    "recommendation_score": 0.89,
                    "reason": "Highly rated by 8 similar premium electronics buyers",
                    "similar_customer_count": 8,
                    "avg_sentiment": 0.85,
                    "source": "collaborative",
                }
            ]
        }
    )

class ProductReview(BaseModel):
    """Product review with sentiment"""

    review_id: str
    product_id: str
    review_text: str
    review_date: str
    sentiment_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Sentiment score (0=negative, 1=positive)"
    )
    sentiment_label: Optional[Literal["positive", "negative", "neutral"]] = None

    model_config = ConfigDict(from_attributes=True)

class ProductDetail(BaseModel):
    """Product details"""

    product_id: str
    product_name: str
    product_category: str
    avg_price: float = Field(ge=0)
    purchase_count: int = Field(default=0, ge=0)
    avg_sentiment: Optional[float] = Field(default=None, ge=0, le=1)
    review_count: int = Field(default=0, ge=0)

    model_config = ConfigDict(from_attributes=True)
