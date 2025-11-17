"""
Recommendation request/response schemas
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .base import TimestampMixin
from .customer import CustomerProfileSummary
from .product import ProductRecommendation

class RecommendationRequest(BaseModel):
    """Request for personalized recommendations"""

    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Natural language query for recommendations",
        examples=["What else would Kenneth Martinez like based on his purchase history?"]
    )
    customer_name: Optional[str] = Field(
        default=None,
        description="Customer name",
        examples=["Kenneth Martinez"]
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Customer ID",
        examples=["887"]
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations to return"
    )
    include_reasoning: bool = Field(
        default=True,
        description="Include natural language reasoning"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters (category, price range, etc.)"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty"""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What else would Kenneth Martinez like based on his purchase history?",
                    "customer_name": "Kenneth Martinez",
                    "top_n": 5,
                    "include_reasoning": True,
                }
            ]
        }
    )

class RecommendationResponse(TimestampMixin):
    """Response with personalized recommendations"""

    query: str
    customer_profile: Optional[CustomerProfileSummary] = None
    recommendations: List[ProductRecommendation]
    reasoning: str
    confidence_score: float = Field(ge=0, le=1)
    processing_time_ms: float = Field(ge=0)

    # Metadata
    similar_customers_analyzed: int = Field(ge=0)
    products_considered: int = Field(ge=0)
    products_filtered_by_sentiment: int = Field(default=0, ge=0)
    recommendation_strategy: str
    agent_execution_order: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What else would Kenneth Martinez like?",
                    "customer_profile": {
                        "customer_id": "887",
                        "customer_name": "Kenneth Martinez",
                        "total_purchases": 5,
                        "avg_purchase_price": 689.99,
                        "favorite_categories": ["Electronics"],
                        "price_segment": "premium",
                    },
                    "recommendations": [
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
                    ],
                    "reasoning": "Based on Kenneth's purchase history...",
                    "confidence_score": 0.85,
                    "processing_time_ms": 847,
                    "similar_customers_analyzed": 20,
                    "products_considered": 45,
                    "recommendation_strategy": "collaborative_filtering",
                }
            ]
        }
    )
