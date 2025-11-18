"""
Customer Profiling Agent

Extracts and calculates customer behavioral metrics from purchase history.
This agent is responsible for building a comprehensive customer profile including
purchase patterns, category preferences, and behavioral segmentation.
"""

from typing import List
from pydantic import BaseModel, Field
import pandas as pd
import logging

from app.capabilities.base import BaseAgent, AgentMetadata, AgentContext
from app.domain.schemas.customer import CustomerProfile, Purchase
from app.repositories.customer_repository import CustomerRepository
from app.core.exceptions import NotFoundException

logger = logging.getLogger(__name__)


class CustomerProfilingInput(BaseModel):
    """Input for customer profiling agent"""

    customer_id: str = Field(description="Customer ID to profile")


class CustomerProfilingOutput(BaseModel):
    """Output from customer profiling agent"""

    profile: CustomerProfile = Field(description="Complete customer profile with behavioral metrics")


class CustomerProfilingAgent(BaseAgent[CustomerProfilingInput, CustomerProfilingOutput]):
    """
    Agent that builds comprehensive customer profiles

    Responsibilities:
    - Fetch customer purchase history from repository
    - Calculate behavioral metrics (total spent, avg price, purchase frequency)
    - Identify favorite categories and brands
    - Segment customer by price tier and frequency
    - Compute profile confidence score

    This agent encapsulates all customer profiling logic that was previously
    embedded in CustomerService.get_customer_profile().
    """

    # Segmentation thresholds (configurable)
    HIGH_FREQUENCY_THRESHOLD = 10
    MEDIUM_FREQUENCY_THRESHOLD = 3
    MID_RANGE_PRICE_MIN = 200
    PREMIUM_PRICE_MIN = 600

    def __init__(self, customer_repository: CustomerRepository):
        """
        Initialize the customer profiling agent

        Args:
            customer_repository: Repository for accessing customer and purchase data
        """
        metadata = AgentMetadata(
            id="customer_profiling",
            name="Customer Profiling Agent",
            description="Extracts behavioral metrics and segments from customer purchase history",
            version="1.0.0",
            input_schema=CustomerProfilingInput,
            output_schema=CustomerProfilingOutput,
            tags=["customer", "profiling", "segmentation"],
        )
        super().__init__(metadata)
        self.customer_repo = customer_repository

    async def _execute(
        self,
        input_data: CustomerProfilingInput,
        context: AgentContext,
    ) -> CustomerProfilingOutput:
        """
        Build customer profile from purchase history

        Args:
            input_data: Contains customer_id to profile
            context: Execution context

        Returns:
            Complete customer profile with behavioral metrics

        Raises:
            NotFoundException: If customer or purchases not found
        """
        customer_id = input_data.customer_id

        # Fetch customer basic info
        customer = self.customer_repo.get_by_id(customer_id)
        if not customer:
            raise NotFoundException("Customer", customer_id)

        # Fetch purchase history
        purchases = self.customer_repo.get_purchases_by_customer_id(customer_id)
        if not purchases:
            raise NotFoundException("Customer purchases", customer_id)

        # Convert to DataFrame for analysis
        df = pd.DataFrame(purchases)

        # Calculate aggregate metrics
        total_purchases = len(purchases)
        total_spent = float(df["price"].sum())
        avg_price = float(df["price"].mean())

        # Identify favorite categories (top 3 by purchase count)
        category_counts = df["product_category"].value_counts()
        favorite_categories = category_counts.head(3).index.tolist()

        # Determine purchase frequency segment
        if total_purchases > self.HIGH_FREQUENCY_THRESHOLD:
            frequency = "high"
        elif total_purchases >= self.MEDIUM_FREQUENCY_THRESHOLD:
            frequency = "medium"
        else:
            frequency = "low"

        # Determine price segment
        if avg_price < self.MID_RANGE_PRICE_MIN:
            price_segment = "budget"
        elif avg_price < self.PREMIUM_PRICE_MIN:
            price_segment = "mid-range"
        else:
            price_segment = "premium"

        # Get recent purchases (last 5)
        recent_purchases = [Purchase(**p) for p in purchases[-5:]]

        # Calculate confidence score (capped at 1.0)
        # More purchases = higher confidence in profile
        confidence = min(total_purchases / 10.0, 1.0)

        # Build complete profile
        profile = CustomerProfile(
            customer_id=customer_id,
            customer_name=customer["customer_name"],
            total_purchases=total_purchases,
            total_spent=total_spent,
            avg_purchase_price=avg_price,
            favorite_categories=favorite_categories,
            favorite_brands=[],  # TODO: Extract from data if available
            price_segment=price_segment,
            purchase_frequency=frequency,
            recent_purchases=recent_purchases,
            confidence=confidence,
        )

        self._logger.info(
            f"Profiled customer {customer_id}: {total_purchases} purchases, "
            f"{frequency} frequency, {price_segment} segment"
        )

        return CustomerProfilingOutput(profile=profile)
