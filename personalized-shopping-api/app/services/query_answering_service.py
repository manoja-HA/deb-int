"""
Query answering service for informational queries
Answers questions about customer data instead of providing recommendations
"""

from typing import Optional, Dict, Any
import logging
from langchain_core.messages import HumanMessage

from app.domain.schemas.customer import CustomerProfile
from app.services.query_intent_classifier import InformationCategory
from app.models.llm_factory import get_llm, LLMType

logger = logging.getLogger(__name__)


class QueryAnsweringService:
    """Service to answer informational queries about customer data"""

    def __init__(self):
        pass

    async def answer_query(
        self,
        query: str,
        profile: CustomerProfile,
        category: InformationCategory,
        extracted_info: Dict[str, Any],
    ) -> str:
        """
        Generate an answer to an informational query

        Args:
            query: The user's question
            profile: Customer profile with all data
            category: Type of information being requested
            extracted_info: Additional parameters extracted from query

        Returns:
            Natural language answer to the question
        """
        logger.info(f"Answering query: category={category}, query='{query[:50]}...'")

        # Generate structured answer based on category
        if category == InformationCategory.TOTAL_PURCHASES:
            answer = self._answer_total_purchases(profile, extracted_info)
        elif category == InformationCategory.SPENDING:
            answer = self._answer_spending(profile, extracted_info)
        elif category == InformationCategory.FAVORITE_CATEGORIES:
            answer = self._answer_favorite_categories(profile, extracted_info)
        elif category == InformationCategory.RECENT_PURCHASES:
            answer = self._answer_recent_purchases(profile, extracted_info)
        elif category == InformationCategory.CUSTOMER_PROFILE:
            answer = self._answer_customer_profile(profile)
        else:
            # Use LLM for general questions
            answer = await self._answer_general(query, profile)

        return answer

    def _answer_total_purchases(
        self,
        profile: CustomerProfile,
        extracted_info: Dict[str, Any]
    ) -> str:
        """Answer questions about total purchases"""
        total = profile.total_purchases

        # Build detailed answer
        answer_parts = [
            f"{profile.customer_name} has made **{total} purchase{'s' if total != 1 else ''}** in total."
        ]

        # Add category breakdown if available
        if profile.favorite_categories:
            top_category = profile.favorite_categories[0]
            answer_parts.append(
                f"Most purchases are in the **{top_category}** category."
            )

        # Add spending info
        if profile.total_spent > 0:
            answer_parts.append(
                f"Total amount spent: **${profile.total_spent:,.2f}**"
            )

        return " ".join(answer_parts)

    def _answer_spending(
        self,
        profile: CustomerProfile,
        extracted_info: Dict[str, Any]
    ) -> str:
        """Answer questions about spending"""
        total_spent = profile.total_spent
        avg_price = profile.avg_purchase_price

        answer_parts = [
            f"{profile.customer_name} has spent **${total_spent:,.2f}** in total across {profile.total_purchases} purchases."
        ]

        # Add average
        answer_parts.append(
            f"Average purchase price: **${avg_price:,.2f}** ({profile.price_segment} segment)."
        )

        return " ".join(answer_parts)

    def _answer_favorite_categories(
        self,
        profile: CustomerProfile,
        extracted_info: Dict[str, Any]
    ) -> str:
        """Answer questions about favorite categories"""
        if not profile.favorite_categories:
            return f"{profile.customer_name} hasn't made enough purchases to determine favorite categories yet."

        limit = extracted_info.get('limit', 3)
        top_categories = profile.favorite_categories[:limit]

        if len(top_categories) == 1:
            answer = f"{profile.customer_name}'s favorite category is **{top_categories[0]}**."
        else:
            categories_str = ", ".join(f"**{cat}**" for cat in top_categories[:-1])
            categories_str += f", and **{top_categories[-1]}**"
            answer = f"{profile.customer_name}'s top categories are: {categories_str}."

        return answer

    def _answer_recent_purchases(
        self,
        profile: CustomerProfile,
        extracted_info: Dict[str, Any]
    ) -> str:
        """Answer questions about recent purchases"""
        if not profile.recent_purchases:
            return f"{profile.customer_name} has no recorded purchases."

        limit = min(extracted_info.get('limit', 5), len(profile.recent_purchases))
        recent = profile.recent_purchases[:limit]

        answer_parts = [
            f"{profile.customer_name}'s recent purchases:\n"
        ]

        for i, purchase in enumerate(recent, 1):
            answer_parts.append(
                f"{i}. **{purchase.product_name}** "
                f"({purchase.product_category}) - "
                f"${purchase.price:,.2f} "
                f"on {purchase.purchase_date}"
            )

        return "\n".join(answer_parts)

    def _answer_customer_profile(self, profile: CustomerProfile) -> str:
        """Answer questions about customer profile"""
        answer_parts = [
            f"## Customer Profile: {profile.customer_name}\n",
            f"- **Customer ID**: {profile.customer_id}",
            f"- **Total Purchases**: {profile.total_purchases}",
            f"- **Total Spent**: ${profile.total_spent:,.2f}",
            f"- **Average Purchase**: ${profile.avg_purchase_price:,.2f}",
            f"- **Price Segment**: {profile.price_segment.title()}",
            f"- **Purchase Frequency**: {profile.purchase_frequency or 'Unknown'}",
        ]

        if profile.favorite_categories:
            categories = ", ".join(profile.favorite_categories[:3])
            answer_parts.append(f"- **Favorite Categories**: {categories}")

        if profile.favorite_brands:
            brands = ", ".join(profile.favorite_brands[:3])
            answer_parts.append(f"- **Favorite Brands**: {brands}")

        return "\n".join(answer_parts)

    async def _answer_general(self, query: str, profile: CustomerProfile) -> str:
        """Use LLM to answer general questions about the customer"""
        try:
            # Build context from profile
            context = self._build_profile_context(profile)

            prompt = f"""You are a helpful assistant answering questions about a customer's purchase history.

Customer Data:
{context}

Question: {query}

Provide a concise, factual answer based only on the data provided. If the data doesn't contain the information needed to answer the question, say so clearly."""

            llm = get_llm(LLMType.RESPONSE)
            response = llm.invoke([HumanMessage(content=prompt)])

            return response.content.strip()

        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return f"I have information about {profile.customer_name}, but I'm unable to answer that specific question at the moment."

    def _build_profile_context(self, profile: CustomerProfile) -> str:
        """Build a text representation of the customer profile"""
        lines = [
            f"Customer: {profile.customer_name} (ID: {profile.customer_id})",
            f"Total Purchases: {profile.total_purchases}",
            f"Total Spent: ${profile.total_spent:,.2f}",
            f"Average Purchase Price: ${profile.avg_purchase_price:,.2f}",
            f"Price Segment: {profile.price_segment}",
            f"Purchase Frequency: {profile.purchase_frequency or 'Unknown'}",
        ]

        if profile.favorite_categories:
            lines.append(f"Favorite Categories: {', '.join(profile.favorite_categories)}")

        if profile.favorite_brands:
            lines.append(f"Favorite Brands: {', '.join(profile.favorite_brands)}")

        if profile.recent_purchases:
            lines.append("\nRecent Purchases:")
            for i, purchase in enumerate(profile.recent_purchases[:5], 1):
                lines.append(
                    f"  {i}. {purchase.product_name} ({purchase.product_category}) - "
                    f"${purchase.price:,.2f} on {purchase.purchase_date}"
                )

        return "\n".join(lines)
