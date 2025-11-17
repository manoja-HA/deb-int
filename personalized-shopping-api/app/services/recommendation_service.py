"""Recommendation service - Orchestrates multi-agent workflow"""

from typing import Optional
import time
import logging
from collections import Counter

from app.domain.schemas.recommendation import RecommendationResponse
from app.domain.schemas.product import ProductRecommendation
from app.domain.schemas.customer import CustomerProfileSummary
from app.services.customer_service import CustomerService
from app.services.product_service import ProductService
from app.repositories.customer_repository import CustomerRepository
from app.repositories.vector_repository import VectorRepository
from app.repositories.review_repository import ReviewRepository
from app.models.sentiment_analyzer import SentimentAnalyzer
from app.models.llm_factory import get_llm, LLMType
from app.core.config import settings
from app.core.exceptions import NotFoundException, ValidationException
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class RecommendationService:
    """Recommendation service orchestrating multi-agent workflow"""

    def __init__(
        self,
        customer_service: CustomerService,
        product_service: ProductService,
        vector_repository: VectorRepository,
    ):
        self.customer_service = customer_service
        self.product_service = product_service
        self.vector_repo = vector_repository
        self.customer_repo = CustomerRepository()
        self.review_repo = ReviewRepository()
        self.sentiment_analyzer = SentimentAnalyzer(method="rule_based")

    async def get_personalized_recommendations(
        self,
        query: str,
        customer_name: Optional[str] = None,
        customer_id: Optional[str] = None,
        top_n: int = 5,
        include_reasoning: bool = True,
    ) -> RecommendationResponse:
        """Get personalized recommendations using multi-agent workflow"""
        start_time = time.time()
        agent_execution = []

        # Validate input
        if not customer_name and not customer_id:
            raise ValidationException("Either customer_name or customer_id must be provided")

        try:
            # AGENT 1: Customer Profiling
            agent_execution.append("customer_profiling")
            logger.info(f"[Agent 1] Profiling customer: {customer_name or customer_id}")

            if not customer_id and customer_name:
                customer_id = self.customer_repo.get_customer_id_by_name(customer_name)
                if not customer_id:
                    raise NotFoundException("Customer", customer_name)

            profile = await self.customer_service.get_customer_profile(customer_id)
            logger.info(f"[Agent 1] Profile: {profile.total_purchases} purchases, {profile.price_segment} segment")

            # AGENT 2: Similar Customer Discovery
            agent_execution.append("similar_customer_discovery")
            logger.info(f"[Agent 2] Finding similar customers")

            similar_customers = await self.customer_service.get_similar_customers(
                customer_id,
                top_k=settings.SIMILARITY_TOP_K,
            )
            logger.info(f"[Agent 2] Found {len(similar_customers)} similar customers")

            if not similar_customers:
                raise ValidationException("No similar customers found")

            # Get candidate products from similar customers
            all_purchases = []
            for sim_customer in similar_customers:
                their_purchases = self.customer_repo.get_purchases_by_customer_id(
                    sim_customer.customer_id
                )
                all_purchases.extend(their_purchases)

            # Exclude already purchased products
            target_product_ids = set(p.product_id for p in profile.recent_purchases)
            candidate_purchases = [p for p in all_purchases if p["product_id"] not in target_product_ids]

            logger.info(f"Found {len(candidate_purchases)} candidate products")

            # AGENT 3: Review-Based Filtering
            agent_execution.append("review_filtering")
            logger.info(f"[Agent 3] Filtering products by sentiment")

            # Aggregate products
            import pandas as pd
            if candidate_purchases:
                df = pd.DataFrame(candidate_purchases)
                products = df.groupby("product_id").agg({
                    "product_name": "first",
                    "product_category": "first",
                    "price": "mean",
                    "transaction_id": "count",
                }).reset_index()
                products.columns = ["product_id", "product_name", "product_category", "avg_price", "purchase_count"]
                candidate_products = products.to_dict("records")
            else:
                candidate_products = []

            # Filter by sentiment
            filtered_products = []
            filtered_out = 0

            for product in candidate_products:
                product_id = str(product["product_id"])
                reviews = self.review_repo.get_by_product_id(product_id)

                if reviews:
                    review_texts = [r["review_text"] for r in reviews]
                    sentiment_data = self.sentiment_analyzer.calculate_average_sentiment(
                        review_texts,
                        min_reviews=settings.MIN_PURCHASES_FOR_PROFILE,
                    )
                    avg_sentiment = sentiment_data["avg_sentiment"]
                else:
                    avg_sentiment = 0.5  # Neutral for products without reviews

                if avg_sentiment >= settings.SENTIMENT_THRESHOLD:
                    product["avg_sentiment"] = avg_sentiment
                    product["review_count"] = len(reviews) if reviews else 0
                    filtered_products.append(product)
                else:
                    filtered_out += 1

            logger.info(f"[Agent 3] Filtered to {len(filtered_products)} products (removed {filtered_out})")

            # AGENT 4: Cross-Category Recommendation
            agent_execution.append("recommendation")
            logger.info(f"[Agent 4] Generating recommendations")

            # Calculate collaborative filtering scores
            product_purchase_counts = Counter(p["product_id"] for p in candidate_purchases)
            max_count = max(product_purchase_counts.values()) if product_purchase_counts else 1

            for product in filtered_products:
                product_id = str(product["product_id"])
                purchase_count = product_purchase_counts.get(product_id, 0)
                product["collab_score"] = purchase_count / max_count
                product["similar_customer_count"] = purchase_count

                # Category affinity
                category_score = self._calculate_category_affinity(
                    profile.favorite_categories,
                    product["product_category"]
                )
                product["category_score"] = category_score

                # Final score
                product["final_score"] = (
                    settings.COLLABORATIVE_WEIGHT * product["collab_score"] +
                    settings.CATEGORY_AFFINITY_WEIGHT * category_score +
                    0.2 * product.get("avg_sentiment", 0.5)
                )

            # Sort and get top N
            sorted_products = sorted(filtered_products, key=lambda x: x["final_score"], reverse=True)

            # Add diversity (max 2 per category)
            category_counts = Counter()
            final_products = []
            for product in sorted_products:
                category = product["product_category"]
                if category_counts[category] < 2:
                    final_products.append(product)
                    category_counts[category] += 1
                if len(final_products) >= top_n:
                    break

            # Build recommendations
            recommendations = []
            for product in final_products:
                # Determine reason
                if product["collab_score"] > 0.7:
                    source = "collaborative"
                    reason = f"Highly popular with {product['similar_customer_count']} similar customers"
                elif product["category_score"] > 0.8:
                    source = "category_affinity"
                    reason = f"Matches your preference for {product['product_category']}"
                else:
                    source = "trending"
                    reason = "Good balance of popularity and category fit"

                if product.get("avg_sentiment", 0) > 0.8:
                    reason += f" with excellent reviews ({product['avg_sentiment']:.0%} positive)"

                recommendations.append(
                    ProductRecommendation(
                        product_id=str(product["product_id"]),
                        product_name=product["product_name"],
                        product_category=product["product_category"],
                        avg_price=float(product["avg_price"]),
                        recommendation_score=product["final_score"],
                        reason=reason,
                        similar_customer_count=product["similar_customer_count"],
                        avg_sentiment=product.get("avg_sentiment", 0.5),
                        source=source,
                    )
                )

            logger.info(f"[Agent 4] Generated {len(recommendations)} recommendations")

            # AGENT 5: Response Generation
            agent_execution.append("response_generation")
            logger.info(f"[Agent 5] Generating natural language response")

            if include_reasoning and recommendations:
                reasoning = await self._generate_reasoning(query, profile, recommendations, similar_customers)
            else:
                reasoning = f"Based on {profile.customer_name}'s purchase history, here are {len(recommendations)} recommendations."

            # Build response
            processing_time_ms = (time.time() - start_time) * 1000

            response = RecommendationResponse(
                query=query,
                customer_profile=CustomerProfileSummary(
                    customer_id=profile.customer_id,
                    customer_name=profile.customer_name,
                    total_purchases=profile.total_purchases,
                    avg_purchase_price=profile.avg_purchase_price,
                    favorite_categories=profile.favorite_categories,
                    price_segment=profile.price_segment,
                    purchase_frequency=profile.purchase_frequency,
                ),
                recommendations=recommendations,
                reasoning=reasoning,
                confidence_score=min(len(similar_customers) / 20.0, 1.0),
                processing_time_ms=processing_time_ms,
                similar_customers_analyzed=len(similar_customers),
                products_considered=len(candidate_products),
                products_filtered_by_sentiment=filtered_out,
                recommendation_strategy="collaborative_with_category_affinity",
                agent_execution_order=agent_execution,
                metadata={
                    "fallback_used": False,
                },
            )

            logger.info(f"[Complete] Processed in {processing_time_ms:.0f}ms")
            return response

        except Exception as e:
            logger.error(f"Recommendation workflow failed: {e}", exc_info=True)
            raise

    def _calculate_category_affinity(self, target_categories: list, product_category: str) -> float:
        """Calculate category affinity score"""
        if not target_categories:
            return 0.5

        if product_category in target_categories:
            position = target_categories.index(product_category)
            return 1.0 - (position * 0.2)

        return 0.2

    async def _generate_reasoning(self, query: str, profile, recommendations, similar_customers) -> str:
        """Generate natural language reasoning using LLM"""
        try:
            context = f"Customer: {profile.customer_name}, {profile.purchase_frequency} buyer, {profile.price_segment} segment, Favorite categories: {', '.join(profile.favorite_categories)}"

            rec_text = "\n".join([
                f"{i+1}. {rec.product_name} (${rec.avg_price:.2f}): {rec.reason}"
                for i, rec in enumerate(recommendations[:3])
            ])

            prompt = f"""Based on the customer's purchase history, provide a brief explanation for these recommendations.

{context}

Recommendations:
{rec_text}

Provide a 2-3 sentence explanation focusing on why these products match the customer's preferences."""

            llm = get_llm(LLMType.RESPONSE)
            response = llm.invoke([HumanMessage(content=prompt)])

            return response.content.strip()

        except Exception as e:
            logger.warning(f"LLM reasoning generation failed: {e}")
            return f"Based on {profile.customer_name}'s purchase history of {', '.join(profile.favorite_categories)} products, these recommendations match their {profile.price_segment} price segment preferences."
