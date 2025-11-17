"""
Query intent classification service
Determines whether a query is asking for information or recommendations
"""

import re
from enum import Enum
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Types of query intents"""
    INFORMATIONAL = "informational"  # Questions about customer data
    RECOMMENDATION = "recommendation"  # Product recommendation requests


class InformationCategory(str, Enum):
    """Categories of informational queries"""
    TOTAL_PURCHASES = "total_purchases"
    SPENDING = "spending"
    FAVORITE_CATEGORIES = "favorite_categories"
    RECENT_PURCHASES = "recent_purchases"
    CUSTOMER_PROFILE = "customer_profile"
    GENERAL = "general"


class QueryIntentClassifier:
    """Classifies user queries to determine intent and extract parameters"""

    # Keywords for informational queries
    INFORMATIONAL_PATTERNS = {
        InformationCategory.TOTAL_PURCHASES: [
            r'\b(how many|total|number of)\s+(purchases?|items?|products?|orders?)\b',
            r'\bpurchase(d|s)?\s+(count|total|number)\b',
            r'\bhow much\s+(did|has).*\b(buy|purchase|bought)\b',
        ],
        InformationCategory.SPENDING: [
            r'\b(how much|total|amount).*\b(spent|spending|paid|money)\b',
            r'\btotal\s+(amount|cost|price|expenditure)\b',
            r'\bspending\s+(total|amount|history)\b',
            r'\baverage\s+(purchase|spending|price)\b',
            r'\bspent\s+(total|in total)\b',
            r'\bhow much.*\bspent\b',
        ],
        InformationCategory.FAVORITE_CATEGORIES: [
            r'\b(favorite|preferred|most\s+bought|top)\s+(categories?|types?)\b',
            r'\bwhat\s+(categories?|types?).*\b(buy|purchase|like)\b',
            r'\bcategor(y|ies)\s+(preference|history)\b',
        ],
        InformationCategory.RECENT_PURCHASES: [
            r'\b(recent|latest|last|previous)\s+(purchases?|orders?|items?)\b',
            r'\bwhat.*\b(recently|lately)\s+(bought|purchased)\b',
            r'\bpurchase\s+history\b',
        ],
        InformationCategory.CUSTOMER_PROFILE: [
            r'\b(profile|information|details|data)\s+(of|for|about)\b',
            r'\btell\s+me\s+about\b',
            r'\bwho\s+is\b',
            r'\bcustomer\s+(profile|info|details)\b',
        ],
    }

    # Keywords for recommendation queries
    RECOMMENDATION_KEYWORDS = [
        r'\brecommend(ations?|ed)?\b',
        r'\bsuggest(ions?|ed)?\b',
        r'\bwhat\s+(else|other|more).*\b(like|buy|purchase)\b',
        r'\bshow\s+me\s+(products?|items?)\b',
        r'\blooking\s+for\b',
        r'\binterested\s+in\b',
        r'\bwould.*\b(like|enjoy|love)\b',
    ]

    # Question words that typically indicate informational queries
    QUESTION_WORDS = [
        r'\bhow many\b',
        r'\bhow much\b',
        r'\bwhat (is|are|was|were)\b',
        r'\bwhen (did|was)\b',
        r'\bwhich\b',
        r'\bwho (is|are)\b',
    ]

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify the query intent and extract parameters

        Returns:
            Dict with:
                - intent: QueryIntent (INFORMATIONAL or RECOMMENDATION)
                - category: InformationCategory (if informational)
                - confidence: float (0-1)
                - extracted_info: Dict of extracted parameters
        """
        query_lower = query.lower().strip()

        # Check for informational patterns
        info_category, info_confidence = self._check_informational(query_lower)

        # Check for recommendation patterns
        rec_confidence = self._check_recommendation(query_lower)

        # Check for question words (increases info confidence)
        has_question_word = any(
            re.search(pattern, query_lower, re.IGNORECASE)
            for pattern in self.QUESTION_WORDS
        )

        if has_question_word:
            info_confidence *= 1.3

        # Determine final intent
        if info_confidence > rec_confidence:
            intent = QueryIntent.INFORMATIONAL
            confidence = min(info_confidence, 1.0)
            category = info_category
        else:
            intent = QueryIntent.RECOMMENDATION
            confidence = min(rec_confidence, 1.0)
            category = None

        logger.info(
            f"Query classified: intent={intent}, category={category}, "
            f"confidence={confidence:.2f}, query='{query[:50]}...'"
        )

        return {
            "intent": intent,
            "category": category,
            "confidence": confidence,
            "extracted_info": self._extract_parameters(query_lower, category),
        }

    def _check_informational(self, query: str) -> tuple[Optional[InformationCategory], float]:
        """Check if query matches informational patterns"""
        best_category = None
        max_confidence = 0.0

        for category, patterns in self.INFORMATIONAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    confidence = 0.8
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_category = category

        # If no specific pattern matched but has question structure
        if max_confidence == 0 and '?' in query:
            max_confidence = 0.3
            best_category = InformationCategory.GENERAL

        return best_category, max_confidence

    def _check_recommendation(self, query: str) -> float:
        """Check if query matches recommendation patterns"""
        matches = 0
        for pattern in self.RECOMMENDATION_KEYWORDS:
            if re.search(pattern, query, re.IGNORECASE):
                matches += 1

        # Base confidence
        if matches > 0:
            confidence = 0.6 + (matches * 0.15)
        else:
            # Default to recommendation if unclear (existing behavior)
            confidence = 0.4

        return min(confidence, 1.0)

    def _extract_parameters(
        self,
        query: str,
        category: Optional[InformationCategory]
    ) -> Dict[str, Any]:
        """Extract parameters from the query"""
        params = {}

        if not category:
            return params

        # Extract time periods
        if re.search(r'\b(recent|latest|last)\b', query):
            params['time_period'] = 'recent'
        elif re.search(r'\ball\s+time\b', query):
            params['time_period'] = 'all_time'

        # Extract limits
        top_match = re.search(r'\btop\s+(\d+)\b', query)
        if top_match:
            params['limit'] = int(top_match.group(1))

        return params
