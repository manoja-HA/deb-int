"""
Input validation and sanitization utilities
"""

import re
from typing import Optional, Dict
import logging

from ..config import config

logger = logging.getLogger(__name__)

def sanitize_text(text: str) -> str:
    """
    Sanitize user input text

    Args:
        text: Input text

    Returns:
        Sanitized text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove special characters that might cause issues
    text = re.sub(r'[<>{}]', '', text)

    return text

def validate_input(
    query: str,
    customer_name: Optional[str] = None,
    customer_id: Optional[str] = None
) -> Dict:
    """
    Validate user input

    Args:
        query: User query
        customer_name: Customer name
        customer_id: Customer ID

    Returns:
        Dictionary with 'valid' flag and 'errors' list
    """
    errors = []

    # Validate query
    if not query or not query.strip():
        errors.append("Query cannot be empty")

    if len(query) > config.input_max_length:
        errors.append(f"Query exceeds maximum length of {config.input_max_length} characters")

    # Validate customer identifier
    if not customer_name and not customer_id:
        errors.append("Either customer_name or customer_id must be provided")

    # Validate customer_id format if provided
    if customer_id:
        if not re.match(r'^\d+$', str(customer_id)):
            errors.append("customer_id must be numeric")

    # Validate customer_name format if provided
    if customer_name:
        if len(customer_name) < 2:
            errors.append("customer_name must be at least 2 characters")

        if not re.match(r'^[a-zA-Z\s\-\.]+$', customer_name):
            errors.append("customer_name contains invalid characters")

    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_config() -> bool:
    """
    Validate configuration settings

    Returns:
        True if valid

    Raises:
        ValueError if invalid
    """
    errors = []

    # Validate paths
    if not config.purchase_data_path:
        errors.append("purchase_data_path not configured")

    if not config.review_data_path:
        errors.append("review_data_path not configured")

    # Validate thresholds
    if not 0 <= config.similarity_threshold <= 1:
        errors.append("similarity_threshold must be between 0 and 1")

    if not 0 <= config.sentiment_threshold <= 1:
        errors.append("sentiment_threshold must be between 0 and 1")

    # Validate counts
    if config.similarity_top_k < 1:
        errors.append("similarity_top_k must be at least 1")

    if config.recommendation_top_n < 1:
        errors.append("recommendation_top_n must be at least 1")

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    return True
