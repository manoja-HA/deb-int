"""Review repository - Data access for product reviews"""

from typing import List, Dict, Optional
import pandas as pd
import logging

from app.core.config import settings
from app.repositories.base import BaseRepository

logger = logging.getLogger(__name__)

class ReviewRepository(BaseRepository):
    """Review data access repository"""

    def __init__(self):
        self._data_cache: Optional[pd.DataFrame] = None

    def get_by_id(self, review_id: str) -> Optional[Dict]:
        """Get review by ID"""
        df = self._load_data()

        matches = df[df["reviewid"] == int(review_id)]
        if len(matches) == 0:
            return None

        row = matches.iloc[0]
        return {
            "review_id": str(row["reviewid"]),
            "product_id": str(row["productid"]),
            "review_text": str(row["reviewtext"]),
            "review_date": str(row["reviewdate"]),
        }

    def get_all(self) -> List[Dict]:
        """Get all reviews"""
        df = self._load_data()
        return df.to_dict("records")

    def get_by_product_id(self, product_id: str) -> List[Dict]:
        """Get all reviews for a product"""
        df = self._load_data()

        matches = df[df["productid"] == int(product_id)]

        reviews = []
        for _, row in matches.iterrows():
            reviews.append({
                "review_id": str(row["reviewid"]),
                "product_id": str(row["productid"]),
                "review_text": str(row["reviewtext"]),
                "review_date": str(row["reviewdate"]),
            })

        return reviews

    def get_by_product_ids(self, product_ids: List[str]) -> Dict[str, List[Dict]]:
        """Get reviews grouped by product ID"""
        df = self._load_data()

        product_ids_int = [int(pid) for pid in product_ids]
        matches = df[df["productid"].isin(product_ids_int)]

        reviews_by_product = {}
        for product_id in product_ids:
            product_reviews = matches[matches["productid"] == int(product_id)]
            reviews = []
            for _, row in product_reviews.iterrows():
                reviews.append({
                    "review_id": str(row["reviewid"]),
                    "product_id": str(row["productid"]),
                    "review_text": str(row["reviewtext"]),
                    "review_date": str(row["reviewdate"]),
                })
            reviews_by_product[product_id] = reviews

        return reviews_by_product

    def _load_data(self) -> pd.DataFrame:
        """Load review data with caching"""
        if self._data_cache is not None:
            return self._data_cache

        try:
            if not settings.REVIEW_DATA_PATH.exists():
                logger.warning(f"Review data not found: {settings.REVIEW_DATA_PATH}")
                return pd.DataFrame()

            df = pd.read_csv(settings.REVIEW_DATA_PATH)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")

            self._data_cache = df
            return df

        except Exception as e:
            logger.error(f"Failed to load review data: {e}")
            return pd.DataFrame()
