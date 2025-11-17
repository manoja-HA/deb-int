"""Product repository - Data access for product information"""

from typing import List, Dict, Optional
import pandas as pd
import logging

from app.core.config import settings
from app.repositories.base import BaseRepository

logger = logging.getLogger(__name__)

class ProductRepository(BaseRepository):
    """Product data access repository"""

    def __init__(self):
        self._data_cache: Optional[pd.DataFrame] = None

    def get_by_id(self, product_id: str) -> Optional[Dict]:
        """Get product by ID"""
        df = self._load_data()

        matches = df[df["productid"] == int(product_id)]
        if len(matches) == 0:
            return None

        return {
            "product_id": product_id,
            "product_name": matches.iloc[0]["productname"],
            "product_category": matches.iloc[0]["productcategory"],
            "avg_price": float(matches["purchaseprice"].mean()),
            "purchase_count": len(matches),
        }

    def get_all(self) -> List[Dict]:
        """Get all unique products"""
        df = self._load_data()

        products = df.groupby("productid").agg({
            "productname": "first",
            "productcategory": "first",
            "purchaseprice": "mean",
            "transactionid": "count",
        }).reset_index()

        return products.to_dict("records")

    def get_products_by_ids(self, product_ids: List[str]) -> List[Dict]:
        """Get multiple products by IDs"""
        df = self._load_data()

        product_ids_int = [int(pid) for pid in product_ids]
        matches = df[df["productid"].isin(product_ids_int)]

        products = matches.groupby("productid").agg({
            "productname": "first",
            "productcategory": "first",
            "purchaseprice": "mean",
            "transactionid": "count",
        }).reset_index()

        return products.to_dict("records")

    def _load_data(self) -> pd.DataFrame:
        """Load purchase data with caching"""
        if self._data_cache is not None:
            return self._data_cache

        try:
            if not settings.PURCHASE_DATA_PATH.exists():
                logger.warning(f"Purchase data not found: {settings.PURCHASE_DATA_PATH}")
                return pd.DataFrame()

            df = pd.read_csv(settings.PURCHASE_DATA_PATH)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")

            self._data_cache = df
            return df

        except Exception as e:
            logger.error(f"Failed to load purchase data: {e}")
            return pd.DataFrame()
