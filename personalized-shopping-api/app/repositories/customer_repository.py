"""
Customer repository - Data access for customer information
"""

from typing import List, Optional, Dict
import pandas as pd
import logging

from app.core.config import settings
from app.repositories.base import BaseRepository

logger = logging.getLogger(__name__)

class CustomerRepository(BaseRepository):
    """
    Customer data access repository

    Responsibilities:
    - Load customer purchase data from CSV/database
    - Filter by customer ID/name
    - Return as plain dictionaries (domain entities)
    - No business logic (pure data access)
    """

    def __init__(self):
        self._data_cache: Optional[pd.DataFrame] = None

    def get_by_id(self, customer_id: str) -> Optional[Dict]:
        """Get customer summary by ID"""
        purchases = self.get_purchases_by_customer_id(customer_id)

        if not purchases:
            return None

        df = pd.DataFrame(purchases)

        return {
            "customer_id": customer_id,
            "customer_name": df.iloc[0]["customer_name"],
            "total_purchases": len(purchases),
            "total_spent": float(df["price"].sum()),
            "country": df.iloc[0]["country"],
        }

    def get_by_name(self, customer_name: str) -> Optional[Dict]:
        """Get customer by name (case-insensitive)"""
        df = self._load_data()

        match = df[df["customername"].str.lower() == customer_name.lower()]

        if len(match) == 0:
            return None

        customer_id = str(match.iloc[0]["customerid"])
        return self.get_by_id(customer_id)

    def get_all(self) -> List[Dict]:
        """Get all unique customers"""
        df = self._load_data()

        unique_customers = df.groupby("customerid").agg({
            "customername": "first",
            "transactionid": "count",
            "purchaseprice": "sum",
        }).reset_index()

        customers = []
        for _, row in unique_customers.iterrows():
            customers.append({
                "customer_id": str(row["customerid"]),
                "customer_name": str(row["customername"]),
                "total_purchases": int(row["transactionid"]),
                "total_spent": float(row["purchaseprice"]),
            })

        return customers

    def get_purchases_by_customer_id(
        self,
        customer_id: str,
        limit_days: Optional[int] = None,
    ) -> List[Dict]:
        """Get all purchases for a customer"""
        df = self._load_data()

        # Filter by customer
        customer_df = df[df["customerid"] == int(customer_id)]

        # Convert to list of dicts
        purchases = []
        for _, row in customer_df.iterrows():
            purchases.append({
                "transaction_id": str(row["transactionid"]),
                "customer_id": str(row["customerid"]),
                "customer_name": str(row["customername"]),
                "product_id": str(row["productid"]),
                "product_name": str(row["productname"]),
                "product_category": str(row["productcategory"]),
                "quantity": int(row["purchasequantity"]),
                "price": float(row["purchaseprice"]),
                "purchase_date": str(row["purchasedate"]),
                "country": str(row["country"]),
            })

        return purchases

    def get_customer_id_by_name(self, customer_name: str) -> Optional[str]:
        """Lookup customer ID by name"""
        customer = self.get_by_name(customer_name)
        return customer["customer_id"] if customer else None

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

            # Data type conversions
            df["purchaseprice"] = df["purchaseprice"].astype(float)
            df["purchasequantity"] = df["purchasequantity"].astype(int)

            self._data_cache = df

            logger.info(f"Loaded {len(df)} purchase records")
            return df

        except Exception as e:
            logger.error(f"Failed to load purchase data: {e}")
            return pd.DataFrame()
