"""
API v1 main router aggregator
"""

from fastapi import APIRouter

from app.api.v1.endpoints import health, recommendations, customers, products

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router)
api_router.include_router(recommendations.router)
api_router.include_router(customers.router)
api_router.include_router(products.router)
