"""
Workflows package - Orchestration layer

Workflows compose agents into end-to-end use cases. Each workflow:
- Calls agents in a specific sequence
- Passes Pydantic models between agents
- Contains no business logic (pure orchestration)
- Returns typed outputs
"""

from app.workflows.personalized_recommendation import PersonalizedRecommendationWorkflow

__all__ = [
    "PersonalizedRecommendationWorkflow",
]
