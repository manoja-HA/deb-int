"""
Base repository abstract class
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List

T = TypeVar("T")

class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository

    Defines common interface for all repositories
    following Repository pattern
    """

    @abstractmethod
    def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID"""
        pass

    @abstractmethod
    def get_all(self) -> List[T]:
        """Get all entities"""
        pass
