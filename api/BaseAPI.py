from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any
from pydantic import BaseModel

from .query import FinancialQuery  # adjust import path as needed

class BaseAPI(ABC):
    """
    Abstract Base Class for financial data provider SDK wrappers.
    
    Subclasses should define their specific query execution logic,
    authentication handling, and session management.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.extra_config = kwargs

    @abstractmethod
    def query(self, query: FinancialQuery) -> Any:
        """
        Execute a synchronous data query against the provider SDK or endpoint.
        """
        pass

    @abstractmethod
    async def query_async(self, query: FinancialQuery) -> Any:
        """
        Execute an asynchronous data query against the provider SDK or endpoint.
        Override if the underlying SDK/REST client supports async/await.
        """
        raise NotImplementedError("Async query is not implemented for this provider.")

    # Context Manager hooks for clean connection handling
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """Close underlying HTTP sessions or client connections if applicable."""
        pass