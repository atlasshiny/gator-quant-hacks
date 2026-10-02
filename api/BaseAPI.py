from abc import ABC, abstractmethod
from pydantic import BaseModel

class Query(BaseModel):
    query: list[str] # List of symbols to querry the backend API for
    # Additional fields can be added here as needed

class BaseAPI(ABC):
    """
        Abstract base class for all API implementations.
        Subclasses must implement the query method at the very least.
    """
    def __init__(self):
        pass

    @abstractmethod
    def query(self, query: Query):
        pass


