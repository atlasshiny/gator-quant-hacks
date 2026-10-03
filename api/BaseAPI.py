from abc import ABC, abstractmethod
from pydantic import BaseModel

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


