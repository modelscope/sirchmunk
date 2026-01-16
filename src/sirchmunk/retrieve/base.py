# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """
    Abstract base class for data retrievers.
    """

    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def retrieve(self, *args, **kwargs) -> Any:
        """
        Abstract method to retrieve data.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Any: Retrieved results.
        """
        raise NotImplementedError("Subclasses must implement this method.")
