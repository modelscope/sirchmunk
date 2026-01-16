# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any


class BaseScanner(ABC):
    """Abstract base class for scanners."""

    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def scan(self, *args, **kwargs) -> Any:
        """Perform a scan operation.

        Returns:
            Any: The result of the scan operation.
        """
        raise NotImplementedError("Subclasses must implement this method.")
