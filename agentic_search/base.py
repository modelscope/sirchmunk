# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod


class BaseSearch(ABC):
    """
    Abstract base class for search implementations.
    """

    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def search(self, *args, **kwargs):
        """
        Perform a search based on the given query.
        """
        raise NotImplementedError("search method not implemented")
