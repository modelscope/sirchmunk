# Copyright (c) ModelScope Contributors. All rights reserved.

__all__ = ["OpenAIChat"]


def __getattr__(name):
    if name == "OpenAIChat":
        from .openai_chat import OpenAIChat
        return OpenAIChat
    raise AttributeError(name)
