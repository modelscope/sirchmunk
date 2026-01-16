# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, Any, Dict, List

from openai import AsyncOpenAI, OpenAI
from sirchmunk.utils import create_logger, LogCallback

if TYPE_CHECKING:
    pass


class OpenAIChat:
    """
    A client for interacting with OpenAI's chat completion API.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        log_callback: LogCallback = None,
        **kwargs,
    ):
        """
        Initialize the OpenAIChat client.

        Args:
            api_key (str): The API key for OpenAI.
            base_url (str): The base URL for the OpenAI API.
            model (str): The model to use for chat completions.
            **kwargs: Additional keyword arguments.
        """
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self._model = model
        self._kwargs = kwargs

        self._logger = create_logger(log_callback=log_callback, enable_async=False)
        self._logger_async = create_logger(log_callback=log_callback, enable_async=True)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = True,
    ) -> str:
        """
        Generate a chat completion synchronously.

        Args:
            messages (List[Dict[str, Any]]): A list of messages for the chat.
            stream (bool): Whether to stream the response.

        Returns:
            str: The generated chat completion.
        """
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, stream=stream, **self._kwargs
        )

        res_content: str = ""

        if stream:
            for chunk in resp:
                delta = chunk.choices[0].delta
                if delta.role:
                    # print(f"[role={delta.role}] ", end="", flush=True)
                    self._logger.info(f"[role={delta.role}] ", end="", flush=True)
                if delta.content:
                    # print(delta.content, end="", flush=True)
                    self._logger.info(delta.content, end="", flush=True)
                    res_content += delta.content
        else:
            res_content = resp.choices[0].message.content

        return res_content

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = True,
    ) -> str:
        """
        Generate a chat completion asynchronously.

        Args:
            messages (List[Dict[str, Any]]): A list of messages for the chat.
            stream (bool): Whether to stream the response.

        Returns:
            str: The generated chat completion.
        """
        resp = await self._async_client.chat.completions.create(
            model=self._model, messages=messages, stream=stream, **self._kwargs
        )

        res_content: str = ""

        if stream:
            async for chunk in resp:
                delta = chunk.choices[0].delta
                if delta.role:
                    await self._logger_async.info(f"[role={delta.role}] ", end="", flush=True)
                if delta.content:
                    await self._logger_async.info(delta.content, end="", flush=True)
                    res_content += delta.content
        else:
            res_content = resp.choices[0].message.content

        return res_content
