# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from dataclasses import dataclass, field

from openai import AsyncOpenAI, OpenAI
from sirchmunk.utils import create_logger, LogCallback

if TYPE_CHECKING:
    pass


@dataclass
class OpenAIChatResponse:
    """
    Data class representing the response from the OpenAI Chat API.
    """
    content: str
    role: str = "assistant"
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = None
    finish_reason: str = None
    logprobs: Any = None

    def __str__(self):
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response to a dictionary.

        Returns:
            Dict[str, Any]: The response as a dictionary.
        """
        return {
            "content": self.content,
            "role": self.role,
            "usage": self.usage,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "logprobs": self.logprobs,
        }


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
            log_callback (LogCallback): Optional callback for logging.
            **kwargs: Additional keyword arguments passed to the OpenAI client create method.
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

        # Initialize synchronous and asynchronous loggers
        self._logger = create_logger(log_callback=log_callback, enable_async=False)
        self._logger_async = create_logger(log_callback=log_callback, enable_async=True)

    def _build_request_kwargs(
            self,
            stream: bool,
            enable_thinking: Optional[bool],
            **kwargs,
    ) -> Dict[str, Any]:
        """Merge instance-level and call-level kwargs for the API request.

        Precedence (highest wins): call-level kwargs > instance-level self._kwargs.
        ``extra_body`` dicts are deep-merged so that fields from both levels coexist.
        """
        request_kwargs = {**self._kwargs, **kwargs}

        extra_body = {
            **(self._kwargs.get("extra_body") or {}),
            **(kwargs.get("extra_body") or {}),
        }
        if enable_thinking is not None:
            extra_body["enable_thinking"] = enable_thinking
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        if stream and "stream_options" not in request_kwargs:
            request_kwargs["stream_options"] = {"include_usage": True}

        return request_kwargs

    def chat(
            self,
            messages: List[Dict[str, Any]],
            stream: bool = True,
            enable_thinking: Optional[bool] = False,
            **kwargs,
    ) -> OpenAIChatResponse:
        """
        Generate a chat completion synchronously.

        Args:
            messages (List[Dict[str, Any]]): A list of messages for the chat.
            stream (bool): Whether to stream the response.
            enable_thinking (Optional[bool]): Whether to enable model thinking/reasoning.
                Sent via ``extra_body``. Defaults to False. Pass None to omit.
            **kwargs: Additional keyword arguments merged with instance-level kwargs
                and forwarded to the OpenAI API. Call-level kwargs take precedence.

        Returns:
            OpenAIChatResponse: The structured response containing content, usage, etc.
        """
        request_kwargs = self._build_request_kwargs(stream, enable_thinking, **kwargs)

        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, stream=stream, **request_kwargs
        )

        res_content: str = ""
        role: str = "assistant"
        usage: Dict[str, int] = {}
        finish_reason: str = None
        response_model: str = self._model

        if stream:
            for chunk in resp:
                # Extract usage if present (usually in the last chunk if stream_options is set)
                if chunk.usage:
                    usage = chunk.usage.model_dump()

                # Update model name if provided in chunks
                if chunk.model:
                    response_model = chunk.model

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Capture role (usually only in the first chunk)
                if delta.role:
                    role = delta.role
                    self._logger.info(f"[role={delta.role}] ", end="", flush=True)

                # Capture content
                if delta.content:
                    self._logger.info(delta.content, end="", flush=True)
                    res_content += delta.content

                # Capture finish reason
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            # Print a newline at the end of streaming for cleaner logs
            self._logger.info("", end="\n", flush=True)

        else:
            # Non-streaming response
            message = resp.choices[0].message
            res_content = message.content or ""
            role = message.role
            finish_reason = resp.choices[0].finish_reason
            response_model = resp.model
            if resp.usage:
                usage = resp.usage.model_dump()

            # Log the full response content since we didn't stream it
            self._logger.info(f"[role={role}] {res_content}")

        return OpenAIChatResponse(
            content=res_content,
            role=role,
            usage=usage,
            model=response_model,
            finish_reason=finish_reason
        )

    async def achat(
            self,
            messages: List[Dict[str, Any]],
            stream: bool = True,
            enable_thinking: Optional[bool] = False,
            **kwargs,
    ) -> OpenAIChatResponse:
        """
        Generate a chat completion asynchronously.

        Args:
            messages (List[Dict[str, Any]]): A list of messages for the chat.
            stream (bool): Whether to stream the response.
            enable_thinking (Optional[bool]): Whether to enable model thinking/reasoning.
                Sent via ``extra_body``. Defaults to False. Pass None to omit.
            **kwargs: Additional keyword arguments merged with instance-level kwargs
                and forwarded to the OpenAI API. Call-level kwargs take precedence.

        Returns:
            OpenAIChatResponse: The structured response containing content, usage, etc.
        """
        request_kwargs = self._build_request_kwargs(stream, enable_thinking, **kwargs)

        resp = await self._async_client.chat.completions.create(
            model=self._model, messages=messages, stream=stream, **request_kwargs
        )

        res_content: str = ""
        role: str = "assistant"
        usage: Dict[str, int] = {}
        finish_reason: str = None
        response_model: str = self._model

        if stream:
            async for chunk in resp:
                # Extract usage if present (usually in the last chunk if stream_options is set)
                if chunk.usage:
                    usage = chunk.usage.model_dump()

                if chunk.model:
                    response_model = chunk.model

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Capture role
                if delta.role:
                    role = delta.role
                    await self._logger_async.info(f"[role={delta.role}] ", end="", flush=True)

                # Capture content
                if delta.content:
                    await self._logger_async.info(delta.content, end="", flush=True)
                    res_content += delta.content

                # Capture finish reason
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            # Print a newline at the end of streaming for cleaner logs
            await self._logger_async.info("", end="\n", flush=True)

        else:
            # Non-streaming response
            message = resp.choices[0].message
            res_content = message.content or ""
            role = message.role
            finish_reason = resp.choices[0].finish_reason
            response_model = resp.model
            if resp.usage:
                usage = resp.usage.model_dump()

            # Log the full response content since we didn't stream it
            await self._logger_async.info(f"[role={role}] {res_content}")

        return OpenAIChatResponse(
            content=res_content,
            role=role,
            usage=usage,
            model=response_model,
            finish_reason=finish_reason
        )
