from typing import TYPE_CHECKING, Any, Dict, List

from openai import OpenAI, AsyncOpenAI

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
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the OpenAIChat client.

        Args:
            api_key (str): The API key for OpenAI.
            base_url (str): The base URL for the OpenAI API.
            model (str): The model to use for chat completions.
            verbose (bool): Whether to enable verbose logging.
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
        self._verbose = verbose

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
            model=self._model,
            messages=messages,
            stream=stream,
            **self._kwargs
        )

        res_content: str = ""

        if stream:
            for chunk in resp:
                delta = chunk.choices[0].delta
                if delta.role and self._verbose:
                    print(f"[role={delta.role}] ", end="", flush=True)
                if delta.content:
                    if self._verbose:
                        print(delta.content, end="", flush=True)
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
            model=self._model,
            messages=messages,
            stream=stream,
            **self._kwargs
        )

        res_content: str = ""

        if stream:
            async for chunk in resp:
                delta = chunk.choices[0].delta
                if delta.role and self._verbose:
                    print(f"[role={delta.role}] ", end="", flush=True)
                if delta.content:
                    if self._verbose:
                        print(delta.content, end="", flush=True)
                    res_content += delta.content
        else:
            res_content = resp.choices[0].message.content

        return res_content
