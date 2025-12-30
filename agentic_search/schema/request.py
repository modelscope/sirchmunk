from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union


@dataclass
class ImageURL:
    """Represents an image URL with optional detail and media type."""

    url: str
    detail: str = "auto"
    media_type: str = "image/jpeg"  # Necessary for Anthropic


@dataclass
class ContentItem:
    """Represents a content item, which can be either text or an image URL."""

    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None

    def to_openai(self):
        if self.type == "text":
            return {"type": "text", "text": self.text}
        return {
            "type": "image_url",
            "image_url": {"url": self.image_url.url, "detail": self.image_url.detail},
        }

    def to_anthropic(self):
        if self.type == "text":
            return {"type": "text", "text": self.text}

        # Strip Base64 prefix if present for Anthropic
        raw_data = self.image_url.url
        if "base64," in raw_data:
            raw_data = raw_data.split("base64,")[1]

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.image_url.media_type,
                "data": raw_data,
            },
        }


@dataclass
class Message:
    """Represents a message in the conversation, system/user/assistant."""

    role: str
    content: Union[str, List[ContentItem]]


@dataclass
class Request:
    """
    Represents a request to Agentic Search API, supporting both OpenAI and Anthropic message formats.
    """

    messages: List[Message]
    system: Optional[str] = "You are a helpful assistant."
    message_format: Literal["openai", "anthropic"] = "openai"

    def get_system(self) -> str:
        """Get the system prompt."""
        return self.system

    def get_user_query(self) -> str:
        """Extract the user query from the messages."""
        for m in self.messages:
            if m.role == "user":
                if isinstance(m.content, str):
                    return m.content
                else:
                    texts = [c.text for c in m.content if c.type == "text" and c.text]
                    return " ".join(texts)
        return ""

    def get_image_urls(self) -> List[str]:
        """Extract image URLs from user messages."""
        image_urls = []
        for m in self.messages:
            if m.role == "user":
                if isinstance(m.content, list):
                    for c in m.content:
                        if c.type == "image_url" and c.image_url:
                            image_urls.append(c.image_url.url)
        return image_urls

    def to_payload(
        self, prompt_template: Optional[str] = None
    ) -> Union[List[Dict], Dict]:
        """Convert messages to the appropriate API payload format based on message_format."""
        if self.message_format == "openai":
            return self._to_openai_payload(prompt_template=prompt_template)
        elif self.message_format == "anthropic":
            return self._to_anthropic_payload(prompt_template=prompt_template)
        else:
            raise ValueError(
                f"Unsupported message format: {self.message_format}, must be 'openai' or 'anthropic'."
            )

    def _to_openai_payload(self, prompt_template: Optional[str] = None) -> List[Dict]:
        """Convert messages to OpenAI API payload format."""
        formatted_msgs = []
        system_msg: Message = Message(role="system", content=self.system)
        self.messages.insert(0, system_msg)

        for m in self.messages:
            if m.role == "user" and prompt_template:
                # Apply prompt template if provided
                if isinstance(m.content, str):
                    content = prompt_template.format(user_input=m.content)
                else:
                    content = []
                    for c in m.content:
                        if c.type == "text":
                            formatted_text = prompt_template.format(user_input=c.text)
                            content.append({"type": "text", "text": formatted_text})
                        else:
                            content.append(c.to_openai())
            else:
                content = (
                    m.content
                    if isinstance(m.content, str)
                    else [c.to_openai() for c in m.content]
                )
            formatted_msgs.append({"role": m.role, "content": content})

        return formatted_msgs

    def _to_anthropic_payload(self, prompt_template: Optional[str] = None) -> Dict:
        """Convert messages to Anthropic API payload format."""
        formatted_msgs = []

        for m in self.messages:
            if m.role == "user" and prompt_template:
                # Apply prompt template if provided
                if isinstance(m.content, str):
                    content = prompt_template.format(user_input=m.content)
                else:
                    content = []
                    for c in m.content:
                        if c.type == "text":
                            formatted_text = prompt_template.format(user_input=c.text)
                            content.append({"type": "text", "text": formatted_text})
                        else:
                            content.append(c.to_anthropic())
            else:
                # Anthropic expects 'system' as a top-level parameter, not in messages
                content = (
                    m.content
                    if isinstance(m.content, str)
                    else [c.to_anthropic() for c in m.content]
                )

            formatted_msgs.append({"role": m.role, "content": content})

        payload = {"system": self.system, "messages": formatted_msgs}

        return payload


if __name__ == "__main__":
    import json

    prompt_template: str = (
        "Please answer the following question carefully based on the given information: {user_input}"
    )

    # Usage for Anthropic
    req_anthropic = Request(
        message_format="anthropic",
        system="Analyze the video frames carefully.",
        messages=[
            Message(
                role="user",
                content=[
                    ContentItem(type="text", text="What is happening here?"),
                    ContentItem(
                        type="image_url",
                        image_url=ImageURL(url="base64_string_here..."),
                    ),
                ],
            ),
        ],
    )
    print(
        f"Anthropic Payload:\n{json.dumps(req_anthropic.to_payload(prompt_template=prompt_template), ensure_ascii=False, indent=2)}"
    )

    # Usage for OpenAI
    req_openai = Request(
        message_format="openai",
        system="You are a helpful assistant.",
        messages=[
            Message(
                role="user",
                content=[
                    ContentItem(type="text", text="What is unusual about this image?"),
                    ContentItem(
                        type="image_url",
                        image_url=ImageURL(
                            url="https://example.com/strange-building.jpg"
                        ),
                    ),
                ],
            ),
        ],
    )
    print(
        f"\nOpenAI Payload:\n{json.dumps(req_openai.to_payload(prompt_template=prompt_template), ensure_ascii=False, indent=2)}"
    )

    print(f"\nUser Query (OpenAI format): {req_openai.get_user_query()}")

    print(f"\nImage URLs (OpenAI format): {req_openai.get_image_urls()}")
