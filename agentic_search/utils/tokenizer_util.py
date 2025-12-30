from typing import List, Optional, Union


class TokenizerUtil:
    """Fast tokenizer utility using modelscope AutoTokenizer."""

    def __init__(self, model_id: Optional[str] = None):
        """
        Tokenizer encoding and counting utility.
        Args:
            model_id: Model ID for loading the tokenizer. Defaults to "Qwen/Qwen3-8B".
        """
        from modelscope import AutoTokenizer

        model_id: str = model_id or "Qwen/Qwen3-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def encode(self, content: str) -> List[int]:
        """Encode text into token IDs.

        Args:
            content: Input text string.

        Returns:
            List of token IDs.
        """
        if not content.strip():
            return []

        return self.tokenizer.encode(content.strip())

    def count_tokens(self, contents: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Batch count tokens for multiple texts.

        Args:
            contents: List of input text strings.

        Returns:
            List of token counts corresponding to each input text, or an integer if a single string is provided.
        """
        if isinstance(contents, str):
            contents = [contents]

        counts = []
        for content in contents:
            if not content.strip():
                counts.append(0)
            else:
                counts.append(len(self.tokenizer.encode(content.strip())))

        if len(contents) == 1:
            return counts[0]
        return counts
