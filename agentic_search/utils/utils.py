import re
from typing import Dict, List, LiteralString, Optional


def extract_fields(
    content: str, tags: Optional[List[str]] = None
) -> Dict[str, LiteralString | None]:
    """
    Extracts specified fields from the LLM output content.
        e.g. <DESCRIPTION>xxx</DESCRIPTION>, <NAME>xxx</NAME>, <CONTENT>xxx</CONTENT>.

    Args:
        content (str): The raw output content from the LLM.
        tags (Optional[List[str]]): List of tags to extract. Defaults to ["DESCRIPTION", "NAME", "CONTENT"].

    Returns:
        Dict[str, LiteralString | None]: A dictionary with extracted fields.
            Keys are the lowercase tag names, and values are the extracted content or None if not found.
    """
    # Define the list of tags to extract
    tags = tags or ["DESCRIPTION", "NAME", "CONTENT"]
    extracted_data = {}

    for tag in tags:
        # Regex Breakdown:
        # <{tag}>: Matches the opening tag
        # (.*?): Non-greedy match to capture everything inside the tags
        # </{tag}>: Matches the closing tag
        # re.DOTALL: Allows the dot (.) to match newlines, handling multi-line content
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            # .strip() removes leading/trailing whitespace or newlines
            extracted_data[tag.lower()] = match.group(1).strip()
        else:
            # Handle cases where the LLM might miss a tag
            extracted_data[tag.lower()] = None

    return extracted_data


if __name__ == "__main__":

    # --- Test Case ---
    llm_raw_output = """
    Some irrelevant preamble from the LLM...
    <DESCRIPTION>
    This document set provides a detailed overview of the installation steps for Open-Agentic-Search.
    It covers environment configuration and core dependencies.
    </DESCRIPTION>
    <NAME>Environment Setup Guide</NAME>
    <CONTENT>
    1. Install Python 3.10+
    2. Run pip install -r requirements.txt
    3. Configure the .env environment variables.
    </CONTENT>
    """

    result = extract_fields(llm_raw_output)

    # Print results
    print(f"Name: {result['name']}")
    print(f"Description: {result['description']}")
    print(f"Content: \n{result['content']}")
