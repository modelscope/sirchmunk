# Copyright (c) ModelScope Contributors. All rights reserved.
"""
ReAct (Reasoning + Acting) search agent.

Implements an iterative loop where the LLM reasons about what information
it needs, selects and calls a retrieval tool, observes the result, and
either continues searching or produces a final answer.  All retrieval
state is tracked via SearchContext (token budget, file dedup, logs).
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from sirchmunk.agentic.prompts import (
    REACT_CONTINUATION_PROMPT,
    REACT_SYSTEM_PROMPT,
)
from sirchmunk.agentic.tools import ToolRegistry
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.schema.search_context import SearchContext
from sirchmunk.utils import LogCallback, create_logger

logger = logging.getLogger(__name__)


# ---- Helpers ----

_ANSWER_PATTERN = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL)


def _extract_answer(text: str) -> Optional[str]:
    """Extract content within <ANSWER>...</ANSWER> tags."""
    m = _ANSWER_PATTERN.search(text)
    return m.group(1).strip() if m else None


def _parse_tool_call(text: str, available_tools: List[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Best-effort extraction of a tool call from free-form LLM output.

    Supports multiple styles:
    1. JSON block with "tool" key: ``{"tool": "keyword_search", "arguments": {...}}``
    2. JSON block with "name" key: ``{"name": "keyword_search", "arguments": {...}}``
    3. Function call style: ``keyword_search({"keywords": [...]})``
    4. Nested JSON in markdown code block: ```json\\n{...}\\n```

    Returns:
        Tuple of (tool_name, arguments_dict) or None if no valid call found.
    """
    # Pre-process: extract JSON from markdown code blocks
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)

    # Combine code block contents with other potential JSON
    search_texts = code_blocks + [text]

    for search_text in search_texts:
        # Strategy 1: look for JSON objects
        json_blocks = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", search_text)
        for block in json_blocks:
            try:
                obj = json.loads(block)
                tool_name = obj.get("tool") or obj.get("name")
                if tool_name and tool_name in available_tools:
                    args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
                    return tool_name, args
            except (json.JSONDecodeError, AttributeError):
                continue

    # Strategy 2: look for function_name({...}) pattern
    for tool_name in available_tools:
        pattern = rf"{re.escape(tool_name)}\s*\(\s*(\{{.*?\}})\s*\)"
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                args = json.loads(m.group(1))
                return tool_name, args
            except json.JSONDecodeError:
                continue

    return None


def _build_tool_descriptions(registry: ToolRegistry) -> str:
    """Build human-readable tool descriptions from the registry.

    Converts tool schemas into a compact text block that the LLM
    can parse to understand available tools and their parameters.
    """
    lines: List[str] = []
    for schema_wrapper in registry.get_all_schemas():
        func = schema_wrapper.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_parts: List[str] = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            req_tag = " (required)" if pname in required else ""
            param_parts.append(f"    - {pname} ({ptype}{req_tag}): {pdesc}")

        lines.append(f"### {name}\n{desc}")
        if param_parts:
            lines.append("  Parameters:\n" + "\n".join(param_parts))
        lines.append("")

    return "\n".join(lines)


class ReActSearchAgent:
    """Iterative ReAct agent for agentic information retrieval.

    Each ``run()`` call executes a self-contained search session with its
    own SearchContext.  The agent interleaves reasoning with tool calls
    until it produces a final answer or exhausts the budget / loop limit.

    Args:
        llm: OpenAI-compatible chat client.
        tool_registry: Registry of available tools.
        max_loops: Maximum number of reasoning-action iterations.
        max_token_budget: Maximum LLM tokens per session.
        log_callback: Optional async logging callback.
    """

    def __init__(
        self,
        llm: OpenAIChat,
        tool_registry: ToolRegistry,
        max_loops: int = 10,
        max_token_budget: int = 64000,
        log_callback: LogCallback = None,
    ) -> None:
        self.llm = llm
        self.registry = tool_registry
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self._logger = create_logger(log_callback=log_callback, enable_async=True)

    # ---- Public API ----

    async def run(
        self,
        query: str,
        images: Optional[List[str]] = None,
        initial_keywords: Optional[List[str]] = None,
    ) -> Tuple[str, SearchContext]:
        """Execute a full ReAct search session.

        Args:
            query: User's search query.
            images: Optional image URLs (reserved for future multimodal support).
            initial_keywords: Optional pre-extracted keywords to use for the
                first keyword_search call, bypassing the LLM's first turn.

        Returns:
            Tuple of (final_answer_text, search_context).
        """
        context = SearchContext(
            max_token_budget=self.max_token_budget,
            max_loops=self.max_loops,
        )

        # Build tool descriptions for the system prompt
        tool_descriptions = _build_tool_descriptions(self.registry)

        # Build the initial conversation
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self._build_system_prompt(context, tool_descriptions),
            },
            {
                "role": "user",
                "content": self._build_user_message(query, images),
            },
        ]

        await self._logger.info(f"[ReAct] Starting search: '{query[:80]}...'")
        await self._logger.info(f"[ReAct] Budget: {context.max_token_budget} tokens, max loops: {context.max_loops}")
        await self._logger.info(f"[ReAct] Tools: {self.registry.tool_names}")

        tool_names = self.registry.tool_names
        final_answer: Optional[str] = None

        # Optionally execute pre-extracted keywords before the first LLM call
        if initial_keywords and "keyword_search" in tool_names:
            context.increment_loop()
            await self._logger.info(
                f"[ReAct] Loop {context.loop_count}/{context.max_loops} | "
                f"Pre-extracted keywords: {initial_keywords}"
            )
            result_text, meta = await self.registry.execute(
                tool_name="keyword_search",
                context=context,
                keywords=initial_keywords,
            )
            if result_text and "No results" not in result_text:
                messages.append({
                    "role": "assistant",
                    "content": (
                        f"I'll start by searching with the pre-extracted keywords: {initial_keywords}\n"
                        f'{{"tool": "keyword_search", "arguments": {{"keywords": {json.dumps(initial_keywords, ensure_ascii=False)}}}}}'
                    ),
                })
                messages.append({
                    "role": "user",
                    "content": (
                        f"**Tool result** (keyword_search):\n{result_text}\n\n"
                        f"{self._build_continuation_prompt(context)}"
                    ),
                })
                await self._logger.info(
                    f"[ReAct] Initial keyword search: {len(result_text)} chars"
                )

        while not context.is_loop_limit_reached() and not context.is_budget_exceeded():
            context.increment_loop()
            await self._logger.info(f"[ReAct] Loop {context.loop_count}/{context.max_loops} | {context.summary()}")

            # Call LLM
            llm_response = await self._call_llm(messages)
            content = llm_response.content or ""

            # Track LLM token usage
            usage = llm_response.usage or {}
            total_tok = usage.get("total_tokens", 0)
            if total_tok == 0:
                total_tok = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            context.add_llm_tokens(total_tok, usage=usage if usage else None)

            # Check for final answer in response
            answer = _extract_answer(content)
            if answer:
                final_answer = answer
                await self._logger.success(f"[ReAct] Answer found at loop {context.loop_count}")
                break

            # Try to extract a tool call
            tool_call = _parse_tool_call(content, tool_names)
            if tool_call is None:
                # LLM didn't call a tool and didn't answer — nudge it
                await self._logger.warning("[ReAct] No tool call or answer detected, nudging...")
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        "You must either call a tool using the JSON format or provide "
                        "a final answer in <ANSWER>...</ANSWER> tags. Please try again.\n\n"
                        f"{self._build_continuation_prompt(context)}"
                    ),
                })
                continue

            tool_name, tool_args = tool_call
            await self._logger.info(f"[ReAct] Calling tool: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})")

            # Execute the tool
            result_text, meta = await self.registry.execute(
                tool_name=tool_name,
                context=context,
                **tool_args,
            )

            # Truncate if the tool returned too much text
            if len(result_text) > 8000:
                result_text = result_text[:8000] + "\n... [output truncated]"

            await self._logger.info(
                f"[ReAct] Tool result: {len(result_text)} chars | "
                f"Budget remaining: {context.budget_remaining}"
            )

            # Append reasoning + tool call + observation to conversation
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"**Tool result** ({tool_name}):\n{result_text}\n\n"
                    f"{self._build_continuation_prompt(context)}"
                ),
            })

        # If loop exited without answer, ask LLM to synthesize
        if final_answer is None:
            await self._logger.warning("[ReAct] Loop limit or budget reached — forcing synthesis")
            messages.append({
                "role": "user",
                "content": (
                    "You have reached the retrieval limit. "
                    "Please synthesize your best answer from ALL evidence collected so far. "
                    "Wrap it in <ANSWER>...</ANSWER> tags."
                ),
            })
            llm_response = await self._call_llm(messages)
            content = llm_response.content or ""
            usage = llm_response.usage or {}
            total_tok = usage.get("total_tokens", 0)
            if total_tok == 0:
                total_tok = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            context.add_llm_tokens(total_tok, usage=usage if usage else None)
            final_answer = _extract_answer(content) or content

        await self._logger.success(f"[ReAct] Completed: {context.summary()}")

        return final_answer, context

    # ---- Internal helpers ----

    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
    ):
        """Call the LLM with the given messages.

        Uses stream=False for tool-calling loops to get complete responses.
        """
        return await self.llm.achat(messages=messages, stream=False)

    @staticmethod
    def _build_system_prompt(context: SearchContext, tool_descriptions: str) -> str:
        """Format the system prompt with tool descriptions and context state."""
        return REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            budget_remaining=context.budget_remaining,
            files_read=len(context.read_file_ids),
            search_count=len(context.search_history),
            loop_count=context.loop_count,
            max_loops=context.max_loops,
        )

    @staticmethod
    def _build_user_message(
        query: str,
        images: Optional[List[str]] = None,
    ) -> str:
        """Build the initial user message."""
        parts = [query]
        if images:
            parts.append(f"\n[Attached {len(images)} image(s) — multimodal analysis not yet supported]")
        return "\n".join(parts)

    @staticmethod
    def _build_continuation_prompt(context: SearchContext) -> str:
        """Build the loop continuation prompt with current state."""
        return REACT_CONTINUATION_PROMPT.format(
            budget_remaining=context.budget_remaining,
            loop_count=context.loop_count,
            max_loops=context.max_loops,
            files_read_count=len(context.read_file_ids),
        )
