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
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from sirchmunk.agentic.prompts import (
    REACT_CONTINUATION_PROMPT,
    REACT_SYSTEM_PROMPT,
)
from sirchmunk.agentic.tools import ToolRegistry
from sirchmunk.llm.openai_chat import LLMTokenBudgetExceeded, OpenAIChat
from sirchmunk.schema.search_context import SearchContext
from sirchmunk.utils import LogCallback, create_logger

logger = logging.getLogger(__name__)


# ---- Helpers ----

_ANSWER_PATTERN = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL)
_SUFFICIENCY_PATTERN = re.compile(
    r"<EVIDENCE_SUFFICIENCY>\s*(sufficient|partial|absent)\s*</EVIDENCE_SUFFICIENCY>",
    re.IGNORECASE,
)
# Optional machine-readable computation disclosure. Captured from the raw
# response (before answer sanitization strips JSON) so the pipeline can
# deterministically re-check the arithmetic behind a numeric answer.
_COMPUTATION_TRACE_PATTERN = re.compile(
    r"<COMPUTATION_TRACE>\s*(\{.*?\})\s*</COMPUTATION_TRACE>",
    re.DOTALL | re.IGNORECASE,
)

# Patterns indicating JSON/code garbage in extracted answers
_GARBAGE_PATTERNS = [
    re.compile(r'^\s*[{\[].*[}\]]\s*$', re.DOTALL),  # JSON object/array
    re.compile(r'^\s*```', re.MULTILINE),              # code fence
    re.compile(r'^\s*\{\s*"', re.DOTALL),             # JSON-like start
]


def _extract_answer(text: str) -> Optional[str]:
    """Extract content within <ANSWER>...</ANSWER> tags.

    Includes post-extraction garbage check: if extracted content looks like
    JSON/code, attempt to find natural language within it.
    """
    m = _ANSWER_PATTERN.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    # Post-extraction check: if content matches JSON/code patterns, clean it
    if raw and _is_garbage_content(raw):
        cleaned = _strip_garbage_from_answer(raw)
        if cleaned:
            return cleaned
    return raw


def _extract_sufficiency(text: str) -> Optional[str]:
    """Extract the agent's own rating of the evidence behind its answer.

    Returned verbatim from the tag so the caller's answer policy can decide how
    to present a weakly supported answer. Absent when the agent omitted the tag,
    which the caller must treat as unknown rather than as any particular rating.
    """
    m = _SUFFICIENCY_PATTERN.search(text or "")
    return m.group(1).strip().lower() if m else None


def _set_telemetry(context: SearchContext, **fields: Any) -> None:
    """Attach diagnostic fields to a search context."""
    telemetry = getattr(context, "telemetry", None)
    if not isinstance(telemetry, dict):
        telemetry = {}
        setattr(context, "telemetry", telemetry)
    telemetry.update(fields)


def _record_sufficiency(context: SearchContext, content: str) -> None:
    """Attach the evidence rating to ``context.telemetry`` when present.

    ``telemetry`` is an ad-hoc bag the pipeline attaches on demand rather than a
    declared field, so it is created here when missing; ``SearchContext.to_dict``
    exports whatever it holds.
    """
    sufficiency = _extract_sufficiency(content)
    if not sufficiency:
        return
    telemetry = getattr(context, "telemetry", None)
    if not isinstance(telemetry, dict):
        telemetry = {}
        setattr(context, "telemetry", telemetry)
    telemetry["evidence_sufficiency"] = sufficiency


def _record_computation_trace(context: SearchContext, content: str) -> None:
    """Stash a raw ``<COMPUTATION_TRACE>`` payload from the final response.

    The trace is captured here, before answer sanitization removes JSON blocks,
    and handed to the deterministic computation verifier via telemetry. It is
    advisory only — absence simply means no arithmetic disclosure was made.
    """
    match = _COMPUTATION_TRACE_PATTERN.search(content or "")
    if not match:
        return
    telemetry = getattr(context, "telemetry", None)
    if not isinstance(telemetry, dict):
        telemetry = {}
        setattr(context, "telemetry", telemetry)
    telemetry["computation_trace"] = match.group(1).strip()


def _is_garbage_content(text: str) -> bool:
    """Check if extracted answer content looks like JSON/code garbage."""
    stripped = text.strip()
    if not stripped:
        return True
    for pat in _GARBAGE_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def _strip_garbage_from_answer(text: str) -> Optional[str]:
    """Try to extract natural language from a garbage answer."""
    # Remove code fences
    cleaned = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL).strip()
    # Remove JSON blocks
    cleaned = re.sub(r'\{[^}]*\}', '', cleaned).strip()
    cleaned = re.sub(r'\[[^\]]*\]', '', cleaned).strip()
    # Remove markdown formatting residuals
    cleaned = re.sub(r'[\*`#]', '', cleaned).strip()
    if cleaned and len(cleaned) > 1:
        return cleaned
    return None


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
        answer_style_instruction: str = "",
    ) -> None:
        self.llm = llm
        self.registry = tool_registry
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        # Optional caller-supplied constraint appended to the system prompt
        # (e.g. a minimal-answer-span contract so evaluated systems share the
        # same output register). Empty by default: behavior is unchanged
        # unless the caller opts in.
        self.answer_style_instruction = (answer_style_instruction or "").strip()
        self._logger = create_logger(log_callback=log_callback, enable_async=True)

    # ---- Public API ----

    async def run(
        self,
        query: str,
        images: Optional[List[str]] = None,
        initial_keywords: Optional[List[str]] = None,
        preloaded_observations: Optional[str] = None,
        subgoals: Optional[List[str]] = None,
        trust_preloaded: bool = False,
    ) -> Tuple[str, SearchContext]:
        """Execute a full ReAct search session.

        Args:
            query: User's search query.
            images: Optional image URLs (reserved for future multimodal support).
            initial_keywords: Optional pre-extracted keywords to use for the
                first keyword_search call, bypassing the LLM's first turn.
            preloaded_observations: Optional evidence block (e.g. prior-warmed
                retrieval results) injected as the first observation so the
                answering agent starts from gathered evidence instead of an
                empty context. The agent still owns every subsequent action.
            subgoals: Optional explicit per-fact requirements the answer must
                satisfy, listed for the agent as a checklist. Complements the
                free-form reasoning with a structured target set.

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
                "content": self._build_user_message(query, images, subgoals),
            },
        ]

        await self._logger.info(f"[ReAct] Starting search: '{query[:80]}...'")
        await self._logger.info(f"[ReAct] Budget: {context.max_token_budget} tokens, max loops: {context.max_loops}")
        await self._logger.info(f"[ReAct] Tools: {self.registry.tool_names}")

        tool_names = self.registry.tool_names
        final_answer: Optional[str] = None

        # Prior-warmed start: inject gathered evidence as the first observation.
        # The agent reads it as if it had just retrieved it, then decides the
        # next action itself — this raises the starting evidence quality without
        # taking control away from the answering agent.
        if preloaded_observations and preloaded_observations.strip():
            preload_limit = min(
                24_000,
                max(4_000, int(self.max_token_budget * 0.375)),
            )
            if len(preloaded_observations) > preload_limit:
                preloaded_observations = preloaded_observations[:preload_limit]
                _set_telemetry(
                    context,
                    preloaded_evidence_truncated=True,
                    preloaded_evidence_chars=preload_limit,
                )
            context.increment_loop()
            messages.append({
                "role": "assistant",
                "content": (
                    "I'll begin from the evidence already gathered for this "
                    "question, then decide what else is needed."
                ),
            })
            if trust_preloaded:
                evidence_instruction = (
                    "This evidence was extracted from a high-confidence unique "
                    "source. Treat verbatim facts and complete table rows as "
                    "confirmed. Answer directly when all subgoals are present; "
                    "call a tool only for genuinely missing information."
                )
            else:
                evidence_instruction = (
                    "Treat this as leads, not conclusions. For each thing you "
                    "still need to establish that is not already stated "
                    "verbatim above, issue a keyword_search with the specific "
                    "entity name to confirm it before answering — do not answer "
                    "from the leads alone if a required fact is unconfirmed."
                )
            messages.append({
                "role": "user",
                "content": (
                    f"**Preloaded evidence** (starting evidence from prior "
                    f"retrieval):\n{preloaded_observations}\n\n"
                    f"{evidence_instruction}\n\n"
                    f"{self._build_continuation_prompt(context)}"
                ),
            })
            await self._logger.info(
                f"[ReAct] Prior-warmed with {len(preloaded_observations)} chars of evidence"
            )

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

            # Call LLM.  The task-local guard in OpenAIChat reserves prompt +
            # completion tokens before network I/O, preventing one oversized
            # call from crossing the session budget.
            try:
                llm_response = await self._call_llm(messages, context)
            except LLMTokenBudgetExceeded as exc:
                _set_telemetry(
                    context,
                    hard_budget_exhausted=True,
                    hard_budget_reason=str(exc),
                )
                await self._logger.warning(f"[ReAct] {exc}")
                break
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
                _record_sufficiency(context, content)
                _record_computation_trace(context, content)
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
                    "Wrap it in <ANSWER>...</ANSWER> tags.\n\n"
                    "CRITICAL: Your answer must be a concise natural language span "
                    "(a name, date, number, or yes/no phrase). "
                    "Do NOT output JSON, code blocks, tool calls, or markdown formatting. "
                    "If you cannot determine the answer, output your best guess as a simple phrase."
                ),
            })
            try:
                llm_response = await self._call_llm(messages, context)
            except LLMTokenBudgetExceeded as exc:
                _set_telemetry(
                    context,
                    hard_budget_exhausted=True,
                    hard_budget_reason=str(exc),
                )
                final_answer = "No results found."
                await self._logger.warning(f"[ReAct] synthesis skipped: {exc}")
                llm_response = None
            if llm_response is not None:
                content = llm_response.content or ""
                usage = llm_response.usage or {}
                total_tok = usage.get("total_tokens", 0)
                if total_tok == 0:
                    total_tok = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                context.add_llm_tokens(total_tok, usage=usage if usage else None)
                final_answer = _extract_answer(content) or content
                _record_sufficiency(context, content)

            # If the forced synthesis still produced garbage, strip and retry
            if final_answer and _is_garbage_content(final_answer):
                cleaned = _strip_garbage_from_answer(final_answer)
                if cleaned:
                    final_answer = cleaned

        await self._logger.success(f"[ReAct] Completed: {context.summary()}")

        return final_answer, context

    # ---- Internal helpers ----

    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        context: SearchContext,
    ):
        """Call the LLM within both local-loop and task-global budgets."""
        prompt_estimate = self.llm._estimate_message_tokens(messages)
        local_remaining = context.budget_remaining
        requested = min(
            int(os.getenv("LENS_MAX_COMPLETION_TOKENS", "4096")),
            max(0, local_remaining - prompt_estimate),
        )
        if requested < 128:
            raise LLMTokenBudgetExceeded(
                f"ReAct budget exhausted: remaining={local_remaining}, "
                f"estimated_prompt={prompt_estimate}"
            )
        return await self.llm.achat(
            messages=messages,
            stream=False,
            max_tokens=requested,
        )

    def _build_system_prompt(self, context: SearchContext, tool_descriptions: str) -> str:
        """Format the system prompt with tool descriptions and context state."""
        prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            budget_remaining=context.budget_remaining,
            files_read=len(context.read_file_ids),
            search_count=len(context.search_history),
            loop_count=context.loop_count,
            max_loops=context.max_loops,
        )
        if self.answer_style_instruction:
            prompt += f"\n## Answer Style\n{self.answer_style_instruction}\n"
        return prompt

    @staticmethod
    def _build_user_message(
        query: str,
        images: Optional[List[str]] = None,
        subgoals: Optional[List[str]] = None,
    ) -> str:
        """Build the initial user message."""
        parts = [query]
        if subgoals:
            checklist = "\n".join(f"- {g}" for g in subgoals if str(g).strip())
            if checklist:
                parts.append(
                    "\nTo answer this, you need to establish each of the "
                    f"following from the evidence:\n{checklist}"
                )
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
