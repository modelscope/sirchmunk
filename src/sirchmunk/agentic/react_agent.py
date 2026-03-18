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
from typing import Any, Dict, List, Optional, Tuple

from sirchmunk.agentic.belief_state import BeliefState
from sirchmunk.agentic.prompts import (
    QUERY_DECOMPOSITION_PROMPT,
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


_MAX_RECENT_ROUNDS = 4
_PRUNED_ROUND_SUMMARY_CHARS = 300


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
        max_loops: int = 5,
        max_token_budget: int = 64000,
        log_callback: LogCallback = None,
        enable_thinking: bool = False,
        enable_decomposition: bool = True,
        enable_belief_tracking: bool = True,
        memory_prior: Any = None,
    ) -> None:
        self.llm = llm
        self.registry = tool_registry
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.enable_thinking = enable_thinking
        self.enable_decomposition = enable_decomposition
        self.enable_belief_tracking = enable_belief_tracking
        self._memory_prior = memory_prior
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
        context.query = query

        if self.enable_belief_tracking:
            belief = BeliefState()
            if self._memory_prior is not None:
                belief.warm_start(self._memory_prior)
            context.belief_state = belief

        # Build tool descriptions for the system prompt
        tool_descriptions = _build_tool_descriptions(self.registry)

        # Adaptive question decomposition: assess complexity and generate
        # a search plan for multi-hop queries.
        decomposition_hint = ""
        if self.enable_decomposition:
            decomposition_hint = await self._maybe_decompose_query(query, context)

        # Build the initial conversation
        user_message = self._build_user_message(query, images)
        if decomposition_hint:
            user_message += f"\n\n{decomposition_hint}"

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self._build_system_prompt(context, tool_descriptions),
            },
            {
                "role": "user",
                "content": user_message,
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
            self._update_beliefs(context, "keyword_search", meta)
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

        budget_exceeded = False
        _NO_PROGRESS_LIMIT = 5
        _consecutive_no_progress = 0
        _prev_known_files = context.total_known_files

        while not context.is_loop_limit_reached() and not context.is_budget_exceeded():
            context.increment_loop()
            await self._logger.info(f"[ReAct] Loop {context.loop_count}/{context.max_loops} | {context.summary()}")

            # Prune older rounds to keep prompt size bounded
            messages = self._prune_messages(messages)

            # Call LLM
            llm_response = await self._call_llm(messages)
            content = llm_response.content or ""

            # Track LLM token usage
            usage = llm_response.usage or {}
            total_tok = usage.get("total_tokens", 0)
            if total_tok == 0:
                total_tok = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            context.add_llm_tokens(total_tok, usage=usage if usage else None)

            # Budget overflow detected after this LLM call
            if context.is_budget_exceeded():
                budget_exceeded = True
                context.add_reasoning(content)
                answer = _extract_answer(content)
                if answer:
                    final_answer = answer
                    await self._logger.success(f"[ReAct] Answer found at budget boundary (loop {context.loop_count})")
                break

            # Check for final answer in response
            answer = _extract_answer(content)
            if answer:
                context.add_reasoning(content)
                final_answer = answer
                await self._logger.success(f"[ReAct] Answer found at loop {context.loop_count}")
                break

            # Try to extract a tool call
            tool_call = _parse_tool_call(content, tool_names)
            if tool_call is None:
                # LLM didn't call a tool and didn't answer — nudge it
                await self._logger.warning("[ReAct] No tool call or answer detected, nudging...")
                context.add_reasoning(content)
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

            # Pre-emptive budget guard: if remaining budget is very low,
            # skip the tool call and force synthesis from existing evidence.
            _BUDGET_GUARD_THRESHOLD = 2000
            if context.budget_remaining < _BUDGET_GUARD_THRESHOLD:
                await self._logger.warning(
                    f"[ReAct] Budget nearly exhausted ({context.budget_remaining} tokens) — "
                    f"skipping tool call, forcing synthesis"
                )
                budget_exceeded = True
                messages.append({"role": "assistant", "content": content})
                break

            # Execute the tool
            result_text, meta = await self.registry.execute(
                tool_name=tool_name,
                context=context,
                **tool_args,
            )

            # Update BA-ReAct belief state from tool observations
            self._update_beliefs(context, tool_name, meta)

            # Truncate if the tool returned too much text
            if len(result_text) > 8000:
                result_text = result_text[:8000] + "\n... [output truncated]"

            await self._logger.info(
                f"[ReAct] Tool result: {len(result_text)} chars | "
                f"Budget remaining: {context.budget_remaining}"
            )

            # Record LLM reasoning for downstream SP matching
            context.add_reasoning(content)

            # Track progress: count read + discovered files; weight by tool type
            cur_known_files = context.total_known_files
            if cur_known_files > _prev_known_files:
                _consecutive_no_progress = 0
                _prev_known_files = cur_known_files
            elif tool_name == "file_read":
                _consecutive_no_progress += 1
            else:
                _consecutive_no_progress += 0.5

            if _consecutive_no_progress >= _NO_PROGRESS_LIMIT:
                await self._logger.warning(
                    f"[ReAct] No progress in {_consecutive_no_progress:.0f} tool calls "
                    f"(limit {_NO_PROGRESS_LIMIT}) — early stop"
                )
                messages.append({"role": "assistant", "content": content})
                break

            # BA-ReAct: ESS-based early stopping when evidence is concentrated
            belief = getattr(context, "belief_state", None)
            if (
                belief
                and belief.should_stop_early()
                and context.loop_count >= 3
            ):
                await self._logger.info(
                    f"[ReAct] Belief ESS suggests evidence convergence "
                    f"(ESS={belief.compute_ess():.1f}) — nudging answer"
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

        if not budget_exceeded and context.is_budget_exceeded():
            budget_exceeded = True

        # If loop exited without answer, ask LLM to synthesize
        if final_answer is None:
            if budget_exceeded:
                await self._logger.warning("[ReAct] Token budget exceeded — forcing synthesis from partial evidence")
                synthesis_prompt = (
                    "⚠️ Token budget exhausted. You MUST answer NOW from the evidence already collected. "
                    "Do NOT request more tools. Synthesize your best answer from ALL tool results above, "
                    "even if incomplete. If uncertain, give your best guess. "
                    "Wrap it in <ANSWER>...</ANSWER> tags."
                )
            else:
                await self._logger.warning("[ReAct] Loop limit reached — forcing synthesis")
                synthesis_prompt = (
                    "You have reached the retrieval limit. "
                    "Please synthesize your best answer from ALL evidence collected so far. "
                    "Wrap it in <ANSWER>...</ANSWER> tags."
                )
            messages.append({"role": "user", "content": synthesis_prompt})
            llm_response = await self._call_llm(messages)
            content = llm_response.content or ""
            usage = llm_response.usage or {}
            total_tok = usage.get("total_tokens", 0)
            if total_tok == 0:
                total_tok = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            context.add_llm_tokens(total_tok, usage=usage if usage else None)
            context.add_reasoning(content)
            final_answer = _extract_answer(content) or content

        await self._logger.success(f"[ReAct] Completed: {context.summary()}")

        return final_answer, context

    # ---- Internal helpers ----

    async def _maybe_decompose_query(
        self,
        query: str,
        context: SearchContext,
    ) -> str:
        """Assess query complexity and optionally decompose into sub-questions.

        Uses a lightweight LLM call to determine whether the query needs
        multi-hop reasoning.  If so, returns a formatted hint string with
        sub-questions to prepend to the user message.  If the query is
        simple or the call fails, returns an empty string.

        The token cost of this call is charged to the search context.
        """
        prompt = QUERY_DECOMPOSITION_PROMPT.format(query=query)
        try:
            resp = await self.llm.achat(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            usage = resp.usage or {}
            total_tok = usage.get("total_tokens", 0)
            if total_tok == 0:
                total_tok = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            context.add_llm_tokens(total_tok, usage=usage if usage else None)

            raw = (resp.content or "").strip()

            # Extract JSON from potential markdown wrapping
            brace_start = raw.find("{")
            brace_end = raw.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                raw = raw[brace_start:brace_end + 1]

            parsed = json.loads(raw)
            needs = parsed.get("needs_decomposition", False)
            ans_fmt = parsed.get("answer_format", "entity")

            if not needs:
                await self._logger.info("[ReAct] Query decomposition: simple query, no decomposition needed")
                if ans_fmt == "yes_no":
                    return "[Answer format: yes/no — respond with ONLY 'yes' or 'no'.]"
                return ""

            sub_qs = parsed.get("sub_questions", [])
            r_type = parsed.get("reasoning_type", "bridge")
            constraints = parsed.get("search_constraints", [])
            if not sub_qs:
                return ""

            await self._logger.info(
                f"[ReAct] Query decomposition: {r_type} type, "
                f"{len(sub_qs)} sub-question(s), "
                f"{len(constraints)} constraint(s), "
                f"answer_format={ans_fmt}"
            )

            lines = [f"[Search plan — {r_type} reasoning]"]
            for i, sq in enumerate(sub_qs, 1):
                lines.append(f"  Step {i}: {sq}")
            lines.append("Follow this plan in order. Each step may depend on the previous one's result.")

            if constraints:
                lines.append("")
                lines.append("[Search constraints — do NOT answer until ALL are satisfied]")
                for c in constraints:
                    lines.append(f"  - {c}")

            _FMT_LABELS = {
                "yes_no": "yes/no — respond with ONLY 'yes' or 'no'",
                "entity": "an entity name (person, place, title, etc.)",
                "number": "a number or quantity",
                "date": "a date or time period",
                "description": "a short descriptive phrase",
            }
            fmt_label = _FMT_LABELS.get(ans_fmt, "a short phrase")
            lines.append(f"\n[Answer format: {fmt_label}]")

            return "\n".join(lines)

        except Exception as exc:
            await self._logger.warning(f"[ReAct] Query decomposition failed: {exc}")
            return ""

    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
    ):
        """Call the LLM with the given messages.

        Uses stream=False for tool-calling loops to get complete responses.
        Passes through ``enable_thinking`` for deep reasoning support.
        """
        import time as _time
        _t0 = _time.time()
        result = await self.llm.achat(
            messages=messages, stream=False,
            enable_thinking=self.enable_thinking,
        )
        _elapsed = _time.time() - _t0
        await self._logger.info(f"[Timing] ReAct LLM call: {_elapsed:.2f}s")
        return result

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
        """Build the loop continuation prompt with current state.

        Appends BA-ReAct advisory signals (e.g., promising unread files,
        evidence concentration, chain hints) when a belief state is
        available.
        """
        base = REACT_CONTINUATION_PROMPT.format(
            budget_remaining=context.budget_remaining,
            loop_count=context.loop_count,
            max_loops=context.max_loops,
            files_read_count=len(context.read_file_ids),
        )
        belief = getattr(context, "belief_state", None)
        if belief:
            advisory = belief.get_advisory()
            if advisory:
                base += f"\n{advisory}"
            # Inject reasoning chain hint in early rounds
            chain = belief.chain_hint
            if chain and context.loop_count <= 2:
                steps = " → ".join(
                    s.get("action", "?") for s in chain[:4]
                )
                base += f"\nSuggested reasoning pattern: {steps}"
        return base

    @staticmethod
    def _update_beliefs(
        context: SearchContext,
        tool_name: str,
        meta: Dict[str, Any],
    ) -> None:
        """Update BA-ReAct belief state from tool execution metadata."""
        belief = getattr(context, "belief_state", None)
        if not belief:
            return
        if tool_name == "keyword_search":
            belief.update_from_search(meta.get("files_discovered", []))
        elif tool_name == "title_lookup":
            paths = meta.get("paths_found", 0)
            title = meta.get("title", "")
            if paths and title:
                belief.update_from_search([title])

    @staticmethod
    def _prune_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sliding-window pruning to keep prompt size bounded.

        Preserves the system message (index 0) and user query (index 1)
        as immutable context.  The remaining messages form assistant/user
        round pairs.  When the number of rounds exceeds
        ``_MAX_RECENT_ROUNDS``, older rounds are collapsed into a compact
        summary message to reduce prompt tokens while retaining key facts.
        """
        if len(messages) <= 2:
            return messages

        head = messages[:2]
        body = messages[2:]

        # Each "round" is an (assistant, user) pair = 2 messages
        n_msgs = len(body)
        keep_msgs = _MAX_RECENT_ROUNDS * 2
        if n_msgs <= keep_msgs:
            return messages

        old_msgs = body[: n_msgs - keep_msgs]
        recent_msgs = body[n_msgs - keep_msgs:]

        # Build compact summary of pruned rounds
        summary_parts: List[str] = ["[Earlier rounds condensed]"]
        for msg in old_msgs:
            role = msg.get("role", "")
            text = msg.get("content", "")
            if role == "assistant":
                snippet = text[:_PRUNED_ROUND_SUMMARY_CHARS].replace("\n", " ")
                if len(text) > _PRUNED_ROUND_SUMMARY_CHARS:
                    snippet += "..."
                summary_parts.append(f"- Assistant: {snippet}")
            elif role == "user" and "**Tool result**" in text:
                tool_match = re.search(r"\*\*Tool result\*\*\s*\((\w+)\)", text)
                tool_name = tool_match.group(1) if tool_match else "tool"
                result_snippet = text[:150].replace("\n", " ")
                summary_parts.append(f"- {tool_name} result: {result_snippet}...")

        summary = "\n".join(summary_parts)
        summary_msg = {"role": "user", "content": summary}

        return head + [summary_msg] + recent_msgs
