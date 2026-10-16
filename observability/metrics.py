"""Per-turn and per-tool metrics for comparing agent variants side by side.

Works for any agent built by agent/agent.py (variant "tools") or
agent/sql_agent.py (variant "sql") without touching their tools: it rides on
the Agents SDK lifecycle hooks, which see every model call and tool call.

Each user message produces one "turn" record, and each tool call one
"tool_call" record, appended as JSON lines to logs/agent_metrics.jsonl
(override with CARTWHEEL_METRICS_LOG). Compare variants with:

    uv run python -m scripts.compare_agents

A turn record holds the message and reply, latency (total, model, tools),
every model call's tokens and latency, tool calls and failures, the state
changes the agent made ("actions"), cost, and what produced it: variant,
model, provider route, prompt version, code version, and settings.

This is local, offline logging, independent of the Langfuse tracing in
observability/instrument.py; both can run at once.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from agents import Agent, Runner
from agents.items import ModelResponse
from agents.lifecycle import RunHooks
from agents.result import RunResult
from agents.run_context import RunContextWrapper
from agents.tool import Tool

from agent.auth import AuthContext
from agent.config import REPO_ROOT

# USD per million tokens: (input, cached input, output). Published list prices
# when this was written; check the provider's pricing page before relying on
# a cost figure. Models missing here are logged with cost_usd = null.
# The small and frontier rows were checked against OpenRouter's model list on
# 2026-10-14; OpenRouter passes through the provider's list price.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "claude-haiku-4-5": (1.00, 0.10, 5.00),
    "claude-haiku-5-5": (0.10, 0.01, 0.50),
    "claude-opus-5-5": (4.00, 0.20, 20.00),
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gemini-4-argon": (2.00, 0.10, 10.00),
}

MAX_LOGGED_TEXT = 20_000
MAX_LOGGED_TOOL_RESULT = 4_000
# Tools that change state; their calls are also listed as the turn's "actions".
WRITE_TOOLS = {"issue_refund", "cancel_order", "escalate_to_human", "write_sql"}


def log_path() -> Path:
    return Path(os.environ.get("CARTWHEEL_METRICS_LOG", REPO_ROOT / "logs" / "agent_metrics.jsonl"))


def _price_key(model: str) -> str:
    """Map a course name or provider id to its price row.

    "openrouter/anthropic/claude-haiku-4.5" and "anthropic/claude-haiku-4-5"
    both price as "claude-haiku-4-5".
    """
    return model.removeprefix("openrouter/").split("/")[-1].replace(".", "-")


def cost_usd(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float | None:
    """Price one or more model calls. Cached input tokens are billed at the cached rate."""
    prices = PRICES_USD_PER_MTOK.get(model) or PRICES_USD_PER_MTOK.get(_price_key(model))
    if prices is None:
        return None
    price_in, price_cached, price_out = prices
    uncached = max(input_tokens - cached_tokens, 0)
    return (uncached * price_in + cached_tokens * price_cached + output_tokens * price_out) / 1_000_000


@lru_cache(maxsize=1)
def code_version() -> str | None:
    """Short git commit of the running code, with "+dirty" for uncommitted edits."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}+dirty" if dirty else sha


def agent_route(agent: Agent[Any]) -> str:
    """Where the agent's model calls go, e.g. "openai/gpt-4o-mini"."""
    model = agent.model
    if isinstance(model, str):
        return f"openai/{model}"
    return getattr(model, "model", None) or type(model).__name__


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _truncate(text: Any, limit: int = MAX_LOGGED_TEXT) -> Any:
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + "…"
    return text


def _as_dict(result: Any) -> dict[str, Any] | None:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _loggable_result(result: Any) -> Any:
    """The tool result as JSON, or a truncated string when it is large."""
    text = json.dumps(result, default=str)
    return result if len(text) <= MAX_LOGGED_TOOL_RESULT else _truncate(text, MAX_LOGGED_TOOL_RESULT)


def _write(record: dict[str, Any]) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


@dataclass
class TurnMetrics:
    """What one user message cost, filled in by MetricsHooks during the run."""

    run_id: str
    variant: str
    model: str
    route: str
    ctx: AuthContext
    session_id: str | None
    turn_index: int | None
    prompt_version: str | None
    settings: dict[str, Any]
    tags: dict[str, Any] = field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)

    def base(self) -> dict[str, Any]:
        return {
            "ts": _now(),
            "run_id": self.run_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "variant": self.variant,
            "model": self.model,
            "role": self.ctx.role,
            "user_id": self.ctx.user_id,
            "store_id": self.ctx.store_id,
            **self.tags,
        }

    def summary(self, *, latency_ms: float, user_input: str, output: Any, error: BaseException | None) -> dict[str, Any]:
        input_tokens = sum(c["input_tokens"] for c in self.llm_calls)
        cached_tokens = sum(c["cached_tokens"] for c in self.llm_calls)
        output_tokens = sum(c["output_tokens"] for c in self.llm_calls)
        failures = [t for t in self.tool_calls if not t["ok"]]
        if error is None and not (output.strip() if isinstance(output, str) else output):
            # The run finished but said nothing, e.g. a provider content filter
            # stopped the reply. Not a success, though no exception was raised.
            status = "empty"
        elif error is None:
            status = "ok"
        elif isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
            status = "cancelled"
        else:
            status = "error"
        return {
            "type": "turn",
            **self.base(),
            "route": self.route,
            "prompt_version": self.prompt_version,
            "code_version": code_version(),
            "settings": self.settings,
            "status": status,
            "error": f"{type(error).__name__}: {error}" if error else None,
            "input": _truncate(user_input),
            "output": _truncate(output if isinstance(output, str) or output is None else str(output)),
            "latency_ms": round(latency_ms, 1),
            "llm_calls": len(self.llm_calls),
            "llm_latency_ms": round(sum(c["latency_ms"] for c in self.llm_calls), 1),
            "llm_call_details": self.llm_calls,
            "tool_calls": len(self.tool_calls),
            "tool_failures": len(failures),
            "tool_errors": [t["error"] for t in failures],
            "tool_latency_ms": round(sum(t["latency_ms"] for t in self.tool_calls), 1),
            "actions": self.actions,
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": sum(c["reasoning_tokens"] for c in self.llm_calls),
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost_usd(self.model, input_tokens, cached_tokens, output_tokens),
        }


class MetricsHooks(RunHooks[AuthContext]):
    """Times every model call and tool call and logs each tool call as it ends."""

    def __init__(self, metrics: TurnMetrics) -> None:
        self.metrics = metrics
        self._llm_started: float | None = None
        self._tool_started: dict[str, tuple[float, Any]] = {}

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:  # noqa: ANN001
        self._llm_started = time.perf_counter()

    async def on_llm_end(self, context: RunContextWrapper[AuthContext], agent: Agent, response: ModelResponse) -> None:
        started = self._llm_started or time.perf_counter()
        usage = response.usage
        input_tokens = usage.input_tokens
        cached = getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
        output_tokens = usage.output_tokens
        self.metrics.llm_calls.append(
            {
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "input_tokens": input_tokens,
                "cached_tokens": cached,
                "output_tokens": output_tokens,
                "reasoning_tokens": getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0,
                "cost_usd": cost_usd(self.metrics.model, input_tokens, cached, output_tokens),
            }
        )

    async def on_tool_start(self, context: RunContextWrapper[AuthContext], agent: Agent, tool: Tool) -> None:
        call_id = getattr(context, "tool_call_id", None) or tool.name
        self._tool_started[call_id] = (time.perf_counter(), getattr(context, "tool_arguments", None))

    async def on_tool_end(self, context: RunContextWrapper[AuthContext], agent: Agent, tool: Tool, result: Any) -> None:
        call_id = getattr(context, "tool_call_id", None) or tool.name
        started, raw_args = self._tool_started.pop(call_id, (time.perf_counter(), None))
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            arguments = raw_args
        payload = _as_dict(result)
        ok = bool(payload.get("ok", True)) if payload is not None else True
        error = None if ok else (payload or {}).get("error", "unknown")
        record = {
            "type": "tool_call",
            **self.metrics.base(),
            "tool": tool.name,
            "call_id": call_id,
            "arguments": arguments,
            "ok": ok,
            "error": error,
            "reason": None if ok else _truncate((payload or {}).get("reason")),
            "result": _loggable_result(payload if payload is not None else result),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        self.metrics.tool_calls.append(record)
        if tool.name in WRITE_TOOLS:
            self.metrics.actions.append(
                {
                    "action": (payload or {}).get("operation", tool.name),
                    "ok": ok,
                    "outcome": (payload or {}).get("status") if ok else error,
                }
            )
        _write(record)


async def run_with_metrics(
    agent: Agent[AuthContext],
    user_input: str,
    *,
    context: AuthContext,
    variant: str,
    model: str,
    session_id: str | None = None,
    turn_index: int | None = None,
    prompt_version: str | None = None,
    defenses: bool = False,
    tags: dict[str, Any] | None = None,
    **runner_kwargs: Any,
) -> tuple[RunResult, dict[str, Any]]:
    """Runner.run with metrics. Returns the result and the logged turn record.

    `tags` are extra fields copied onto every record of this turn, e.g. the
    replay id and the source conversation when replaying recorded chats.

    A run that raises, or is interrupted with Ctrl-C, still logs its turn
    (status "error" or "cancelled") before re-raising, so failed turns count
    in the comparison.
    """
    metrics = TurnMetrics(
        run_id=uuid.uuid4().hex[:12],
        variant=variant,
        model=model,
        route=agent_route(agent),
        ctx=context,
        session_id=session_id,
        turn_index=turn_index,
        prompt_version=prompt_version,
        settings={
            "defenses": defenses,
            "kill_switch": os.environ.get("CARTWHEEL_KILL_SWITCH") or "off",
            "max_turns": runner_kwargs.get("max_turns"),
        },
        tags=tags or {},
    )
    started = time.perf_counter()
    try:
        result = await Runner.run(agent, user_input, context=context, hooks=MetricsHooks(metrics), **runner_kwargs)
    except BaseException as exc:
        turn = metrics.summary(
            latency_ms=(time.perf_counter() - started) * 1000, user_input=user_input, output=None, error=exc
        )
        _write(turn)
        raise
    turn = metrics.summary(
        latency_ms=(time.perf_counter() - started) * 1000,
        user_input=user_input,
        output=result.final_output,
        error=None,
    )
    _write(turn)
    return result, turn


def format_turn(turn: dict[str, Any]) -> str:
    """One-line footer for the CLI: latency, tool calls, tokens, cost."""
    failed = f", {turn['tool_failures']} failed" if turn["tool_failures"] else ""
    cost = f"${turn['cost_usd']:.4f}" if turn["cost_usd"] is not None else "cost n/a"
    return (
        f"[{turn['variant']} · {turn['model']} · {turn['latency_ms'] / 1000:.1f}s · "
        f"{turn['tool_calls']} tool calls{failed} · {turn['total_tokens']:,} tokens · {cost}]"
    )
