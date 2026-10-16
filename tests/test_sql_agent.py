"""Tests for the SQL variant (read_sql / write_sql) and the side-by-side metrics.

Offline: the agent loop runs against a scripted FakeModel, no API keys.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest
from agents import Agent
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from agent import cli
from agent.auth import AuthContext
from agent.sql_agent import SQL_TOOLS, build_sql_agent, render_sql_system_prompt
from agent.sql_tools import READ_ROW_LIMIT, read_sql, write_sql
from observability.metrics import cost_usd, run_with_metrics
from scripts import compare_agents
from tests.eval.fake_model import FakeModel, text_message, tool_call

SHOPPER_1 = AuthContext(user_id=1, role="shopper")
SHOPPER_2 = AuthContext(user_id=2, role="shopper")
MERCHANT_STORE_1 = AuthContext(user_id=9001, role="merchant", store_id=1)
MERCHANT_STORE_2 = AuthContext(user_id=9002, role="merchant", store_id=2)
SUPPORT = AuthContext(user_id=9501, role="support")


def rows(ctx: AuthContext, query: str) -> list[list]:
    result = read_sql(ctx, query)
    assert result["ok"] is True, result
    return result["rows"]


# ---------------------------------------------------------------------------
# read_sql: scope
# ---------------------------------------------------------------------------


def test_read_sql_shows_only_the_callers_orders(world: dict) -> None:
    conn = sqlite3.connect(world["db"])
    try:
        own = conn.execute("SELECT count(*) FROM orders WHERE user_id = 1").fetchone()[0]
        store_2 = conn.execute("SELECT count(*) FROM orders WHERE store_id = 2").fetchone()[0]
    finally:
        conn.close()
    assert rows(SHOPPER_1, "SELECT count(*) FROM orders") == [[own]]
    assert rows(SHOPPER_1, "SELECT count(*) FROM orders WHERE user_id <> 1") == [[0]]
    assert rows(MERCHANT_STORE_2, "SELECT count(*) FROM orders") == [[store_2]]
    assert rows(SUPPORT, "SELECT count(*) FROM orders") == [[10_000]]


def test_read_sql_hides_other_stores_orders(world: dict) -> None:
    assert rows(MERCHANT_STORE_2, "SELECT * FROM orders WHERE id = 4127") == []
    assert rows(SHOPPER_2, "SELECT * FROM orders WHERE id = 4127") == []
    assert rows(MERCHANT_STORE_1, "SELECT id FROM orders WHERE id = 4127") == [[4127]]
    assert rows(SUPPORT, "SELECT id FROM orders WHERE id = 4127") == [[4127]]


def test_read_sql_scopes_users_refunds_and_escalations(world: dict) -> None:
    assert rows(SHOPPER_1, "SELECT id FROM users") == [[1]]
    assert rows(MERCHANT_STORE_2, "SELECT id FROM users") == [[9002]]
    assert rows(SUPPORT, "SELECT count(*) FROM users") == [[525]]
    assert rows(SHOPPER_1, "SELECT count(*) FROM refunds WHERE order_id NOT IN (SELECT id FROM orders)") == [[0]]
    assert rows(SHOPPER_1, "SELECT count(*) FROM escalations WHERE user_id <> 1") == [[0]]
    assert rows(MERCHANT_STORE_2, "SELECT count(*) FROM escalations WHERE store_id <> 2 AND user_id <> 9002") == [[0]]


def test_read_sql_has_public_and_helper_tables(world: dict) -> None:
    assert rows(SHOPPER_1, "SELECT today FROM world") == [["2026-07-01"]]
    assert rows(SHOPPER_1, "SELECT value FROM facts WHERE key = 'refund_auto_approve_threshold_usd'") == [["100"]]
    assert rows(SHOPPER_1, "SELECT count(*) FROM stores") == [[20]]
    assert rows(SHOPPER_1, "SELECT count(*) FROM products") == [[800]]
    body = rows(SHOPPER_1, "SELECT body FROM policies WHERE policy_id = 'cw-returns'")[0][0]
    assert "30" in body
    assert rows(SUPPORT, "SELECT name FROM sqlite_master WHERE name = 'data_quality_cases'") == []


# ---------------------------------------------------------------------------
# read_sql: guardrails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,error",
    [
        ("UPDATE orders SET status = 'cancelled' WHERE id = 4127", "read_only"),
        ("DELETE FROM orders", "read_only"),
        ("PRAGMA table_info(orders)", "read_only"),
        ("ATTACH DATABASE 'x.db' AS x", "read_only"),
        ("WITH x AS (SELECT 1) DELETE FROM orders", "read_only"),
        ("SELECT 1; DELETE FROM orders", "invalid_argument"),
        ("SELECT date('now')", "invalid_argument"),
        ("SELECT * FROM orders WHERE ordered_at > CURRENT_DATE", "invalid_argument"),
        ("", "invalid_argument"),
        ("SELECT nope FROM orders", "sql_error"),
        ("SELECT * FROM src.orders", "sql_error"),
    ],
)
def test_read_sql_rejects(world: dict, query: str, error: str) -> None:
    result = read_sql(SUPPORT, query)
    assert result["ok"] is False
    assert result["error"] == error


def test_read_sql_errors_carry_a_schema_hint(world: dict) -> None:
    result = read_sql(SHOPPER_1, "SELECT nope FROM orders")
    assert "orders(id, user_id" in result["hint"]


def test_read_sql_caps_rows(world: dict) -> None:
    result = read_sql(SUPPORT, "SELECT id FROM orders")
    assert result["row_count"] == READ_ROW_LIMIT
    assert result["truncated"] is True
    assert "note" in result


def test_read_sql_never_changes_the_real_database(world_copy) -> None:
    read_sql(SUPPORT, "WITH x AS (SELECT 1) DELETE FROM orders")
    conn = sqlite3.connect(world_copy)
    try:
        assert conn.execute("SELECT count(*) FROM orders").fetchone()[0] == 10_000
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# write_sql: the three accepted statements and their rules
# ---------------------------------------------------------------------------


def _placed_order(db_path) -> tuple[int, int, int]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT id, user_id, store_id FROM orders WHERE status = 'placed' ORDER BY id LIMIT 1"
        ).fetchone()
    finally:
        conn.close()


def test_write_sql_cancels_a_placed_order(world_copy) -> None:
    order_id, owner_id, _ = _placed_order(world_copy)
    owner = AuthContext(user_id=owner_id, role="shopper")
    stranger = AuthContext(user_id=owner_id + 1, role="shopper")
    statement = f"UPDATE orders SET status = 'cancelled' WHERE id = {order_id};"

    denied = write_sql(stranger, statement)
    assert denied["error"] == "permission_denied"
    assert denied["operation"] == "cancel_order"

    result = write_sql(owner, statement)
    assert result == {"operation": "cancel_order", "ok": True, "order_id": order_id, "status": "cancelled"}
    assert rows(owner, f"SELECT status FROM orders WHERE id = {order_id}") == [["cancelled"]]


def test_write_sql_refuses_to_cancel_a_delivered_order(world_copy) -> None:
    result = write_sql(SHOPPER_1, "update orders set status='cancelled' where id=4127")
    assert result["error"] == "not_eligible"


def test_write_sql_queues_a_refund_above_the_threshold(world_copy) -> None:
    result = write_sql(
        SHOPPER_1,
        "INSERT INTO refunds (order_id, amount_cents, reason) VALUES (4455, 24000, 'it''s cracked')",
    )
    assert result["ok"] is True
    assert result["operation"] == "issue_refund"
    assert result["status"] == "queued_for_approval"
    assert result["amount_cents"] == 24000
    assert rows(SHOPPER_1, "SELECT status FROM orders WHERE id = 4455") == [["delivered"]]
    assert rows(SHOPPER_1, "SELECT amount_cents, status, reason FROM refunds WHERE order_id = 4455") == [
        [24000, "queued_for_approval", "it's cracked"]
    ]


def test_write_sql_auto_approves_a_small_refund(world_copy) -> None:
    # Column order does not matter.
    result = write_sql(
        SHOPPER_1, "INSERT INTO refunds (reason, order_id, amount_cents) VALUES ('changed my mind', 4127, 8400)"
    )
    assert result["status"] == "auto_approved"
    assert rows(SHOPPER_1, "SELECT status FROM orders WHERE id = 4127") == [["refunded"]]


@pytest.mark.parametrize(
    "ctx,statement,error",
    [
        (SHOPPER_1, "INSERT INTO refunds (order_id, amount_cents, reason) VALUES (3980, 5200, 'late')", "not_eligible"),
        (MERCHANT_STORE_2, "INSERT INTO refunds (order_id, amount_cents, reason) VALUES (4455, 1000, 'x')", "permission_denied"),
        (SHOPPER_1, "INSERT INTO refunds (order_id, amount_cents, reason) VALUES (4127, 84.00, 'x')", "invalid_argument"),
        (SHOPPER_1, "INSERT INTO refunds (order_id, amount_cents, reason) VALUES (4127, 99999, 'x')", "invalid_argument"),
        (SHOPPER_1, "INSERT INTO refunds (order_id, amount_cents, reason, status) VALUES (4127, 100, 'x', 'auto_approved')", "unsupported_statement"),
        (SHOPPER_1, "INSERT INTO refunds (order_id, amount_cents, reason) VALUES (999999, 100, 'x')", "not_found"),
        (SHOPPER_1, "UPDATE orders SET status = 'refunded' WHERE id = 4127", "unsupported_statement"),
        (SHOPPER_1, "UPDATE orders SET status = 'cancelled' WHERE user_id = 1", "unsupported_statement"),
        (SHOPPER_1, "UPDATE orders SET status = 'cancelled' WHERE id = 4127 OR 1=1", "unsupported_statement"),
        (SHOPPER_1, "DELETE FROM orders WHERE id = 4127", "unsupported_statement"),
        (SHOPPER_1, "INSERT INTO users (id, name, role) VALUES (5, 'x', 'support')", "unsupported_statement"),
        (SHOPPER_1, "SELECT * FROM orders", "unsupported_statement"),
        (SHOPPER_1, "INSERT INTO refunds (order_id, amount_cents, reason) VALUES (4127, 100)", "invalid_argument"),
    ],
)
def test_write_sql_rejects(world_copy, ctx, statement, error) -> None:
    result = write_sql(ctx, statement)
    assert result["ok"] is False
    assert result["error"] == error


def test_write_sql_unsupported_lists_accepted_forms(world_copy) -> None:
    result = write_sql(SHOPPER_1, "DELETE FROM orders")
    assert len(result["accepted_forms"]) == 3


def test_write_sql_escalates_with_identity_from_the_session(world_copy) -> None:
    result = write_sql(
        MERCHANT_STORE_1,
        "INSERT INTO escalations (summary, context, order_id) VALUES ('refund above limit', 'wants $240 back', 4455)",
    )
    assert result["ok"] is True and result["operation"] == "escalate"
    assert result["sla_hours"] == 24
    conn = sqlite3.connect(world_copy)
    try:
        user_id, store_id, order_id, context = conn.execute(
            "SELECT user_id, store_id, order_id, context FROM escalations WHERE id = ?", (result["ticket_id"],)
        ).fetchone()
    finally:
        conn.close()
    assert (user_id, store_id, order_id) == (9001, 1, 4455)
    assert json.loads(context) == {"role": "merchant", "context": "wants $240 back"}

    no_order = write_sql(SHOPPER_1, "INSERT INTO escalations (summary, context) VALUES ('question', 'about returns')")
    assert no_order["ok"] is True


def test_write_sql_escalation_checks_order_scope(world_copy) -> None:
    result = write_sql(
        MERCHANT_STORE_2, "INSERT INTO escalations (summary, context, order_id) VALUES ('x', 'y', 4127)"
    )
    assert result["error"] == "permission_denied"


def test_write_sql_respects_the_kill_switch(world_copy, monkeypatch) -> None:
    monkeypatch.setenv("CARTWHEEL_KILL_SWITCH", "readonly")
    result = write_sql(SHOPPER_1, "INSERT INTO refunds (order_id, amount_cents, reason) VALUES (4127, 8400, 'x')")
    assert result["error"] == "paused"


# ---------------------------------------------------------------------------
# The SQL agent
# ---------------------------------------------------------------------------


def test_sql_agent_has_only_the_two_sql_tools(monkeypatch) -> None:
    from agent import sql_agent

    monkeypatch.setattr(sql_agent, "resolve_model", lambda _: FakeModel())
    agent = build_sql_agent(SHOPPER_1)
    assert [t.name for t in agent.tools] == ["read_sql", "write_sql"]
    assert "read_sql" in agent.instructions and "User id: 1" in agent.instructions
    assert agent.instructions == render_sql_system_prompt(SHOPPER_1)


def test_sql_tool_descriptions_state_the_rules() -> None:
    read, write = SQL_TOOLS
    for phrase in ["(SELECT today FROM world)", "integer cents", "refund_eligible", "policy_id", "100 rows"]:
        assert phrase in read.description
    for phrase in ["UPDATE orders SET status = 'cancelled'", "INSERT INTO refunds", "INSERT INTO escalations",
                   "queued_for_approval", "permission_denied", "unsupported_statement"]:
        assert phrase in write.description
    # Evaluation inputs stay out of prompts: no demo order ids in the descriptions.
    for demo_order in ("4127", "3980", "4455"):
        assert demo_order not in read.description + write.description


# ---------------------------------------------------------------------------
# Metrics: both variants, side by side
# ---------------------------------------------------------------------------


class MeteredFakeModel(FakeModel):
    """FakeModel that reports token usage like a real provider."""

    async def get_response(self, *args, **kwargs) -> ModelResponse:
        response = await super().get_response(*args, **kwargs)
        response.usage = Usage(
            requests=1,
            input_tokens=1000,
            input_tokens_details=InputTokensDetails(cached_tokens=400),
            output_tokens=50,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=1050,
        )
        return response


def _read_log(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_cost_uses_cached_and_output_rates() -> None:
    assert cost_usd("gpt-4o-mini", 1_000_000, 0, 0) == pytest.approx(0.15)
    assert cost_usd("gpt-4o-mini", 1_000_000, 1_000_000, 1_000_000) == pytest.approx(0.075 + 0.60)
    assert cost_usd("unknown-model", 10, 0, 10) is None


def test_run_with_metrics_logs_turns_and_tool_calls(world_copy, metrics_log, monkeypatch) -> None:
    from agent import sql_agent

    model = MeteredFakeModel()
    model.set_next_output([tool_call("read_sql", {"query": "SELECT id, status FROM orders WHERE id = 4127"})])
    model.set_next_output([tool_call("write_sql", {"statement": "DELETE FROM orders"})])
    model.set_next_output([text_message("Order 4127 was delivered.")])
    monkeypatch.setattr(sql_agent, "resolve_model", lambda _: model)
    agent = build_sql_agent(SHOPPER_1)

    result, turn = asyncio.run(
        run_with_metrics(agent, "Where is order 4127?", context=SHOPPER_1, variant="sql", model="gpt-4o-mini")
    )
    assert result.final_output == "Order 4127 was delivered."

    records = _read_log(metrics_log)
    calls = [r for r in records if r["type"] == "tool_call"]
    assert [(c["tool"], c["ok"], c["error"]) for c in calls] == [
        ("read_sql", True, None),
        ("write_sql", False, "unsupported_statement"),
    ]
    assert calls[0]["arguments"] == {"query": "SELECT id, status FROM orders WHERE id = 4127"}
    assert all(c["latency_ms"] >= 0 and c["variant"] == "sql" for c in calls)

    assert calls[0]["result"]["rows"] == [[4127, "delivered"]]
    assert calls[1]["result"]["accepted_forms"]

    assert records[-1] == turn
    assert turn["status"] == "ok"
    assert turn["route"] == "MeteredFakeModel"
    assert turn["settings"] == {"defenses": False, "kill_switch": "off", "max_turns": None}
    assert turn["actions"] == [{"action": "write_sql", "ok": False, "outcome": "unsupported_statement"}]
    assert [c["input_tokens"] for c in turn["llm_call_details"]] == [1000, 1000, 1000]
    assert turn["llm_calls"] == 3
    assert (turn["tool_calls"], turn["tool_failures"], turn["tool_errors"]) == (2, 1, ["unsupported_statement"])
    assert (turn["input_tokens"], turn["cached_tokens"], turn["output_tokens"]) == (3000, 1200, 150)
    assert turn["cost_usd"] == pytest.approx(cost_usd("gpt-4o-mini", 3000, 1200, 150))
    assert turn["latency_ms"] >= turn["tool_latency_ms"]


def test_write_actions_are_listed_on_the_turn(world_copy, metrics_log, monkeypatch) -> None:
    from agent import sql_agent

    model = FakeModel()
    model.set_next_output([tool_call("write_sql", {
        "statement": "INSERT INTO refunds (order_id, amount_cents, reason) VALUES (4455, 24000, 'cracked')"})])
    model.set_next_output([text_message("A person will review your refund.")])
    monkeypatch.setattr(sql_agent, "resolve_model", lambda _: model)
    _, turn = asyncio.run(run_with_metrics(
        build_sql_agent(SHOPPER_1), "Refund 4455", context=SHOPPER_1, variant="sql", model="gpt-4o-mini",
        turn_index=2))
    assert turn["actions"] == [{"action": "issue_refund", "ok": True, "outcome": "queued_for_approval"}]
    assert turn["turn_index"] == 2


def test_frontier_models_prefer_the_providers_own_key(monkeypatch) -> None:
    from agent.agent import model_route

    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY or OPENROUTER_API_KEY"):
        model_route("gemini-4-argon")
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")
    assert model_route("gemini-4-argon") == "openrouter/google/gemini-4-argon"
    assert model_route("claude-opus-5-5") == "openrouter/anthropic/claude-opus-5.5"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    assert model_route("claude-opus-5-5") == "anthropic/claude-opus-5-5"
    assert model_route("gpt-6-astra") == "openai/gpt-6-astra"
    # Course models are unchanged.
    assert model_route("gpt-4o-mini") == "openai/gpt-4o-mini"
    assert model_route("claude-haiku-4-5") == "anthropic/claude-haiku-4-5"


def test_every_named_model_has_a_price() -> None:
    from agent.agent import FRONTIER_MODELS

    for name in [*FRONTIER_MODELS, "gpt-4o-mini", "gpt-5-nano", "claude-haiku-4-5"]:
        assert cost_usd(name, 1000, 0, 1000) is not None
    # Provider and OpenRouter ids price like the course name.
    assert cost_usd("openrouter/anthropic/claude-haiku-4.5", 10**6, 0, 0) == pytest.approx(1.00)
    assert cost_usd("anthropic/claude-opus-5-5", 0, 0, 10**6) == pytest.approx(20.00)


def test_empty_reply_is_not_a_success(world: dict, metrics_log) -> None:
    # A provider content filter ends the run with no text and no exception.
    model = FakeModel()
    model.set_next_output([])
    agent = Agent(name="filtered", model=model)
    _, turn = asyncio.run(run_with_metrics(agent, "hi", context=SHOPPER_1, variant="tools", model="gpt-4o-mini"))
    assert turn["status"] == "empty"


def test_failed_run_is_still_logged(world: dict, metrics_log) -> None:
    model = FakeModel()  # no scripted output: the first model call raises
    agent = Agent(name="broken", model=model)
    with pytest.raises(AssertionError):
        asyncio.run(run_with_metrics(agent, "hi", context=SHOPPER_1, variant="tools", model="gpt-4o-mini"))
    turn = _read_log(metrics_log)[-1]
    assert turn["type"] == "turn" and turn["status"] == "error"
    assert "ran out of scripted outputs" in turn["error"]


def test_cli_sql_variant_prints_a_metrics_footer(world_copy, metrics_log, tmp_path, monkeypatch, capsys) -> None:
    from agent import sql_agent

    model = MeteredFakeModel()
    model.set_next_output([tool_call("read_sql", {"query": "SELECT count(*) FROM orders"})])
    model.set_next_output([text_message("You have some orders.")])
    monkeypatch.setattr(sql_agent, "resolve_model", lambda _: model)
    monkeypatch.setattr(cli, "SESSIONS_DB", tmp_path / "sessions.db")
    monkeypatch.setattr(cli, "resolve_auth", lambda *args: SHOPPER_1)
    monkeypatch.setattr(cli, "load_env", lambda: None)
    monkeypatch.setattr("sys.argv", ["agent.cli", "--variant", "sql", "--model", "gpt-4o-mini"])
    messages = iter(["How many orders do I have?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(messages))

    cli.main()
    output = capsys.readouterr().out
    assert "variant=sql" in output
    assert "agent> You have some orders." in output
    assert "[sql · " in output and "1 tool calls" in output and "2,100 tokens" in output
    turn = _read_log(metrics_log)[-1]
    assert turn["variant"] == "sql" and turn["session_id"].startswith("cli-sql-shopper-1-")


def test_cli_rejects_defenses_with_the_sql_variant(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["agent.cli", "--variant", "sql", "--defenses"])
    with pytest.raises(SystemExit):
        cli.main()


def test_compare_report_puts_variants_side_by_side(metrics_log, monkeypatch, capsys) -> None:
    def turn(variant: str, latency: float, tools: int, failures: int, cost: float) -> dict:
        return {"type": "turn", "ts": "2026-10-14T00:00:00", "variant": variant, "model": "gpt-4o-mini",
                "role": "shopper", "user_id": 1, "input": "Where is order 4127?", "status": "ok",
                "latency_ms": latency, "llm_calls": 2, "tool_calls": tools, "tool_failures": failures,
                "input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100, "cost_usd": cost}

    def call(variant: str, tool: str, ok: bool, error: str | None = None) -> dict:
        return {"type": "tool_call", "ts": "2026-10-14T00:00:00", "variant": variant, "model": "gpt-4o-mini",
                "role": "shopper", "tool": tool, "ok": ok, "error": error, "latency_ms": 5.0}

    lines = [turn("tools", 1500, 1, 0, 0.0002), turn("sql", 2500, 2, 1, 0.0004),
             call("tools", "get_order", True), call("sql", "read_sql", False, "sql_error"), call("sql", "read_sql", True)]
    metrics_log.write_text("".join(json.dumps(r) + "\n" for r in lines))
    monkeypatch.setattr("sys.argv", ["compare_agents"])

    compare_agents.main()
    output = capsys.readouterr().out
    assert "tools · gpt-4o-mini" in output and "sql · gpt-4o-mini" in output
    assert "sql_error ×1" in output
    assert "1.5s / 2.5s" in output  # the paired message
