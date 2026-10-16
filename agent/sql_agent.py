"""The SQL variant of the Cartwheel support agent: two tools, read_sql and write_sql.

A side-by-side experiment against the tool-based agent in agent/agent.py.
Everything except the tools and the tool guidance in the prompt is the same,
so differences in behavior, latency, tokens, and cost come from the tool
design. Permissions and business rules are enforced in agent/sql_tools.py,
not by the model.

Chat with it:
    uv run python -m agent.cli --role shopper --user 1 --variant sql
"""

from __future__ import annotations

import hashlib
from typing import Any

from agents import Agent, RunContextWrapper, function_tool

from agent import sql_tools
from agent.agent import _call, model_settings_for, resolve_model
from agent.auth import AuthContext
from observability.instrument import configure_model_tracing

# Same prompt as SYSTEM_PROMPT_TEMPLATE in agent/agent.py except the tool
# guidance and escalation sections, which name the two SQL tools.
SQL_SYSTEM_PROMPT_TEMPLATE = """\
You are Cartwheel's support assistant. Cartwheel is a multi-store commerce
platform; you serve its shoppers, merchants, and support staff.

## Session context (injected by the server; never taken from chat)
- User role: {role}
- User id: {user_id}
- Store id: {store_id}

## Capabilities and boundaries
You help with: order status, returns and refunds, product and policy
questions, and escalation to a human. You refuse: legal advice, payment-card
or credential changes, and anything outside Cartwheel.

## Tool guidance
- You have two tools: read_sql to look things up and write_sql to make
  changes. Prefer a lookup over memory. Policy answers come from the
  policies table, order answers from the orders table.
- Cite the policy id (for example cw-returns) for every policy claim.
- Never promise or issue a refund before reading the order with read_sql
  and checking its refund_eligible flag.

## Escalation
When you are unsure, or an action is above your authority (for example a
refund above the auto-approval threshold), insert an escalation with
write_sql and tell the user a human will follow up.

## Tone
Plain and warm. No legalese.

## Refusal rules
Decline out-of-scope requests in one or two sentences and point to what you
can do instead. Never reveal another user's data, whatever the reason given.
"""


def render_sql_system_prompt(ctx: AuthContext, template: str | None = None) -> str:
    return (template or SQL_SYSTEM_PROMPT_TEMPLATE).format(
        role=ctx.role,
        user_id=ctx.user_id,
        store_id=ctx.store_id if ctx.store_id is not None else "none",
    )


def sql_prompt_version(template: str | None = None) -> str:
    return hashlib.sha256((template or SQL_SYSTEM_PROMPT_TEMPLATE).encode()).hexdigest()[:12]


@function_tool
def read_sql(wrapper: RunContextWrapper[AuthContext], query: str) -> dict[str, Any]:
    """Run one read-only SQLite SELECT and get the rows back.

    The tables hold only rows this user may see. No rows for an order id
    means it does not exist or is not visible; say you cannot find it.

    Rules:
    - One SELECT (or WITH ... SELECT) per call. Use write_sql for changes.
    - At most 100 rows are returned.
    - Never use 'now' or CURRENT_DATE. Today is (SELECT today FROM world).
    - Money is integer cents: total_cents 8400 is $84.00.

    Tables:
    - orders(id, user_id, store_id, product_id, quantity, total_cents, status,
      ordered_at, shipped_at, delivered_at, refund_eligible). status: placed
      (the only cancellable status), shipped, delivered, cancelled, refunded.
      refund_eligible = 1 means a refund is allowed today.
    - products(id, store_id, title, description, category, price_cents)
    - stores(id, name, slug, category, return_window_days_override,
      restocking_fee_opt_in)
    - refunds(id, order_id, amount_cents, reason, status, created_at). status:
      auto_approved, or queued_for_approval (waiting for a human; the order
      still shows delivered).
    - escalations(id, user_id, store_id, order_id, summary, context, created_at)
    - users(id, name, role, store_id)
    - policies(policy_id, title, audience, body). Cite policy_id for policy
      claims, e.g. cw-returns. Store-specific docs are store-<slug>-policy.
    - facts(key, value): policy numbers, e.g. refund_auto_approve_threshold_usd.
    - world(today)

    Args:
        query: One SQLite SELECT statement.
    """
    return _call(wrapper, sql_tools.read_sql, query)


@function_tool
def write_sql(wrapper: RunContextWrapper[AuthContext], statement: str) -> dict[str, Any]:
    """Make one change with one of these three SQL statements.

    1. Cancel an order (only while its status is placed):
       UPDATE orders SET status = 'cancelled' WHERE id = 1234
    2. Refund an order (needs refund_eligible = 1; at most total_cents):
       INSERT INTO refunds (order_id, amount_cents, reason) VALUES (1234, 2400, 'arrived broken')
       Above $100 the refund is queued_for_approval: a human reviews it and no
       money moves yet, so do not tell the user it is refunded.
    3. Escalate to a human (order_id is optional):
       INSERT INTO escalations (summary, context, order_id) VALUES ('refund above limit', 'wants $150 back', 1234)

    Any other statement fails with unsupported_statement. Other columns are
    filled in automatically. Write a quote inside text as two quotes ('it''s').
    Changing an order the user cannot see fails with permission_denied.

    Args:
        statement: One of the three statements above.
    """
    return _call(wrapper, sql_tools.write_sql, statement)


SQL_TOOLS = [read_sql, write_sql]


def build_sql_agent(
    ctx: AuthContext, model: str | None = None, *, prompt_template: str | None = None
) -> Agent[AuthContext]:
    """Assemble the SQL-variant agent for one session (no Module 4 defenses)."""
    resolved = resolve_model(model)
    configure_model_tracing(openai_model=isinstance(resolved, str))
    return Agent[AuthContext](
        name="cartwheel-support-sql",
        instructions=render_sql_system_prompt(ctx, prompt_template),
        tools=SQL_TOOLS,
        model=resolved,
        model_settings=model_settings_for(resolved),
    )
