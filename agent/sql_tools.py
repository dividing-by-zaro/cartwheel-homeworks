"""Logic for the SQL variant of the support agent (agent/sql_agent.py).

The SQL agent has two tools instead of eleven: `read_sql` and `write_sql`.
The model writes SQL, but the permissions and business rules stay in code:

  - read_sql runs the query against a private, in-memory copy of the world
    that holds only the caller's rows (the access matrix in SPEC.md, applied
    before the model's SQL ever runs). The copy is read-only and the query
    may only SELECT. Even a query that tries to read someone else's order
    finds no row to read.
  - write_sql accepts exactly three statement shapes (cancel an order, insert
    a refund, insert an escalation). It parses the statement and hands it to
    the same validated logic the tool-based agent uses (cancel_order,
    issue_refund_logic, the escalation insert), so scope, the pre-shipment
    rule, refund eligibility, the $100 approval threshold, and the kill switch
    all still apply. The model's SQL is never executed against the real
    database.

The model-facing documentation (schema, rules, examples) lives in the tool
docstrings in agent/sql_agent.py, because that is what the model reads.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import Any

from agent import db
from agent import tools as hw_tools
from agent.auth import AuthContext, can_view_order, permission_denied
from agent.config import db_path, load_facts
from agent.helpcenter import load_policy_docs

READ_ROW_LIMIT = 100
READ_TIMEOUT_SECONDS = 2.0

# Columns per table in the caller's private copy, used in error hints so the
# model can fix a bad query in one retry.
SCHEMA_HINT = (
    "Tables: orders(id, user_id, store_id, product_id, quantity, total_cents, status, "
    "ordered_at, shipped_at, delivered_at, refund_eligible); products(id, store_id, title, "
    "description, category, price_cents); stores(id, name, slug, category, "
    "return_window_days_override, restocking_fee_opt_in); refunds(id, order_id, amount_cents, "
    "reason, status, created_at); escalations(id, user_id, store_id, order_id, summary, context, "
    "created_at); users(id, name, role, store_id); policies(policy_id, title, audience, body); "
    "facts(key, value); world(today)."
)

_FORBIDDEN_NOW = re.compile(r"'now'|\bcurrent_(date|time|timestamp)\b", re.IGNORECASE)

# SQLite authorizer action codes a SELECT legitimately needs.
_READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
_READ_ACTIONS.add(getattr(sqlite3, "SQLITE_RECURSIVE", 33))  # WITH RECURSIVE


def _error(code: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": code, "reason": reason, **extra}


# ---------------------------------------------------------------------------
# read_sql
# ---------------------------------------------------------------------------


def _scope_filters(ctx: AuthContext) -> dict[str, tuple[str, tuple]]:
    """WHERE clauses that cut each private table down to the caller's rows.

    This is the access matrix from SPEC.md, applied in code. `refunds` follows
    `orders` (a refund is visible exactly when its order is).
    """
    if ctx.role == "shopper":
        return {
            "orders": ("user_id = ?", (ctx.user_id,)),
            "escalations": ("user_id = ?", (ctx.user_id,)),
            "users": ("id = ?", (ctx.user_id,)),
        }
    if ctx.role == "merchant":
        return {
            "orders": ("store_id = ?", (ctx.store_id,)),
            "escalations": ("store_id = ? OR user_id = ?", (ctx.store_id, ctx.user_id)),
            "users": ("id = ?", (ctx.user_id,)),
        }
    if ctx.role == "support":
        return {"orders": ("1", ()), "escalations": ("1", ()), "users": ("1", ())}
    raise ValueError(f"unknown role: {ctx.role!r}")


def _flatten_facts(facts: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    rows = []
    for key, value in facts.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            rows.extend(_flatten_facts(value, f"{name}."))
        else:
            rows.append((name, str(value)))
    return rows


def _scoped_world(ctx: AuthContext) -> sqlite3.Connection:
    """Build the caller's private, in-memory copy of the world."""
    filters = _scope_filters(ctx)
    conn = sqlite3.connect("file::memory:", uri=True)
    conn.execute("ATTACH DATABASE ? AS src", (db_path().resolve().as_uri() + "?mode=ro",))
    conn.execute("CREATE TABLE stores AS SELECT * FROM src.stores")
    conn.execute("CREATE TABLE products AS SELECT * FROM src.products")
    for table in ("orders", "escalations", "users"):
        where, params = filters[table]
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM src.{table} WHERE {where}", params)
    conn.execute(
        "CREATE TABLE refunds AS SELECT * FROM src.refunds WHERE order_id IN (SELECT id FROM orders)"
    )
    asof = conn.execute("SELECT value FROM src.meta WHERE key = 'world_asof'").fetchone()[0]
    conn.execute("DETACH DATABASE src")

    conn.execute("CREATE TABLE world (today TEXT)")
    conn.execute("INSERT INTO world VALUES (?)", (asof,))
    conn.execute("CREATE TABLE facts (key TEXT, value TEXT)")
    conn.executemany("INSERT INTO facts VALUES (?, ?)", _flatten_facts(load_facts()))
    conn.execute("CREATE TABLE policies (policy_id TEXT, title TEXT, audience TEXT, body TEXT)")
    conn.executemany(
        "INSERT INTO policies VALUES (?, ?, ?, ?)",
        [(d.policy_id, d.title, d.audience, d.body) for d in load_policy_docs()],
    )
    conn.commit()
    return conn


def _read_only_authorizer(action: int, *_args: Any) -> int:
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


def read_sql(ctx: AuthContext, query: str) -> dict[str, Any]:
    """Run one SELECT against the caller's private copy of the world.

    Returns {"ok": True, "columns": [...], "rows": [[...], ...],
    "row_count": int, "truncated": bool}, at most READ_ROW_LIMIT rows.
    Failures: invalid_argument (empty, not a SELECT, uses the real clock),
    read_only (the statement tried to change something), timeout, sql_error.
    """
    query = query.strip().rstrip(";").strip()
    if not query:
        return _error("invalid_argument", "query is empty")
    first_word = query.split(None, 1)[0].lower()
    if first_word not in {"select", "with"}:
        hint = " Use write_sql for changes." if first_word in {"insert", "update", "delete"} else ""
        return _error("read_only", f"read_sql runs only SELECT (or WITH ... SELECT) queries.{hint}")
    if _FORBIDDEN_NOW.search(query):
        return _error(
            "invalid_argument",
            "do not use the real clock ('now', CURRENT_DATE). The world's date is fixed.",
            hint="Use (SELECT today FROM world), e.g. julianday((SELECT today FROM world)) - julianday(delivered_at).",
        )

    conn = _scoped_world(ctx)
    try:
        conn.set_authorizer(_read_only_authorizer)
        deadline = time.monotonic() + READ_TIMEOUT_SECONDS
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cursor = conn.execute(query)
            rows = cursor.fetchmany(READ_ROW_LIMIT + 1)
        except sqlite3.ProgrammingError as exc:
            return _error("invalid_argument", f"{exc}. Send exactly one statement per call.")
        except sqlite3.DatabaseError as exc:
            message = str(exc)
            if message == "interrupted":
                return _error("timeout", f"query ran longer than {READ_TIMEOUT_SECONDS:g}s; add filters or a LIMIT")
            if "not authorized" in message:
                return _error("read_only", "only SELECT is allowed; PRAGMA, ATTACH, and writes are blocked")
            return _error("sql_error", message, hint=SCHEMA_HINT)
        columns = [c[0] for c in cursor.description or []]
    finally:
        conn.close()
    truncated = len(rows) > READ_ROW_LIMIT
    rows = [list(r) for r in rows[:READ_ROW_LIMIT]]
    result = {"ok": True, "columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated}
    if truncated:
        result["note"] = f"only the first {READ_ROW_LIMIT} rows are returned; add a WHERE, ORDER BY, or LIMIT"
    return result


# ---------------------------------------------------------------------------
# write_sql
# ---------------------------------------------------------------------------

ACCEPTED_WRITES = [
    "UPDATE orders SET status = 'cancelled' WHERE id = <order id>",
    "INSERT INTO refunds (order_id, amount_cents, reason) VALUES (<order id>, <cents>, '<reason>')",
    "INSERT INTO escalations (summary, context[, order_id]) VALUES ('<summary>', '<context>'[, <order id>])",
]

_CANCEL = re.compile(
    r"UPDATE\s+orders\s+SET\s+status\s*=\s*'(?P<status>[^']*)'\s+WHERE\s+id\s*=\s*(?P<id>\d+)",
    re.IGNORECASE,
)
_UPDATE_ORDERS = re.compile(r"UPDATE\s+orders\b", re.IGNORECASE)
_INSERT = re.compile(
    r"INSERT\s+INTO\s+(?P<table>\w+)\s*\((?P<cols>[^)]*)\)\s*VALUES\s*\((?P<vals>.*)\)",
    re.IGNORECASE | re.DOTALL,
)
_VALUE = re.compile(r"\s*('(?:[^']|'')*'|-?\d+\.\d+|-?\d+|NULL)\s*(,|$)", re.IGNORECASE | re.DOTALL)


def _parse_values(text: str) -> list[Any] | None:
    """Parse a VALUES list of string, integer, decimal, and NULL literals."""
    values: list[Any] = []
    pos = 0
    while pos < len(text):
        match = _VALUE.match(text, pos)
        if match is None:
            return None
        token = match.group(1)
        if token.startswith("'"):
            values.append(token[1:-1].replace("''", "'"))
        elif token.upper() == "NULL":
            values.append(None)
        elif "." in token:
            values.append(float(token))
        else:
            values.append(int(token))
        pos = match.end()
        if match.group(2) == "":
            break
    return values if pos >= len(text.rstrip()) else None


def _unsupported(reason: str) -> dict[str, Any]:
    return _error("unsupported_statement", reason, accepted_forms=ACCEPTED_WRITES)


def write_sql(ctx: AuthContext, statement: str) -> dict[str, Any]:
    """Apply one of the three accepted write statements through validated logic."""
    statement = statement.strip().rstrip(";").strip()
    if not statement:
        return _error("invalid_argument", "statement is empty")
    if statement.split(None, 1)[0].lower() in {"select", "with"}:
        return _unsupported("write_sql does not run queries; use read_sql to look things up")

    cancel = _CANCEL.fullmatch(statement)
    if cancel:
        if cancel["status"] != "cancelled":
            return _unsupported(
                f"status can only be set to 'cancelled' (got '{cancel['status']}'); "
                "refunds are made with INSERT INTO refunds"
            )
        order_id = int(cancel["id"])
        return {"operation": "cancel_order", **hw_tools.cancel_order(ctx, order_id, "write_sql")}
    if _UPDATE_ORDERS.match(statement):
        return _unsupported(
            "the only accepted UPDATE is SET status = 'cancelled' WHERE id = <one order id>; "
            "no other columns, conditions, or multiple orders"
        )

    insert = _INSERT.fullmatch(statement)
    if insert is None:
        return _unsupported("statement is not one of the accepted forms")
    table = insert["table"].lower()
    cols = [c.strip().lower() for c in insert["cols"].split(",")]
    values = _parse_values(insert["vals"])
    if values is None or len(values) != len(cols):
        return _error(
            "invalid_argument",
            "could not read the VALUES list: use one value per column, with text in single quotes "
            "(write a quote inside text as two quotes, e.g. 'it''s broken')",
        )
    row = dict(zip(cols, values))
    if table == "refunds":
        return _insert_refund(ctx, row)
    if table == "escalations":
        return _insert_escalation(ctx, row)
    return _unsupported(f"inserts into '{table}' are not allowed; only refunds and escalations")


def _insert_refund(ctx: AuthContext, row: dict[str, Any]) -> dict[str, Any]:
    if set(row) != {"order_id", "amount_cents", "reason"}:
        return _unsupported(
            "a refund insert names exactly these columns: (order_id, amount_cents, reason). "
            "id, status, and created_at are filled in automatically"
        )
    order_id, amount, reason = row["order_id"], row["amount_cents"], row["reason"]
    if not isinstance(order_id, int):
        return _error("invalid_argument", "order_id must be an integer")
    if not isinstance(amount, int):
        return _error("invalid_argument", "amount_cents must be a whole number of cents, e.g. 2400 for $24.00")
    if not isinstance(reason, str) or not reason.strip():
        return _error("invalid_argument", "reason must be non-empty text in single quotes")
    # issue_refund_logic enforces the kill switch, scope, amount, eligibility,
    # and the approval threshold, and writes the refund record.
    from agent.agent import issue_refund_logic

    result = issue_refund_logic(ctx, order_id, amount / 100, reason)
    if result.get("ok"):
        result["amount_cents"] = amount
    return {"operation": "issue_refund", **result}


def _insert_escalation(ctx: AuthContext, row: dict[str, Any]) -> dict[str, Any]:
    allowed, required = {"summary", "context", "order_id"}, {"summary", "context"}
    if not required <= set(row) <= allowed:
        return _unsupported(
            "an escalation insert names (summary, context) or (summary, context, order_id). "
            "id, user_id, store_id, and created_at are filled in automatically"
        )
    summary, context, order_id = row["summary"], row["context"], row.get("order_id")
    if not isinstance(summary, str) or not summary.strip() or not isinstance(context, str):
        return _error("invalid_argument", "summary and context must be text in single quotes")
    if order_id is not None and not isinstance(order_id, int):
        return _error("invalid_argument", "order_id must be an integer or NULL")
    facts = load_facts()
    with db.connection() as conn:
        if order_id is not None:
            order = db.get_order(conn, order_id)
            if order is None:
                return {"operation": "escalate", **_error("not_found", f"no order #{order_id}")}
            if not can_view_order(ctx, order.user_id, order.store_id):
                return {
                    "operation": "escalate",
                    **permission_denied(
                        f"role '{ctx.role}' (user {ctx.user_id}) may not escalate order #{order_id}"
                    ),
                }
        ticket_id = db.insert_escalation(
            conn,
            user_id=ctx.user_id,
            store_id=ctx.store_id,
            order_id=order_id,
            summary=summary,
            context=json.dumps({"role": ctx.role, "context": context}),
            created_at=db.world_asof(conn).isoformat(),
        )
    return {
        "operation": "escalate",
        "ok": True,
        "ticket_id": ticket_id,
        "sla_hours": facts["support_escalation_sla_hours"],
    }
