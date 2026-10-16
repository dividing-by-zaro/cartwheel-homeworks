"""Homework 1: the remaining commerce-agent tools.

The three lecture tools (`search_help_center`, `get_order`, `issue_refund`)
are implemented in agent/agent.py and are worked examples of the pattern:
check permissions first, go through agent/db.py for data, and return a
structured dict, never a prose error. The homework tools follow the same
pattern. agent/agent.py already wraps each function below as an SDK tool, so
once a function works here it works in chat with no further wiring.

Result convention (see agent/auth.py):
  - Success: a dict with "ok": True plus the payload fields named in each
    docstring.
  - Failure: {"ok": False, "error": <code>, "reason": <human-readable str>}.

Run the contract tests with: uv run pytest tests/test_hw_holes.py -k hw1
They are marked xfail and flip to passing as you implement each function.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from agent import db
from agent.auth import AuthContext, can_cancel_order, can_view_order, permission_denied
from agent.config import load_facts
from agent.helpcenter import load_policy_docs
from agent.killswitch import kill_switch

MAX_SEARCH_LIMIT = 25
DEFAULT_ORDER_LIMIT = 20
FIND_ORDER_LIMIT = 5


def get_policy(ctx: AuthContext, policy_id: str) -> dict[str, Any]:
    """Fetch one policy doc by its exact id. Risk tier: read.

    Every role may read every policy doc (the corpus is public help-center
    content), so this tool needs no permission check.

    Args:
        ctx: The caller's auth context. Unused here, but every tool takes it.
        policy_id: An exact policy id, e.g. "cw-returns" or
            "store-juniper-home-goods-policy". Matching is exact and
            case-sensitive; ids are the `policy_id` front-matter field of the
            files in data/policies/.

    Returns:
        On success: {"ok": True, "policy_id": str, "title": str,
        "audience": str, "body": str} where body is the markdown body of the
        doc without the front matter.
        If no doc has that id: {"ok": False, "error": "not_found",
        "reason": ...} naming the id that was requested.

    Implementation notes:
        agent.helpcenter.load_policy_docs() returns every parsed doc.
    """
    for doc in load_policy_docs():
        if doc.policy_id == policy_id:
            return {
                "ok": True,
                "policy_id": doc.policy_id,
                "title": doc.title,
                "audience": doc.audience,
                "body": doc.body,
            }
    return {"ok": False, "error": "not_found", "reason": f"no policy doc with id '{policy_id}'"}


def search_products(
    ctx: AuthContext,
    query: str,
    store: str | None = None,
    max_price_usd: float | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search the product catalog. Risk tier: read.

    Every role may search products. Matching is deterministic keyword
    matching, not semantic search: a product matches when every whitespace
    token of `query` appears case-insensitively as a substring of the
    product's title or description.

    Args:
        ctx: The caller's auth context.
        query: Free-text query. Must be non-empty after stripping whitespace;
            otherwise return {"ok": False, "error": "invalid_argument",
            "reason": ...}.
        store: Optional store filter. Matched with
            agent.db.get_store_by_name (case-insensitive name or slug). If
            given and no store matches, return {"ok": False, "error":
            "not_found", "reason": ...} naming the store string.
        max_price_usd: Optional inclusive price ceiling. If given and not
            strictly positive, return an "invalid_argument" error.
        limit: Maximum products to return. Clamp to the range
            [1, MAX_SEARCH_LIMIT]; do not error on out-of-range values.

    Returns:
        {"ok": True, "products": [...], "count": <len(products)>} where each
        product is {"product_id": int, "store_id": int, "title": str,
        "price_usd": float}. Sort matches by price_usd ascending, then by
        product_id ascending, and truncate to `limit`. No matches is still a
        success: {"ok": True, "products": [], "count": 0}.

    Implementation notes:
        agent.db.list_products(conn, store_id) gives the candidate set.
        Use `with db.connection() as conn:` to close the database automatically.
    """
    tokens = query.lower().split()
    if not tokens:
        return {"ok": False, "error": "invalid_argument", "reason": "query must not be empty"}
    if max_price_usd is not None and max_price_usd <= 0:
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": "max_price_usd must be greater than zero",
        }
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))

    with db.connection() as conn:
        store_id = None
        if store is not None:
            found = db.get_store_by_name(conn, store)
            if found is None:
                return {"ok": False, "error": "not_found", "reason": f"no store named '{store}'"}
            store_id = found.id
        candidates = db.list_products(conn, store_id)

    matches = [
        p
        for p in candidates
        if all(t in p.title.lower() or t in p.description.lower() for t in tokens)
        and (max_price_usd is None or p.price_usd <= max_price_usd)
    ]
    matches.sort(key=lambda p: (p.price_usd, p.id))
    products = [
        {"product_id": p.id, "store_id": p.store_id, "title": p.title, "price_usd": p.price_usd}
        for p in matches[:limit]
    ]
    return {"ok": True, "products": products, "count": len(products)}


def list_my_orders(ctx: AuthContext) -> dict[str, Any]:
    """List recent orders in the caller's own scope. Risk tier: read.

    Role behavior, straight from the access matrix in SPEC.md:
        - shopper: the caller's own orders.
        - merchant: the caller's store's orders (ctx.store_id).
        - support: support staff have no orders of their own and look up
          specific orders with get_order instead, so return {"ok": False,
          "error": "invalid_argument", "reason": ...} saying exactly that.

    Returns:
        For shopper and merchant: {"ok": True, "orders": [...],
        "count": <len(orders)>} where each order is
        agent.db.Order.to_public_dict() and the list holds at most
        DEFAULT_ORDER_LIMIT orders, newest first (agent.db.list_orders_for_user
        and list_orders_for_store already sort and limit this way).

    Implementation notes:
        No permission check is needed beyond the role dispatch, because the
        scope is baked into which query you run. That is the point of the
        tool: the model cannot ask for someone else's orders through it.
    """
    if ctx.role == "support":
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": "support staff have no orders of their own; look up a specific order with get_order",
        }
    with db.connection() as conn:
        if ctx.role == "shopper":
            orders = db.list_orders_for_user(conn, ctx.user_id, DEFAULT_ORDER_LIMIT)
        else:
            orders = db.list_orders_for_store(conn, ctx.store_id, DEFAULT_ORDER_LIMIT)
    return {"ok": True, "orders": [o.to_public_dict() for o in orders], "count": len(orders)}


def cancel_order(ctx: AuthContext, order_id: int, reason: str) -> dict[str, Any]:
    """Cancel an order. Risk tier: write.

    This is the homework's write tool, and it must enforce two independent
    rules in this order:

    1. The access matrix (scope): use agent.auth.can_cancel_order. Shoppers
       may cancel only their own orders, merchants only their own store's
       orders, support any order. On failure return
       agent.auth.permission_denied(...) with a reason naming the role and
       the order id. Scope is checked before the status rule so that an
       out-of-scope caller learns nothing about the order's state.
    2. The pre-shipment rule (facts.yaml `cancel_cutoff`): only orders whose
       status is exactly "placed" can be cancelled, for every role. If the
       order is in scope but its status is not "placed", return
       {"ok": False, "error": "not_eligible", "reason": ...} that names the
       current status and states that orders can be cancelled only before
       shipment.

    Args:
        ctx: The caller's auth context.
        order_id: The order to cancel.
        reason: Free-text reason from the user; not validated.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found",
        "reason": ...}.
        On success: {"ok": True, "order_id": order_id, "status": "cancelled"}
        after persisting the new status with agent.db.set_order_status.

    Implementation notes:
        Fetch with agent.db.get_order. Note the argument order of
        can_cancel_order(ctx, order_user_id, order_store_id).

    The Module 4 kill switch is checked first (before the scope and
    status rules and before your code), so that a paused write tool touches
    nothing. It is provided; the default ("off") returns None and falls
    through to your implementation.
    """
    paused = kill_switch("cancel_order")
    if paused is not None:
        return {"ok": False, "error": "paused", "reason": paused}
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"no order #{order_id}"}
        if not can_cancel_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not cancel order #{order_id}"
            )
        if order.status != "placed":
            return {
                "ok": False,
                "error": "not_eligible",
                "reason": (
                    f"order #{order_id} is '{order.status}'; orders can be cancelled "
                    f"only before shipment, while they are 'placed'"
                ),
            }
        db.set_order_status(conn, order_id, "cancelled")
    return {"ok": True, "order_id": order_id, "status": "cancelled"}


def find_order(ctx: AuthContext, query: str) -> dict[str, Any]:
    """Search the caller's orders by product name. Risk tier: read.

    Takes a natural-language query (e.g., "earmuffs I bought last week")
    and searches the authenticated user's orders for products whose name
    matches. Use fuzzy string matching (e.g., thefuzz.fuzz.partial_ratio
    or case-insensitive substring matching) to find orders whose product name is close to the
    query.

    Access rules: a shopper searches only the shopper's own orders, a
    merchant searches orders from the merchant's store, and support staff
    can search any orders. Use agent.db.list_order_search_candidates with
    user_id=ctx.user_id for shoppers, store_id=ctx.store_id for merchants,
    or all_orders=True only for support. Derive the scope from ctx, never
    from the query; reject unsupported roles or missing required identity.
    Use agent.db.list_products to map product IDs to product titles.

    The helper returns the complete authorised scope, newest first with
    order ID descending as the tie-breaker. Match product names first,
    preserve that order, then return at most five matches. Do not search
    only the 20 most recent orders. Convert matches with to_public_dict().

    Args:
        ctx: The caller's auth context.
        query: A natural-language description of the product.

    Returns:
        {"ok": True, "orders": [...]} with a list of matching orders
        (at most 5), each as the dict returned by agent.db. If no orders
        match, return {"ok": True, "orders": []}.
    """
    with db.connection() as conn:
        if ctx.role == "shopper" and ctx.user_id is not None:
            candidates = db.list_order_search_candidates(conn, user_id=ctx.user_id)
        elif ctx.role == "merchant" and ctx.store_id is not None:
            candidates = db.list_order_search_candidates(conn, store_id=ctx.store_id)
        elif ctx.role == "support":
            candidates = db.list_order_search_candidates(conn, all_orders=True)
        else:
            return permission_denied(f"role '{ctx.role}' may not search orders")
        titles = {p.id: p.title for p in db.list_products(conn)}

    query_words = _content_words(query)
    if not query_words:
        return {"ok": True, "orders": []}

    # Count how many query words each order's product title matches, then keep
    # only the best-matching tier so "midnight planter" prefers Midnight Planter
    # over every other Midnight product. The helper's newest-first order is kept.
    scored = []
    for order in candidates:
        title_words = _content_words(titles.get(order.product_id, ""))
        hits = sum(any(_words_match(q, t) for t in title_words) for q in query_words)
        scored.append((order, hits))
    best = max((hits for _, hits in scored), default=0)
    if best == 0:
        return {"ok": True, "orders": []}
    matches = [order for order, hits in scored if hits == best]
    return {"ok": True, "orders": [o.to_public_dict() for o in matches[:FIND_ORDER_LIMIT]]}


# Words that describe the purchase rather than the product, as in
# "the earmuffs I bought last week".
_FILLER_WORDS = {
    "a", "an", "the", "i", "my", "me", "we", "our", "that", "this", "those", "these",
    "it", "some", "from", "for", "of", "on", "in", "with", "and", "or", "to", "was",
    "bought", "buy", "ordered", "order", "orders", "purchased", "purchase", "got", "get",
    "last", "week", "month", "year", "ago", "yesterday", "today", "recently", "recent",
    "few", "days", "day", "weeks", "months", "one", "thing", "item",
}


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _FILLER_WORDS]


def _words_match(a: str, b: str) -> bool:
    """Fuzzy word match that tolerates plurals and small typos ("mug"/"mugs")."""
    if a == b:
        return True
    if min(len(a), len(b)) < 3:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.85


# ---------------------------------------------------------------------------
# Additional tools (HW1 Part A): gaps found while examining the agent.
# ---------------------------------------------------------------------------


def get_refund_status(ctx: AuthContext, order_id: int) -> dict[str, Any]:
    """Report the refunds requested on one order. Risk tier: read.

    Fills a gap in the lecture tools: an above-threshold refund is queued in
    the refunds table but the order keeps its "delivered" status, so get_order
    alone cannot tell a shopper whether their refund is pending or done.

    Scope matches get_order: shoppers see their own orders, merchants their
    store's orders, support any order. Scope is checked before any refund
    details are read.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found", ...}.
        Out of scope: permission_denied(...).
        Otherwise: {"ok": True, "order_id": int, "order_status": str,
        "refunds": [...], "count": int}, oldest refund first, where each refund
        is {"refund_id", "amount_usd", "status", "reason", "created_at",
        "next_step"}. An order with no refunds returns an empty list.
    """
    facts = load_facts()
    next_steps = {
        "auto_approved": (
            f"approved; the money goes back to the original payment method in "
            f"{facts['refund_processing_days_min']} to "
            f"{facts['refund_processing_days_max']} business days"
        ),
        "queued_for_approval": (
            f"above the ${facts['refund_auto_approve_threshold_usd']} auto-approval "
            f"threshold; waiting for a human support agent to review it, and no "
            f"money has moved yet"
        ),
    }
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"no order #{order_id}"}
        if not can_view_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not view refunds for order #{order_id}"
            )
        refunds = db.list_refunds_for_order(conn, order_id)
    return {
        "ok": True,
        "order_id": order_id,
        "order_status": order.status,
        "refunds": [
            {
                "refund_id": r.id,
                "amount_usd": r.amount_usd,
                "status": r.status,
                "reason": r.reason,
                "created_at": r.created_at.isoformat(),
                "next_step": next_steps.get(r.status, "status unknown; escalate to a human"),
            }
            for r in refunds
        ],
        "count": len(refunds),
    }


def audit_my_products(ctx: AuthContext) -> dict[str, Any]:
    """Check the merchant's own catalog for listing problems. Risk tier: read.

    Merchant only: the store comes from ctx.store_id, never from the model, so
    a merchant cannot audit another store. Other roles get permission_denied.

    Flags three problems, matching the seeded data-quality cases:
        - "missing_title": the title is empty or only whitespace.
        - "invalid_price": the price is zero or negative.
        - "duplicate_title": two or more products in the store share a title
          (case-insensitive). Reported once per group, listing every product.

    Returns:
        {"ok": True, "store_id": int, "products_checked": int,
        "issues": [...], "count": int} where each issue is
        {"issue": str, "product_ids": [int, ...], "detail": str}. Issues are
        sorted by their first product id. A clean catalog returns an empty
        list.
    """
    if ctx.role != "merchant" or ctx.store_id is None:
        return permission_denied(
            f"role '{ctx.role}' may not audit a catalog; only merchants can audit their own store"
        )
    with db.connection() as conn:
        products = db.list_products(conn, store_id=ctx.store_id)

    issues = []
    by_title: dict[str, list[int]] = {}
    for p in products:
        title = p.title.strip()
        if not title:
            issues.append({"issue": "missing_title", "product_ids": [p.id],
                           "detail": f"product {p.id} has no title"})
        else:
            by_title.setdefault(title.lower(), []).append(p.id)
        if p.price_cents <= 0:
            issues.append({"issue": "invalid_price", "product_ids": [p.id],
                           "detail": f"product {p.id} is priced at {'-' if p.price_cents < 0 else ''}${abs(p.price_usd):.2f}"})
    for ids in by_title.values():
        if len(ids) > 1:
            title = next(p.title for p in products if p.id == ids[0])
            issues.append({"issue": "duplicate_title", "product_ids": ids,
                           "detail": f"products {', '.join(map(str, ids))} share the title '{title}'"})
    issues.sort(key=lambda i: (i["product_ids"][0], i["issue"]))
    return {
        "ok": True,
        "store_id": ctx.store_id,
        "products_checked": len(products),
        "issues": issues,
        "count": len(issues),
    }
