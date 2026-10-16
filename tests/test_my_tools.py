"""Tests for the additional HW1 tools: get_refund_status and audit_my_products.

Offline, no LLM, no API keys.
"""

from __future__ import annotations

import sqlite3

from agent import tools
from agent.agent import TOOLS_BY_ROLE, issue_refund_logic
from agent.auth import AuthContext

SHOPPER_1 = AuthContext(user_id=1, role="shopper")
SHOPPER_2 = AuthContext(user_id=2, role="shopper")
MERCHANT_STORE_1 = AuthContext(user_id=9001, role="merchant", store_id=1)
MERCHANT_STORE_2 = AuthContext(user_id=9002, role="merchant", store_id=2)
SUPPORT = AuthContext(user_id=9501, role="support")


def _tool_names(role: str) -> set[str]:
    return {t.name for t in TOOLS_BY_ROLE[role]}


def test_tools_are_registered_for_the_right_roles() -> None:
    for role in ("shopper", "merchant", "support"):
        assert "get_refund_status" in _tool_names(role)
    assert "audit_my_products" in _tool_names("merchant")
    assert "audit_my_products" not in _tool_names("shopper")
    assert "audit_my_products" not in _tool_names("support")


# ---------------------------------------------------------------------------
# get_refund_status
# ---------------------------------------------------------------------------


def test_refund_status_with_no_refunds(world: dict) -> None:
    result = tools.get_refund_status(SHOPPER_1, 4127)
    assert result == {
        "ok": True,
        "order_id": 4127,
        "order_status": "delivered",
        "refunds": [],
        "count": 0,
    }


def test_refund_status_shows_queued_refund_above_threshold(world_copy) -> None:
    queued = issue_refund_logic(SHOPPER_1, 4455, 240.0, "arrived cracked")
    assert queued["status"] == "queued_for_approval"

    result = tools.get_refund_status(SHOPPER_1, 4455)
    assert result["ok"] is True
    # The order itself still reads "delivered"; only the refund record knows.
    assert result["order_status"] == "delivered"
    assert result["count"] == 1
    refund = result["refunds"][0]
    assert refund["refund_id"] == queued["refund_id"]
    assert refund["amount_usd"] == 240.0
    assert refund["status"] == "queued_for_approval"
    assert "human" in refund["next_step"]


def test_refund_status_shows_auto_approved_refund(world_copy) -> None:
    approved = issue_refund_logic(SHOPPER_1, 4127, 84.0, "changed my mind")
    assert approved["status"] == "auto_approved"

    result = tools.get_refund_status(SHOPPER_1, 4127)
    assert result["order_status"] == "refunded"
    assert result["refunds"][0]["status"] == "auto_approved"
    assert "business days" in result["refunds"][0]["next_step"]


def test_refund_status_scope(world_copy) -> None:
    issue_refund_logic(SHOPPER_1, 4455, 240.0, "arrived cracked")

    for outsider in (SHOPPER_2, MERCHANT_STORE_2):
        denied = tools.get_refund_status(outsider, 4455)
        assert denied["ok"] is False
        assert denied["error"] == "permission_denied"
        assert "refunds" not in denied  # nothing about the refund leaks

    assert tools.get_refund_status(MERCHANT_STORE_1, 4455)["count"] == 1
    assert tools.get_refund_status(SUPPORT, 4455)["count"] == 1


def test_refund_status_unknown_order(world: dict) -> None:
    assert tools.get_refund_status(SUPPORT, 999_999)["error"] == "not_found"


# ---------------------------------------------------------------------------
# audit_my_products
# ---------------------------------------------------------------------------


def _issues_by_type(result: dict) -> dict[str, list[list[int]]]:
    grouped: dict[str, list[list[int]]] = {}
    for issue in result["issues"]:
        grouped.setdefault(issue["issue"], []).append(issue["product_ids"])
    return grouped


def test_audit_finds_the_seeded_product_defects(world: dict) -> None:
    result = tools.audit_my_products(MERCHANT_STORE_1)
    assert result["ok"] is True
    assert result["store_id"] == 1
    assert result["products_checked"] == 40
    assert result["count"] == len(result["issues"])

    grouped = _issues_by_type(result)
    assert [3] in grouped["missing_title"]
    assert [4] in grouped["invalid_price"]
    # The seed renamed product 2 to collide with product 1. The dev catalog also
    # reuses generated titles, so that group can hold more products than 1 and 2.
    assert any({1, 2} <= set(group) for group in grouped["duplicate_title"])


def test_audit_covers_every_seeded_product_case_for_the_store(world: dict) -> None:
    conn = sqlite3.connect(world["db"])
    try:
        cases = conn.execute(
            "SELECT d.issue_type, d.entity_id FROM data_quality_cases d "
            "JOIN products p ON p.id = d.entity_id "
            "WHERE d.entity_type = 'product' AND p.store_id = 1"
        ).fetchall()
    finally:
        conn.close()
    assert cases, "seed should include product defects in store 1"

    result = tools.audit_my_products(MERCHANT_STORE_1)
    flagged = {(i["issue"], pid) for i in result["issues"] for pid in i["product_ids"]}
    for issue_type, product_id in cases:
        assert (issue_type, product_id) in flagged


def test_audit_stays_inside_the_merchants_store(world: dict) -> None:
    store_1_ids = {pid for i in tools.audit_my_products(MERCHANT_STORE_1)["issues"] for pid in i["product_ids"]}
    result = tools.audit_my_products(MERCHANT_STORE_2)
    assert result["ok"] is True
    assert result["store_id"] == 2
    store_2_ids = {pid for i in result["issues"] for pid in i["product_ids"]}
    assert store_1_ids.isdisjoint(store_2_ids)


def test_audit_is_merchant_only(world: dict) -> None:
    for ctx in (SHOPPER_1, SUPPORT):
        denied = tools.audit_my_products(ctx)
        assert denied["ok"] is False
        assert denied["error"] == "permission_denied"
