"""Offline tests for scripts/replay_conversations.py (no model calls)."""

from __future__ import annotations

import json

from scripts import replay_conversations as replay


def _write(path, records) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_load_conversations_groups_cli_turns_in_order(tmp_path) -> None:
    log = tmp_path / "agent_metrics.jsonl"
    base = {"type": "turn", "role": "shopper", "user_id": 1, "store_id": None}
    _write(log, [
        {**base, "session_id": "cli-tools-shopper-1-2", "ts": "2026-10-14T02", "turn_index": 1, "input": "second"},
        {**base, "session_id": "cli-tools-shopper-1-2", "ts": "2026-10-14T01", "turn_index": 0, "input": "first"},
        {**base, "session_id": "cli-tools-shopper-1-1", "ts": "2026-10-14T00", "turn_index": 0, "input": "earlier chat"},
        {**base, "session_id": "cli-tools-shopper-1-3", "ts": "2026-10-14T03", "turn_index": 0, "input": "a replay",
         "replay_id": "x"},
        {"type": "tool_call", "session_id": "cli-tools-shopper-1-1", "ts": "2026-10-14T00"},
        {**base, "session_id": "replay-abc", "ts": "2026-10-14T04", "turn_index": 0, "input": "not a cli chat"},
    ])
    conversations = replay.load_conversations(log)
    assert [c["messages"] for c in conversations] == [["earlier chat"], ["first", "second"]]
    assert conversations[1]["source_session_id"] == "cli-tools-shopper-1-2"


def test_combine_keeps_only_the_last_attempt(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(replay, "REPLAYS_DIR", tmp_path)
    out_dir = tmp_path / "20261015-000000"
    out_dir.mkdir()
    key = {"model": "gemini-4-argon", "variant": "sql", "conversation": 0, "turn_index": 0}
    _write(out_dir / "gemini-4-argon__sql.jsonl", [
        {"type": "turn", **key, "attempt": 1, "status": "error"},
        {"type": "tool_call", **key, "attempt": 2, "ok": True},
        {"type": "turn", **key, "attempt": 2, "status": "ok"},
        {"type": "turn", **key, "turn_index": 1, "attempt": 1, "status": "ok"},
    ])
    combined = replay.combine(out_dir)
    kept = [json.loads(line) for line in combined.read_text().splitlines()]
    assert [(r["type"], r["turn_index"], r["attempt"]) for r in kept] == [
        ("tool_call", 0, 2), ("turn", 0, 2), ("turn", 1, 1)]
    assert (tmp_path / "latest.jsonl").read_text() == combined.read_text()


def test_headline_matches_messages_and_averages_models_equally() -> None:
    import importlib.util
    from pathlib import Path

    import pytest

    path = Path(__file__).parent.parent / "explorer" / "build.py"
    if not path.exists():  # explorer/ is a local, untracked tool
        pytest.skip("explorer/build.py not present")
    spec = importlib.util.spec_from_file_location("explorer_build", path)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)

    def turn(model, variant, conv, tokens, status="ok"):
        return {"model": model, "variant": variant, "conversation": conv, "turn_index": 0, "status": status,
                "total_tokens": tokens, "latency_ms": tokens, "cost_usd": tokens / 1000}

    turns = [
        # Model A: SQL uses 2x the tokens; model B: 0.5x. Equal weight -> geometric mean 1.0.
        *[turn("a", "tools", c, 100) for c in range(4)], *[turn("a", "sql", c, 200) for c in range(4)],
        *[turn("b", "tools", c, 1000) for c in range(4)], *[turn("b", "sql", c, 500) for c in range(4)],
        # Unmatched and failed messages are ignored.
        turn("a", "sql", 9, 99999), turn("b", "tools", 5, 1, status="error"), turn("b", "sql", 5, 99999),
    ]
    h = build.headline(turns)
    assert h["pairs"] == 8 and h["models"] == 2 and h["conversations"] == 4
    tokens = h["metrics"]["tokens"]
    assert abs(tokens["ratio"] - 1.0) < 1e-9
    assert tokens["models_higher"] == 1
    assert {m["model"]: round(m["ratio"], 3) for m in tokens["per_model"]} == {"a": 2.0, "b": 0.5}
    assert tokens["low"] <= tokens["ratio"] <= tokens["high"]
