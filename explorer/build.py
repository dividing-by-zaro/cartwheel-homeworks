"""Build explorer/cartwheel-explorer.html from the seeded Cartwheel world and the agent code.

Run from the repository root after seeding (rerun after a reseed, a refund,
or a cancellation to see the current state):

    uv run python explorer/build.py
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
from pathlib import Path

import tiktoken
import yaml

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
TABLES = ["stores", "products", "users", "orders", "refunds", "escalations", "data_quality_cases"]
FACT_KEYS = [
    "return_window_days",
    "refund_auto_approve_threshold_usd",
    "restocking_fee_max_percent",
    "support_escalation_sla_hours",
    "dispute_window_days",
]


def load_tables(db_path: Path) -> tuple[dict, dict]:
    conn = sqlite3.connect(db_path)
    try:
        tables = {}
        for name in TABLES:
            cursor = conn.execute(f"SELECT * FROM {name} ORDER BY 1")
            cols = [c[0] for c in cursor.description]
            tables[name] = {"cols": cols, "rows": [list(r) for r in cursor.fetchall()]}
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    finally:
        conn.close()
    return tables, meta


def load_policies(directory: Path) -> dict:
    rows = []
    for path in sorted(directory.glob("*.md")):
        _, raw_front, body = path.read_text().split("---", 2)
        front = yaml.safe_load(raw_front)
        rows.append([front["policy_id"], front["title"], front.get("audience", "all"), body.strip()])
    return {"cols": ["id", "title", "audience", "body"], "rows": rows}


def load_agents() -> dict:
    """Exactly what each agent variant sends the model: prompt and tool definitions.

    Token counts use the o200k_base tokenizer (gpt-4o-mini). Each tool is
    measured as the function definition sent to the API.
    """
    from agent.agent import SYSTEM_PROMPT_TEMPLATE, TOOLS_BY_ROLE
    from agent.sql_agent import SQL_SYSTEM_PROMPT_TEMPLATE, SQL_TOOLS

    enc = tiktoken.get_encoding("o200k_base")
    tools: dict[str, dict] = {}

    def add(tool) -> str:
        definition = {"type": "function", "name": tool.name, "description": tool.description,
                      "parameters": tool.params_json_schema}
        tools[tool.name] = {**definition, "tokens": len(enc.encode(json.dumps(definition)))}
        return tool.name

    return {
        "tokenizer": "o200k_base (gpt-4o-mini)",
        "tools": tools,
        "variants": {
            "tools": {
                "prompt": SYSTEM_PROMPT_TEMPLATE,
                "prompt_tokens": len(enc.encode(SYSTEM_PROMPT_TEMPLATE)),
                "roles": {role: [add(t) for t in role_tools] for role, role_tools in TOOLS_BY_ROLE.items()},
            },
            "sql": {
                "prompt": SQL_SYSTEM_PROMPT_TEMPLATE,
                "prompt_tokens": len(enc.encode(SQL_SYSTEM_PROMPT_TEMPLATE)),
                "roles": {role: [add(t) for t in SQL_TOOLS] for role in TOOLS_BY_ROLE},
            },
        },
    }


HEADLINE_METRICS = {"tokens": "total_tokens", "time": "latency_ms", "cost": "cost_usd"}


def headline(turns: list[dict]) -> dict:
    """SQL agent vs course agent, matched on model and message.

    For each model, compare the two agents on the messages both answered:
    ratio = SQL total / course total. The headline is the geometric mean of
    the per-model ratios, so every model counts equally and a 2x and a 0.5x
    cancel out. The 95% range comes from a bootstrap that resamples whole
    conversations (messages in one conversation are not independent).
    """
    answered = {(t["model"], t["variant"], t["conversation"], t["turn_index"]): t
                for t in turns if t["status"] == "ok"}
    by_conv: dict[str, dict[int, list]] = {}
    for (model, variant, conv, idx), t in answered.items():
        sql = answered.get((model, "sql", conv, idx))
        if variant == "tools" and sql:
            by_conv.setdefault(model, {}).setdefault(conv, []).append((t, sql))
    models = sorted(by_conv)
    convs = sorted({c for per in by_conv.values() for c in per})

    def ratios(sample: list[int], key: str) -> list[float]:
        out = []
        for model in models:
            rows = [pair for c in sample for pair in by_conv[model].get(c, [])]
            tools = sum(t[key] or 0 for t, _ in rows)
            sql = sum(s[key] or 0 for _, s in rows)
            if tools > 0 and sql > 0:
                out.append(sql / tools)
        return out

    def geomean(xs: list[float]) -> float:
        return math.exp(sum(math.log(x) for x in xs) / len(xs))

    rng = random.Random(20261015)
    samples = [[rng.choice(convs) for _ in convs] for _ in range(2000)]
    metrics = {}
    for name, key in HEADLINE_METRICS.items():
        per_model = []
        for model in models:
            rows = [pair for c in convs for pair in by_conv[model].get(c, [])]
            tools = sum(t[key] or 0 for t, _ in rows) / len(rows)
            sql = sum(s[key] or 0 for _, s in rows) / len(rows)
            per_model.append({"model": model, "tools": tools, "sql": sql, "ratio": sql / tools if tools else None,
                              "pairs": len(rows)})
        boot = sorted(geomean(r) for r in (ratios(s, key) for s in samples) if r)
        metrics[name] = {
            "ratio": geomean([m["ratio"] for m in per_model if m["ratio"]]),
            "low": boot[int(0.025 * len(boot))],
            "high": boot[int(0.975 * len(boot)) - 1],
            "models_higher": sum(1 for m in per_model if m["ratio"] and m["ratio"] > 1),
            "per_model": per_model,
        }
    return {
        "pairs": sum(len(rows) for per in by_conv.values() for rows in per.values()),
        "conversations": len(convs),
        "models": len(models),
        "metrics": metrics,
    }


def load_replay() -> dict | None:
    """The latest replay (scripts/replay_conversations.py) plus the original chats it replayed."""
    latest = ROOT / "logs" / "replays" / "latest.jsonl"
    if not latest.exists():
        return None
    records = [json.loads(line) for line in latest.read_text().splitlines() if line.strip()]
    if not records:
        return None
    replay_id = records[0]["replay_id"]
    conversations = json.loads((ROOT / "logs" / "replays" / replay_id / "conversations.json").read_text())

    def turn(r: dict) -> dict:
        if r.get("status") == "ok" and not (r.get("output") or "").strip():
            # Logged before empty replies had their own status.
            r = {**r, "status": "empty"}
        return {k: r.get(k) for k in (
            "model", "variant", "conversation", "turn_index", "status", "error", "output", "latency_ms", "prompt_version",
            "llm_calls", "llm_latency_ms", "tool_latency_ms", "tool_calls", "tool_failures", "tool_errors", "actions", "input_tokens", "cached_tokens",
            "output_tokens", "total_tokens", "cost_usd")}

    def tool_call(r: dict) -> dict:
        result = json.dumps(r.get("result"), default=str)
        return {**{k: r.get(k) for k in ("model", "variant", "conversation", "turn_index", "tool", "arguments",
                                          "ok", "error", "latency_ms")},
                "result": result if len(result) <= 1500 else result[:1500] + "…"}

    # The original replies from your own CLI chats, for side-by-side reading.
    sources = {c["source_session_id"]: i for i, c in enumerate(conversations)}
    originals, original_calls = [], []
    metrics_log = ROOT / "logs" / "agent_metrics.jsonl"
    if metrics_log.exists():
        for line in metrics_log.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("session_id") in sources and not r.get("replay_id"):
                r = {**r, "conversation": sources[r["session_id"]]}
                if r["type"] == "turn":
                    originals.append(turn(r))
                elif r["type"] == "tool_call":
                    original_calls.append(tool_call(r))

    return {
        "replay_id": replay_id,
        "conversations": conversations,
        "turns": (turns := [turn(r) for r in records if r["type"] == "turn"]),
        "headline": headline(turns),
        "tool_calls": [tool_call(r) for r in records if r["type"] == "tool_call"],
        "originals": originals,
        "original_tool_calls": original_calls,
    }


def main() -> None:
    db_path = ROOT / "data" / "cartwheel.db"
    if not db_path.exists():
        raise SystemExit("data/cartwheel.db not found. Run: uv run python -m seed.generate")
    tables, meta = load_tables(db_path)
    tables["policies"] = load_policies(ROOT / "data" / "policies")
    facts = yaml.safe_load((ROOT / "facts.yaml").read_text())
    data = {"tables": tables, "meta": meta, "facts": {k: facts.get(k) for k in FACT_KEYS}, "agents": load_agents(),
            "replay": load_replay()}

    # Escape "</" so record text can never close the <script> tag early.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", payload, 1)
    out = HERE / "cartwheel-explorer.html"
    out.write_text(html)
    print(f"Wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
