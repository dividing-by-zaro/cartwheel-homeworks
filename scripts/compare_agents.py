"""Compare agent variants side by side from logs/agent_metrics.jsonl.

    uv run python -m scripts.compare_agents
    uv run python -m scripts.compare_agents --model gpt-4o-mini --since 2026-10-14

Three tables:
  1. Overview per variant and model: latency, tool use, failures, tokens, cost.
  2. Per tool: how often each tool is called, fails, and how long it takes.
  3. Paired turns: the same message from the same user sent to both variants
     (for example, the Part B cases run once with each), compared directly.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from observability.metrics import log_path


def load(path: Path, model: str | None, since: str | None, role: str | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"{path} does not exist yet. Chat with an agent first (uv run python -m agent.cli ...).")
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if model and r.get("model") != model:
            continue
        if since and r.get("ts", "") < since:
            continue
        if role and r.get("role") != role:
            continue
        records.append(r)
    return records


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    cells = [headers] + [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)), "  ".join("-" * w for w in widths)]
    for row in cells[1:]:
        lines.append("  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths))))
    return "\n".join(lines)


def overview(turns: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in turns:
        groups[(t["variant"], t["model"])].append(t)
    rows = []
    for (variant, model), ts in sorted(groups.items()):
        latencies = [t["latency_ms"] / 1000 for t in ts]
        tool_calls = sum(t["tool_calls"] for t in ts)
        tool_failures = sum(t["tool_failures"] for t in ts)
        costs = [t["cost_usd"] for t in ts if t["cost_usd"] is not None]
        rows.append([
            f"{variant} · {model}",
            len(ts),
            sum(t["status"] == "error" for t in ts),
            f"{pct(latencies, 0.5):.1f}s",
            f"{pct(latencies, 0.95):.1f}s",
            f"{statistics.mean(t['llm_calls'] for t in ts):.1f}",
            f"{tool_calls / len(ts):.1f}",
            f"{(tool_failures / tool_calls * 100) if tool_calls else 0:.0f}%",
            f"{statistics.mean(sum(a['ok'] for a in t.get('actions', [])) for t in ts):.1f}",
            f"{statistics.mean(t['input_tokens'] for t in ts):,.0f}",
            f"{statistics.mean(t['output_tokens'] for t in ts):,.0f}",
            money(statistics.mean(costs) if costs else None),
            money(sum(costs) if costs else None),
        ])
    return table(
        ["Variant · model", "Turns", "Errors", "p50", "p95", "LLM calls", "Tools/turn",
         "Tool fail", "Writes/turn", "In tok", "Out tok", "Cost/turn", "Total"],
        rows,
    )


def per_tool(calls: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in calls:
        groups[(c["variant"], c["tool"])].append(c)
    rows = []
    for (variant, tool), cs in sorted(groups.items()):
        failures = [c for c in cs if not c["ok"]]
        errors = Counter(c["error"] for c in failures).most_common(3)
        rows.append([
            variant,
            tool,
            len(cs),
            f"{len(failures) / len(cs) * 100:.0f}%",
            f"{statistics.mean(c['latency_ms'] for c in cs):.0f}ms",
            ", ".join(f"{e} ×{n}" for e, n in errors) or "—",
        ])
    return table(["Variant", "Tool", "Calls", "Fail", "Avg time", "Top errors"], rows)


def paired(turns: list[dict[str, Any]]) -> str:
    latest: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for t in turns:  # later turns overwrite earlier ones: compare the newest run of each
        key = (t["role"], t["user_id"], t["input"].strip().lower())
        latest[key][t["variant"]] = t
    rows = []
    for (role, user, text), by_variant in latest.items():
        if not {"tools", "sql"} <= set(by_variant):
            continue
        a, b = by_variant["tools"], by_variant["sql"]
        label = f"{role} {user}: {text[:48]}{'…' if len(text) > 48 else ''}"
        rows.append([
            label,
            f"{a['latency_ms'] / 1000:.1f}s / {b['latency_ms'] / 1000:.1f}s",
            f"{a['tool_calls']}({a['tool_failures']}) / {b['tool_calls']}({b['tool_failures']})",
            f"{a['total_tokens']:,} / {b['total_tokens']:,}",
            f"{money(a['cost_usd'])} / {money(b['cost_usd'])}",
        ])
    if not rows:
        return "No message has been sent to both variants yet (same role, user, and text)."
    return table(["Message (tools / sql)", "Latency", "Tool calls (failed)", "Tokens", "Cost"], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=None, help="metrics file (default: logs/agent_metrics.jsonl)")
    parser.add_argument("--model", default=None, help="only this model")
    parser.add_argument("--since", default=None, help="only records at or after this ISO date, e.g. 2026-10-14")
    parser.add_argument("--role", choices=["shopper", "merchant", "support"], default=None)
    args = parser.parse_args()

    records = load(args.log or log_path(), args.model, args.since, args.role)
    turns = [r for r in records if r["type"] == "turn"]
    calls = [r for r in records if r["type"] == "tool_call"]
    if not turns:
        raise SystemExit("No turns match these filters.")
    print("\nOverview (p50/p95 = median and 95th-percentile latency per message)\n")
    print(overview(turns))
    print("\nPer tool\n")
    print(per_tool(calls) if calls else "No tool calls.")
    print("\nPaired messages\n")
    print(paired(turns))
    print()


if __name__ == "__main__":
    main()
