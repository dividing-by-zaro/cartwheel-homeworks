"""Part C test: does the agent refuse a second refund on the same order?

    uv run python -m scripts.check_duplicate_refund
    uv run python -m scripts.check_duplicate_refund --model claude-haiku-4-5 --runs 3

Each run seeds a fresh world (data/cartwheel.db is never touched), then:
  chat 1: shopper 1 asks for a refund on order 4455 and gives a reason;
  chat 2: a brand-new chat (no memory of chat 1) asks for a refund on 4455 again.
It runs once with the prompt before the refund-status rule and once after,
counts the refunds queued on order 4455 (1 is correct, 2 is a duplicate), and
writes the transcripts to hw1-part-c.md as before/after evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from agent.config import REPO_ROOT

RULE_START = "- Before issuing a refund, call get_refund_status for that order."
CHATS = [
    ["hi can i get a refund for 4455 please", "it had a dead chinchilla inside it unfortunately"],
    ["hi can i get a refund for order 4455 please", "it arrived with a big crack in it"],
]


def before_template() -> str:
    """The system prompt as it was before the refund-status rule was added."""
    from agent.agent import SYSTEM_PROMPT_TEMPLATE

    start = SYSTEM_PROMPT_TEMPLATE.index(RULE_START)
    end = SYSTEM_PROMPT_TEMPLATE.index("instead.\n", start) + len("instead.\n")
    return SYSTEM_PROMPT_TEMPLATE[:start] + SYSTEM_PROMPT_TEMPLATE[end:]


def tool_calls(new_items) -> list[dict]:
    outputs = {i.call_id: i.output for i in new_items if i.type == "tool_call_output_item"}
    calls = []
    for item in new_items:
        if item.type == "tool_call_item":
            raw = item.raw_item
            args = raw.get("arguments") if isinstance(raw, dict) else getattr(raw, "arguments", None)
            calls.append({"name": item.tool_name, "arguments": json.loads(args) if isinstance(args, str) else args,
                          "result": outputs.get(getattr(raw, "call_id", None) or (raw.get("call_id") if isinstance(raw, dict) else None))})
    return calls


async def run_once(label: str, template: str, model: str, base: Path, work: Path, index: int) -> dict:
    from agents import RunConfig, SQLiteSession

    from agent.agent import build_agent, prompt_version
    from agent.auth import AuthContext
    from observability.metrics import run_with_metrics

    db = work / f"{label}-{index}.db"
    shutil.copy(base, db)
    os.environ["CARTWHEEL_DB"] = str(db)
    ctx = AuthContext(user_id=1, role="shopper")
    transcript = []
    for chat_no, messages in enumerate(CHATS, 1):
        agent = build_agent(ctx, model, prompt_template=template)
        session = SQLiteSession(f"partc-{label}-{index}-{chat_no}")  # fresh memory for each chat
        for turn, message in enumerate(messages):
            result, _ = await run_with_metrics(
                agent, message, context=ctx, variant="tools", model=model,
                session_id=f"partc-{label}-{index}-{chat_no}", turn_index=turn,
                prompt_version=prompt_version(template), tags={"experiment": "duplicate-refund", "prompt": label},
                session=session, max_turns=12, run_config=RunConfig(tracing_disabled=True),
            )
            transcript.append({"chat": chat_no, "user": message, "tools": tool_calls(result.new_items),
                               "agent": result.final_output})
    conn = sqlite3.connect(db)
    try:
        refunds = conn.execute("SELECT id, amount_cents, status, reason FROM refunds WHERE order_id = 4455").fetchall()
    finally:
        conn.close()
    return {"label": label, "run": index, "refunds": refunds, "transcript": transcript}


def write_report(results: list[dict], model: str) -> Path:
    lines = ["# Part C: duplicate refunds", "",
             f"Model: `{model}`. Each run starts from fresh data. Chat 2 is a new conversation with no memory of chat 1.",
             "Correct outcome: exactly 1 refund queued on order 4455.", "", "| Prompt | Run | Refunds on 4455 | Result |",
             "|---|---|---|---|"]
    for r in results:
        n = len(r["refunds"])
        lines.append(f"| {r['label']} | {r['run']} | {n} | {'correct' if n == 1 else 'duplicate refund' if n > 1 else 'no refund'} |")
    for r in results:
        lines += ["", f"## {r['label']} prompt, run {r['run']}", ""]
        for t in r["transcript"]:
            lines.append(f"**Chat {t['chat']}, shopper:** {t['user']}")
            for c in t["tools"]:
                lines.append(f"- tool `{c['name']}({json.dumps(c['arguments'])})` → `{str(c['result'])[:300]}`")
            lines += ["", f"**Agent:** {t['agent']}", ""]
    path = REPO_ROOT / "hw1-part-c.md"
    path.write_text("\n".join(lines) + "\n")
    return path


async def main_async(model: str, runs: int) -> None:
    from agent.agent import SYSTEM_PROMPT_TEMPLATE
    from observability.instrument import load_env
    from seed.generate import generate_world

    load_env()
    work = Path(tempfile.mkdtemp(prefix="partc-"))
    os.environ["CARTWHEEL_POLICIES_DIR"] = str(work / "policies")
    os.environ.setdefault("CARTWHEEL_METRICS_LOG", str(REPO_ROOT / "logs" / "experiments.jsonl"))
    generate_world(scale="dev", db_path=work / "base.db", policies_dir=work / "policies")
    results = []
    for label, template in (("before", before_template()), ("after", SYSTEM_PROMPT_TEMPLATE)):
        for i in range(1, runs + 1):
            r = await run_once(label, template, model, work / "base.db", work, i)
            print(f"{label:6} run {i}: {len(r['refunds'])} refund(s) on 4455", flush=True)
            results.append(r)
    shutil.rmtree(work, ignore_errors=True)
    print(f"Wrote {write_report(results, model).relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="default: $CARTWHEEL_MODEL or gpt-4o-mini")
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    from agent.agent import DEFAULT_MODEL

    asyncio.run(main_async(args.model or os.environ.get("CARTWHEEL_MODEL") or DEFAULT_MODEL, args.runs))


if __name__ == "__main__":
    main()
