"""Replay recorded chat conversations through every model and agent variant.

    uv run python -m scripts.replay_conversations
    uv run python -m scripts.replay_conversations --models gpt-4o-mini claude-haiku-5-5 --variants sql
    uv run python -m scripts.replay_conversations --replace-in <REPLAY_ID> --models gemini-4-argon --concurrency 1
    uv run python -m scripts.replay_conversations --append-to <REPLAY_ID> --conversations scripts/extra_conversations.json

Reads the conversations you had in the CLI from logs/agent_metrics.jsonl and
sends the same user messages, in order, to each (model, variant) pair. Each
pair runs in its own worker process, and every conversation starts from a
freshly seeded copy of the world, so a refund in one conversation never
affects another and data/cartwheel.db is never touched.

Results go to logs/replays/<replay_id>.jsonl and logs/replays/latest.jsonl,
in the same record format as the CLI metrics. Rebuild the explorer to see
them in the dashboard: uv run python explorer/build.py

Messages are replayed as recorded. When a model's reply differs from the
original, a later message ("yes, do it") can read oddly; that is part of
what the comparison shows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import REPO_ROOT

DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-5-nano",
    "claude-haiku-4-5",
    "claude-haiku-5-5",
    "gemini-4-argon",
    "claude-opus-5-5",
    "gpt-6-astra",
]
VARIANTS = ["tools", "sql"]
MAX_TURNS = 12
TURN_TIMEOUT_SECONDS = 180
# Rate limits and dropped connections are retried with exponential backoff.
# Only each message's last attempt is kept in the combined results.
MAX_ATTEMPTS = 5
RETRY_BASE_SECONDS = 15
TRANSIENT_ERRORS = ("RateLimit", "InternalServer", "ServiceUnavailable", "APIConnection", "Timeout", "Overloaded")
REPLAYS_DIR = REPO_ROOT / "logs" / "replays"


def load_conversations(log: Path) -> list[dict[str, Any]]:
    """Group the CLI turns in the metrics log into conversations, oldest first."""
    sessions: dict[str, dict[str, Any]] = {}
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") != "turn" or r.get("replay_id") or not str(r.get("session_id", "")).startswith("cli-"):
            continue
        conv = sessions.setdefault(r["session_id"], {
            "source_session_id": r["session_id"],
            "role": r["role"],
            "user_id": r["user_id"],
            "store_id": r["store_id"],
            "started": r["ts"],
            "turns": [],
        })
        conv["turns"].append((r.get("turn_index") or 0, r["input"]))
    conversations = sorted(sessions.values(), key=lambda c: c["started"])
    for conv in conversations:
        conv["messages"] = [text for _, text in sorted(conv.pop("turns"), key=lambda t: t[0])]
    return conversations


def scripted_conversations(path: Path, offset: int = 0) -> list[dict[str, Any]]:
    """Load hand-written conversations; merchants get their store from the users table."""
    from agent import db

    items = json.loads(path.read_text())
    with db.connection() as conn:
        out = []
        for i, item in enumerate(items):
            user = db.get_user(conn, item["user_id"])
            if user is None or user.role != item["role"]:
                raise SystemExit(f"{path}: user {item['user_id']} is not a {item['role']}")
            out.append({
                "source_session_id": f"scripted-{offset + i}",
                "role": user.role,
                "user_id": user.id,
                "store_id": user.store_id,
                "started": "",
                "topic": item.get("topic"),
                "messages": item["messages"],
            })
    return out


# ---------------------------------------------------------------------------
# Worker: one (model, variant) pair, every conversation
# ---------------------------------------------------------------------------


async def run_worker(model: str, variant: str, out_dir: Path, max_cost: float) -> None:
    from agents import RunConfig, SQLiteSession

    from agent.agent import build_agent, prompt_version
    from agent.auth import AuthContext
    from agent.sql_agent import build_sql_agent, sql_prompt_version
    from observability.instrument import load_env
    from observability.metrics import run_with_metrics

    load_env()
    world = Path(os.environ.get("CARTWHEEL_REPLAY_WORLD", out_dir / "world"))
    os.environ["CARTWHEEL_POLICIES_DIR"] = str(world / "policies")
    conversations = json.loads((out_dir / "conversations.json").read_text())
    replay_id = out_dir.name
    version = sql_prompt_version() if variant == "sql" else prompt_version()
    spent = 0.0

    start = int(os.environ.get("CARTWHEEL_REPLAY_START", "0"))
    for index, conv in enumerate(conversations):
        if index < start:
            continue
        db_copy = world / f"{model}__{variant}__{index}.db"
        shutil.copy(world / "base.db", db_copy)
        os.environ["CARTWHEEL_DB"] = str(db_copy)
        ctx = AuthContext(user_id=conv["user_id"], role=conv["role"], store_id=conv["store_id"])
        agent = build_sql_agent(ctx, model) if variant == "sql" else build_agent(ctx, model)
        session_id = f"replay-{replay_id}-{model}-{variant}-{index}"
        session = SQLiteSession(session_id)
        for turn_index, message in enumerate(conv["messages"]):
            if spent > max_cost:
                print(f"STOP cost cap ${max_cost:.2f} reached", flush=True)
                return
            for attempt in range(1, MAX_ATTEMPTS + 1):
                tags = {
                    "replay_id": replay_id,
                    "conversation": index,
                    "source_session_id": conv["source_session_id"],
                    "attempt": attempt,
                }
                history = len(await session.get_items())
                try:
                    _, turn = await asyncio.wait_for(
                        run_with_metrics(
                            agent, message, context=ctx, variant=variant, model=model,
                            session_id=session_id, turn_index=turn_index, prompt_version=version, tags=tags,
                            session=session, max_turns=MAX_TURNS, run_config=RunConfig(tracing_disabled=True),
                        ),
                        TURN_TIMEOUT_SECONDS,
                    )
                    spent += turn["cost_usd"] or 0
                    print(f"OK   conv {index} turn {turn_index} {turn['latency_ms'] / 1000:.1f}s "
                          f"tools={turn['tool_calls']}({turn['tool_failures']}) ${spent:.4f}", flush=True)
                    break
                except (Exception, asyncio.TimeoutError) as exc:  # logged as an error turn
                    transient = any(word in type(exc).__name__ for word in TRANSIENT_ERRORS)
                    print(f"FAIL conv {index} turn {turn_index} attempt {attempt} {type(exc).__name__}: "
                          f"{str(exc)[:160]}", flush=True)
                    if not transient or attempt == MAX_ATTEMPTS:
                        break
                    # Undo what the failed attempt added to the conversation, then wait and retry.
                    while len(await session.get_items()) > history:
                        await session.pop_item()
                    await asyncio.sleep(RETRY_BASE_SECONDS * 2 ** (attempt - 1))
        db_copy.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


async def run_all(models: list[str], variants: list[str], out_dir: Path, world: Path, concurrency: int,
                  max_cost: float, start: int = 0, log_suffix: str = "") -> list[Path]:
    semaphore = asyncio.Semaphore(concurrency)
    outputs = []

    async def one(model: str, variant: str) -> None:
        log = out_dir / f"{model}__{variant}{log_suffix}.jsonl"
        if not log_suffix:
            log.unlink(missing_ok=True)  # a rerun replaces this pair's earlier results
        outputs.append(log)
        async with semaphore:
            env = {**os.environ, "CARTWHEEL_METRICS_LOG": str(log), "CARTWHEEL_REPLAY_WORLD": str(world),
                   "CARTWHEEL_REPLAY_START": str(start)}
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "scripts.replay_conversations", "--worker", model, variant, str(out_dir),
                "--max-cost-per-pair", str(max_cost),
                cwd=REPO_ROOT, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode().rstrip()
                if line.startswith(("OK", "FAIL", "STOP", "Traceback")) or "Error" in line[:40]:
                    print(f"[{model} · {variant}] {line}", flush=True)
            await proc.wait()
            print(f"[{model} · {variant}] done (exit {proc.returncode})", flush=True)

    await asyncio.gather(*(one(m, v) for m in models for v in variants))
    return outputs


def combine(out_dir: Path) -> Path:
    """Merge every pair's log into logs/replays/<id>.jsonl and latest.jsonl.

    A retried message keeps only its last attempt, so a rate-limited try that
    succeeded later does not count as an error.
    """
    records = []
    for log in sorted(out_dir.glob("*__*.jsonl")):
        records += [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    key = lambda r: (r["model"], r["variant"], r.get("conversation"), r.get("turn_index"))
    last: dict[tuple, int] = {}
    for r in records:
        last[key(r)] = max(last.get(key(r), 0), r.get("attempt", 1))
    kept = [r for r in records if r.get("attempt", 1) == last[key(r)]]
    combined = REPLAYS_DIR / f"{out_dir.name}.jsonl"
    combined.write_text("".join(json.dumps(r, default=str) + "\n" for r in kept))
    shutil.copy(combined, REPLAYS_DIR / "latest.jsonl")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--log", type=Path, default=REPO_ROOT / "logs" / "agent_metrics.jsonl",
                        help="where the recorded CLI conversations are")
    parser.add_argument("--concurrency", type=int, default=7, help="worker processes at once")
    parser.add_argument("--max-cost-per-pair", type=float, default=3.0,
                        help="stop a model/variant pair once it has spent this many dollars")
    parser.add_argument("--replace-in", metavar="REPLAY_ID", default=None,
                        help="rerun the chosen models/variants inside an existing replay and replace their results")
    parser.add_argument("--append-to", metavar="REPLAY_ID", default=None,
                        help="add the --conversations file to an existing replay and run only those")
    parser.add_argument("--conversations", type=Path, default=None,
                        help="JSON list of {role, user_id, messages[, topic]} to replay instead of the CLI log")
    parser.add_argument("--worker", nargs=3, metavar=("MODEL", "VARIANT", "OUT_DIR"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        model, variant, out_dir = args.worker
        asyncio.run(run_worker(model, variant, Path(out_dir), args.max_cost_per_pair))
        return

    from agent.agent import model_route
    from observability.instrument import load_env
    from seed.generate import generate_world

    load_env()
    for model in args.models:
        model_route(model)  # fail fast on a missing key, before any spending
    start, log_suffix = 0, ""
    if args.append_to:
        if not args.conversations:
            raise SystemExit("--append-to needs --conversations")
        replay_id = args.append_to
        out_dir = REPLAYS_DIR / replay_id
        conversations = json.loads((out_dir / "conversations.json").read_text())
        start = len(conversations)
        conversations += scripted_conversations(args.conversations, offset=start)
        (out_dir / "conversations.json").write_text(json.dumps(conversations, indent=2))
        log_suffix = f"__from{start}"
    elif args.replace_in:
        replay_id = args.replace_in
        out_dir = REPLAYS_DIR / replay_id
        conversations = json.loads((out_dir / "conversations.json").read_text())
    else:
        conversations = (scripted_conversations(args.conversations) if args.conversations
                         else load_conversations(args.log))
        if not conversations:
            raise SystemExit(f"No CLI conversations found in {args.log}")
        replay_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = REPLAYS_DIR / replay_id
        out_dir.mkdir(parents=True)
        (out_dir / "conversations.json").write_text(json.dumps(conversations, indent=2))
    # Each invocation seeds its own world, so a rerun never disturbs a replay still in progress.
    world = out_dir / f"world-{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
    world.mkdir()
    generate_world(scale="dev", db_path=world / "base.db", policies_dir=world / "policies")
    turns = sum(len(c["messages"]) for c in conversations[start:])
    print(f"Replay {replay_id}: {len(conversations) - start} conversations, {turns} messages, "
          f"{len(args.models)} models x {len(args.variants)} variants = "
          f"{turns * len(args.models) * len(args.variants)} turns", flush=True)

    started = time.monotonic()
    asyncio.run(run_all(args.models, args.variants, out_dir, world, args.concurrency, args.max_cost_per_pair,
                        start=start, log_suffix=log_suffix))
    combined = combine(out_dir)
    shutil.rmtree(world, ignore_errors=True)

    records = [json.loads(line) for line in combined.read_text().splitlines() if line.strip()]
    done = [r for r in records if r["type"] == "turn"]
    cost = sum(r["cost_usd"] or 0 for r in done)
    errors = sum(r["status"] != "ok" for r in done)
    print(f"\nFinished in {time.monotonic() - started:.0f}s: {len(done)} turns, {errors} errors, "
          f"${cost:.2f}. Wrote {combined.relative_to(REPO_ROOT)} and logs/replays/latest.jsonl")


if __name__ == "__main__":
    main()
