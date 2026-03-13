"""
DaVinci CLI — run with ``python -m davinci <command> [options]``.

Commands
--------
remember       Store a new memory
recall         Retrieve a memory by UUID
search         Search memories by content
search-fts     FTS5 full-text search
forget         Prune memories by classification
decay          Run the decay cycle
consolidate    Run the consolidation engine
merge          Merge similar memories
stats          Show memory statistics
memories       List all (or filtered) memories
migrate        Run migration check
ask            Ask the LLM a question (full pipeline)
model          Show active LLM model
model-switch   Switch active LLM model
model-toggle   Cycle to next LLM model
episodic-status  Show episodic memory statistics
episodic-decay   Run episodic importance decay
episodic-prune   Prune low-importance episodic entries
review         Show pending auto-learn facts
approve        Approve a pending fact by index
approve-all    Approve all pending facts
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from davinci.interface.api import DaVinci


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_PREVIEW_LEN = 60


def _preview(text: str) -> str:
    """Return the first ``_PREVIEW_LEN`` characters, appending '…' if trimmed."""
    if len(text) <= _PREVIEW_LEN:
        return text
    return text[:_PREVIEW_LEN] + "…"


def _print_node_table(nodes: list[Any], title: str = "") -> None:
    """Print a table of memory nodes to stdout."""
    if title:
        print(title)
    if not nodes:
        print("  (no memories)")
        return
    fmt = "  {:<36}  {:<10}  {:>5}  {}"
    print(fmt.format("ID", "CLASS", "FREQ", "CONTENT"))
    print("  " + "-" * 80)
    for node in nodes:
        node_id = getattr(node, "id", "—")
        print(fmt.format(
            node_id,
            node.classification,
            node.frequency,
            _preview(node.content),
        ))


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_remember(dv: DaVinci, args: argparse.Namespace) -> None:
    zoom_levels: dict[int, str] | None = None
    if args.zoom1 or args.zoom2 or args.zoom3:
        zoom_levels = {
            1: args.zoom1 or args.content,
            2: args.zoom2 or args.content,
            3: args.zoom3 or args.content,
        }
    mid = dv.remember(args.content, zoom_levels=zoom_levels)
    print(f"Stored memory with ID: {mid}")


def cmd_recall(dv: DaVinci, args: argparse.Namespace) -> None:
    node = dv.recall(args.id)
    if node is None:
        print(f"No memory found with ID: {args.id}", file=sys.stderr)
        sys.exit(1)
    print(f"ID:             {getattr(node, 'id', args.id)}")
    print(f"Classification: {node.classification}")
    print(f"Frequency:      {node.frequency}")
    print(f"Content:        {node.content}")
    print(f"Zoom level 1:   {node.zoom_levels.get(1, '')}")
    print(f"Zoom level 2:   {node.zoom_levels.get(2, '')}")
    print(f"Zoom level 3:   {node.zoom_levels.get(3, '')}")


def cmd_search(dv: DaVinci, args: argparse.Namespace) -> None:
    results = dv.search(args.query, limit=args.limit)
    _print_node_table(results, title=f"Search results for '{args.query}' ({len(results)} found):")


def cmd_search_fts(dv: DaVinci, args: argparse.Namespace) -> None:
    results = dv.search_fts(args.query, limit=args.limit)
    _print_node_table(results, title=f"FTS5 search results for '{args.query}' ({len(results)} found):")


def cmd_forget(dv: DaVinci, args: argparse.Namespace) -> None:
    count = dv.forget(args.classification)
    print(f"Deleted {count} memories with classification '{args.classification}'.")


def cmd_decay(dv: DaVinci, _args: argparse.Namespace) -> None:
    changed = dv.decay()
    if not changed:
        print("Decay cycle complete. No memories reclassified.")
    else:
        print("Decay cycle complete. Reclassified memories:")
        for cls, count in sorted(changed.items()):
            print(f"  → {cls}: {count}")


def cmd_consolidate(dv: DaVinci, args: argparse.Namespace) -> None:
    count = dv.consolidate(strategy=args.strategy)
    print(f"Consolidation complete. {count} memories updated (strategy: '{args.strategy}').")


def cmd_merge(dv: DaVinci, args: argparse.Namespace) -> None:
    count = dv.merge_similar(threshold=args.threshold)
    print(f"Merge complete. {count} memories merged (threshold: {args.threshold}).")


def cmd_stats(dv: DaVinci, _args: argparse.Namespace) -> None:
    s = dv.stats()
    by_cls = s.get("by_classification", {})
    avg_freq = s.get("avg_frequency") or 0.0
    print("=" * 40)
    print("  DaVinci Memory Statistics")
    print("=" * 40)
    print(f"  Total memories  : {s.get('total', 0)}")
    print(f"  Avg frequency   : {avg_freq:.2f}")
    print(f"  Oldest memory   : {s.get('oldest_timestamp') or '—'}")
    print(f"  Newest memory   : {s.get('newest_timestamp') or '—'}")
    print()
    print("  By classification:")
    for cls in ("core", "boundary", "decay", "forget"):
        print(f"    {cls:<10}: {by_cls.get(cls, 0)}")
    print("=" * 40)


def cmd_memories(dv: DaVinci, args: argparse.Namespace) -> None:
    nodes = dv.memories(classification=args.classification)
    title = (
        f"Memories (classification='{args.classification}')"
        if args.classification
        else "All memories"
    )
    _print_node_table(nodes, title=f"{title} ({len(nodes)} total):")


def cmd_migrate(dv: DaVinci, _args: argparse.Namespace) -> None:
    result = dv.migrate()
    if not result:
        print("Migration check complete. No memories needed reclassification.")
    else:
        print("Migration complete. Reclassified memories:")
        for cls, ids in sorted(result.items()):
            print(f"  → {cls} ({len(ids)}):")
            for mid in ids:
                print(f"      {mid}")


def cmd_ask(dv: DaVinci, args: argparse.Namespace) -> None:
    answer = dv.ask(args.question)
    print(answer)


def cmd_model(dv: DaVinci, _args: argparse.Namespace) -> None:
    status = dv.model_status()
    if not status:
        print("LLM not available.")
        return
    print(f"Active model  : {status.get('active', '—')}")
    print(f"Model ID      : {status.get('model', '—')}")
    print(f"Base URL      : {status.get('base_url', '—')}")
    print(f"Available     : {status.get('available', False)}")


def cmd_model_switch(dv: DaVinci, args: argparse.Namespace) -> None:
    ok = dv.model_switch(args.name)
    if ok:
        print(f"Switched to model: {args.name}")
    else:
        print(f"Unknown model: {args.name}", file=sys.stderr)
        sys.exit(1)


def cmd_model_toggle(dv: DaVinci, _args: argparse.Namespace) -> None:
    new_name = dv.model_toggle()
    if new_name:
        print(f"Toggled to model: {new_name}")
    else:
        print("LLM not available.")


def cmd_episodic_status(dv: DaVinci, _args: argparse.Namespace) -> None:
    s = dv.episodic_status()
    print("=" * 40)
    print("  Episodic Memory Statistics")
    print("=" * 40)
    print(f"  Episodes        : {s.get('count', 0)}")
    print(f"  Avg importance  : {s.get('avg_importance', 0.0):.3f}")
    print(f"  Oldest episode  : {s.get('oldest_timestamp') or '—'}")
    print("=" * 40)


def cmd_episodic_decay(dv: DaVinci, args: argparse.Namespace) -> None:
    count = dv.episodic_decay(rate=args.rate)
    print(f"Episodic decay complete. {count} entries updated.")


def cmd_episodic_prune(dv: DaVinci, args: argparse.Namespace) -> None:
    count = dv.episodic_prune(threshold=args.threshold)
    print(f"Episodic prune complete. {count} entries deleted.")


def cmd_review(dv: DaVinci, _args: argparse.Namespace) -> None:
    pending = dv.review_pending()
    if not pending:
        print("No pending facts to review.")
        return
    print(f"Pending facts ({len(pending)}):")
    for i, entry in enumerate(pending):
        print(f"  [{i}] {entry['fact']}")


def cmd_approve(dv: DaVinci, args: argparse.Namespace) -> None:
    ok = dv.approve_fact(args.index)
    if ok:
        print(f"Fact [{args.index}] approved and stored.")
    else:
        print(f"Invalid index: {args.index}", file=sys.stderr)
        sys.exit(1)


def cmd_approve_all(dv: DaVinci, _args: argparse.Namespace) -> None:
    count = dv.approve_all_facts()
    print(f"Approved and stored {count} facts.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m davinci",
        description="DaVinci fractal memory system — command-line interface.",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default="davinci_memory.db",
        help="Path to the SQLite database file (default: davinci_memory.db).",
    )
    parser.add_argument(
        "--profile",
        metavar="PATH",
        default=None,
        help="Path to a JSON profile file for LLM configuration.",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # remember
    p_remember = sub.add_parser("remember", help="Store a new memory.")
    p_remember.add_argument("content", help="Text content to remember.")
    p_remember.add_argument("--zoom1", default=None, metavar="TEXT", help="Zoom level 1 (summary).")
    p_remember.add_argument("--zoom2", default=None, metavar="TEXT", help="Zoom level 2 (detail).")
    p_remember.add_argument("--zoom3", default=None, metavar="TEXT", help="Zoom level 3 (full).")

    # recall
    p_recall = sub.add_parser("recall", help="Retrieve a memory by UUID.")
    p_recall.add_argument("id", help="UUID of the memory to recall.")

    # search
    p_search = sub.add_parser("search", help="Search memories by content.")
    p_search.add_argument("query", help="Search query string.")
    p_search.add_argument("--limit", type=int, default=10, metavar="N", help="Maximum results (default: 10).")

    # search-fts
    p_search_fts = sub.add_parser("search-fts", help="FTS5 full-text search.")
    p_search_fts.add_argument("query", help="Search query string.")
    p_search_fts.add_argument("--limit", type=int, default=10, metavar="N", help="Maximum results (default: 10).")

    # forget
    p_forget = sub.add_parser("forget", help="Prune memories by classification.")
    p_forget.add_argument(
        "--classification",
        default="forget",
        metavar="CLASS",
        help="Classification to delete (default: forget).",
    )

    # decay
    sub.add_parser("decay", help="Run the decay cycle.")

    # consolidate
    p_consolidate = sub.add_parser("consolidate", help="Run the consolidation engine.")
    p_consolidate.add_argument(
        "--strategy",
        default="frequency",
        metavar="STRATEGY",
        help="Consolidation strategy (default: frequency).",
    )

    # merge
    p_merge = sub.add_parser("merge", help="Merge similar memories.")
    p_merge.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        metavar="FLOAT",
        help="Jaccard similarity threshold 0–1 (default: 0.8).",
    )

    # stats
    sub.add_parser("stats", help="Show memory statistics.")

    # memories
    p_memories = sub.add_parser("memories", help="List all memories.")
    p_memories.add_argument(
        "--classification",
        default=None,
        metavar="CLASS",
        help="Filter by classification (core, boundary, decay, forget).",
    )

    # migrate
    sub.add_parser("migrate", help="Run migration check.")

    # ask
    p_ask = sub.add_parser("ask", help="Ask the LLM a question (full pipeline).")
    p_ask.add_argument("question", help="The question to ask.")
    p_ask.add_argument(
        "--mode",
        choices=["fast", "deep", "auto"],
        default="auto",
        metavar="MODE",
        help="Answer mode: fast, deep, or auto (default: auto).",
    )

    # model
    sub.add_parser("model", help="Show active LLM model.")

    # model-switch
    p_ms = sub.add_parser("model-switch", help="Switch active LLM model.")
    p_ms.add_argument("name", help="Model key name (e.g. qwen, qwen35, model3).")

    # model-toggle
    sub.add_parser("model-toggle", help="Cycle to next LLM model.")

    # episodic-status
    sub.add_parser("episodic-status", help="Show episodic memory statistics.")

    # episodic-decay
    p_ed = sub.add_parser("episodic-decay", help="Run episodic importance decay.")
    p_ed.add_argument(
        "--rate",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help="Decay rate per day (default: 0.05).",
    )

    # episodic-prune
    p_ep = sub.add_parser("episodic-prune", help="Prune low-importance episodic entries.")
    p_ep.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        metavar="FLOAT",
        help="Importance threshold below which to prune (default: 0.2).",
    )

    # review
    sub.add_parser("review", help="Show pending auto-learn facts.")

    # approve
    p_approve = sub.add_parser("approve", help="Approve a pending fact by index.")
    p_approve.add_argument("index", type=int, help="Zero-based index of the fact to approve.")

    # approve-all
    sub.add_parser("approve-all", help="Approve all pending auto-learn facts.")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMANDS = {
    "remember": cmd_remember,
    "recall": cmd_recall,
    "search": cmd_search,
    "search-fts": cmd_search_fts,
    "forget": cmd_forget,
    "decay": cmd_decay,
    "consolidate": cmd_consolidate,
    "merge": cmd_merge,
    "stats": cmd_stats,
    "memories": cmd_memories,
    "migrate": cmd_migrate,
    "ask": cmd_ask,
    "model": cmd_model,
    "model-switch": cmd_model_switch,
    "model-toggle": cmd_model_toggle,
    "episodic-status": cmd_episodic_status,
    "episodic-decay": cmd_episodic_decay,
    "episodic-prune": cmd_episodic_prune,
    "review": cmd_review,
    "approve": cmd_approve,
    "approve-all": cmd_approve_all,
}


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    with DaVinci(db_path=args.db, profile_path=args.profile) as dv:
        handler(dv, args)


if __name__ == "__main__":
    main()
