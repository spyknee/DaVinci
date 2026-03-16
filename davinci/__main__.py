"""
DaVinci CLI — run with ``python -m davinci <command> [options]``.

Commands
--------
remember  Store a new memory
recall    Retrieve a memory by UUID
search    Search memories by content
forget    Prune memories by classification
decay     Run the decay cycle
consolidate  Run the consolidation engine
merge     Merge similar memories
stats     Show memory statistics
memories  List all (or filtered) memories
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


def cmd_forget(dv: DaVinci, args: argparse.Namespace) -> None:
    count = dv.forget(args.classification)
    print(f"Deleted {count} memories with classification '{args.classification}'.")


def cmd_decay(dv: DaVinci, _args: argparse.Namespace) -> None:
    changed = dv.decay()
    if not changed:
        print("Decay cycle complete. No memories reclassified.")
    else:
        print("Decay cycle complete. Reclassified memories:")
        for cls, ids in sorted(changed.items()):
            print(f"  → {cls}: {len(ids)}")


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

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMANDS = {
    "remember": cmd_remember,
    "recall": cmd_recall,
    "search": cmd_search,
    "forget": cmd_forget,
    "decay": cmd_decay,
    "consolidate": cmd_consolidate,
    "merge": cmd_merge,
    "stats": cmd_stats,
    "memories": cmd_memories,
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

    with DaVinci(db_path=args.db) as dv:
        handler(dv, args)


if __name__ == "__main__":
    main()
