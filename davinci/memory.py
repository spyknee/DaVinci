"""Memory node and database operations for DaVinci v0.5.2."""
from __future__ import annotations

import ast
import sqlite3
import uuid
import time
import json
import math
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

from davinci.fractals import escape_time, normalize_escape_time

_ZOOM_LEVELS_DEFAULT = {1: "", 2: "", 3: ""}


def _load_zoom_levels(value: str) -> dict:
    """Deserialize zoom_levels from a DB string.

    Tries JSON first (new format), falls back to ast.literal_eval for old
    Python-dict strings, and returns the default dict if both fail.
    """
    if not value:
        return dict(_ZOOM_LEVELS_DEFAULT)
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return dict(_ZOOM_LEVELS_DEFAULT)


@dataclass
class MemoryNode:
    id: str
    content: str
    zoom_levels: Dict[int, str]
    classification: str
    frequency: float
    created_at: float
    last_accessed: float
    context_c: Optional[str] = None
    speaker: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['tags'] = list(self.tags)
        return d

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> MemoryNode:
        tags_str = row.get("tags", "[]")
        try:
            tags = json.loads(tags_str) if tags_str else []
        except json.JSONDecodeError:
            tags = []

        return cls(
            id=row["id"],
            content=row["content"],
            zoom_levels=_load_zoom_levels(row["zoom_levels"]),
            classification=row["classification"],
            frequency=float(row["frequency"]),
            created_at=float(row["created_at"]),
            last_accessed=float(row["last_accessed"]),
            context_c=row.get("context_c"),
            speaker=row.get("speaker"),
            source=row.get("source"),
            tags=tags,
        )


class MemoryDB:
    SCHEMA_VERSION = 3

    def __init__(self, db_path: str = "davinci_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self):
        with self.conn:
            cur = self.conn.cursor()
            cur.execute("PRAGMA user_version")
            version = cur.fetchone()[0]

            if version == 0:
                self._create_tables(cur)
            elif version < self.SCHEMA_VERSION:
                self._migrate(cur)

            cur.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")

    def _create_tables(self, cursor):
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                zoom_levels TEXT NOT NULL,
                classification TEXT NOT NULL DEFAULT 'core',
                frequency REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                context_c TEXT,
                speaker TEXT,
                source TEXT,
                tags TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_classification ON memories(classification);
            CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_speaker_source ON memories(speaker, source);
        """)

    def _migrate(self, cursor):
        # v2 → v3: add provenance columns
        for col in ("speaker", "source"):
            try:
                cursor.execute(f"ALTER TABLE memories ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass  # column exists or migration done

        try:
            cursor.execute("ALTER TABLE memories ADD COLUMN tags TEXT DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass

    def store(
        self,
        content: str,
        zoom_levels: Optional[Dict[int, str]] = None,
        context_embedding: Optional[List[float]] = None,
        speaker: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        node_id = str(uuid.uuid4())
        now = time.time()
        if zoom_levels is None:
            zoom_levels = {1: "", 2: "", 3: ""}
        if tags is None:
            tags = []

        c_complex = self._compute_context_c(context_embedding)
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO memories 
                    (id, content, zoom_levels, classification, frequency,
                     created_at, last_accessed, context_c, speaker, source, tags)
                VALUES (?, ?, ?, 'core', 1.0, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    content,
                    json.dumps(zoom_levels),
                    now,
                    now,
                    str(c_complex),
                    speaker or "",
                    source or "manual",
                    json.dumps(tags) if tags else "[]",
                ),
            )
        return node_id

    def _compute_context_c(self, embedding: Optional[List[float]]) -> complex:
        if not embedding:
            return complex(0.25, 0)
        try:
            x = float(embedding[0])
            y = float(embedding[1]) if len(embedding) > 1 else 0.0
            return complex(x * 0.5, y * 0.5)
        except (IndexError, ValueError):
            return complex(0.25, 0)

    def recall(self, node_id: str) -> Optional[MemoryNode]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM memories WHERE id = ?", (node_id,))
        row = cur.fetchone()
        return MemoryNode.from_row(row) if row else None

    def decay(self, max_iter: int = 100, tau_base: float = 86400.0) -> Dict[str, List[str]]:
        now = time.time()
        changed = {"decay": [], "forget": []}
        cur = self.conn.cursor()

        cur.execute(
            "SELECT * FROM memories WHERE classification != 'forget'"
        )
        rows = cur.fetchall()

        with self.conn:
            for row in rows:
                node = MemoryNode.from_row(row)
                age = now - node.last_accessed

                # Get context c
                if node.context_c:
                    try:
                        c_ctx = complex(node.context_c)
                    except (ValueError, TypeError):
                        c_ctx = complex(0.25, 0)
                else:
                    c_ctx = complex(0.25, 0)

                z0_norm = min(node.frequency / 10.0, 2.0) * math.exp(1j * (now % 6.283))
                tau = fractal_decay_factor(z0_norm, c_ctx, max_iter=max_iter, base_tau=tau_base)

                # Apply decay
                f_new = node.frequency * math.exp(-age / tau) if tau > 0 else node.frequency

                # Reclassify
                old_class = node.classification
                if f_new < 0.5:
                    new_class = "forget"
                elif f_new < 2.0:
                    new_class = "decay"
                else:
                    new_class = "boundary" if f_new < 4.0 else "core"

                if old_class != new_class:
                    cur.execute(
                        """
                        UPDATE memories 
                        SET classification = ?, frequency = ?,
                            last_accessed = ?
                        WHERE id = ?
                        """,
                        (new_class, f_new, now, node.id),
                    )
                    changed[old_class].append(node.id)
                    if new_class in ("decay", "forget"):
                        changed[new_class].append(node.id)

        return {k: v for k, v in changed.items() if v}

    def search(self, query: str, limit: int = 10) -> List[MemoryNode]:
        cur = self.conn.cursor()
        pattern = f"%{query}%"
        cur.execute(
            """
            SELECT * FROM memories 
            WHERE content LIKE ? OR zoom_levels LIKE ?
            ORDER BY last_accessed DESC
            LIMIT ?
            """,
            (pattern, pattern, limit),
        )
        return [MemoryNode.from_row(row) for row in cur.fetchall()]

    def delete_by_classification(self, classification: str = "forget") -> int:
        with self.conn:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM memories WHERE classification = ?", (classification,))
            return cur.rowcount

    def get_all(self) -> List[MemoryNode]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM memories ORDER BY created_at")
        return [MemoryNode.from_row(row) for row in cur.fetchall()]

    def update_frequency(self, node_id: str, delta: float = 0.0) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE memories SET frequency = frequency + ?, last_accessed = ? WHERE id = ?",
                (delta, time.time(), node_id),
            )

    def update_zoom(self, node_id: str, level: int, text: str) -> None:
        node = self.recall(node_id)
        if not node:
            return
        zooms = node.zoom_levels.copy()
        zooms[level] = text
        with self.conn:
            self.conn.execute(
                "UPDATE memories SET zoom_levels = ? WHERE id = ?",
                (json.dumps(zooms), node_id),
            )

    def add_tags(self, node_id: str, tags: List[str]) -> bool:
        node = self.recall(node_id)
        if not node:
            return False
        new_tags = list(set(node.tags + [t for t in tags if t]))
        with self.conn:
            self.conn.execute(
                "UPDATE memories SET tags = ? WHERE id = ?",
                (json.dumps(new_tags), node_id),
            )
        return True

    def close(self):
        self.conn.close()


# Utility function (outside class)
def fractal_decay_factor(z0: complex, c_context: complex, max_iter: int = 100, base_tau: float = 86400.0) -> float:
    t_esc = escape_time(z0, c_context, max_iter=max_iter)
    alpha = 3.0
    tau = base_tau * (1.0 + alpha * normalize_escape_time(t_esc, max_iter))
    return tau
