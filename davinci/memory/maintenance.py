"""
DaVinci Memory — Layer 3: Background Maintenance
=================================================
Runs decay_cycle(), merge_similar(), and prune() periodically in a
background thread so memory stays clean without manual CLI calls.

No external dependencies — stdlib threading only.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from davinci.memory.consolidation import ConsolidationEngine
from davinci.memory.store import MemoryStore

__all__ = ["MemoryMaintenance"]

logger = logging.getLogger(__name__)

# Defaults
_DEFAULT_INTERVAL_SECONDS = 300          # 5 minutes
_DEFAULT_SIMILARITY_THRESHOLD = 0.8


class MemoryMaintenance:
    """Runs memory maintenance operations on a background timer loop.

    Parameters
    ----------
    store:
        Injected :class:`~davinci.memory.store.MemoryStore` instance.
        ``MemoryMaintenance`` never creates its own store.
    interval:
        Seconds between maintenance cycles (default: 300).
    similarity_threshold:
        Passed to :meth:`~davinci.memory.consolidation.ConsolidationEngine.merge_similar`
        (default: 0.8).
    on_cycle:
        Optional callback invoked after each cycle with a stats dict
        ``{"decayed": ..., "merged": ..., "pruned": ...}``.
    """

    def __init__(
        self,
        store: MemoryStore,
        interval: float = _DEFAULT_INTERVAL_SECONDS,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
        on_cycle: Callable[[dict], None] | None = None,
    ) -> None:
        self._store = store
        self._interval = interval
        self._similarity_threshold = similarity_threshold
        self._on_cycle = on_cycle
        self._running = False
        self._timer: threading.Timer | None = None
        self._engine = ConsolidationEngine(store)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background maintenance loop."""
        self._running = True
        self._schedule_next()
        logger.info("Memory maintenance started (interval=%ss)", self._interval)

    def stop(self) -> None:
        """Stop the background maintenance loop."""
        self._running = False
        if self._timer is not None:
            self._timer.cancel()
        logger.info("Memory maintenance stopped")

    def run_once(self) -> dict:
        """Run one full maintenance cycle synchronously.

        Steps performed in order:

        1. :meth:`~davinci.memory.store.MemoryStore.decay_cycle`
        2. :meth:`~davinci.memory.consolidation.ConsolidationEngine.merge_similar`
        3. :meth:`~davinci.memory.store.MemoryStore.prune`

        Returns
        -------
        dict
            ``{"decayed": <changed>, "merged": <count>, "pruned": <count>}``
        """
        changed = self._store.decay_cycle()
        logger.debug("decay_cycle: %s", changed)
        merged = self._engine.merge_similar(self._similarity_threshold)
        logger.debug("merge_similar: %d", merged)
        pruned = self._store.prune("forget")
        logger.debug("prune: %d", pruned)
        return {"decayed": changed, "merged": merged, "pruned": pruned}

    # ------------------------------------------------------------------
    # Internal timer helpers
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        """Schedule the next tick if the loop is still running."""
        if not self._running:
            return
        self._timer = threading.Timer(self._interval, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self) -> None:
        """Execute one maintenance cycle then reschedule."""
        try:
            result = self.run_once()
            if self._on_cycle is not None:
                self._on_cycle(result)
        except Exception:
            logger.error("Error in maintenance cycle", exc_info=True)
        finally:
            self._schedule_next()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "MemoryMaintenance":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
