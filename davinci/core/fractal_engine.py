"""
DaVinci Core — Layer 1: Fractal Engine
=======================================
Memory addressing and organization based on the Mandelbrot set's z² + c
iteration.

Every memory node is assigned a complex number c derived from its access
patterns.  Iterating z → z² + c determines whether that node:

  * Stays bounded (inside the set) → **core** — permanent memory
  * Escapes late                   → **boundary** — hot / actively used
  * Escapes mid-range              → **decay** — aging data
  * Escapes early                  → **forget** — irrelevant, prune it

Recency normalisation range: −2 … 0.25 (real axis inside the Mandelbrot
set), so the most recently accessed node maps toward the stable interior
and the stalest node maps toward the escaping boundary.

No external dependencies — pure Python + stdlib only.
"""

from __future__ import annotations

import time
from typing import Any

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "normalize",
    "compute_c",
    "iterate",
    "classify",
    "MemoryNode",
    "batch_classify",
]

# Classification thresholds (as a fraction of max_iter)
_BOUNDARY_THRESHOLD = 0.80   # > 80 % → boundary
_DECAY_THRESHOLD = 0.20      # 20–80 % → decay; < 20 % → forget


# ---------------------------------------------------------------------------
# 1. normalize
# ---------------------------------------------------------------------------

def normalize(
    value: float,
    min_val: float,
    max_val: float,
    target_min: float = -2.0,
    target_max: float = 2.0,
) -> float:
    """Map *value* from [min_val, max_val] into [target_min, target_max].

    When ``min_val == max_val`` (degenerate range) the midpoint of the target
    range is returned so that a single data-point doesn't cause a
    zero-division error.

    Parameters
    ----------
    value:      Raw input value.
    min_val:    Minimum of the source range.
    max_val:    Maximum of the source range.
    target_min: Lower bound of the output range (default −2).
    target_max: Upper bound of the output range (default  2).

    Returns
    -------
    float
        The value linearly interpolated into [target_min, target_max] and
        clamped to that range.
    """
    if min_val == max_val:
        return (target_min + target_max) / 2.0

    # Linear interpolation
    ratio = (value - min_val) / (max_val - min_val)
    result = target_min + ratio * (target_max - target_min)

    # Clamp to target range so out-of-source-range inputs stay well-behaved
    return max(target_min, min(target_max, result))


# ---------------------------------------------------------------------------
# 2. compute_c
# ---------------------------------------------------------------------------

def compute_c(
    frequency: float,
    recency: float,
    freq_range: tuple[float, float],
    recency_range: tuple[float, float],
) -> complex:
    """Build the complex c value that positions a node on the complex plane.

    ``c = normalize(recency, target=[-2, 0.25]) + normalize(frequency, target=[-1, 1]) * 1j``

    Recency drives the real axis.  The meaningful interior of the Mandelbrot
    set on the real axis lies in ``[-2, 0.25]``, so high recency (recently
    accessed) maps toward ``+0.25`` (inside the set → ``core``) while stale
    memories map toward ``-2`` (escapes → ``forget``).

    Frequency modulates depth on the imaginary axis within ``[-1, 1]``, which
    keeps the combined ``c`` value well inside the set for active nodes.

    Parameters
    ----------
    frequency:      Raw access-count for the memory node.
    recency:        Raw timestamp (or similar metric) of last access.
    freq_range:     (min, max) of the frequency dimension across all nodes.
    recency_range:  (min, max) of the recency dimension across all nodes.

    Returns
    -------
    complex
        A complex number whose real part encodes normalised recency
        (mapped to ``[-2.0, 0.25]``) and whose imaginary part encodes
        normalised frequency (mapped to ``[-1.0, 1.0]``).
    """
    norm_recency = normalize(recency, recency_range[0], recency_range[1],
                             target_min=-2.0, target_max=0.25)
    norm_freq = normalize(frequency, freq_range[0], freq_range[1],
                          target_min=-1.0, target_max=1.0)
    return complex(norm_recency, norm_freq)


# ---------------------------------------------------------------------------
# 3. iterate
# ---------------------------------------------------------------------------

def iterate(c: complex, max_iter: int = 1000) -> tuple[int, bool]:
    """Perform the Mandelbrot iteration z → z² + c starting from z = 0.

    Iteration stops as soon as |z| > 2 (escape condition) or after
    *max_iter* steps (inside the set).

    Parameters
    ----------
    c:        Complex parameter — position of the memory node.
    max_iter: Upper bound on iteration count (default 1 000).

    Returns
    -------
    (iteration_count, escaped)
        *iteration_count* — number of iterations executed.
        *escaped*         — ``True`` if the orbit escaped the set.
    """
    z = 0 + 0j
    for i in range(max_iter):
        z = z * z + c
        if abs(z) > 2.0:
            return i + 1, True
    return max_iter, False


# ---------------------------------------------------------------------------
# 4. classify
# ---------------------------------------------------------------------------

def classify(c: complex, max_iter: int = 1000) -> str:
    """Classify a memory node by its escape behaviour.

    Categories
    ----------
    ``"core"``
        Never escapes — inside the Mandelbrot set.  Retain permanently.
    ``"boundary"``
        Escapes after > 80 % of max_iter iterations.  Hot / active data.
    ``"decay"``
        Escapes in the mid range (20–80 % of max_iter).  Aging data.
    ``"forget"``
        Escapes early (≤ 20 % of max_iter).  Irrelevant — prune it.

    Parameters
    ----------
    c:        Complex position of the memory node.
    max_iter: Iteration limit passed through to :func:`iterate`.

    Returns
    -------
    str
        One of ``"core"``, ``"boundary"``, ``"decay"``, or ``"forget"``.
    """
    iteration_count, escaped = iterate(c, max_iter)

    if not escaped:
        return "core"

    ratio = iteration_count / max_iter
    if ratio > _BOUNDARY_THRESHOLD:
        return "boundary"
    if ratio > _DECAY_THRESHOLD:
        return "decay"
    return "forget"


# ---------------------------------------------------------------------------
# 5. MemoryNode
# ---------------------------------------------------------------------------

class MemoryNode:
    """A single unit of fractal memory.

    Each node holds a piece of content and tracks how often and how recently
    it has been accessed.  Its position on the complex plane (``c_value``)
    and Mandelbrot classification are recomputed automatically whenever
    :meth:`update_access` is called.

    Parameters
    ----------
    content:       The actual data stored in this node.
    frequency:     Initial access count (default 0).
    recency:       Timestamp of last access (default: *now*).
    freq_range:    (min, max) used to normalise frequency.  The store always
                   provides a meaningful range; the ``(0, 0)`` default is only
                   used when constructing nodes outside the store context.
    recency_range: (min, max) used to normalise recency.  The store always
                   provides a meaningful range (synthetic 1-second window when
                   the DB is empty) so that new nodes map to ``target_max``
                   and are classified as ``"core"``.
    max_iter:      Mandelbrot iteration limit (default 1 000).
    zoom_levels:   Pre-populated zoom-level content dict (optional).

    Attributes
    ----------
    content:        The stored data.
    c_value:        Complex position on the Mandelbrot plane.
    classification: ``"core"``, ``"boundary"``, ``"decay"``, or ``"forget"``.
    iteration_count: Iterations before escape (or max_iter if inside set).
    zoom_levels:    Dict keyed 1/2/3 with content at different resolutions.
    frequency:      Access count.
    recency:        Unix timestamp of last access.
    created_at:     Unix timestamp of node creation.
    """

    def __init__(
        self,
        content: str,
        frequency: int = 0,
        recency: float | None = None,
        freq_range: tuple[float, float] = (0.0, 0.0),
        recency_range: tuple[float, float] = (0.0, 0.0),
        max_iter: int = 1000,
        zoom_levels: dict[int, str] | None = None,
    ) -> None:
        self.content: str = content
        self.frequency: int = frequency
        self.recency: float = recency if recency is not None else time.time()
        self.created_at: float = time.time()
        self._freq_range: tuple[float, float] = freq_range
        self._recency_range: tuple[float, float] = recency_range
        self._max_iter: int = max_iter

        self.zoom_levels: dict[int, str] = zoom_levels if zoom_levels is not None else {
            1: content,
            2: content,
            3: content,
        }

        # Compute initial position and classification
        self._recompute()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recompute(self) -> None:
        """Recompute c_value, iteration_count, and classification."""
        self.c_value: complex = compute_c(
            self.frequency,
            self.recency,
            self._freq_range,
            self._recency_range,
        )
        self.iteration_count, _ = iterate(self.c_value, self._max_iter)
        self.classification: str = classify(self.c_value, self._max_iter)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def update_access(self) -> None:
        """Record an access event.

        Increments :attr:`frequency`, refreshes :attr:`recency` to the
        current time, and recomputes the node's position and classification.
        """
        self.frequency += 1
        self.recency = time.time()
        self._recompute()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the node to a plain Python dictionary.

        Returns
        -------
        dict
            All public attributes in JSON-serialisable form.
        """
        return {
            "content": self.content,
            "c_value": {"real": self.c_value.real, "imag": self.c_value.imag},
            "classification": self.classification,
            "iteration_count": self.iteration_count,
            "zoom_levels": {str(k): v for k, v in self.zoom_levels.items()},
            "frequency": self.frequency,
            "recency": self.recency,
            "created_at": self.created_at,
            "_freq_range": list(self._freq_range),
            "_recency_range": list(self._recency_range),
            "_max_iter": self._max_iter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryNode":
        """Deserialise a node from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data: dict
            Dictionary as returned by :meth:`to_dict`.

        Returns
        -------
        MemoryNode
            Reconstructed node with the same state as when it was serialised.
        """
        # zoom_levels keys are stored as strings for JSON compatibility
        # and converted back to ints on deserialisation.
        zoom_raw = data.get("zoom_levels", {})
        zoom_levels = {int(k): v for k, v in zoom_raw.items()}

        raw_freq = data.get("_freq_range", [0.0, 0.0])
        raw_recency = data.get("_recency_range", [0.0, 0.0])
        freq_range: tuple[float, float] = (float(raw_freq[0]), float(raw_freq[1]))
        recency_range: tuple[float, float] = (float(raw_recency[0]), float(raw_recency[1]))

        node = cls(
            content=data["content"],
            frequency=data["frequency"],
            recency=data["recency"],
            freq_range=freq_range,
            recency_range=recency_range,
            max_iter=data.get("_max_iter", 1000),
            zoom_levels=zoom_levels,
        )
        # Restore creation timestamp rather than using time.time()
        node.created_at = data["created_at"]
        return node

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MemoryNode(classification={self.classification!r}, "
            f"frequency={self.frequency}, "
            f"c_value={self.c_value!r})"
        )


# ---------------------------------------------------------------------------
# 6. batch_classify
# ---------------------------------------------------------------------------

def batch_classify(nodes: list[MemoryNode]) -> dict[str, list[MemoryNode]]:
    """Classify multiple :class:`MemoryNode` objects and group them.

    Each node is classified using its current :attr:`~MemoryNode.classification`
    attribute (which is kept up-to-date automatically).

    Parameters
    ----------
    nodes: list[MemoryNode]
        Nodes to classify.

    Returns
    -------
    dict[str, list[MemoryNode]]
        A dict with keys ``"core"``, ``"boundary"``, ``"decay"``,
        ``"forget"``, each mapping to the list of nodes in that category.
    """
    result: dict[str, list[MemoryNode]] = {
        "core": [],
        "boundary": [],
        "decay": [],
        "forget": [],
    }
    for node in nodes:
        result[node.classification].append(node)
    return result
