"""Fractal memory utilities inspired by Julia and Mandelbrot sets."""
from __future__ import annotations

import math


def escape_time(z0: complex, c: complex, max_iter: int = 100, escape_radius: float = 2.0) -> float:
    """
    Compute smoothed escape time for z_{n+1} = z_n^2 + c starting at z0.
    
    Returns a real-valued 'time' (not just integer iterations):
    - Smaller values → faster escape (less stable)
    - Larger values → slower escape (more stable, near Julia boundary)
    
    Uses the standard escape-time algorithm with logarithmic smoothing:
        t = n - log_2(log_2(|z_n|))
    to give fractional iteration counts.
    
    If bounded for all iterations, returns max_iter (fully stable).
    """
    z = z0
    for n in range(1, max_iter + 1):
        z = z * z + c
        if abs(z) > escape_radius:
            # Smoothed escape time: fractional iteration count
            return n - math.log2(math.log2(abs(z)))
    # Fully bounded (within Julia set or its interior)
    return float(max_iter)


def normalize_escape_time(t: float, max_iter: int = 100) -> float:
    """
    Normalize escape time to [0, 1], where:
      - 0 → escapes immediately (unstable, forgets fast)
      - 1 → never escapes (highly stable, retains long)
    
    This becomes our *retention potential*.
    """
    if max_iter <= 0:
        return 1.0
    # Clamp and scale
    t_clamped = min(max(0.0, t), float(max_iter))
    return t_clamped / float(max_iter)


def fractal_decay_factor(
    z0: complex,
    c_context: complex,
    max_iter: int = 100,
    base_tau: float = 86400.0,  # 1 day in seconds
) -> float:
    """
    Compute effective memory retention factor τ (time constant) using fractal dynamics.
    
    Returns τ: larger τ → slower decay (longer retention).
    Model: retention ∝ exp(-t / τ), where τ = base_tau * (1 + α * T_escape)
    
    Intuition:
      - If |z0| near Julia boundary for given c_context → high escape time → big τ
      - Far from boundary → quick escape → small τ
    """
    t_esc = escape_time(z0, c_context, max_iter=max_iter)
    # Scaling: more iterations = stronger retention; use linear scaling + offset
    alpha = 3.0  # tunable — how much escape time boosts retention
    tau = base_tau * (1.0 + alpha * normalize_escape_time(t_esc, max_iter))
    return tau


def context_to_complex(embedding: list[float] | None) -> complex:
    """
    Convert a 2D embedding (e.g., from LLM vectorizer) to a complex number c = x + iy.
    
    If embedding length != 2, pad or truncate to fit.
    If no embedding provided, return (0+0j) — neutral context.
    """
    if not embedding:
        return complex(0.0, 0.0)
    
    # Ensure exactly two dimensions
    x = float(embedding[0])
    y = float(embedding[1]) if len(embedding) > 1 else 0.0
    
    # Optional: scale to [-2, 2] range (Mandelbrot view)
    # x *= 2.0; y *= 2.0
    return complex(x, y)
