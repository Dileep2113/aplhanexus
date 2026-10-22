"""W.D. Gann Square of 9 computation engine.

All mathematics are computed to exactly 2 decimal places using Python's
built-in math library — no external dependencies.

Formula:
    base_factor = floor(sqrt(CMP))
    Price(angle) = (base_factor + angle / 180) ** 2

Targets:
    T1 — 90°  vector  (first cardinal cross)
    T2 — 180° vector  (half cycle)
    T3 — 360° vector  (full master cycle)

Invalidation Floor:
    Structural 45-degree angle below base factor:
    Price = (base_factor - 0.5) ** 2
"""

from __future__ import annotations

import math
from typing import Dict, Any


def compute_gann_square9(cmp: float) -> Dict[str, Any]:
    """Compute W.D. Gann Square of 9 price targets from the current market price.

    Parameters
    ----------
    cmp : float
        Current Market Price (last traded price) as the seed value.

    Returns
    -------
    dict with keys:
        seed_price          — the CMP used as seed
        base_factor         — floor(sqrt(CMP))
        t1_price            — 90° target price (rounded to 2dp)
        t2_price            — 180° target price (rounded to 2dp)
        t3_price            — 360° master cycle target price (rounded to 2dp)
        invalidation_floor  — structural 45° below base (hard stop reference)
        t1_upside_pct       — % distance from CMP to T1
        t2_upside_pct       — % distance from CMP to T2
        t3_upside_pct       — % distance from CMP to T3
        floor_downside_pct  — % distance from CMP to invalidation floor
        t1_horizon_days     — approximate temporal horizon for T1 (days)
        t2_horizon_days     — approximate temporal horizon for T2 (days)
        t3_horizon_days     — approximate temporal horizon for T3 (days)
    """
    if cmp <= 0:
        raise ValueError(f"CMP must be positive; got {cmp}")

    base = math.floor(math.sqrt(cmp))

    def gann_price(angle_deg: float) -> float:
        return round((base + angle_deg / 180.0) ** 2, 2)

    t1 = gann_price(90)
    t2 = gann_price(180)
    t3 = gann_price(360)
    floor_val = round((base - 0.5) ** 2, 2)

    def pct(target: float) -> float:
        return round((target - cmp) / cmp * 100.0, 2)

    return {
        "seed_price": round(cmp, 2),
        "base_factor": base,
        "t1_price": t1,
        "t2_price": t2,
        "t3_price": t3,
        "invalidation_floor": floor_val,
        "t1_upside_pct": pct(t1),
        "t2_upside_pct": pct(t2),
        "t3_upside_pct": pct(t3),
        "floor_downside_pct": pct(floor_val),
        # Gann time projections (empirical: ~1 degree ≈ 0.25 trading day)
        "t1_horizon_days": 22,    # ~90° / 4 ≈ 22 trading days (~1 month)
        "t2_horizon_days": 45,    # ~180° / 4 ≈ 45 trading days (~2 months)
        "t3_horizon_days": 90,    # ~360° / 4 ≈ 90 trading days (~1 quarter)
        "t1_catalyst": "Volume surge > 1.5× 20-day average required for confirmation",
        "t2_catalyst": "Fundamental trigger: earnings beat or sector re-rating event",
        "t3_catalyst": "Full sector re-rating + macro tailwind alignment required",
    }
