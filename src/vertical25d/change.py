"""Building change between an early and a late period (Section 4.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .floors import estimate


def classify_change(chg: pd.DataFrame, bld: pd.DataFrame, predict, s) -> pd.DataFrame:
    chg = chg.merge(bld[["bid", "footprint_m2", "lat", "lon"]], on="bid", how="left")
    for c in ["h_early", "p_early", "h_late", "p_late"]:
        chg[c] = pd.to_numeric(chg[c], errors="coerce")
    chg["floors_early"] = estimate(chg["h_early"], predict, s.min_height_m)
    chg["floors_late"] = estimate(chg["h_late"], predict, s.min_height_m)
    chg["d_height"] = chg["h_late"] - chg["h_early"]
    chg["d_floors"] = (chg["floors_late"] - chg["floors_early"]).fillna(0)
    new = (chg["p_early"] < s.p_absent) & (chg["p_late"] >= s.p_present) & (chg["h_late"] >= s.min_height_m)
    grew = ~new & (chg["p_early"] >= s.p_present) & (chg["d_height"] >= s.min_dh_m) & (chg["d_floors"] >= 1)
    lower = ~new & (chg["p_early"] >= s.p_present) & (chg["d_height"] <= -s.min_dh_m) & (chg["d_floors"] <= -1)
    chg["change"] = np.select([new, grew, lower], ["new", "taller", "lower"], "stable")
    chg["added_floor_area_m2"] = np.where(new, chg["footprint_m2"] * chg["floors_late"].fillna(1),
                                          np.where(grew, chg["footprint_m2"] * chg["d_floors"], 0))
    both = chg["p_early"].ge(s.p_present) & chg["p_late"].ge(s.p_present)
    chg.attrs["median_drift_m"] = float(chg.loc[both, "d_height"].median()) if both.any() else float("nan")
    return chg
