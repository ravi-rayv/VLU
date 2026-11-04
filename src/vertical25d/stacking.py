"""Floor-by-floor land use inside each building ("how uses are stacked").

Uses the per-POI floor-band probabilities (known floors are certain, others are
predicted) to build (a) expected POI counts per building x floor band x land use and
(b) 3D floor slabs coloured by the dominant land use of each band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BANDS, BAND_START, LANDUSE

P_COLS = [f"p_{b}" for b in BANDS]

# 11 distinguishable colours (land-use order of config.LANDUSE)
LANDUSE_COLOURS = {
    "Retail": "#e15759", "Food & drink": "#f28e2b", "Office": "#4e79a7", "Personal services": "#b07aa1",
    "Health": "#59a14f", "Finance": "#1b9e9e", "Education": "#edc948", "Hospitality": "#ff9da7",
    "Leisure": "#9c755f", "Civic": "#76b7b2", "Religious": "#8c564b"}


def hex_rgb(h, a=235):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


def band_ranges(n_floors):
    """(start, end) floor numbers of each above-ground band, capped at the building top."""
    top = int(n_floors) if pd.notna(n_floors) and n_floors >= 1 else 1
    out = {}
    for i, b in enumerate(BANDS):
        if b == "Basement":
            continue
        start = BAND_START[i]
        end = BAND_START[i + 1] if i + 1 < len(BANDS) else 10 ** 6
        if start >= top:
            continue
        end = min(end, top)
        if end > start:
            out[b] = (start, end)
    return out


def building_band_uses(pp: pd.DataFrame) -> pd.DataFrame:
    """Expected POIs per building, band and land use; also known-floor counts."""
    d = pp[pp["bid"].notna() & pp["landuse"].isin(LANDUSE)].copy()
    if d.empty or not set(P_COLS).issubset(d.columns):
        return pd.DataFrame(columns=["bid", "band", "landuse", "expected", "n_known"])
    long = d.melt(id_vars=["bid", "landuse", "band_obs"], value_vars=P_COLS, var_name="band", value_name="p")
    long["band"] = long["band"].str[2:]
    long["known"] = (long["band_obs"] >= 0) & (long["band"] == long["band_obs"].map(
        lambda v: BANDS[int(v)] if v >= 0 else None))
    g = long.groupby(["bid", "band", "landuse"]).agg(expected=("p", "sum"), n_known=("known", "sum")).reset_index()
    return g[g["expected"] > 0.01]


def poi_floor_table(pp: pd.DataFrame, bid) -> pd.DataFrame:
    d = pp[pp["bid"] == bid].copy()
    if d.empty:
        return d
    pm = d[P_COLS].values
    best = pm.argmax(1)
    d["floor band"] = [BANDS[i] for i in best]
    d["probability"] = pm.max(1).round(2)
    d["source"] = np.where(d["band_obs"] >= 0, "known (OSM tag / address)", "predicted")
    d["floor"] = d["floor"].map(lambda v: "" if pd.isna(v) else ("G" if v == 0 else f"{v:.0f}"))
    cols = ["name", "landuse", "floor band", "floor", "source", "probability"]
    d["name"] = d["name"].fillna("(unnamed)") if "name" in d else "(unnamed)"
    order = {b: i for i, b in enumerate(BANDS)}
    return d[cols].sort_values("floor band", key=lambda s: s.map(order), ascending=False)


def slabs(buildings, pp: pd.DataFrame, min_expected: float = 0.5, max_buildings: int = 3000):
    """3D slabs: one per building x band with >= min_expected POIs, plus a grey shell per building.

    Returns (slab_df, shell_df) for pydeck PolygonLayer; polygons carry z = base height (m).
    """
    bu = building_band_uses(pp)
    if bu.empty:
        return pd.DataFrame(), pd.DataFrame()
    tot = bu.groupby("bid")["expected"].sum().sort_values(ascending=False)
    keep = tot.index[:max_buildings]
    bld = buildings.set_index("bid").loc[lambda x: x.index.isin(keep)]
    bld = bld[bld["h_median"].fillna(0) > 0]
    rows, shells = [], []
    for bid, r in bld.iterrows():
        floors = r["floors_final"] if pd.notna(r["floors_final"]) and r["floors_final"] >= 1 else 1
        fh = float(r["h_median"]) / float(floors)
        geom = r.geometry
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        ring = [list(c[:2]) for c in geom.exterior.coords]
        name = r.get("name") or "Unnamed building"
        shells.append({"polygon": ring, "height": float(r["h_median"]), "name": name,
                       "floors": int(floors), "band": "", "uses": "", "colour": [190, 190, 190, 60]})
        sub = bu[bu["bid"] == bid]
        for band, (f0, f1) in band_ranges(floors).items():
            s = sub[sub["band"] == band]
            if s["expected"].sum() < min_expected:
                continue
            s = s.sort_values("expected", ascending=False)
            main = s.iloc[0]["landuse"]
            uses = ", ".join(f"{u} {e:.1f}" for u, e in zip(s["landuse"], s["expected"]))
            z0, z1 = f0 * fh, f1 * fh
            gap = min(0.4, fh * 0.1)
            rows.append({"polygon": [c + [z0] for c in ring], "height": max(z1 - z0 - gap, 0.5),
                         "name": name, "floors": int(floors),
                         "band": band + (f" (floor {f0})" if f1 - f0 == 1 else f" (floors {f0}–{f1 - 1})"),
                         "uses": uses, "main": main, "known": int(s["n_known"].sum()),
                         "colour": hex_rgb(LANDUSE_COLOURS[main])})
    return pd.DataFrame(rows), pd.DataFrame(shells)
