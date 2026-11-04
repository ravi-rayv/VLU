"""Zone-level statistics (e.g. wards uploaded by the user): built form, POI
density, optional population, vertical indices and Spearman correlations."""
from __future__ import annotations

import pandas as pd
import geopandas as gpd


def zone_table(zones, name_col, bld, poi, z_idx, utm, pop_col=None, min_poi=10, change=None):
    w = zones[[name_col, "geometry"] + ([pop_col] if pop_col and pop_col in zones.columns else [])].copy()
    w = w.rename(columns={name_col: "zone"})
    w["area_km2"] = w.to_crs(utm).area.values / 1e6
    bpts = gpd.GeoDataFrame(bld[["floors_final", "footprint_m2"]],
                            geometry=gpd.points_from_xy(bld["lon"], bld["lat"]), crs=4326)
    bj = gpd.sjoin(bpts, w[["zone", "geometry"]], predicate="within", how="inner")
    bj["floor_area"] = bj["footprint_m2"] * bj["floors_final"].fillna(1)
    bstat = bj.groupby("zone").agg(buildings=("floors_final", "size"), mean_floors=("floors_final", "mean"),
                                   share_4plus=("floors_final", lambda v: (v >= 4).mean()),
                                   floor_area_m2=("floor_area", "sum"))
    w = w.merge(bstat, left_on="zone", right_index=True, how="left")
    if z_idx is not None:
        w = w.merge(z_idx.rename(columns={"id": "zone"}), on="zone", how="left")
    w["far"] = w["floor_area_m2"] / (w["area_km2"] * 1e6)
    w["poi_density"] = poi["zone"].value_counts().reindex(w["zone"]).fillna(0).values / w["area_km2"]
    if pop_col and pop_col in w.columns:
        w["pop_density"] = pd.to_numeric(w[pop_col], errors="coerce") / w["area_km2"]
    if change is not None and len(change):
        cp = gpd.GeoDataFrame(change[["change", "added_floor_area_m2"]],
                              geometry=gpd.points_from_xy(change["lon"], change["lat"]), crs=4326)
        cj = gpd.sjoin(cp, w[["zone", "geometry"]], predicate="within", how="inner")
        cs = cj.groupby("zone").agg(share_new=("change", lambda v: (v == "new").mean()),
                                    share_taller=("change", lambda v: (v == "taller").mean()),
                                    added_floor_area_m2=("added_floor_area_m2", "sum"))
        w = w.merge(cs, left_on="zone", right_index=True, how="left")
        w["added_far"] = w["added_floor_area_m2"] / (w["area_km2"] * 1e6)
    return w


def correlations(w: pd.DataFrame, min_poi=10) -> pd.DataFrame:
    if "VSI_excess" not in w.columns:
        return pd.DataFrame()
    targets = [c for c in ["pop_density", "poi_density", "mean_floors", "share_4plus", "far", "added_far",
                           "share_taller", "share_new"] if c in w.columns]
    wv = w[w["n_poi"] >= min_poi]
    if len(wv) < 5:
        return pd.DataFrame()
    idx = ["H_use", "H_band", "VSI", "VSI_excess"]
    return wv[idx + targets].corr(method="spearman").loc[idx, targets]

