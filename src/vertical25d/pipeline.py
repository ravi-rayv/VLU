"""End-to-end pipeline: study area -> buildings with floors -> POIs on floors
-> vertical land-use indices -> files, figures and a zip of everything.

Used by the Streamlit portal and the command-line tool. Network steps (OSM,
Earth Engine) can be replaced by pre-computed inputs, which is how the demo
mode and the tests run without an internet connection.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import geopandas as gpd

from . import aoi as aoi_mod, change as change_mod, floors as floors_mod, indices as idx_mod
from . import osm as osm_mod, plots, pois as poi_mod, zones as zones_mod
from .config import Settings, LANDUSE


@dataclass
class Inputs:
    aoi: gpd.GeoDataFrame
    ee_project: str | None = None
    service_account_json: str | None = None
    zones: gpd.GeoDataFrame | None = None
    zone_name_col: str | None = None
    zone_pop_col: str | None = None
    pois: gpd.GeoDataFrame | None = None          # user-supplied POIs (else downloaded from OSM)
    buildings: gpd.GeoDataFrame | None = None     # pre-downloaded OSM buildings (demo / offline)
    heights: pd.DataFrame | None = None           # pre-computed heights: bid, h_median, h_p90 (demo / offline)
    change_layers: pd.DataFrame | None = None     # pre-computed: bid, h_early, p_early, h_late, p_late
    name: str = "study_area"


@dataclass
class Result:
    out_dir: str
    zip_path: str
    summary: dict
    buildings: gpd.GeoDataFrame
    pois: gpd.GeoDataFrame
    building_index: pd.DataFrame | None
    grid: gpd.GeoDataFrame | None
    zones: gpd.GeoDataFrame | None
    profile: pd.DataFrame
    figures: dict = field(default_factory=dict)
    poi_bands: pd.DataFrame | None = None   # classified POIs with floor-band probabilities p_<band>


STEPS = [("aoi", 0.02), ("buildings", 0.13), ("heights", 0.30), ("floors", 0.05), ("pois", 0.10),
         ("link", 0.05), ("predict", 0.07), ("indices", 0.18), ("change", 0.05), ("outputs", 0.05)]


class Progress:
    def __init__(self, callback=None):
        self.cb, self.done = callback, 0.0
        self.weights = dict(STEPS)

    def step(self, name):
        def inner(frac):
            if self.cb:
                self.cb(min(self.done + self.weights[name] * frac, 1.0), name)
        return inner

    def finish(self, name):
        self.done += self.weights[name]
        if self.cb:
            self.cb(min(self.done, 1.0), name)


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "study_area"


def run(inp: Inputs, s: Settings | None = None, out_dir: str = "outputs", log=print, progress=None) -> Result:
    s = s or Settings()
    t0 = time.time()
    P = Progress(progress)
    name = safe_name(inp.name)
    out_dir = os.path.join(out_dir, name)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    summary = {"study_area": name, "settings": s.to_dict(), "package_version": _version()}

    # 1. study area ---------------------------------------------------------------------------
    info = aoi_mod.check_aoi(inp.aoi, s.max_area_km2)
    geom = aoi_mod.aoi_geometry(inp.aoi)
    utm = inp.aoi.estimate_utm_crs()
    summary["aoi"] = info
    log(f"Study area: {info['area_km2']} km², UTM EPSG:{info['utm_epsg']}")
    inp.aoi.to_file(os.path.join(out_dir, "study_area.gpkg"), driver="GPKG")
    P.finish("aoi")

    # 2. buildings ----------------------------------------------------------------------------
    if inp.buildings is not None:
        bld = osm_mod.clean_buildings(inp.buildings, geom, utm)
    else:
        bld = osm_mod.fetch_buildings(geom, utm, log=log)
    log(f"Buildings: {len(bld):,} OSM footprints")
    P.finish("buildings")

    # 3. heights ------------------------------------------------------------------------------
    need_ee = inp.heights is None or (s.run_change and inp.change_layers is None)
    if need_ee:
        from . import heights as h_mod
        h_mod.init_earth_engine(inp.ee_project, inp.service_account_json)
        log("Earth Engine: connected")
    if inp.heights is not None:
        hts = inp.heights[["bid", "h_median", "h_p90"]]
    else:
        from . import heights as h_mod
        hts = h_mod.building_heights(bld, geom, s, log=log, progress=P.step("heights"))
    from .heights import attach_heights, add_geometry_stats
    res = add_geometry_stats(attach_heights(bld, hts), utm)
    P.finish("heights")

    # 4. floors -------------------------------------------------------------------------------
    cal = floors_mod.calibrate(res, s, log=log)
    res = floors_mod.assign_floors(res, cal, s)
    summary["floors"] = {
        "model": cal["model"], "n_tagged": cal["n_tagged"], "n_plausible": cal.get("n_plausible"),
        "n_excluded": cal.get("n_excluded"), "intercept": cal.get("intercept"), "slope": cal.get("slope"),
        "m_per_floor": cal.get("storey_m"),
        "cv": cal["cv"].round(3).reset_index().to_dict("records") if len(cal["cv"]) else [],
        "source_counts": res["floor_source"].value_counts().to_dict(),
        "median_height_m": float(res.loc[res["h_median"] >= s.min_height_m, "h_median"].median()) if len(res) else None,
        "share_4plus": float((res["floors_final"] >= 4).mean()),
    }
    P.finish("floors")

    # 5. POIs ---------------------------------------------------------------------------------
    raw = (osm_mod.standardise_pois(inp.pois, geom) if inp.pois is not None
           else osm_mod.fetch_pois(geom, log=log))
    poi = poi_mod.extract_floors(poi_mod.classify(raw))
    log(f"POIs: {len(raw):,} points, {len(poi):,} classified into land uses; "
        f"floor known for {poi['floor'].notna().mean():.1%}")
    P.finish("pois")

    # 6. link ---------------------------------------------------------------------------------
    poi = poi_mod.link_to_buildings(gpd.GeoDataFrame(poi, geometry="geometry", crs=4326), res, utm, s.match_dist_m)
    has_zones = inp.zones is not None and inp.zone_name_col
    if has_zones:
        poi = poi_mod.assign_zones(poi, inp.zones, inp.zone_name_col)
    summary["pois"] = {"n_points": int(len(raw)), "n_classified": int(len(poi)),
                       "landuse_counts": poi["landuse"].value_counts().reindex(LANDUSE).fillna(0).astype(int).to_dict(),
                       "match_counts": poi["match"].value_counts().to_dict(),
                       "floor_known_share": float(poi["floor"].notna().mean()),
                       "floor_source_counts": poi["floor_source"].value_counts().to_dict()}
    P.finish("link")

    # 7. floor prediction -----------------------------------------------------------------------
    pp = idx_mod.prepare(poi)            # all classified POIs, as in the paper
    if len(pp) < 3:
        raise RuntimeError("Too few POIs linked to buildings with floors to compute vertical indices")
    Pmat, cv = idx_mod.predict_bands(pp, s, log=log)
    summary["floor_prediction"] = {k: v for k, v in cv.items() if not isinstance(v, list)}
    P.finish("predict")

    # 8. indices --------------------------------------------------------------------------------
    res_idx, band_counts, cells = idx_mod.monte_carlo(pp, Pmat, s, utm, has_zones=bool(has_zones), log=log,
                                                      progress=P.step("indices"))
    profile = idx_mod.vertical_profile(band_counts)
    pp["cell"] = cells.values
    P.finish("indices")

    # 9. change (optional) --------------------------------------------------------------------
    chg = None
    if s.run_change:
        if inp.change_layers is not None:
            layers = inp.change_layers
        else:
            from . import heights as h_mod
            layers = h_mod.building_change(bld, geom, s, log=log, progress=P.step("change"))
        chg = change_mod.classify_change(layers, res, cal["predict"], s)
        summary["change"] = {"counts": chg["change"].value_counts().to_dict(),
                             "added_floor_area_km2": float(chg["added_floor_area_m2"].sum() / 1e6),
                             "median_drift_m": chg.attrs.get("median_drift_m")}
        chg.to_csv(os.path.join(out_dir, "building_change.csv"), index=False)
    P.finish("change")

    # 10. outputs -------------------------------------------------------------------------------
    bcols = ["bid", "osmid", "name", "building", "building_levels", "h_median", "h_p90", "floor_model",
             "floors_est", "floors_est_p90", "floors_final", "floor_source", "footprint_m2", "lat", "lon", "qc_flags"]
    res[bcols].to_csv(os.path.join(out_dir, "buildings_floors.csv"), index=False)
    res[bcols + ["geometry"]].to_file(os.path.join(out_dir, "buildings_floors.gpkg"), driver="GPKG")

    pcols = [c for c in ["osm_id", "name", "zone", "landuse", "category", "brand", "addr_text", "level", "addr_floor",
                         "floor", "floor_source", "floor_above_building", "bid", "match", "match_dist_m",
                         "floors_final", "bld_floor_source", "h_median", "lon", "lat"] if c in poi.columns]
    poi_out = gpd.GeoDataFrame(poi[pcols], geometry=poi.geometry, crs=4326)
    poi_out.to_file(os.path.join(out_dir, "poi_buildings.gpkg"), driver="GPKG")
    poi_out.drop(columns="geometry").to_csv(os.path.join(out_dir, "poi_buildings.csv"), index=False)
    band_cols = [f"p_{b}" for b in idx_mod.BANDS]
    pd.concat([pp[["osm_id", "landuse", "bid", "band_obs"]].reset_index(drop=True),
               pd.DataFrame(Pmat, columns=band_cols)], axis=1).to_csv(
        os.path.join(out_dir, "poi_band_probabilities.csv"), index=False)
    profile.to_csv(os.path.join(out_dir, "vertical_profile.csv"))
    for j, c in enumerate(band_cols):
        pp[c] = Pmat[:, j]
    from . import stacking as stack_mod
    stack_mod.building_band_uses(pp).round(3).to_csv(os.path.join(out_dir, "building_floor_uses.csv"), index=False)

    b_idx = grid = zgdf = None
    scales = {}
    if "building" in res_idx:
        b_idx = res_idx["building"].rename(columns={"id": "bid"}).merge(
            res[["bid", "name", "floors_final", "h_median", "lat", "lon"]], on="bid", how="left")
        top = pp.groupby("bid")["landuse"].agg(lambda v: v.value_counts().index[0])
        b_idx["main_use"] = b_idx["bid"].map(top)
        b_idx.to_csv(os.path.join(out_dir, "vertical_index_buildings.csv"), index=False)
    if "grid" in res_idx:
        grid = idx_mod.grid_polygons(res_idx["grid"], s.grid_m, utm)
        grid.to_file(os.path.join(out_dir, "vertical_index_grid.gpkg"), driver="GPKG")
        grid.drop(columns="geometry").to_csv(os.path.join(out_dir, "vertical_index_grid.csv"), index=False)
    if has_zones:
        zgdf = zones_mod.zone_table(inp.zones, inp.zone_name_col, res, pp, res_idx.get("zone"), utm,
                                    pop_col=inp.zone_pop_col, min_poi=s.min_poi_area, change=chg)
        zgdf.to_file(os.path.join(out_dir, "vertical_index_zones.gpkg"), driver="GPKG")
        zgdf.drop(columns="geometry").to_csv(os.path.join(out_dir, "vertical_index_zones.csv"), index=False)
        corr = zones_mod.correlations(zgdf, s.min_poi_area)
        if len(corr):
            corr.round(3).to_csv(os.path.join(out_dir, "zone_correlations_spearman.csv"))
            summary["zone_correlations"] = corr.round(3).to_dict()
    for k, t in res_idx.items():
        scales[k] = {"units": int(len(t)), "mean_n_poi": float(t["n_poi"].mean()),
                     **{c: float(np.nanmean(t[c])) for c in ["H_use", "H_band", "VSI", "VSI_null", "VSI_excess", "VSI_conf"]},
                     "share_excess_positive": float((t["VSI_excess"] > 0).mean())}
    summary["indices"] = scales
    summary["vertical_profile"] = profile.round(4).to_dict()

    figs = {}
    try:
        figs["floors"] = plots.floors_distribution(res, fig_dir, name.replace("_", " ").title())
        figs["calibration"] = plots.calibration(cal, fig_dir)
        figs["profile"] = plots.vertical_profile(profile, fig_dir)
        figs["grid"] = plots.grid_map(grid, inp.aoi, fig_dir)
        figs["buildings"] = plots.buildings_mix(b_idx, fig_dir)
    except Exception as e:  # noqa: BLE001 - figures must never break a run
        log(f"Figures: {e}")
    figs = {k: v for k, v in figs.items() if v}

    summary["runtime_min"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(RESULTS_README)
    zip_path = shutil.make_archive(out_dir, "zip", out_dir)
    P.finish("outputs")
    log(f"Done in {summary['runtime_min']} min. Results: {out_dir}")
    return Result(out_dir, zip_path, summary, res, poi, b_idx, grid, zgdf, profile, figs, pp)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    return str(o)


def _version():
    try:
        from importlib.metadata import version
        return version("vertical25d")
    except Exception:  # noqa: BLE001
        return "dev"


RESULTS_README = """Vertical land-use stratification: result files
==================================================
study_area.gpkg                 the study area used
buildings_floors.csv/.gpkg      OSM buildings with satellite height (h_median, h_p90), estimated floors,
                                final floors (floors_final), floor_source (osm/estimated/none) and quality flags
poi_buildings.csv/.gpkg         POIs with land use, parsed floor, linked building (bid) and match type
poi_band_probabilities.csv      floor-band probabilities used in the Monte Carlo draws
vertical_profile.csv            share of each land use on each floor band (mean of the draws)
building_floor_uses.csv         expected POIs per building x floor band x land use (n_known = known floors)
vertical_index_buildings.csv    indices for buildings with >= min_poi_building POIs
vertical_index_grid.csv/.gpkg   indices for grid cells with >= min_poi_area POIs
vertical_index_zones.csv/.gpkg  (if zones were given) indices + built form + POI density per zone
zone_correlations_spearman.csv  (if zones were given) Spearman correlations
building_change.csv             (if change analysis was run) new / taller / lower / stable buildings
summary.json                    all settings and headline numbers of the run
figures/                        PNG and SVG figures

Index columns: H_use and H_band are entropies normalised by ln K and ln B; VSI = MI / H_use;
VSI_null = VSI after shuffling floor bands within the unit; VSI_excess = VSI - VSI_null
(stratification beyond chance); VSI_conf = share of draws with VSI > VSI_null.
"""
