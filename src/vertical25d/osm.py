"""Download OpenStreetMap building footprints and points of interest (POIs)
for a study area, trying several Overpass servers in turn."""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import geopandas as gpd

from .config import OVERPASS_URLS

POI_TAGS = {"amenity": True, "shop": True, "office": True, "tourism": True,
            "leisure": True, "healthcare": True, "craft": True}

POI_KEYS = ["amenity", "shop", "office", "tourism", "leisure", "healthcare", "craft", "sport",
            "level", "addr:floor", "addr:unit", "addr:full", "addr:housenumber", "addr:street",
            "brand", "cuisine", "name"]


def _features(polygon, tags, log=print, timeout=600):
    import osmnx as ox
    ox.settings.requests_timeout = timeout
    ox.settings.use_cache = True
    last = None
    for url in OVERPASS_URLS:
        try:
            ox.settings.overpass_url = url
            log(f"OSM: querying {url}")
            return ox.features_from_polygon(polygon, tags=tags)
        except Exception as e:  # noqa: BLE001 - try the next server
            msg = str(e)
            if "No matching features" in msg or "InsufficientResponse" in type(e).__name__:
                return gpd.GeoDataFrame(geometry=[], crs=4326)
            last = e
            log(f"OSM: {url} failed ({type(e).__name__}); trying the next server")
            time.sleep(3)
    raise RuntimeError(f"All Overpass servers failed: {last}")


def _flatten_index(gdf):
    gdf = gdf.reset_index()
    return gdf.rename(columns={"id": "osmid", "element": "osm_type", "element_type": "osm_type"})


def fetch_buildings(aoi_geom, utm, log=print) -> gpd.GeoDataFrame:
    """OSM building polygons whose representative point lies in the study area."""
    gdf = _features(aoi_geom, {"building": True}, log=log)
    if gdf.empty:
        raise RuntimeError("No OSM buildings found in the study area")
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = _flatten_index(gdf)
    return clean_buildings(gdf, aoi_geom, utm)


def clean_buildings(gdf, aoi_geom, utm) -> gpd.GeoDataFrame:
    """Standardise columns: bid, osmid, osm_type, building, building_levels, name, geometry."""
    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    gdf.columns = [c.replace(":", "_") for c in gdf.columns]
    inside = gdf.to_crs(utm).representative_point().to_crs(4326).within(aoi_geom).values
    gdf = gdf[inside]
    for c in ["osm_type", "osmid", "building", "building_levels", "name"]:
        if c not in gdf.columns:
            gdf[c] = np.nan
    if gdf["osmid"].isna().all():
        gdf["osmid"] = np.arange(len(gdf))
        gdf["osm_type"] = "x"
    if "bid" not in gdf.columns:
        gdf["bid"] = gdf["osm_type"].astype(str).str[0] + gdf["osmid"].astype(str)
    gdf["building_levels"] = pd.to_numeric(gdf["building_levels"], errors="coerce")
    gdf["name"] = gdf["name"].fillna("").astype(str)
    gdf["building"] = gdf["building"].fillna("yes").astype(str)
    gdf = gdf[["bid", "osmid", "osm_type", "building", "building_levels", "name", "geometry"]]
    return gdf.drop_duplicates("bid").reset_index(drop=True)


def fetch_pois(aoi_geom, log=print) -> gpd.GeoDataFrame:
    """OSM POIs (nodes, and centroids of tagged ways) with the tags the method needs."""
    gdf = _features(aoi_geom, POI_TAGS, log=log)
    if gdf.empty:
        raise RuntimeError("No OSM POIs found in the study area")
    gdf = _flatten_index(gdf)
    return standardise_pois(gdf, aoi_geom)


KV = re.compile(r'"([^"]+)"=>"((?:[^"\\]|\\.)*)"')


def standardise_pois(gdf, aoi_geom) -> gpd.GeoDataFrame:
    """Turn any POI layer (OSM download, QGIS export with ``other_tags``,
    GeoJSON/GPKG) into points with columns amenity, shop, office, ...,
    addr_floor, level, name."""
    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    if "other_tags" in gdf.columns:  # QGIS/ogr OSM export: hstore-like string
        tags = gdf["other_tags"].fillna("").map(lambda s: dict(KV.findall(str(s))))
        for k in POI_KEYS:
            if k.replace(":", "_") not in gdf.columns and k not in gdf.columns:
                gdf[k] = tags.map(lambda d, k=k: d.get(k))
    gdf.columns = [c.replace(":", "_") for c in gdf.columns]
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    for k in POI_KEYS:
        c = k.replace(":", "_")
        if c not in gdf.columns:
            gdf[c] = None
    # polygons (shops mapped as buildings, malls...) -> representative point
    poly = ~gdf.geom_type.isin(["Point"])
    if poly.any():
        utm = gdf.estimate_utm_crs()
        gdf.loc[poly, "geometry"] = gdf.loc[poly].to_crs(utm).representative_point().to_crs(4326).values
    gdf = gdf[gdf.geom_type == "Point"]
    gdf = gdf[gdf.within(aoi_geom)]
    if "osmid" in gdf.columns:
        gdf["osm_id"] = gdf.get("osm_type", "n").astype(str).str[0] + gdf["osmid"].astype(str)
    elif "osm_id" not in gdf.columns:
        gdf["osm_id"] = [f"p{i}" for i in range(len(gdf))]
    for c in gdf.columns:
        if c != "geometry" and gdf[c].dtype == object:
            gdf[c] = gdf[c].map(lambda v: v if (v is None or isinstance(v, str)) else (None if pd.isna(v) else str(v)))
    return gdf.reset_index(drop=True)
