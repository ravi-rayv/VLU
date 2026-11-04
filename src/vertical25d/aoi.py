"""Study area (area of interest) from a KML/KMZ, GeoJSON, zipped shapefile,
GeoPackage or a polygon drawn on the web map.

KML is parsed directly (no GDAL KML driver needed), so it works on any install.
"""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import unary_union


class AOIError(ValueError):
    """Raised when a study area cannot be read or is unsuitable."""


# ----------------------------------------------------------------------------- KML
def _coords(text: str):
    pts = []
    for tok in re.split(r"\s+", (text or "").strip()):
        if not tok:
            continue
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def _strip_ns(root):
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def polygons_from_kml(kml_bytes: bytes):
    """Return a list of shapely Polygons found in a KML document."""
    root = _strip_ns(ET.fromstring(kml_bytes))
    polys = []
    for pg in root.iter("Polygon"):
        outer = pg.find("outerBoundaryIs/LinearRing/coordinates")
        if outer is None:
            continue
        shell = _coords(outer.text)
        holes = [_coords(c.text) for c in pg.findall("innerBoundaryIs/LinearRing/coordinates")]
        if len(shell) >= 3:
            polys.append(Polygon(shell, [h for h in holes if len(h) >= 3]))
    if not polys:  # a closed LineString drawn as a boundary also counts
        for ls in root.iter("LineString"):
            c = ls.find("coordinates")
            pts = _coords(c.text if c is not None else "")
            if len(pts) >= 4 and pts[0] == pts[-1]:
                polys.append(Polygon(pts))
    return polys


# ----------------------------------------------------------------------------- readers
def read_aoi(source, filename: str | None = None) -> gpd.GeoDataFrame:
    """Read a study area.

    ``source`` may be a path, raw bytes (uploaded file), a GeoJSON dict
    (e.g. the polygon drawn in the portal) or a shapely geometry.
    Returns a one-row GeoDataFrame in EPSG:4326.
    """
    if hasattr(source, "geom_type"):
        geoms = [source]
    elif isinstance(source, dict):
        geoms = _geoms_from_geojson(source)
    else:
        if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
            filename = filename or str(source)
            with open(source, "rb") as f:
                data = f.read()
        elif isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        else:
            raise AOIError("Unsupported study-area input")
        geoms = _geoms_from_bytes(data, (filename or "").lower())
    geom = unary_union([g for g in geoms if g is not None and not g.is_empty])
    geom = geom.buffer(0)
    if geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise AOIError("No polygon found. Draw a polygon or upload a file with polygon(s).")
    return gpd.GeoDataFrame({"name": ["study_area"]}, geometry=[geom], crs=4326)


def _geoms_from_geojson(gj: dict):
    if gj.get("type") == "FeatureCollection":
        return [shape(f["geometry"]) for f in gj["features"] if f.get("geometry")]
    if gj.get("type") == "Feature":
        return [shape(gj["geometry"])]
    return [shape(gj)]


def _geoms_from_bytes(data: bytes, name: str):
    if name.endswith(".kmz") or (data[:2] == b"PK" and not name.endswith(".zip")):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if kml_names:
                return polygons_from_kml(z.read(kml_names[0]))
        # not a KMZ: fall through to zip handling
    if name.endswith(".kml") or data.lstrip()[:5] in (b"<?xml", b"<kml ", b"<kml>"):
        return polygons_from_kml(data)
    if name.endswith((".geojson", ".json")):
        return _geoms_from_geojson(json.loads(data.decode("utf-8")))
    # zipped shapefile / GeoPackage / anything GDAL can read
    with tempfile.TemporaryDirectory() as tmp:
        if data[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(tmp)
            cands = [os.path.join(r, f) for r, _, fs in os.walk(tmp) for f in fs
                     if f.lower().endswith((".shp", ".gpkg", ".geojson", ".kml"))]
            if not cands:
                raise AOIError("The zip contains no .shp, .gpkg, .geojson or .kml file")
            path = cands[0]
            if path.lower().endswith(".kml"):
                with open(path, "rb") as f:
                    return polygons_from_kml(f.read())
        else:
            path = os.path.join(tmp, os.path.basename(name) or "aoi.gpkg")
            with open(path, "wb") as f:
                f.write(data)
        g = gpd.read_file(path)
        if g.crs is None:
            g = g.set_crs(4326)
        return list(g.to_crs(4326).geometry)


# ----------------------------------------------------------------------------- checks
def area_km2(aoi: gpd.GeoDataFrame) -> float:
    return float(aoi.to_crs(aoi.estimate_utm_crs()).area.sum() / 1e6)


def check_aoi(aoi: gpd.GeoDataFrame, max_area_km2: float) -> dict:
    a = area_km2(aoi)
    if a > max_area_km2:
        raise AOIError(f"Study area is {a:,.0f} km², larger than the limit of {max_area_km2:,.0f} km². "
                       "Split it into parts or raise max_area_km2 when running locally.")
    minx, miny, maxx, maxy = aoi.total_bounds
    return {"area_km2": round(a, 2), "bounds": [round(v, 5) for v in (minx, miny, maxx, maxy)],
            "utm_epsg": aoi.estimate_utm_crs().to_epsg()}


def aoi_geometry(aoi: gpd.GeoDataFrame):
    g = aoi.geometry
    return (g.union_all() if hasattr(g, "union_all") else g.unary_union).buffer(0)


def read_zones(source, filename: str | None, name_col: str | None = None):
    """Optional zones (e.g. wards) for zone-level statistics. Keeps all attributes."""
    if isinstance(source, (bytes, bytearray)):
        with tempfile.TemporaryDirectory() as tmp:
            nm = (filename or "zones").lower()
            if nm.endswith(".kml") or nm.endswith(".kmz"):
                geoms = _geoms_from_bytes(bytes(source), nm)
                z = gpd.GeoDataFrame({"zone": [f"zone_{i+1}" for i in range(len(geoms))]}, geometry=geoms, crs=4326)
                return z, "zone"
            if bytes(source)[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(bytes(source))) as zf:
                    zf.extractall(tmp)
                cands = [os.path.join(r, f) for r, _, fs in os.walk(tmp) for f in fs
                         if f.lower().endswith((".shp", ".gpkg", ".geojson"))]
                if not cands:
                    raise AOIError("The zones zip contains no .shp, .gpkg or .geojson")
                path = cands[0]
            else:
                path = os.path.join(tmp, os.path.basename(nm))
                with open(path, "wb") as f:
                    f.write(bytes(source))
            z = gpd.read_file(path)
    else:
        z = gpd.read_file(source)
    if z.crs is None:
        z = z.set_crs(4326)
    z = z.to_crs(4326)
    z = z[z.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if name_col is None or name_col not in z.columns:
        text_cols = [c for c in z.columns if c != "geometry" and z[c].dtype == object and z[c].is_unique]
        name_col = text_cols[0] if text_cols else None
    if name_col is None:
        z["zone"] = [f"zone_{i+1}" for i in range(len(z))]
        name_col = "zone"
    z[name_col] = z[name_col].astype(str)
    return z, name_col
