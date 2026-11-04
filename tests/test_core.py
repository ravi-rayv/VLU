"""Offline tests: no internet or Earth Engine needed."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Polygon, Point, box

from vertical25d import Settings, read_aoi, AOIError
from vertical25d.aoi import polygons_from_kml, check_aoi
from vertical25d import pois as P, floors as F, indices as I

SQUARE = [(77.55, 13.00), (77.56, 13.00), (77.56, 13.01), (77.55, 13.01), (77.55, 13.00)]
KML = ("<?xml version='1.0'?><kml xmlns='http://www.opengis.net/kml/2.2'><Document><Placemark><Polygon>"
       "<outerBoundaryIs><LinearRing><coordinates>"
       + " ".join(f"{x},{y},0" for x, y in SQUARE) +
       "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>").encode()


# ----------------------------------------------------------------------------- study area
def test_kml_and_kmz():
    assert polygons_from_kml(KML)[0].equals(Polygon(SQUARE))
    a = read_aoi(KML, "area.kml")
    assert a.crs.to_epsg() == 4326 and a.geometry.iloc[0].area > 0
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("doc.kml", KML)
    assert read_aoi(buf.getvalue(), "area.kmz").geometry.iloc[0].equals(a.geometry.iloc[0])


def test_drawn_geojson_and_limits():
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
          "geometry": {"type": "Polygon", "coordinates": [SQUARE]}}]}
    a = read_aoi(gj)
    info = check_aoi(a, 400)
    assert 1.0 < info["area_km2"] < 1.3
    with pytest.raises(AOIError):
        check_aoi(a, 0.5)
    with pytest.raises(AOIError):
        read_aoi({"type": "Point", "coordinates": [77.5, 13.0]})


# ----------------------------------------------------------------------------- POIs
@pytest.mark.parametrize("text,floor", [("Shop 4, 3rd Floor, Brigade Rd", 3), ("Floor No. 2", 2),
                                        ("First floor, MG Road", 1), ("UGF-12", 0), ("LGF", -1),
                                        ("Basement 1", -1), ("Terrace", np.nan), ("Main road", np.nan)])
def test_floor_text(text, floor):
    v = P.parse_text(text)
    assert (np.isnan(v) and np.isnan(floor)) or v == floor


def test_level_tag_and_classify():
    assert P.parse_level_tag("0;1") == 0 and P.parse_level_tag("2-4") == 2 and P.parse_level_tag("-1") == -1 and np.isnan(P.parse_level_tag(""))
    assert P.classify_row(amenity="restaurant") == "Food & drink"
    assert P.classify_row(shop="hairdresser") == "Personal services"
    assert P.classify_row(office="government") == "Civic"
    assert P.classify_row(amenity="bench") is None
    assert P.classify_row(shop="clothes") == "Retail"


def test_link_inside_and_nearest():
    bld = gpd.GeoDataFrame({"bid": ["a", "b"], "floors_final": [3.0, 10.0], "floor_source": ["osm", "estimated"],
                            "h_median": [9.0, 31.0], "footprint_m2": [100.0, 400.0]},
                           geometry=[box(77.5500, 13.0000, 77.5502, 13.0002), box(77.5510, 13.0000, 77.5512, 13.0002)],
                           crs=4326)
    poi = gpd.GeoDataFrame({"floor": [1.0, 5.0, np.nan]},
                           geometry=[Point(77.5501, 13.0001), Point(77.55125, 13.0001), Point(77.56, 13.01)], crs=4326)
    out = P.link_to_buildings(poi, bld, 32643, 20)
    assert list(out["match"]) == ["inside", "nearest", "no_building"]
    assert list(out["bid"][:2]) == ["a", "b"]
    assert not out["floor_above_building"].iloc[0]


# ----------------------------------------------------------------------------- floors
def test_calibration_recovers_storey_height():
    rng = np.random.default_rng(0)
    lv = rng.integers(1, 12, 400).astype(float)
    h = 1.0 + 3.3 * lv + rng.normal(0, 0.8, 400)
    res = pd.DataFrame({"bid": range(400), "building_levels": lv, "h_median": h, "h_p90": h + 1,
                        "building": "yes", "footprint_m2": 200.0})
    s = Settings()
    cal = F.calibrate(res, s, log=lambda m: None)
    assert abs(cal["storey_m"] - 3.3) < 0.2
    out = F.assign_floors(res, cal, s)
    assert set(out["floor_source"]) <= {"osm", "estimated", "none"}
    assert (out["floors_final"] >= 1).all()


def test_keep_higher_osm_tags():
    res = pd.DataFrame({"bid": [1], "building_levels": [32.0], "h_median": [49.4], "h_p90": [72.0],
                        "building": ["commercial"], "footprint_m2": [3800.0]})
    cal = {"predict": lambda x: 0.67 + 0.301 * np.asarray(x), "models": {}, "model": "linear"}
    assert F.assign_floors(res, cal, Settings())["floors_final"].iloc[0] == 16
    assert F.assign_floors(res, cal, Settings(keep_higher_osm_tags=True))["floors_final"].iloc[0] == 32


# ----------------------------------------------------------------------------- indices
def test_vsi_extremes():
    unit = np.zeros(8, int)
    use = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    *_, vsi_sep = I._indices(unit, 1, np.array([1, 1, 1, 1, 2, 2, 2, 2]), use)   # each use on its own floor
    *_, vsi_mix = I._indices(unit, 1, np.array([1, 2, 1, 2, 1, 2, 1, 2]), use)   # same mix on every floor
    assert vsi_sep[0] == pytest.approx(1.0) and vsi_mix[0] == pytest.approx(0.0)


def test_demo_pipeline(tmp_path):
    from vertical25d import run
    from vertical25d.demo import demo_inputs
    r = run(demo_inputs(), Settings(n_sim=20, grid_m=250, run_change=True), out_dir=str(tmp_path),
            log=lambda m: None)
    assert len(r.buildings) > 2000 and r.summary["pois"]["n_classified"] > 200
    assert {"building", "grid"} <= set(r.summary["indices"])
    assert abs(r.profile.drop(columns="expected_POIs").sum(axis=1) - 1).max() < 1e-6
    with open(f"{r.out_dir}/summary.json") as f:
        assert json.load(f)["floors"]["model"] == "linear"


def test_osmnx_style_pois_and_buildings():
    """osmnx returns a (element, id) MultiIndex and 'addr:floor'-style columns."""
    from vertical25d.osm import standardise_pois, clean_buildings
    idx = pd.MultiIndex.from_tuples([("node", 1), ("way", 2), ("node", 3)], names=["element", "id"])
    raw = gpd.GeoDataFrame({"amenity": ["cafe", None, "bench"], "shop": [None, "clothes", None],
                            "addr:floor": ["2", None, None], "name": ["A", "B", "C"]},
                           geometry=[Point(77.551, 13.001), box(77.552, 13.002, 77.5522, 13.0022), Point(77.553, 13.003)],
                           index=idx, crs=4326)
    aoi = box(77.55, 13.0, 77.56, 13.01)
    pts = standardise_pois(raw.reset_index().rename(columns={"id": "osmid", "element": "osm_type"}), aoi)
    assert len(pts) == 3 and (pts.geom_type == "Point").all() and "addr_floor" in pts.columns
    out = P.extract_floors(P.classify(pts))
    assert list(out["landuse"]) == ["Food & drink", "Retail"] and out["floor"].iloc[0] == 2
    bld = gpd.GeoDataFrame({"building": ["yes"], "building:levels": ["4"]},
                           geometry=[box(77.551, 13.001, 77.5512, 13.0012)],
                           index=pd.MultiIndex.from_tuples([("way", 9)], names=["element", "id"]), crs=4326)
    b = clean_buildings(bld.reset_index().rename(columns={"id": "osmid", "element": "osm_type"}), aoi, 32643)
    assert b["bid"].iloc[0] == "w9" and b["building_levels"].iloc[0] == 4
