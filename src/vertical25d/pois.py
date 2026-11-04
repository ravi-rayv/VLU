"""POI land-use classification, floor extraction from tags and address text,
and linking of POIs to buildings (Section 4.2 of the manuscript)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import geopandas as gpd

from .config import LANDUSE

# ----------------------------------------------------------------------------- land use
DROP_AMENITY = {"bench", "waste_basket", "toilets", "parking", "parking_entrance", "parking_space", "bicycle_parking",
                "motorcycle_parking", "drinking_water", "vending_machine", "atm", "telephone", "post_box", "clock",
                "fountain", "shelter", "recycling", "waste_disposal", "charging_station", "taxi", "bus_station",
                "water_point", "watering_place", "grave_yard", "hunting_stand", "loading_dock", "compressed_air",
                "letter_box", "bbq", "feeding_place", "smoking_area", "public_bookcase", "give_box",
                "device_charging_station"}
FOOD = {"restaurant", "fast_food", "cafe", "bar", "pub", "food_court", "ice_cream", "biergarten", "juice_bar"}
HEALTH = {"hospital", "clinic", "doctors", "dentist", "pharmacy", "veterinary", "nursing_home"}
EDU = {"school", "college", "university", "kindergarten", "library", "training", "language_school",
       "driving_school", "music_school", "research_institute", "childcare", "prep_school", "tuition"}
FINANCE = {"bank", "bureau_de_change", "money_transfer", "payment_centre"}
CIVIC = {"police", "townhall", "courthouse", "fire_station", "post_office", "community_centre", "social_facility",
         "public_building", "embassy", "prison"}
LEISURE_A = {"cinema", "theatre", "nightclub", "arts_centre", "events_venue", "casino", "studio", "conference_centre"}
WORSHIP = {"place_of_worship", "monastery"}
PERSONAL_SHOP = {"hairdresser", "beauty", "laundry", "dry_cleaning", "tailor", "massage", "tattoo", "optician",
                 "travel_agency", "copyshop", "mobile_phone_repair", "funeral_directors"}
FOOD_SHOP = {"bakery", "pastry", "confectionery", "coffee", "tea"}
HOSPITALITY = {"hotel", "guest_house", "hostel", "motel", "apartment"}
LEISURE_L = {"fitness_centre", "sports_centre", "dance", "bowling_alley", "amusement_arcade",
             "escape_game", "sports_hall", "trampoline_park"}


def _s(v):
    return v if isinstance(v, str) and v.strip() else None


def classify_row(amenity=None, shop=None, office=None, tourism=None, leisure=None, healthcare=None, craft=None):
    a, s, o, t, l, h, c = map(_s, (amenity, shop, office, tourism, leisure, healthcare, craft))
    if a and a in DROP_AMENITY:
        return None
    if o:
        return "Civic" if o in {"government", "diplomatic"} else "Office"
    if a:
        if a in FOOD: return "Food & drink"
        if a in HEALTH: return "Health"
        if a in EDU: return "Education"
        if a in FINANCE: return "Finance"
        if a in CIVIC: return "Civic"
        if a in LEISURE_A: return "Leisure"
        if a in WORSHIP: return "Religious"
        if a == "coworking_space": return "Office"
    if h: return "Health"
    if t and t in HOSPITALITY: return "Hospitality"
    if l and l in LEISURE_L: return "Leisure"
    if s:
        if s in PERSONAL_SHOP: return "Personal services"
        if s in FOOD_SHOP: return "Food & drink"
        return "Retail"
    if c: return "Personal services"
    return None


def classify(poi: pd.DataFrame) -> pd.DataFrame:
    poi = poi.copy()
    cols = ["amenity", "shop", "office", "tourism", "leisure", "healthcare", "craft"]
    for c in cols:
        if c not in poi.columns:
            poi[c] = None
    lu = [classify_row(*vals) for vals in poi[cols].itertuples(index=False, name=None)]
    if "landuse" in poi.columns:  # POIs that arrive already classified keep their land use
        given = poi["landuse"].where(poi["landuse"].isin(LANDUSE))
        lu = pd.Series(lu, index=poi.index).fillna(given).tolist()
    poi["landuse"] = lu
    poi = poi[poi["landuse"].isin(LANDUSE)].copy()
    cat = poi["amenity"]
    for c in cols[1:]:
        cat = cat.fillna(poi[c])
    poi["category"] = cat.fillna(poi["category"]) if "category" in poi.columns else cat
    return poi


# ----------------------------------------------------------------------------- floors
WORD_NUM = {"ground": 0, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12}
RX_NUM = re.compile(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:floor|flr|fl)\b", re.I)
RX_WORD = re.compile(r"\b(" + "|".join(WORD_NUM) + r")\s*(?:floor|flr|fl)\b", re.I)
RX_FLOOR_N = re.compile(r"\bfloor\s*(?:no\.?\s*)?(\d{1,2})\b", re.I)
RX_CODES = {"ugf": 0, "upper ground": 0, "gf": 0, "grd": 0, "lgf": -1, "lower ground": -1,
            "basement": -1, "cellar": -1, "mezzanine": 0, "terrace": None}
RX_CODE = re.compile(r"\b(" + "|".join(sorted(RX_CODES, key=len, reverse=True)) + r")\b", re.I)


def parse_level_tag(v):
    """OSM ``level`` / ``addr:floor``: '0;1' or '0-2' -> lowest floor."""
    if not isinstance(v, str) or not v.strip():
        if isinstance(v, (int, float)) and not pd.isna(v):
            return float(v)
        return np.nan
    v = re.sub(r"(?<=\d)\s*[-–]\s*(?=\d)", ";", v)   # '0-2' is a range, not 0 and -2
    nums = re.findall(r"-?\d+(?:\.\d+)?", v)
    return float(min(float(n) for n in nums)) if nums else np.nan


def parse_text(s):
    """Floor number from free text: '3rd Floor', 'Floor No. 3', 'First floor', GF, LGF, basement..."""
    if not isinstance(s, str) or not s.strip():
        return np.nan
    for rx in (RX_NUM, RX_FLOOR_N):
        m = rx.search(s)
        if m:
            return float(m.group(1))
    m = RX_WORD.search(s)
    if m:
        return float(WORD_NUM[m.group(1).lower()])
    m = RX_CODE.search(s)
    if m:
        v = RX_CODES[m.group(1).lower()]
        return np.nan if v is None else float(v)
    return np.nan


def extract_floors(poi: pd.DataFrame) -> pd.DataFrame:
    poi = poi.copy()
    for c in ["level", "addr_floor"]:
        if c not in poi.columns:
            poi[c] = None
    text_cols = [c for c in ["name", "address", "addr_full", "addr_unit", "addr_housenumber", "addr_street"]
                 if c in poi.columns]
    poi["addr_text"] = (poi[text_cols].fillna("").astype(str).agg(" | ".join, axis=1)
                        if text_cols else "")
    poi["floor_tag"] = poi["level"].map(parse_level_tag).fillna(poi["addr_floor"].map(parse_level_tag))
    poi["floor_text"] = poi["addr_text"].map(parse_text)
    poi["floor"] = poi["floor_tag"].fillna(poi["floor_text"])
    poi["floor_source"] = np.select([poi["floor_tag"].notna(), poi["floor_text"].notna()],
                                    ["osm_tag", "address_text"], "unknown")
    return poi


# ----------------------------------------------------------------------------- linking
def link_to_buildings(poi: gpd.GeoDataFrame, bld: gpd.GeoDataFrame, utm, match_dist_m=20.0) -> gpd.GeoDataFrame:
    """POI inside a footprint, else nearest footprint within ``match_dist_m``."""
    poi = poi.copy().reset_index(drop=True)
    b = bld[["bid", "geometry"]].to_crs(utm)
    pts = poi.to_crs(utm)
    inside = gpd.sjoin(pts[["geometry"]], b, how="left", predicate="within")
    inside = inside[~inside.index.duplicated()]
    poi["bid"] = inside["bid"].values
    poi["match"] = np.where(poi["bid"].notna(), "inside", None)
    poi["match_dist_m"] = np.where(poi["bid"].notna(), 0.0, np.nan)
    miss = pts[poi["bid"].isna().values]
    if len(miss):
        near = gpd.sjoin_nearest(miss[["geometry"]], b, how="left", max_distance=match_dist_m, distance_col="d")
        near = near[~near.index.duplicated()]
        ok = near["bid"].notna()
        poi.loc[near.index[ok], "bid"] = near.loc[ok, "bid"].values
        poi.loc[near.index[ok], "match"] = "nearest"
        poi.loc[near.index[ok], "match_dist_m"] = near.loc[ok, "d"].round(1).values
    poi["match"] = poi["match"].fillna("no_building")
    bcols = ["bid", "floors_final", "floor_source", "h_median", "footprint_m2"]
    poi = poi.merge(bld[bcols].rename(columns={"floor_source": "bld_floor_source"}), on="bid", how="left")
    poi["floor_above_building"] = (poi["floor"].notna() & poi["floors_final"].notna()
                                   & (poi["floor"] > poi["floors_final"] - 1))
    poi["lon"], poi["lat"] = poi.geometry.x.round(6), poi.geometry.y.round(6)
    return gpd.GeoDataFrame(poi, geometry="geometry", crs=4326)


def assign_zones(poi, zones, name_col):
    j = gpd.sjoin(poi[["geometry"]], zones[[name_col, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    out = poi.copy()
    out["zone"] = j[name_col].values
    return out
