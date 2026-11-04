"""All tunable settings in one place.

Defaults reproduce the Bengaluru study (CEUS manuscript). Every value can be
changed from the Streamlit portal, the command line (``--set key=value``) or
Python (``Settings(grid_m=300)``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Settings:
    # --- Earth Engine heights -------------------------------------------------
    year: int = 2023                  # Open Buildings 2.5D Temporal layer used for heights
    chunk_size: int = 2000            # buildings per Earth Engine request (< 10 MB payload)
    workers: int = 6                  # parallel Earth Engine requests
    gee_scale_m: int = 4              # reduceRegions scale (dataset's effective resolution)

    # --- floor model ----------------------------------------------------------
    min_height_m: float = 2.5         # below this, a building has no usable height
    ground_floor_m: float = 4.0       # only used by the simple 'formula' model
    floor_model: str = "linear"       # 'linear' (Huber), 'isotonic', 'formula' or 'auto'
    min_calibration: int = 30         # fewer tagged buildings -> fall back to formula
    keep_higher_osm_tags: bool = False
    # False reproduces the paper: an OSM tag that disagrees strongly with the
    # satellite height is replaced by the estimate. True keeps OSM tags that are
    # HIGHER than the estimate (satellite heights saturate on slender towers).

    # --- POIs -----------------------------------------------------------------
    match_dist_m: float = 20.0        # POIs outside every footprint snap to nearest building within this distance

    # --- vertical indices -----------------------------------------------------
    n_sim: int = 200                  # Monte Carlo draws of unknown floors
    min_poi_building: int = 3         # buildings need at least this many POIs
    min_poi_area: int = 10            # grid cells / zones need at least this many POIs
    grid_m: int = 500                 # grid cell size (metres)
    seed: int = 42
    min_known_floors: int = 30        # fewer POIs with a known floor -> use default priors

    # --- change 2016 -> 2023 (optional) ----------------------------------------
    run_change: bool = False
    early_years: list = field(default_factory=lambda: [2016, 2017])
    late_years: list = field(default_factory=lambda: [2022, 2023])
    p_absent: float = 0.3
    p_present: float = 0.5
    min_dh_m: float = 3.0

    # --- safety limits for the web portal ---------------------------------------
    max_area_km2: float = 400.0

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, **kw) -> "Settings":
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(f"Unknown setting: {k}")
            cur = getattr(self, k)
            if isinstance(cur, bool) and isinstance(v, str):
                v = v.lower() in ("1", "true", "yes", "y")
            elif isinstance(cur, int) and not isinstance(cur, bool) and isinstance(v, str):
                v = int(v)
            elif isinstance(cur, float) and isinstance(v, str):
                v = float(v)
            setattr(self, k, v)
        return self


LANDUSE = ["Retail", "Food & drink", "Office", "Personal services", "Health", "Finance",
           "Education", "Hospitality", "Leisure", "Civic", "Religious"]

BANDS = ["Basement", "Ground", "1st", "2nd–3rd", "4th–7th", "8th+"]
BAND_START = [-99, 0, 1, 2, 4, 8]      # lowest floor number in each band (ground = 0)

GROUPS = {"Retail & food": ["Retail", "Food & drink"], "Office": ["Office"],
          "Services": ["Personal services", "Health", "Finance", "Education", "Civic", "Religious"],
          "Leisure & hotels": ["Leisure", "Hospitality"]}

OVERPASS_URLS = ["https://overpass-api.de/api",
                 "https://overpass.private.coffee/api",
                 "https://overpass.kumi.systems/api",
                 "https://maps.mail.ru/osm/tools/overpass/api"]
