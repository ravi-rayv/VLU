"""Offline demo: 680 m around World Trade Center Bengaluru (Brigade Gateway,
Malleshwaram / Rajajinagar). Buildings and POIs © OpenStreetMap contributors
(ODbL); heights from Google Open Buildings 2.5D Temporal (CC BY 4.0), 2023.

The demo needs no internet connection and no Earth Engine account.
"""
from __future__ import annotations

import os

import geopandas as gpd
import pandas as pd

from .aoi import read_aoi
from .pipeline import Inputs

DEMO_DIR = os.path.join(os.path.dirname(__file__), "data", "demo")


def demo_inputs() -> Inputs:
    aoi = read_aoi(os.path.join(DEMO_DIR, "aoi.kml"))
    return Inputs(aoi=aoi,
                  buildings=gpd.read_file(os.path.join(DEMO_DIR, "buildings.gpkg")),
                  heights=pd.read_csv(os.path.join(DEMO_DIR, "heights.csv")),
                  pois=gpd.read_file(os.path.join(DEMO_DIR, "pois.gpkg")),
                  change_layers=pd.read_csv(os.path.join(DEMO_DIR, "change_layers.csv")),
                  name="demo_brigade_gateway")
