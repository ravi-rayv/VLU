"""vertical25d: measure vertical land-use stratification from open POIs and
satellite building heights (2.5D) for any study area.

Authors: Ravi Satyappa Dabbanavar and Rohini N, Nitte School of Architecture,
Planning and Design, Nitte University, Bengaluru, India (ravisd@nittesoa.ac.in).

Quick start::

    from vertical25d import Settings, Inputs, read_aoi, run
    aoi = read_aoi("my_area.kml")
    result = run(Inputs(aoi=aoi, ee_project="my-ee-project", name="my_area"), Settings())
    print(result.summary["indices"])
"""
from .config import Settings, LANDUSE, BANDS
from .aoi import read_aoi, read_zones, area_km2, AOIError
from .pipeline import Inputs, Result, run

__all__ = ["Settings", "LANDUSE", "BANDS", "read_aoi", "read_zones", "area_km2", "AOIError",
           "Inputs", "Result", "run"]
__version__ = "0.1.0"
__author__ = "Ravi Satyappa Dabbanavar, Rohini N"
__email__ = "ravisd@nittesoa.ac.in"
