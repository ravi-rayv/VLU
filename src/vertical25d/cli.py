"""Command-line tool.

Examples::

    vertical25d run --aoi my_area.kml --ee-project my-ee-project
    vertical25d run --aoi wards.zip --zones wards.zip --zone-name name_en --zone-pop population \
                    --ee-project my-ee-project --set n_sim=100 --set grid_m=300 --change
"""
from __future__ import annotations

import argparse
import json
import sys

from .aoi import read_aoi, read_zones
from .config import Settings
from .pipeline import Inputs, run


def main(argv=None):
    p = argparse.ArgumentParser(prog="vertical25d", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline for a study area")
    r.add_argument("--aoi", required=True, help="study area: .kml, .kmz, .geojson, .gpkg or zipped shapefile")
    r.add_argument("--ee-project", help="your Google Earth Engine project ID")
    r.add_argument("--ee-key", help="optional: your own Earth Engine service-account key (JSON file)")
    r.add_argument("--name", help="name used for the output folder (default: file name)")
    r.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    r.add_argument("--zones", help="optional zones (e.g. wards) for zone statistics")
    r.add_argument("--zone-name", help="column with the zone name")
    r.add_argument("--zone-pop", help="optional column with zone population")
    r.add_argument("--pois", help="optional POI layer instead of downloading from OSM")
    r.add_argument("--change", action="store_true", help="also run the 2016-2023 change analysis")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a setting, e.g. --set n_sim=100 (see vertical25d settings)")
    d = sub.add_parser("demo", help="run the offline demo (no internet or Earth Engine needed)")
    d.add_argument("--out", default="outputs")
    sub.add_parser("settings", help="print all settings and their defaults")
    a = p.parse_args(argv)

    if a.cmd == "settings":
        print(json.dumps(Settings().to_dict(), indent=2))
        return 0

    if a.cmd == "demo":
        from .demo import demo_inputs
        res = run(demo_inputs(), Settings(grid_m=250, run_change=True), out_dir=a.out)
        print("Results zip:", res.zip_path)
        return 0

    s = Settings()
    for kv in a.set:
        k, v = kv.split("=", 1)
        s.update(**{k.strip(): v.strip()})
    if a.change:
        s.run_change = True
    aoi = read_aoi(a.aoi)
    zones = name_col = None
    if a.zones:
        zones, name_col = read_zones(a.zones, a.zones, a.zone_name)
    pois = None
    if a.pois:
        import geopandas as gpd
        pois = gpd.read_file(a.pois)
    key = open(a.ee_key).read() if a.ee_key else None
    import os
    name = a.name or os.path.splitext(os.path.basename(a.aoi))[0]
    res = run(Inputs(aoi=aoi, ee_project=a.ee_project, service_account_json=key, zones=zones,
                     zone_name_col=name_col, zone_pop_col=a.zone_pop, pois=pois, name=name),
              s, out_dir=a.out)
    print("Results zip:", res.zip_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
