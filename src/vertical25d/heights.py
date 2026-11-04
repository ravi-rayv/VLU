"""Building heights from the Open Buildings 2.5D Temporal dataset in Google
Earth Engine. Each user signs in with their own Earth Engine project.

Only the building ID and a slightly simplified outline are sent to Earth
Engine; all other attributes stay on the user's machine.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

DATASET = "GOOGLE/Research/open-buildings-temporal/v1"


def init_earth_engine(project: str, service_account_json: str | bytes | None = None):
    """Initialise Earth Engine with the user's own project.

    * Local use: run ``earthengine authenticate`` once, then pass the project ID.
    * Hosted portal: the user may upload their OWN service-account key (JSON);
      it is used in memory only and never written to disk.
    """
    import ee
    if not project:
        raise ValueError("An Earth Engine project ID is required")
    if service_account_json:
        info = json.loads(service_account_json if isinstance(service_account_json, str)
                          else service_account_json.decode("utf-8"))
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=json.dumps(info))
    else:
        creds = None
        try:  # saved login from `earthengine authenticate`
            creds = ee.data.get_persistent_credentials()
        except Exception:  # noqa: BLE001
            creds = None
    # The saved login can carry an old quota/billing project; force the one the user typed.
    if creds is not None and hasattr(creds, "with_quota_project"):
        try:
            creds = creds.with_quota_project(project)
        except Exception:  # noqa: BLE001
            pass
    try:
        ee.Reset()  # drop any earlier session in this server process
    except Exception:  # noqa: BLE001
        pass
    if creds is not None:
        ee.Initialize(creds, project=project)
    else:
        ee.Initialize(project=project)
    ee.Number(1).getInfo()  # fail early if the project is not registered
    return True


def _collection(aoi_geom):
    import ee
    from shapely.geometry import mapping
    aoi_ee = ee.Geometry(mapping(aoi_geom.simplify(0.0005)))
    return ee.ImageCollection(DATASET).filterBounds(aoi_ee)


def _year_image(col, year, bands):
    return col.filterDate(f"{year}-01-01", f"{year + 1}-01-01").select(bands).mosaic()


def _run_batches(image, reducer, gdf, cols, chunk_size, workers, scale, log, progress, label):
    import ee
    send = gdf[["bid", "geometry"]].copy()
    send["geometry"] = send.geometry.simplify(0.000005)  # ~0.5 m, shrinks request size
    chunks = [send.iloc[i:i + chunk_size] for i in range(0, len(send), chunk_size)]
    out, failed = [], []

    def run(i):
        fc = ee.FeatureCollection(json.loads(chunks[i].to_json()))
        st = (image.reduceRegions(collection=fc, reducer=reducer, scale=scale, tileScale=4)
              .map(lambda f: f.select(["bid"] + cols[1:], None, False)))
        err = ""
        for attempt in range(4):
            try:
                rows = [f["properties"] for f in st.getInfo()["features"]]
                return i, pd.DataFrame(rows, columns=cols)
            except Exception as e:  # noqa: BLE001 - retry with back-off
                err = str(e)
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(err[:200])

    log(f"Earth Engine: {label} for {len(send):,} buildings in {len(chunks)} batches")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run, i): i for i in range(len(chunks))}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                out.append(fut.result()[1])
            except Exception as e:  # noqa: BLE001
                failed.append(futs[fut])
                log(f"Earth Engine: batch {futs[fut]} failed: {e}")
            if progress:
                progress(n / len(chunks))
    if failed:
        log(f"Earth Engine: {len(failed)} batches failed; those buildings get no height")
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True).drop_duplicates("bid")


def building_heights(gdf, aoi_geom, s, log=print, progress=None) -> pd.DataFrame:
    """Median and 90th-percentile height per building for ``s.year``."""
    import ee
    col = _collection(aoi_geom)
    h = _year_image(col, s.year, "building_height")
    reducer = ee.Reducer.median().combine(ee.Reducer.percentile([90]), "", True)
    df = _run_batches(h, reducer, gdf, ["bid", "median", "p90"], s.chunk_size, s.workers,
                      s.gee_scale_m, log, progress, f"heights {s.year}")
    return df.rename(columns={"median": "h_median", "p90": "h_p90"})


def building_change(gdf, aoi_geom, s, log=print, progress=None) -> pd.DataFrame:
    """Mean height and presence for an early and a late period."""
    import ee
    col = _collection(aoi_geom)

    def period(years, tag):
        imgs = [_year_image(col, y, ["building_height", "building_presence"]) for y in years]
        return ee.ImageCollection(imgs).mean().rename([f"h_{tag}", f"p_{tag}"])

    img = period(s.early_years, "early").addBands(period(s.late_years, "late"))
    cols = ["bid", "h_early", "p_early", "h_late", "p_late"]
    return _run_batches(img, ee.Reducer.median(), gdf, cols, s.chunk_size, s.workers,
                        s.gee_scale_m, log, progress, "change layers")


def attach_heights(bld, hts):
    res = bld.merge(hts, on="bid", how="left")
    res[["h_median", "h_p90"]] = res[["h_median", "h_p90"]].astype(float).fillna(0)
    return res


def add_geometry_stats(res, utm):
    m = res.to_crs(utm)
    res["footprint_m2"] = m.area.round(1).values
    cen = m.centroid.to_crs(4326)
    res["lon"], res["lat"] = cen.x.round(6).values, cen.y.round(6).values
    return res

