"""Streamlit portal: draw or upload a study area, run the 2.5D pipeline and
download the results.

Run locally:   streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))  # run from a clone without installing

from vertical25d import Settings, Inputs, run, read_aoi, read_zones, AOIError  # noqa: E402
from vertical25d.aoi import check_aoi  # noqa: E402
from vertical25d.config import LANDUSE  # noqa: E402
from vertical25d.basemap import SOURCES, add_folium_layers, stitched_basemap  # noqa: E402


@st.cache_data(show_spinner="Loading basemap…", ttl=24 * 3600)
def cached_basemap(bounds, source):
    return stitched_basemap(bounds, source)

st.set_page_config(page_title="Vertical land-use stratification (2.5D)", page_icon="🏙️", layout="wide")

FLOOR_COLOURS = [(1, [205, 226, 251]), (3, [134, 182, 239]), (6, [42, 120, 214]),
                 (10, [28, 92, 171]), (20, [16, 66, 129]), (999, [13, 54, 107])]
DEFAULT_CENTER = [12.9716, 77.5946]  # Bengaluru


def floor_colour(f):
    if f is None or (isinstance(f, float) and np.isnan(f)):
        return [195, 194, 183]
    for t, c in FLOOR_COLOURS:
        if f <= t:
            return c
    return FLOOR_COLOURS[-1][1]


# ============================================================================ sidebar
with st.sidebar:
    st.header("1 · Earth Engine")
    mode = st.radio("Data source", ["My study area", "Demo (no login needed)"],
                    help="The demo uses stored data for 680 m around World Trade Center Bengaluru.")
    ee_project, sa_json = None, None
    if mode == "My study area":
        ee_project = st.text_input("Your Earth Engine project ID", placeholder="e.g. my-ee-project-123 (filled from the key if left empty)",
                                   help="Free for research: register at code.earthengine.google.com/register")
        auth = st.radio("Sign-in method", ["This computer (earthengine authenticate)",
                                           "Upload my own service-account key (JSON)"])
        if auth.startswith("Upload"):
            key = st.file_uploader("Service-account key", type=["json"],
                                   help="Used only in memory for this session; never stored.")
            sa_json = key.getvalue().decode("utf-8") if key else None
            if sa_json:
                try:
                    key_project = json.loads(sa_json).get("project_id")
                except ValueError:
                    key_project = None
                    st.error("This file is not a valid JSON key.")
                if key_project and not ee_project:
                    ee_project = key_project
                    st.info(f"Using the key's project: {key_project}")
                elif key_project and ee_project != key_project:
                    st.warning(f"This key belongs to project '{key_project}', but you entered '{ee_project}'. "
                               f"Change the project ID to '{key_project}' unless you gave this key access "
                               f"to '{ee_project}'.")
        if st.button("Test connection", disabled=not ee_project):
            try:
                from vertical25d.heights import init_earth_engine
                init_earth_engine(ee_project, sa_json)
                st.success("Connected to Earth Engine")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not connect: {e}")

    st.header("2 · Settings")
    s = Settings()
    with st.expander("Analysis settings", expanded=False):
        s.year = st.selectbox("Height year", list(range(2023, 2015, -1)), index=0)
        s.grid_m = st.select_slider("Grid cell size (m)", [200, 250, 300, 400, 500, 750, 1000], value=500)
        s.n_sim = st.select_slider("Monte Carlo draws", [50, 100, 200, 500], value=200)
        s.min_poi_building = st.number_input("Min. POIs per building", 2, 20, 3)
        s.min_poi_area = st.number_input("Min. POIs per grid cell / zone", 3, 100, 10)
        s.match_dist_m = st.number_input("POI-to-building snap distance (m)", 0.0, 100.0, 20.0)
        s.floor_model = st.selectbox("Floor model", ["linear", "isotonic", "formula", "auto"], index=0)
        s.keep_higher_osm_tags = st.checkbox(
            "Keep OSM floor tags higher than the satellite estimate", value=False,
            help="Satellite heights can be far too low for slender towers. Off = method as in the paper.")
        s.run_change = st.checkbox("Also analyse change 2016–17 → 2022–23", value=False)
    st.markdown("**Authors**  \nRavi Satyappa Dabbanavar · Rohini N  \nNitte School of Architecture, Planning "
                "and Design, Nitte University, Bengaluru  \n✉️ ravisd@nittesoa.ac.in")
    st.caption("Method: OSM footprints + Open Buildings 2.5D heights → floors; OSM POIs → floor bands; "
               "chance-corrected vertical stratification index (VSI).")

# ============================================================================ header
st.title("Vertical land-use stratification")
st.markdown("Place open points of interest on the floors of satellite-measured buildings and measure how "
            "land uses are **stacked**, not just mixed. Choose a study area, run, and download every result.")

# ============================================================================ study area
aoi = None
if mode == "Demo (no login needed)":
    from vertical25d.demo import demo_inputs
    demo = demo_inputs()
    aoi = demo.aoi
    st.info("Demo area: 680 m around World Trade Center Bengaluru (1.45 km²). "
            "Buildings and POIs © OpenStreetMap contributors; heights © Google Open Buildings (CC BY 4.0).")
else:
    st.subheader("3 · Study area")
    tab_draw, tab_upload = st.tabs(["✏️ Draw on map", "📁 Upload KML / KMZ / GeoJSON / zipped shapefile"])
    with tab_draw:
        import folium
        from folium.plugins import Draw
        from streamlit_folium import st_folium
        c1, c2 = st.columns([3, 1])
        place = c1.text_input("Go to a place", placeholder="e.g. Koramangala, Bengaluru")
        if c2.button("Search", width="stretch") and place:
            try:
                import osmnx as ox
                st.session_state["center"] = list(ox.geocode(place))
            except Exception as e:  # noqa: BLE001
                st.warning(f"Place not found: {e}")
        center = st.session_state.get("center", DEFAULT_CENTER)
        m = folium.Map(location=center, zoom_start=14, tiles=None)
        add_folium_layers(m)
        Draw(export=False, draw_options={"polyline": False, "circle": False, "marker": False,
                                         "circlemarker": False, "polygon": True, "rectangle": True},
             edit_options={"edit": True, "remove": True}).add_to(m)
        out = st_folium(m, height=480, use_container_width=True, returned_objects=["all_drawings"],
                        key=f"draw_{center}")
        drawings = (out or {}).get("all_drawings") or []
        if drawings:
            try:
                aoi = read_aoi({"type": "FeatureCollection", "features": drawings})
            except AOIError as e:
                st.error(str(e))
        else:
            st.caption("Use the polygon or rectangle tool on the left of the map. Several shapes are merged.")
    with tab_upload:
        up = st.file_uploader("Study-area file", type=["kml", "kmz", "geojson", "json", "zip", "gpkg"])
        if up is not None:
            try:
                aoi = read_aoi(up.getvalue(), up.name)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the file: {e}")

if aoi is not None:
    try:
        info = check_aoi(aoi, s.max_area_km2)
        msg = f"Study area: **{info['area_km2']:,} km²**"
        if info["area_km2"] > 50:
            msg += " — large areas can take 30+ minutes (OSM download and Earth Engine)."
        st.success(msg)
    except AOIError as e:
        st.error(str(e)); aoi = None

# ============================================================================ optional inputs
zones = zone_col = pop_col = user_pois = None
if mode == "My study area":
    with st.expander("Optional: zones (e.g. wards) and your own POIs"):
        zf = st.file_uploader("Zones layer for zone statistics (KML, GeoJSON, GPKG or zipped shapefile)",
                              type=["kml", "kmz", "geojson", "json", "zip", "gpkg"], key="zones")
        if zf is not None:
            try:
                zones, guess = read_zones(zf.getvalue(), zf.name)
                cols = [c for c in zones.columns if c != "geometry"]
                zone_col = st.selectbox("Zone name column", cols, index=cols.index(guess) if guess in cols else 0)
                num = ["(none)"] + [c for c in cols if pd.api.types.is_numeric_dtype(zones[c])]
                p = st.selectbox("Population column (optional)", num)
                pop_col = None if p == "(none)" else p
                if aoi is not None:
                    zones = zones[zones.intersects(aoi.geometry.iloc[0])]
                st.caption(f"{len(zones)} zones overlap the study area")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read zones: {e}")
        pf = st.file_uploader("Your own POIs instead of OSM (GeoJSON, GPKG or zipped shapefile with OSM-style tags)",
                              type=["geojson", "json", "gpkg", "zip"], key="pois")
        if pf is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, pf.name)
                with open(path, "wb") as f:
                    f.write(pf.getvalue())
                user_pois = gpd.read_file(f"zip://{path}" if pf.name.endswith(".zip") else path)
            st.caption(f"{len(user_pois):,} POIs loaded")

# ============================================================================ run
st.subheader("4 · Run")
ready = aoi is not None and (mode != "My study area" or bool(ee_project))
if mode == "My study area" and not ee_project:
    st.warning("Enter your Earth Engine project ID in the sidebar.")
name = st.text_input("Name for this run (used in file names)",
                     value="demo_brigade_gateway" if mode.startswith("Demo") else "my_study_area")

if st.button("▶ Run analysis", type="primary", disabled=not ready):
    bar = st.progress(0.0, text="Starting…")
    with st.status("Running…", expanded=True) as status:
        lines = []

        def log(msg):
            lines.append(msg)
            st.write(msg)

        def prog(frac, step):
            bar.progress(min(max(frac, 0.0), 1.0), text=f"{step} · {frac:.0%}")

        try:
            if mode.startswith("Demo"):
                inp = demo_inputs()
                inp.name = name
            else:
                inp = Inputs(aoi=aoi, ee_project=ee_project, service_account_json=sa_json, zones=zones,
                             zone_name_col=zone_col, zone_pop_col=pop_col, pois=user_pois, name=name)
            out_root = os.path.join(tempfile.gettempdir(), "vertical25d_runs", str(int(time.time())))
            res = run(inp, s, out_dir=out_root, log=log, progress=prog)
            st.session_state["result"] = res
            status.update(label="Finished", state="complete")
        except Exception as e:  # noqa: BLE001
            status.update(label="Failed", state="error")
            st.exception(e)

# ============================================================================ results
res = st.session_state.get("result")
if res is not None:
    S = res.summary
    st.subheader("5 · Results")
    c = st.columns(6)
    c[0].metric("Buildings", f"{len(res.buildings):,}")
    mh = S["floors"].get("median_height_m")
    c[1].metric("Median height", f"{mh:.1f} m" if mh else "–")
    c[2].metric("≥ 4 floors", f"{S['floors']['share_4plus']:.0%}")
    c[3].metric("POIs", f"{S['pois']['n_classified']:,}")
    c[4].metric("Floor known", f"{S['pois']['floor_known_share']:.0%}")
    g = S["indices"].get("grid", {})
    c[5].metric("Grid VSI_excess (mean)", f"{g.get('VSI_excess', float('nan')):.3f}" if g else "–")

    tabs = st.tabs(["🏢 3D buildings", "🧱 Land-use stacking", "🗺️ Stratification map", "📊 Vertical profile", "📐 Floor model",
                    "🏘️ Zones", "⬇️ Downloads"])

    with tabs[0]:
        import pydeck as pdk
        b = res.buildings[res.buildings["h_median"] > 0].copy()
        if len(b) > 40000:
            st.caption(f"Showing the 40,000 largest of {len(b):,} buildings to keep the browser responsive.")
            b = b.nlargest(40000, "footprint_m2")
        b["fill"] = b["floors_final"].map(floor_colour)
        b["label"] = b["name"].replace("", "Unnamed building")
        b["floors_txt"] = b["floors_final"].map(lambda v: "–" if pd.isna(v) else f"{v:.0f}")
        gj = json.loads(b[["label", "h_median", "floors_txt", "floor_source", "fill", "geometry"]]
                        .assign(h_median=b["h_median"].round(1)).to_json())
        tall = b.loc[b["h_median"].idxmax()] if len(b) else None
        view = pdk.ViewState(latitude=float(tall["lat"]) if tall is not None else DEFAULT_CENTER[0],
                             longitude=float(tall["lon"]) if tall is not None else DEFAULT_CENTER[1],
                             zoom=15, pitch=50, bearing=-20)
        layer = pdk.Layer("GeoJsonLayer", gj, extruded=True, wireframe=True, pickable=True,
                          get_elevation="properties.h_median", get_fill_color="properties.fill",
                          get_line_color=[255, 255, 255])
        bm_choice = st.radio("Basemap", list(SOURCES), horizontal=True, key="bm3d")
        layers = [layer]
        bm = cached_basemap(tuple(round(v, 5) for v in b.total_bounds), bm_choice)
        if bm is not None:
            layers.insert(0, pdk.Layer("BitmapLayer", data=None, image=pdk.types.String(bm[0]), bounds=bm[1], opacity=0.9))
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_provider=None, map_style=None,
                                 tooltip={"text": "{label}\nHeight: {h_median} m\nFloors: {floors_txt} ({floor_source})"}),
                        width="stretch")
        st.caption("Colour = floors (light 1 → dark 21+); grey = no usable height. "
                   + (SOURCES[bm_choice][1] if bm is not None else "Basemap could not be loaded."))

    with tabs[1]:
        import altair as alt
        import pydeck as pdk
        from vertical25d import stacking as stk
        from vertical25d.config import BANDS as _BANDS
        pp = res.poi_bands
        if pp is None or pp.empty:
            st.info("No POIs linked to buildings in this area.")
        else:
            st.markdown("Each building is cut into its floor bands. A coloured slab means POIs are on those floors; "
                        "the colour is the main land use there. Grey shells are the full building. "
                        "Known floors come from OSM tags or addresses; the rest are the model's most likely floors.")
            min_e = st.slider("Show a floor band when it holds at least this many POIs (expected)", 0.2, 3.0, 0.5, 0.1,
                              key="stk_min")
            slab_df, shell_df = stk.slabs(res.buildings, pp, min_expected=min_e)
            if slab_df.empty:
                st.info("No floor band reaches this threshold. Lower the slider.")
            else:
                top = slab_df.loc[slab_df["polygon"].map(lambda r: r[0][2]).idxmax()]
                view = pdk.ViewState(latitude=float(top["polygon"][0][1]), longitude=float(top["polygon"][0][0]),
                                     zoom=16, pitch=55, bearing=-20)
                lyr_shell = pdk.Layer("PolygonLayer", shell_df, get_polygon="polygon", extruded=True,
                                      get_elevation="height", get_fill_color="colour", wireframe=True,
                                      get_line_color=[150, 150, 150], pickable=False)
                lyr_slab = pdk.Layer("PolygonLayer", slab_df, get_polygon="polygon", extruded=True,
                                     get_elevation="height", get_fill_color="colour", wireframe=True,
                                     get_line_color=[255, 255, 255], pickable=True)
                layers = [lyr_shell, lyr_slab]
                bm = cached_basemap(tuple(round(v, 5) for v in res.buildings.total_bounds),
                                    st.session_state.get("bm3d", list(SOURCES)[0]))
                if bm is not None:
                    layers.insert(0, pdk.Layer("BitmapLayer", data=None, image=pdk.types.String(bm[0]),
                                               bounds=bm[1], opacity=0.85))
                st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_provider=None, map_style=None,
                                         tooltip={"text": "{name}\n{band}\nMain use: {main}\n"
                                                          "Expected POIs: {uses}\nKnown-floor POIs: {known}"}),
                                width="stretch")
                present = [u for u in stk.LANDUSE_COLOURS if u in set(slab_df["main"])]
                st.markdown(" ".join(f"<span style='display:inline-block;width:12px;height:12px;background:"
                                     f"{stk.LANDUSE_COLOURS[u]};border-radius:2px;margin:0 4px 0 10px'></span>{u}"
                                     for u in present), unsafe_allow_html=True)

            st.subheader("Inspect one building")
            bu = stk.building_band_uses(pp)
            cnt = pp.groupby("bid").size().sort_values(ascending=False)
            cnt = cnt[cnt >= 2]
            if cnt.empty:
                st.caption("No building has two or more POIs.")
            else:
                info = res.buildings.set_index("bid")
                def _label(b):
                    nm = info.at[b, "name"] if b in info.index and info.at[b, "name"] else "Unnamed building"
                    fl = info.at[b, "floors_final"] if b in info.index else np.nan
                    return f"{nm} · {cnt[b]} POIs · {'?' if pd.isna(fl) else int(fl)} floors · {b}"
                pick = st.selectbox("Building (most POIs first)", list(cnt.index[:500]), format_func=_label)
                sub = bu[bu["bid"] == pick].copy()
                order = [b for b in reversed(_BANDS)]
                chart = (alt.Chart(sub).mark_bar().encode(
                    y=alt.Y("band:N", sort=order, title="Floor band"),
                    x=alt.X("sum(expected):Q", title="Expected number of POIs"),
                    color=alt.Color("landuse:N", title="Land use",
                                    scale=alt.Scale(domain=list(stk.LANDUSE_COLOURS),
                                                    range=list(stk.LANDUSE_COLOURS.values()))),
                    tooltip=["band", "landuse", alt.Tooltip("expected:Q", format=".2f"), "n_known"])
                    .properties(height=260))
                st.altair_chart(chart, width="stretch")
                st.dataframe(stk.poi_floor_table(pp, pick), width="stretch", hide_index=True)

    with tabs[2]:
        import folium
        from streamlit_folium import st_folium
        if res.grid is not None and len(res.grid):
            gr = res.grid.copy()
            ctr = (gr.geometry.union_all() if hasattr(gr.geometry, "union_all") else gr.geometry.unary_union).centroid
            fm = folium.Map(location=[ctr.y, ctr.x], zoom_start=13, tiles=None)
            add_folium_layers(fm)
            v = float(np.nanmax(np.abs(gr["VSI_excess"]))) or 0.1
            import branca.colormap as bcm
            cmap = bcm.LinearColormap(["#d03b3b", "#e98a86", "#f0efec", "#86b6ef", "#2a78d6"], vmin=-v, vmax=v,
                                      caption="VSI_excess (blue = uses separated by floor)")
            folium.GeoJson(json.loads(gr.to_json()),
                           style_function=lambda f: {"fillColor": cmap(f["properties"]["VSI_excess"])
                                                     if f["properties"]["VSI_excess"] is not None else "#cccccc",
                                                     "color": "white", "weight": 0.5, "fillOpacity": 0.75},
                           tooltip=folium.GeoJsonTooltip(["n_poi", "H_use", "H_band", "VSI", "VSI_excess", "VSI_conf"],
                                                         localize=True)).add_to(fm)
            cmap.add_to(fm)
            st_folium(fm, height=520, use_container_width=True, returned_objects=[], key="gridmap")
        else:
            st.info("No grid cell has enough POIs. Lower 'Min. POIs per grid cell' or use larger cells.")
        if res.building_index is not None and len(res.building_index):
            st.markdown("**Buildings with the most POIs**")
            st.dataframe(res.building_index.sort_values("n_poi", ascending=False)
                         [["name", "main_use", "n_poi", "floors_final", "H_use", "VSI", "VSI_null", "VSI_excess", "VSI_conf"]]
                         .head(30).round(3), width="stretch", hide_index=True)

    with tabs[3]:
        if res.figures.get("profile"):
            st.image(res.figures["profile"])
        st.dataframe(res.profile.round(3), width="stretch")
        if res.figures.get("buildings"):
            st.image(res.figures["buildings"])

    with tabs[4]:
        if res.figures.get("calibration"):
            st.image(res.figures["calibration"])
        f = S["floors"]
        st.write(f"Model: **{f['model']}** · tagged buildings used: {f['n_tagged']:,} "
                 f"(plausible {f.get('n_plausible') or 0:,}, excluded {f.get('n_excluded') or 0:,})")
        if f.get("slope"):
            st.write(f"floors = {f['intercept']:.2f} + {f['slope']:.3f} × height (≈ {f['m_per_floor']:.2f} m per floor)")
        if f["cv"]:
            st.dataframe(pd.DataFrame(f["cv"]), hide_index=True)
        if res.figures.get("floors"):
            st.image(res.figures["floors"])
        st.write("Floor source:", f["source_counts"])
        if "change" in S:
            st.markdown("**Change 2016–2023**")
            st.write(S["change"])

    with tabs[5]:
        if res.zones is not None:
            st.dataframe(res.zones.drop(columns="geometry").round(3), width="stretch", hide_index=True)
            if "zone_correlations" in S:
                st.markdown("**Spearman correlations across zones**")
                st.dataframe(pd.DataFrame(S["zone_correlations"]).round(2))
        else:
            st.info("Upload a zones layer (e.g. wards) under 'Optional' to get zone statistics.")

    with tabs[6]:
        with open(res.zip_path, "rb") as f:
            st.download_button("⬇️ Download all results (zip)", f.read(),
                               file_name=os.path.basename(res.zip_path), mime="application/zip", type="primary")
        st.caption("CSV + GeoPackage files, figures (PNG/SVG) and summary.json with all settings.")
        files = sorted(os.listdir(res.out_dir))
        st.write(", ".join(fn for fn in files if not os.path.isdir(os.path.join(res.out_dir, fn))))

st.divider()
st.caption("Developed by **Ravi Satyappa Dabbanavar** and **Rohini N**, Nitte School of Architecture, Planning and Design, "
           "Nitte University, Bengaluru · Contact: ravisd@nittesoa.ac.in")
st.caption("Data: © OpenStreetMap contributors (ODbL) · Google Open Buildings 2.5D Temporal (CC BY 4.0) via "
           "Google Earth Engine. Please cite: Dabbanavar, R. S., & N, R. (2026). vertical25d (v0.1.0).")
