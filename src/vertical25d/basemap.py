"""Key-free basemaps for the portal.

CARTO tiles now need an API key, so the app uses OpenStreetMap (street map)
and Esri World Imagery (satellite). For the 3D view, tiles are fetched on the
server and stitched into one image that pydeck draws under the buildings.
"""
from __future__ import annotations

import base64
import io
import math

OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_ATTR = "© OpenStreetMap contributors"
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Imagery © Esri, Maxar, Earthstar Geographics"
USER_AGENT = "vertical25d-portal/0.1 (+https://github.com/ravi-rayv/vertical-landuse-stratification)"

SOURCES = {"Street map (OpenStreetMap)": (OSM_URL, OSM_ATTR),
           "Satellite (Esri)": (ESRI_URL, ESRI_ATTR)}


def add_folium_layers(m, default: str = "Street map (OpenStreetMap)"):
    """Add both basemaps to a folium map (with a layer switcher)."""
    import folium
    for name, (url, attr) in SOURCES.items():
        folium.TileLayer(tiles=url, attr=attr, name=name, max_zoom=19,
                         show=(name == default), overlay=False, control=True).add_to(m)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    return m


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Fractional XYZ tile coordinates (Web Mercator)."""
    lat = max(min(lat, 85.0511), -85.0511)
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x: float, y: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def tile_range(bounds, max_tiles: int = 64, max_zoom: int = 17):
    """Choose the highest zoom whose tile grid over `bounds` has <= max_tiles tiles."""
    minx, miny, maxx, maxy = bounds
    for z in range(max_zoom, 0, -1):
        x0, y0 = lonlat_to_tile(minx, maxy, z)
        x1, y1 = lonlat_to_tile(maxx, miny, z)
        tx0, ty0, tx1, ty1 = int(x0), int(y0), int(x1), int(y1)
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= max_tiles:
            return z, tx0, ty0, tx1, ty1
    return 1, 0, 0, 1, 1


def stitched_basemap(bounds, source: str = "Street map (OpenStreetMap)", pad: float = 0.15,
                     max_tiles: int = 64, timeout: float = 8.0):
    """Return (data_url, [W, S, E, N]) for a stitched tile image, or None on failure."""
    try:
        import requests
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None
    url, _ = SOURCES[source]
    minx, miny, maxx, maxy = bounds
    dx, dy = (maxx - minx) * pad, (maxy - miny) * pad
    z, tx0, ty0, tx1, ty1 = tile_range((minx - dx, miny - dy, maxx + dx, maxy + dy), max_tiles=max_tiles)
    img = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), (242, 242, 240))
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    ok = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            try:
                r = sess.get(url.format(z=z, x=tx, y=ty), timeout=timeout)
                if r.status_code == 200:
                    img.paste(Image.open(io.BytesIO(r.content)).convert("RGB"), ((tx - tx0) * 256, (ty - ty0) * 256))
                    ok += 1
            except Exception:  # noqa: BLE001
                continue
    if ok == 0:
        return None
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    west, north = tile_to_lonlat(tx0, ty0, z)
    east, south = tile_to_lonlat(tx1 + 1, ty1 + 1, z)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), [west, south, east, north]
