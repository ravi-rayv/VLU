from vertical25d.basemap import lonlat_to_tile, tile_to_lonlat, tile_range


def test_tile_roundtrip():
    x, y = lonlat_to_tile(77.5946, 12.9716, 15)
    lon, lat = tile_to_lonlat(x, y, 15)
    assert abs(lon - 77.5946) < 1e-9 and abs(lat - 12.9716) < 1e-9


def test_tile_range_limits_tiles():
    z, x0, y0, x1, y1 = tile_range((77.55, 12.95, 77.60, 12.99), max_tiles=64)
    assert (x1 - x0 + 1) * (y1 - y0 + 1) <= 64 and z >= 13
