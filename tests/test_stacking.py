import numpy as np
import pandas as pd
from vertical25d import stacking as stk
from vertical25d.config import BANDS


def test_band_ranges_capped():
    r = stk.band_ranges(3)
    assert r["Ground"] == (0, 1) and r["1st"] == (1, 2) and r["2nd–3rd"] == (2, 3)
    assert "4th–7th" not in r


def test_building_band_uses_sums_to_poi_count():
    p = np.zeros((3, len(BANDS))); p[0, 1] = 1; p[1, 2] = 1; p[2, 1:3] = 0.5
    pp = pd.DataFrame(p, columns=stk.P_COLS)
    pp["bid"] = "b1"; pp["landuse"] = ["Retail", "Office", "Retail"]; pp["band_obs"] = [1, 2, -1]
    g = stk.building_band_uses(pp)
    assert abs(g["expected"].sum() - 3) < 1e-9
    assert g["n_known"].sum() == 2
