"""Height-to-floor calibration against OSM ``building:levels`` tags, and the
final floor count per building (Section 4.1 of the manuscript)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold


def to_floors(raw):
    return np.clip(np.round(np.asarray(raw, dtype=float)), 1, None)


def fit_models(h: pd.Series, lv: pd.Series, ground_floor_m: float):
    """Three ways to turn satellite height into floors.

    Returns ({name: f(height) -> continuous floors}, storey_height_formula, huber_model).
    """
    h, lv = pd.Series(h, dtype=float), pd.Series(lv, dtype=float)
    imp = (h - ground_floor_m) / (lv - 1).where(lv > 1)
    ok = imp[(imp > 2) & (imp < 6)]
    storey = float(np.nanmedian(ok)) if len(ok) else 3.2
    hub = HuberRegressor().fit(h.values.reshape(-1, 1), lv.values)
    iso = IsotonicRegression(out_of_bounds="clip").fit(h.values, lv.values)
    models = {
        "formula": lambda x, s=storey: (np.asarray(x, dtype=float) - ground_floor_m) / s + 1,
        "linear": lambda x, m=hub: m.predict(np.asarray(x, dtype=float).reshape(-1, 1)),
        "isotonic": lambda x, m=iso: m.predict(np.asarray(x, dtype=float)),
    }
    return models, storey, hub


def plausible_mask(est, tag):
    return (est - tag).abs() <= np.maximum(3, 0.4 * tag)


def calibrate(res: pd.DataFrame, s, log=print) -> dict:
    """Fit and cross-validate the floor models. Returns a dict with the chosen
    model, its predict function, the CV table and diagnostics."""
    lv = res["building_levels"].where(res["building_levels"].between(1, 60))
    cal = res.assign(building_levels=lv)
    cal = cal[cal["building_levels"].notna() & (cal["h_median"] >= s.min_height_m)].copy()
    out = {"n_tagged": int(len(cal))}

    if len(cal) < max(s.min_calibration, 5):
        log(f"Floors: only {len(cal)} tagged buildings with a usable height; using the fixed formula "
            "(3.2 m per floor above a 4 m ground floor)")
        storey = 3.2
        f = (lambda x: (np.asarray(x, dtype=float) - s.ground_floor_m) / storey + 1)
        out.update(model="formula", predict=f, models={"formula": f}, storey_m=storey,
                   cv=pd.DataFrame(), n_plausible=len(cal), n_excluded=0, intercept=None, slope=None)
        return out

    models, _, _ = fit_models(cal["h_median"], cal["building_levels"], s.ground_floor_m)
    resid_est = pd.Series(to_floors(models["linear"](cal["h_median"])), index=cal.index)
    cal["plausible"] = plausible_mask(resid_est, cal["building_levels"])
    good = cal[cal["plausible"]]
    out.update(n_plausible=int(len(good)), n_excluded=int((~cal["plausible"]).sum()))

    rows = []
    if len(good) >= s.min_calibration:
        for tr, te in KFold(5, shuffle=True, random_state=0).split(good):
            m, _, _ = fit_models(good["h_median"].iloc[tr], good["building_levels"].iloc[tr], s.ground_floor_m)
            for name, f in m.items():
                e = to_floors(f(good["h_median"].iloc[te])) - good["building_levels"].iloc[te].values
                rows.append({"model": name, "MAE": np.abs(e).mean(),
                             "within_1": (np.abs(e) <= 1).mean(), "bias": e.mean()})
    cv = (pd.DataFrame(rows).groupby("model").mean().reindex(["formula", "linear", "isotonic"])
          if rows else pd.DataFrame())
    choice = s.floor_model
    if choice == "auto":
        choice = cv["MAE"].idxmin() if len(cv) else "linear"
    models, storey, hub = fit_models(good["h_median"], good["building_levels"], s.ground_floor_m)
    out.update(model=choice, predict=models[choice], models=models, cv=cv, storey_formula_m=storey,
               intercept=float(hub.intercept_), slope=float(hub.coef_[0]),
               storey_m=float(1 / hub.coef_[0]) if hub.coef_[0] > 0 else storey,
               calibration=cal[["bid", "h_median", "building_levels", "plausible"]])
    log(f"Floors: {choice} model; linear fit floors = {hub.intercept_:.2f} + {hub.coef_[0]:.3f} × height "
        f"(≈ {out['storey_m']:.2f} m per floor); {out['n_plausible']:,} plausible tags, "
        f"{out['n_excluded']:,} excluded")
    return out


def estimate(height, predict, min_height_m):
    height = pd.Series(height, dtype=float)
    f = pd.Series(to_floors(predict(height.fillna(0))), index=height.index)
    return f.where(height >= min_height_m)


def assign_floors(res: pd.DataFrame, cal: dict, s) -> pd.DataFrame:
    """Floors per building with quality flags and the source of the floor count."""
    res = res.copy()
    res.loc[~res["building_levels"].between(1, 60), "building_levels"] = np.nan
    res["floors_est"] = estimate(res["h_median"], cal["predict"], s.min_height_m)
    res["floors_est_p90"] = estimate(res["h_p90"], cal["predict"], s.min_height_m)
    if "formula" in cal["models"]:
        res["floors_est_formula"] = estimate(res["h_median"], cal["models"]["formula"], s.min_height_m)
    res["floor_model"] = cal["model"]

    h, lv, fe, fa = res["h_median"], res["building_levels"], res["floors_est"], res["footprint_m2"]
    conflict = lv.notna() & fe.notna() & ((fe - lv).abs() > np.maximum(3, 0.4 * lv))
    rules = {
        "no_height": h == 0,
        "very_low": (h > 0) & (h < s.min_height_m),
        "levels_conflict": conflict,
        "possible_complex": fa > 5000,
        "tag_check": (res["building"] == "house") & (h > 20),
    }
    res["qc_flags"] = ""
    for name, mask in rules.items():
        res.loc[mask, "qc_flags"] += name + ";"
    res["qc_flags"] = res["qc_flags"].str.rstrip(";")

    keep_tag = lv.notna() & ~conflict
    if s.keep_higher_osm_tags:  # satellite heights saturate on tall, slender towers
        keep_tag = keep_tag | (lv.notna() & conflict & (lv > fe))
    res["floors_final"] = np.where(keep_tag, lv, fe)
    res["floor_source"] = np.where(keep_tag, "osm", np.where(fe.notna(), "estimated", "none"))
    return res
