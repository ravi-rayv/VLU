"""Floor-band prediction for POIs without a floor, Monte Carlo propagation and
the vertical land-use indices with permutation-null chance correction
(Sections 4.3-4.4 of the manuscript)."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from .config import BANDS, BAND_START, LANDUSE

B, K = len(BANDS), len(LANDUSE)
_BS = np.array(BAND_START)
LU_CODE = {u: i for i, u in enumerate(LANDUSE)}


def band_of(floor):
    f = np.asarray(floor, dtype=float)
    b = np.searchsorted(_BS, f, side="right") - 1
    return np.where(np.isnan(f), -1, b)


def default_priors() -> pd.DataFrame:
    """Floor-band shares per land use from Bengaluru POIs with a known floor
    (2,618 POIs). Used when a study area has too few known floors."""
    path = os.path.join(os.path.dirname(__file__), "data", "default_band_priors.csv")
    return pd.read_csv(path, index_col=0).reindex(LANDUSE)


# ----------------------------------------------------------------------------- prepare
def prepare(poi: pd.DataFrame) -> pd.DataFrame:
    poi = poi[poi["landuse"].isin(LANDUSE)].reset_index(drop=True).copy()
    too_high = poi["floor"].notna() & poi["floors_final"].notna() & (poi["floor"] > poi["floors_final"] - 1)
    poi.loc[too_high, "floor"] = np.nan
    poi["band_obs"] = band_of(poi["floor"])
    poi["n_poi_bld"] = poi.groupby("bid")["bid"].transform("size") if poi["bid"].notna().any() else 1
    poi["has_brand"] = poi["brand"].notna().astype(int) if "brand" in poi.columns else 0
    return poi


def _features(df):
    return np.column_stack([df["landuse"].map(LU_CODE).values, df["floors_final"].values, df["h_median"].values,
                            np.log1p(df["footprint_m2"].values), df["n_poi_bld"].fillna(1).values,
                            df["has_brand"].values, df["lon"].values, df["lat"].values]).astype(float)


def _feasible(df):
    top = (df["floors_final"] - 1).values.astype(float)
    m = np.ones((len(df), B), bool)
    known = ~np.isnan(top)
    m[known, 1:] = _BS[1:][None, :] <= top[known, None]
    m[:, 1] = True
    return m


def _model(seed):
    return HistGradientBoostingClassifier(categorical_features=[0], max_iter=250, learning_rate=0.06,
                                          max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)


def _normalise(p, mask):
    p = p * mask
    s = p.sum(1, keepdims=True)
    return np.where(s > 0, p / np.where(s == 0, 1, s), np.eye(B)[1])


def _proba(model, X, mask):
    p = np.zeros((len(X), B))
    p[:, model.classes_] = model.predict_proba(X)
    return _normalise(p, mask)


def predict_bands(poi: pd.DataFrame, s, log=print):
    """Band probabilities for every POI (known floors fixed). Returns (P, cv_summary)."""
    known = poi["band_obs"].values >= 0
    train = poi[known]
    n_classes = len(np.unique(train["band_obs"])) if len(train) else 0
    cv = {"n_known": int(known.sum()), "method": None}

    if known.sum() >= s.min_known_floors and n_classes >= 2:
        X, y, M = _features(train), train["band_obs"].values, _feasible(train)
        counts = np.bincount(y)
        k = int(min(5, counts[counts > 0].min()))
        rows, oof = [], np.zeros((len(train), B))
        if k >= 2:
            for tr, te in StratifiedKFold(k, shuffle=True, random_state=s.seed).split(X, y):
                m = _model(s.seed).fit(X[tr], y[tr])
                p = _proba(m, X[te], M[te]); oof[te] = p
                base = train.iloc[tr].groupby("landuse")["band_obs"].agg(lambda v: v.mode()[0])
                yb = train.iloc[te]["landuse"].map(base).fillna(1).values
                rows.append({"model_accuracy": (p.argmax(1) == y[te]).mean(),
                             "baseline_accuracy": (yb == y[te]).mean(),
                             "log_loss": -np.mean(np.log(np.clip(p[np.arange(len(te)), y[te]], 1e-9, 1)))})
        model = _model(s.seed).fit(X, y)
        P = _proba(model, _features(poi), _feasible(poi))
        cv.update(method="gradient_boosting", **(pd.DataFrame(rows).mean().to_dict() if rows else {}))
        cv["observed_share"] = (np.bincount(y, minlength=B) / len(y)).tolist()
        cv["predicted_share"] = oof.mean(0).tolist() if rows else None
        log(f"Floor bands: gradient-boosted classifier trained on {known.sum():,} POIs with a known floor")
    else:
        pri = default_priors()
        if known.sum() >= 10:  # blend local evidence with the Bengaluru priors
            loc = pd.crosstab(train["landuse"], train["band_obs"]).reindex(index=LANDUSE, columns=range(B), fill_value=0)
            pri = (pri.values * 10 + loc.values) / (10 + loc.values.sum(1, keepdims=True))
            pri = pd.DataFrame(pri, index=LANDUSE)
        P = _normalise(pri.reindex(poi["landuse"]).values.astype(float), _feasible(poi))
        cv["method"] = "default_priors"
        log(f"Floor bands: only {known.sum()} POIs with a known floor; using land-use priors from Bengaluru")
    P[known] = np.eye(B)[poi.loc[known, "band_obs"].values]
    return P, cv


# ----------------------------------------------------------------------------- indices
def _H(p, axis):
    return -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(axis)


def _indices(unit, n_units, band, use):
    c = np.bincount(unit * B * K + band * K + use, minlength=n_units * B * K).reshape(n_units, B, K).astype(float)
    n = c.sum((1, 2)); nn = np.where(n == 0, 1, n)
    h_use = _H(c.sum(1) / nn[:, None], 1)
    h_band = _H(c.sum(2) / nn[:, None], 1)
    h_join = _H((c / nn[:, None, None]).reshape(n_units, -1), 1)
    mi = h_use + h_band - h_join
    vsi = np.where(h_use > 1e-9, mi / np.where(h_use > 1e-9, h_use, 1), np.nan)
    return h_use, h_band, h_join, mi, vsi


def _shuffle_within(unit, band, rng):
    order = np.lexsort((rng.random(len(unit)), unit))
    base = np.argsort(unit, kind="stable")
    out = np.empty_like(band); out[base] = band[order]
    return out


def monte_carlo(poi: pd.DataFrame, P: np.ndarray, s, utm, has_zones=False, log=print, progress=None):
    """Run ``s.n_sim`` draws. Returns ({scale: DataFrame}, band_counts[n_sim, B, K])."""
    rng = np.random.default_rng(s.seed)
    use = poi["landuse"].map(LU_CODE).values
    cumP = P.cumsum(1)
    pts = gpd.GeoSeries(gpd.points_from_xy(poi["lon"], poi["lat"]), crs=4326).to_crs(utm)
    poi = poi.copy()
    poi["cell"] = ((pts.x // s.grid_m).astype(int).astype(str) + "_" + (pts.y // s.grid_m).astype(int).astype(str)).values

    scales = {}
    plan = [("building", "bid", s.min_poi_building), ("grid", "cell", s.min_poi_area)]
    if has_zones:
        plan.append(("zone", "zone", s.min_poi_area))
    for name, col, mn in plan:
        if col not in poi.columns or poi[col].isna().all():
            continue
        cnt = poi[col].value_counts()
        keep = cnt[cnt >= mn].index
        sel = poi[col].isin(keep).values
        if sel.sum() == 0:
            continue
        ids, unit = np.unique(poi.loc[sel, col].astype(str).values, return_inverse=True)
        scales[name] = dict(sel=sel, ids=ids, unit=unit, n=len(ids))
        log(f"Indices: {name}: {len(ids):,} units with ≥ {mn} POIs")

    store = {k: {m: np.zeros((s.n_sim, d["n"])) for m in ["H_band", "MI", "VSI", "VSI_null", "H_joint"]}
             for k, d in scales.items()}
    static = {}
    band_counts = np.zeros((s.n_sim, B, K))
    for sim in range(s.n_sim):
        u = rng.random(len(P))[:, None]
        band = np.minimum((u > cumP).sum(1), B - 1)
        band_counts[sim] = np.bincount(band * K + use, minlength=B * K).reshape(B, K)
        for k, d in scales.items():
            b_, u_ = band[d["sel"]], use[d["sel"]]
            hu, hb, hj, mi, vsi = _indices(d["unit"], d["n"], b_, u_)
            _, _, _, _, vsi0 = _indices(d["unit"], d["n"], _shuffle_within(d["unit"], b_, rng), u_)
            st = store[k]
            st["H_band"][sim], st["MI"][sim], st["VSI"][sim], st["VSI_null"][sim], st["H_joint"][sim] = hb, mi, vsi, vsi0, hj
            static[k] = hu
        if progress and (sim + 1) % 10 == 0:
            progress((sim + 1) / s.n_sim)

    out = {}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, d in scales.items():
            st = store[k]
            t = pd.DataFrame({"id": d["ids"], "n_poi": np.bincount(d["unit"], minlength=d["n"]),
                              "H_use": static[k] / np.log(K),
                              "H_band": np.nanmean(st["H_band"], 0) / np.log(B),
                              "H_joint": np.nanmean(st["H_joint"], 0) / np.log(B * K),
                              "MI": np.nanmean(st["MI"], 0),
                              "VSI": np.nanmean(st["VSI"], 0),
                              "VSI_p05": np.nanpercentile(st["VSI"], 5, 0),
                              "VSI_p95": np.nanpercentile(st["VSI"], 95, 0),
                              "VSI_null": np.nanmean(st["VSI_null"], 0)})
            t["VSI_excess"] = t["VSI"] - t["VSI_null"]
            t["VSI_conf"] = np.nanmean(st["VSI"] > st["VSI_null"], 0)
            out[k] = t
    return out, band_counts, poi["cell"]


def grid_polygons(g_idx: pd.DataFrame, grid_m: int, utm) -> gpd.GeoDataFrame:
    xy = g_idx["id"].str.split("_", expand=True).astype(int) * grid_m
    geom = [box(x, y, x + grid_m, y + grid_m) for x, y in zip(xy[0], xy[1])]
    return gpd.GeoDataFrame(g_idx.rename(columns={"id": "cell"}), geometry=geom, crs=utm).to_crs(4326)


def vertical_profile(band_counts: np.ndarray) -> pd.DataFrame:
    """Mean share of each land use on each floor band across draws (rows = bands)."""
    mean = band_counts.mean(0)
    share = mean / np.where(mean.sum(1, keepdims=True) == 0, 1, mean.sum(1, keepdims=True))
    df = pd.DataFrame(share, index=BANDS, columns=LANDUSE)
    df["expected_POIs"] = mean.sum(1)
    return df
