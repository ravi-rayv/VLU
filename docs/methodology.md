# Methodology

The workflow has three stages. Notation: buildings *i*, POIs *p*, land-use classes *k* ∈ {1,…,K} with K = 11, floor bands *b* ∈ {1,…,B} with B = 6, and analysis units *u* (a building, grid cell or zone).

## A. 3D building stock

**A1 Heights.** For each OSM footprint F<sub>i</sub>, the building height h<sub>i</sub> is the median of the Open Buildings 2.5D Temporal height raster over the pixels inside the footprint. h<sub>i</sub><sup>90</sup> is the 90th percentile. Both are computed in Earth Engine at 4 m scale for the chosen year (default 2023). Buildings with h<sub>i</sub> < 2.5 m have no usable height.

**A2 Calibration set.** Buildings with an OSM tag L<sub>i</sub> ∈ [1, 60] and a usable height. A first linear fit flags tags with |Ĥ<sub>i</sub> − L<sub>i</sub>| > max(3, 0.4 L<sub>i</sub>) as likely tagging or height errors. The remaining tags are "plausible".

**A3 Models.** Three height-to-floor functions are compared by 5-fold cross-validation, measuring MAE, the share within ±1 floor, and bias:

- **Formula:** f(h) = (h − g)/s + 1, with g = 4 m and s the median implied storey height.
- **Huber linear:** f(h) = β₀ + β₁h, robust to bad tags.
- **Isotonic:** a monotone step function.

Floors are rounded: Ĥ<sub>i</sub> = max(1, round(f(h<sub>i</sub>))).

**A4–A5 Final floors.** F<sub>i</sub> = L<sub>i</sub> where a valid tag agrees with the height, and Ĥ<sub>i</sub> otherwise. With `keep_higher_osm_tags=True`, a tag that is higher than the estimate is kept even when it disagrees, because satellite heights saturate on slender towers.

**A6 Change (optional).** Height and presence are averaged over 2016–17 and 2022–23. Each building is classed as:
- **new:** presence < 0.3 early and ≥ 0.5 late;
- **taller:** Δh ≥ 3 m and at least one floor more;
- **lower:** Δh ≤ −3 m and at least one floor fewer;
- **stable:** everything else.

Added floor area per zone: ΔFAR = (Σ<sub>new</sub> a<sub>i</sub> F̂(h<sub>i</sub><sup>late</sup>) + Σ<sub>taller</sub> a<sub>i</sub> ΔF<sub>i</sub>) / A. The median height change of buildings present in both periods is reported as drift.

## B. Activities on floors

**B1 Classification.** OSM tags (`amenity`, `shop`, `office`, `tourism`, `leisure`, `healthcare`, `craft`) are mapped to 11 classes: Retail, Food & drink, Office, Personal services, Health, Finance, Education, Hospitality, Leisure, Civic and Religious. Street furniture (benches, ATMs, parking, …) is dropped.

**B2 Linking.** A POI is linked to the footprint that contains it. If no footprint contains it, it is linked to the nearest footprint within 20 m; otherwise it stays unlinked.

**B3 Floor extraction.**
- **Tags:** the OSM `level` and `addr:floor` tags are read first. For multi-level values the lowest floor is used: `0;1` → 0, and `0-2` → 0.
- **Text:** otherwise, regular expressions search the name and address for numbered floors ("3rd Floor", "Floor No. 3"), written floors ("First floor") and codes (GF, UGF and mezzanine = 0; LGF and basement = −1).
- **Check:** a floor at or above the building's top floor is set to missing.

**B4 Bands.** Floors are grouped into bands with lower bounds σ = (−∞, 0, 1, 2, 4, 8): Basement, Ground, 1st, 2nd–3rd, 4th–7th, 8th+.

**B5 Prediction.**
- **Classifier:** POIs without a floor get band probabilities from a gradient-boosted classifier (250 iterations, learning rate 0.06, 15 leaves, L2 = 1.0).
- **Features:** land use, building floors and height, log footprint area, POIs in the building, brand presence, and coordinates.
- **Feasibility:** bands above the building's top floor are excluded and the probabilities renormalised. Basement and Ground are always allowed.
- **Evaluation:** stratified 5-fold cross-validation (exact-band accuracy, log loss), compared with a baseline that assigns each land use its most frequent band.
- **Fallback:** if a study area has fewer than 30 POIs with a known floor, Bengaluru land-use priors are used instead, blended with any local floors.

**B6 Monte Carlo.** In each of S draws (default 200), every POI receives a band drawn from its probabilities. POIs with a known floor keep it.

## C. Vertical stratification

**C1 Units:**
- buildings with ≥ 3 POIs;
- grid cells (default 500 m) with ≥ 10 POIs;
- zones with ≥ 10 POIs.

**C2 Indices** (per draw). With p<sub>kb</sub> the share of a unit's POIs of use *k* in band *b*:

- H<sub>use</sub> = −Σ p<sub>k</sub> ln p<sub>k</sub>
- H<sub>band</sub> = −Σ p<sub>b</sub> ln p<sub>b</sub>
- H<sub>joint</sub> = −Σ p<sub>kb</sub> ln p<sub>kb</sub>
- MI = H<sub>use</sub> + H<sub>band</sub> − H<sub>joint</sub>
- VSI = MI / H<sub>use</sub> ∈ [0, 1]

**C3 Chance correction.** The plug-in MI estimator is biased upward in small samples, by about (K′ − 1)(B′ − 1)/(2N) nats (Treves & Panzeri, 1995). In each draw, the band labels are shuffled among a unit's POIs, which keeps both marginals fixed and gives VSI<sub>null</sub>. The reported quantities are:
- **VSI_excess** = mean VSI − mean VSI<sub>null</sub>;
- **VSI_conf** = the share of draws with VSI > VSI<sub>null</sub>;
- **intervals:** 5th–95th percentiles across draws.

**C4 Zone statistics.** For each zone: population density, POI density, mean floors, share of buildings with ≥ 4 floors, and FAR = Σ a<sub>i</sub>F<sub>i</sub> / A. Associations with VSI_excess are measured with Spearman's ρ.

## References

- Sirko, W. et al. (2023). High-resolution building and road detection from Sentinel-2. arXiv:2310.11622.
- Treves, A., & Panzeri, S. (1995). The upward bias in measures of information derived from limited data samples. *Neural Computation*, 7(2), 399–407.
- Boeing, G. (2017). OSMnx. *Computers, Environment and Urban Systems*, 65, 126–139.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
