# Nyange Soil Erosion Susceptibility Model

Machine learning pipeline for mapping soil erosion susceptibility in the Nyange sector using topographic, hydrological, soil, and satellite-derived predictors, with Random Forest and XGBoost classifiers and SHAP-based feature attribution.

## Pipeline overview

1. **Topographic & hydrological derivatives** — Slope, aspect, profile curvature, Topographic Wetness Index (TWI), and distance-to-streams computed from a 10 m DEM (gradient-based topography, `pysheds` flow routing for hydrology).
2. **Satellite & environmental data acquisition** — Sentinel-2 (median composite, 2024, <10% cloud cover), SoilGrids (clay/sand/silt, 0–5 cm), and CHIRPS rainfall, pulled via Google Earth Engine.
3. **Target label construction** — Binary erosion label derived from NDVI and Bare Soil Index (BSI) thresholds on Sentinel-2 bands (low NDVI + high BSI → eroded/bare; high NDVI → vegetated/stable).
4. **Multicollinearity check** — Pearson correlation matrix across all predictors; `Rainfall` dropped due to high correlation with `Elevation` (r ≈ 0.73).
5. **Model training** — Random Forest and XGBoost classifiers on a class-balanced sample, evaluated with accuracy, precision, recall, F1, and ROC-AUC.
6. **Explainability** — SHAP TreeExplainer summary plot for the XGBoost model.
7. **Spatial inference** — Trained models applied back across the full raster grid to produce erosion risk maps (single-model and RF/XGBoost soft-voting ensemble).

## Known limitation: target leakage via `Land_Use`

**This is an open issue, not yet resolved in this version of the notebook.**

The target label (step 3) is derived from NDVI/BSI computed on Sentinel-2 imagery. The `Land_Use` predictor used in training is very likely derived from, or strongly correlated with, that same spectral information — meaning it functions less as an independent predictor and more as a near-restatement of the label.

Evidence for this:
- Reported model performance (≈97% accuracy, ROC-AUC ≈ 0.99) is implausibly high for an 8-feature topographic/soil susceptibility model on a real-world target — published erosion susceptibility models in the literature typically land in the 0.75–0.90 AUC range.
- SHAP analysis shows `Elevation` and `Land_Use` dominating feature attribution, consistent with the model leaning on a near-circular signal rather than genuine causal drivers.

**Before treating these results as final:** retrain without `Land_Use` (or any predictor derived from the same source imagery as the label) and compare performance. A drop toward a more literature-consistent AUC range would confirm the leakage diagnosis. A spatially-blocked train/test split (rather than random pixel split) is also recommended, given spatial autocorrelation in raster-derived training data.

## Repository structure

```
.
├── Final_.ipynb          # Full pipeline notebook
└── README.md
```

Expected local data layout (not included in this repo — paths are relative):

```
./SOIL_EROSION_ML/
├── final_model_features_final/final_model_features/Master_Elevation_10m.tif
├── dataset_aligned/aligned_UTM_soilgrid.tif
├── FINAL_V2_STACK/        # intermediate derived rasters
├── NYANGE_CLEAN_STACK/    # final feature stack + balanced inventory CSV
├── rf_model.pkl
└── xgb_model.pkl
```

## Requirements

```
rasterio
numpy
pandas
scikit-learn
xgboost
shap
seaborn
matplotlib
pysheds
scipy
earthengine-api
joblib
```

Note: the Earth Engine cell (`ee.Initialize()`, `FeatureCollection('projects/certain1/assets/nyange')`) references a Google Cloud project and asset that are specific to the original author's account. To re-run that step, point to your own GEE project and AOI asset.

## License

Open source
