# Project: Explainable Physics-Guided Ensemble for Groundwater Vulnerability Mapping Under Climate Uncertainty

# 📁 Folder Structure Overview
# This repo is modular and reproducible. You can track raw data, modeling scripts, explainability code, map generation, and a public-facing dashboard.

# ─────────────────────────────────────────────
# 📂 groundwater_ai_project/
# ├── 📁 data/
# │   ├── raw/                       # GRACE, Sentinel, ERA5, CMIP6, etc.
# │   ├── processed/                 # Aligned, cleaned inputs
# │   └── static/                    # Static vars like LULC, soil, DEM
# │
# ├── 📁 preprocessing/
# │   ├── download_data.py          # Downloader scripts
# │   ├── resample_regrid.py        # Harmonize spatial resolution
# │   ├── sequence_builder.py       # Time series builder for ConvLSTM
# │   └── feature_engineering.py    # Combine static + dynamic vars
# │
# ├── 📁 models/
# │   ├── convlstm_model.py         # Spatiotemporal neural network
# │   ├── rf_model.py               # Random Forest with physics constraints
# │   ├── xgboost_model.py          # Optional ensemble member
# │   └── hybrid_pipeline.py        # Full ensemble orchestrator
# │
# ├── 📁 training/
# │   ├── train_convlstm.py         # ConvLSTM trainer script
# │   ├── train_rf.py               # RF + SHAP model training
# │   └── ensemble_trainer.py       # Combine models
# │
# ├── 📁 explainability/
# │   ├── compute_shap.py           # SHAP value calculation for RF
# │   └── shap_to_raster.py         # Turns SHAP values into spatial maps
# │
# ├── 📁 visualization/
# │   ├── generate_maps.py          # Risk maps, PNGs, GeoTIFFs
# │   └── dashboard.py              # Streamlit map explorer
# │
# ├── 📁 results/
# │   ├── shap_values.csv
# │   ├── shap_rasters/             # SHAP heatmaps
# │   ├── convlstm_predictions.nc   # NetCDF-style spatiotemporal output
# │   ├── risk_map_ssp245_2050.tif
# │   └── traffic_light_map.png     # PNG from predictions
# │
# ├── 📄 requirements.txt           # All dependencies listed here
# ├── 📄 README.md                  # You are here.
# ├── 📄 proposal_abstract.md       # Short concept write-up
# ├── 📄 convlstm_model.py          # ConvLSTM model structure
# ├── 📄 sequence_builder.py        # Data pipeline to sequence array
# ├── 📄 compute_shap.py            # SHAP feature importance
# ├── 📄 shap_to_raster.py          # Rasterize feature importance
# ├── 📄 generate_maps.py           # Risk PNG & GeoTIFF generator
# └── 📄 dashboard.py               # Streamlit interface
# ─────────────────────────────────────────────

# 📖 README.md

## 🌍 Project Title:
**Explainable Physics-Guided Ensemble for Groundwater Vulnerability Mapping Under Climate Uncertainty**

---

## 🔎 What Is This?
A powerful machine learning pipeline that:
- Predicts groundwater vulnerability across space and time
- Explains what factors cause that risk (like drought, irrigation, or land use)
- Shows how that risk will evolve under future climate scenarios (e.g., SSP2-4.5, SSP5-8.5)
- Outputs maps and dashboards usable by scientists, policymakers, and planners

---

## 💡 Why This Project Is Unique
| Feature                  | Our Work                        | Most Papers                       |
|--------------------------|----------------------------------|------------------------------------|
| Core Model               | ConvLSTM + RF ensemble          | Single model (RF or LSTM only)     |
| Climate Future           | Uses CMIP6 SSP scenarios        | Only historical data               |
| Explainability           | SHAP + spatial raster maps      | No explainability                  |
| Deployment-Ready Maps    | PNG + GeoTIFF + Web dashboard   | Static figures only                |
| Physics Integration      | Darcy’s law–penalty in RF model | Pure data-driven only              |

---

## 🧱 Architecture Summary

```
Data In
│
├── Sentinel-2 NDVI, soil moisture
├── ERA5 climate (temp, precip, evap)
├── CMIP6 (SSP245, SSP585)
├── GRACE/GLDAS for validation
└── Static: LULC, DEM, aquifer types

↓↓
Preprocessing (resample → align → sequence)
↓↓
ConvLSTM → predicts spatiotemporal groundwater
↓↓
Physics-Aware RF → improves + explains risk
↓↓
SHAP → feature attribution → maps
↓↓
GeoTIFF/PNG → Web dashboard
```

---

## 🚀 How to Run
```bash
# Step 1: Setup
pip install -r requirements.txt

# Step 2: Prepare sequences
python preprocessing/sequence_builder.py

# Step 3: Train ConvLSTM
python training/train_convlstm.py

# Step 4: Train RF and SHAP
python training/train_rf.py
python explainability/compute_shap.py

# Step 5: Convert SHAP → Rasters
python explainability/shap_to_raster.py

# Step 6: Generate maps
python visualization/generate_maps.py

# Step 7: Launch dashboard
streamlit run visualization/dashboard.py
```

---

## 📦 Dependencies
- TensorFlow / Keras
- XGBoost / Scikit-learn
- SHAP
- Rasterio
- Matplotlib
- Streamlit
- NetCDF4

---

## 📫 Contact
Iman Mahato  
`<add your email or GitHub if public>`  
Indian Institute of Science Education and Research (IISER) Berhampur  
Collaborators: Dr. Deepak Mishra (IIST), Prof. Gnanappazham L.

---

✅ You're now ready to reproduce, share, or present this end-to-end pipeline.
