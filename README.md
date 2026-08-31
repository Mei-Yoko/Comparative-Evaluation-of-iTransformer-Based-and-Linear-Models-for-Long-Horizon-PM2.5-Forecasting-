# 🌫️ PM2.5 Time-Series Forecasting with iTransformer ($M=5$)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Supported-F9AB00.svg)](https://colab.research.google.com/)

An end-to-end Multivariate Time-Series Forecasting framework using **iTransformer** to predict PM2.5 concentrations across multiple monitoring stations in Thailand (2021–2025). 

This repository demonstrates spatial-temporal analysis, data cleaning & outlier treatment, model hyperparameter tuning, and performance evaluation under different loss functions (**MSE** vs. **MAE**).

---

## 📌 Features & Highlights

* **Multivariate Forecasting ($M=5$):** Uses 5 distinct spatial monitoring stations concurrently to capture cross-channel attention and spatial relationships.
* **Data Cleaning & Outlier Mitigation:** Features robust preprocessing including 99.5th percentile quantile clipping, zero/negative filtering, and time-based interpolation.
* **iTransformer Architecture:** Leverages inverted attention across variate dimensions for effective long-sequence multivariate time-series predictions (`SEQ_LEN=60`, `PRED_LEN=30`).
* **Loss Function Analysis:** Comparative study on how **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** impact peak under-estimation and macro-trend learning.

---

## 🛠️ Data Pipeline & Preprocessing

The historical dataset spans **2021 to 2025**, containing raw daily sensor readings across multiple stations.

1. **Spatial Filtering:** Selected top 5 primary monitoring stations ($M=5$).
2. **Quality Control:** 
   * Filtered out invalid sensor anomalies ($\le 0 \mu g/m^3$).
   * Clipped extreme outlier noise using a **99.5% Quantile Upper Threshold**.
3. **Interpolation:** Re-sampled daily series (`resample('D')`) and filled missing values using time-weighted interpolation (`interpolate(method='time')`).

---

## 📊 Methodology & Model Architecture
Input (60 Days, 5 Stations) ➔ [ Data Scaler ] ➔ [ iTransformer Encoder ] ➔ Output (30 Days, 5 Stations)
| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Lookback Window (`SEQ_LEN`)** | `60` days | Past sequence used for temporal features |
| **Forecast Horizon (`PRED_LEN`)** | `30` days | Future target window |
| **Number of Channels ($M$)** | `5` | Spatial monitoring stations |
| **`d_model`** | `64` | Embedding dimension |
| **`e_layers`** | `2` | Encoder layers |
| **`dropout`** | `0.1` | Regularization |
| **Optimizer & Epochs** | Adam (lr=`1e-3`), `40` Epochs | Batch size: 32 |

---

## 📈 Experiments & Visualizations

### 1. Daily PM2.5 Spatial Mean Timeline (2021–2025)
> Visualizing historical seasonality and pollution spikes in Thailand over a 5-year period.
![PM2.5 Cleaned Timeline](pm25_cleaned_timeline.png)

### 2. Tuned iTransformer with MSE Loss vs. MAE Loss
* **MSE Loss ($L_2$ Norm):** Effectively captures the macro-trend and overall seasonal transitions, though it tends to damp extreme peak values (Peak Under-estimation).
* **MAE Loss ($L_1$ Norm):** Demonstrates higher robustness against localized noise and provides sharper responses near sudden trend changes.

| MSE Result | MAE Result |
| :---: | :---: |
| ![MSE Forecast](itransformer_m5_mse_result.png) | ![MAE Forecast](itransformer_m5_mae_result.png) |

### 3. Trend Analysis (Rolling Smoothed Forecast)
> Post-processing centered 3-day moving average filter applied to prediction sequences to capture baseline trends cleanly.
![Smoothed Trend](itransformer_m5_result_smoothed.png)

---

## 🚀 Quick Start (Google Colab)

### 1. Prerequisites & Installation

Ensure you have your `.whl` package (`forecast2win`) uploaded to your runtime environment: but this runtime is now closing you cantact me for the runtime
2. Running the Pipeline
Clone the repo and run the main script or notebook:
import torch
import pandas as pd
import numpy as np
from -.models import get_model

# Load preprocessed sliding window data X_train (N, 60, 5) & Y_train (N, 30, 5)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize iTransformer
model = get_model(
    "itransformer",
    seq_len=60,
    pred_len=30,
    n_channels=5,
    d_model=64,
    e_layers=2,
    dropout=0.1
)

# Train using MAE or MSE Loss
model.fit(
    X_train,
    y=Y_train,
    loss="mae",
    epochs=40,
    batch_size=32,
    scale=True,
    device=DEVICE,
    lr=1e-3
)

# Predict
predictions = model.predict(X_test)

#💡 Key Takeaways & Limitations
Macro-Trend Learning: iTransformer demonstrates strong capacity in modeling seasonal transitions (dry/dust season vs. monsoon season).

Peak Under-estimation: Time-series inputs alone without external meteorological triggers (e.g., wind speed, humidity, hotspot counts) limit the model's ability to anticipate sharp, single-day pollution spikes.
