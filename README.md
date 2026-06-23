# Anomaly Detection on Aircraft Sensor Data

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg)](https://scikit-learn.org/)
[![Bokeh](https://img.shields.io/badge/Bokeh-3.x-1A1A2E.svg)](https://bokeh.org/)

> Unsupervised anomaly detection on multivariate aircraft sensor time-series, combining classical ML (Isolation Forest, LOF) and Deep Learning (Autoencoder). Developed as part of the **Advanced Master in Artificial Intelligence (AIBT)** at **ISAE-SUPAERO**.

---

## Overview

Aircraft generate large volumes of sensor data per flight cycle. Detecting abnormal flight windows is critical for **predictive maintenance**, **safety analysis**, and **operational decision support**, yet labelled examples of failures are extremely rare.

This project tackles that exact constraint: **no labels are available**, and the goal is to flag suspicious flight windows based purely on the *shape* of the sensor signals.

The work was carried out as part of the Anomaly Detection module of the AIBT Advanced Master at ISAE-SUPAERO (January 2026).

---

## Dataset

- **Source:** `dataset.csv` provided by the ISAE-SUPAERO Machine Learning teaching team & Airbus Commercial Aircraft.
- **Granularity:** indexed by `day_cycle_window` (a unique key combining flight day, cycle, and time window).
- **Features:** 11 anonymised sensor channels (`p1` to `p11`) — speed, temperature, pressure, electrical current, etc.
- **Labels:** none. The task is fully **unsupervised**.
- **Assumption:** anomalous windows exhibit signal shapes (slope, volatility, amplitude) that diverge from the rest of the fleet.

---

## Pipeline

```
                +----------------------------+
                |  Raw sensor data (p1..p11) |
                +-------------+--------------+
                              |
                  Exploratory Data Analysis
                   (distributions, missing
                    values, descriptive stats)
                              |
                              v
                +----------------------------+
                |  PCA on raw data (check)   |  --> Only 79% variance with 3 PCs
                +-------------+--------------+      => Raw data too complex
                              |
                              v
                +----------------------------+
                |  Feature Engineering       |
                |  (mean, std, amplitude,    |
                |   slope per sensor)        |  --> 44 features per window
                +-------------+--------------+
                              |
                              v
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
  +-----------------+ +---------------+ +-----------------+
  | Isolation Forest| |      LOF      | |  Autoencoder    |
  | (contamination  | | (density,     | |  (PyTorch       |
  |  = 3%)          | |  k=20)        | |   32->8->32)    |
  +--------+--------+ +-------+-------+ +--------+--------+
           |                  |                  |
           +---------+--------+------------------+
                     |
                     v
         +-------------------------+
         |  Cross-model agreement  |
         |  + visual validation    |
         +-------------------------+
                     |
                     v
         +-------------------------+
         |  Ranked anomaly list    |
         |  + interactive timeline |
         +-------------------------+
```

---

## Methodology

### 1. Exploratory Data Analysis
- Statistical profiling of all 11 sensors (mean, std, percentiles, missing values).
- Interactive distribution plots with **Bokeh** to assess scale and shape.
- Per-window signal visualisation to develop an intuition of what a *normal* flight looks like.

### 2. Strategic pivot: PCA on raw data
- Applied PCA directly on the raw sensor matrix.
- First 3 components explained only **79 %** of the variance — confirming visual intuition that the data is too complex to reduce trivially.
- Decision: build hand-crafted features that capture the **shape** of the signals.

### 3. Feature engineering
For each window and each sensor we extract four shape descriptors:

| Feature      | Captures                                    |
|--------------|----------------------------------------------|
| `mean`       | Average operating level                      |
| `std`        | Vibration / volatility                       |
| `amplitude`  | `max - min` — overall range                  |
| `slope`      | Linear regression slope — trend direction    |

Total: **44 engineered features** (4 descriptors x 11 sensors).

### 4. Modelling
Three complementary detectors are trained on the engineered, standardised features:

- **Isolation Forest** (`contamination=0.03`) — tree-based, isolates points that are easy to separate from the rest.
- **Local Outlier Factor** (`n_neighbors=20`, `contamination=0.03`) — density-based, flags points in low-density regions.
- **Autoencoder** (PyTorch, architecture `44 -> 32 -> 8 -> 32 -> 44`, MSE loss, Adam, 30 epochs) — non-linear reconstruction, anomalies are points the network fails to reconstruct (above 97th-percentile reconstruction error).

### 5. Validation
- **Cross-model agreement** between Isolation Forest, LOF and the Autoencoder.
- **Feature importance** via per-feature mean difference (anomaly vs normal).
- **Visual proof**: each flagged anomaly is plotted against the "normal corridor" (mean +/- 2σ across normal windows).
- **Temporal analysis**: clusters of consecutive anomalous windows are highlighted as the most credible failure candidates.
- **Counter-example study**: a "high coefficient-of-variation but normal" window vs a "low CV but anomalous" window — demonstrating that *shape* matters more than raw variability.

---

## Key results

- **PCA-based reduction is insufficient** on raw data: feature engineering on signal shape is the right strategy.
- **Three independent detectors converge** on the same windows, which gives high confidence in the flagged anomalies despite the absence of labels.
- **Anomalies form temporal clusters** within the same day-cycle, suggesting persistent abnormal regimes rather than isolated noise.
- A simple statistical heuristic (coefficient of variation) is **not** a reliable proxy for anomaly: classifiers learn higher-order shape patterns.

---

## Tech stack

- **Language:** Python 3.10+
- **Data:** `pandas`, `numpy`, `scipy.stats`
- **Classical ML:** `scikit-learn` (`IsolationForest`, `LocalOutlierFactor`, `PCA`, `StandardScaler`)
- **Deep Learning:** `PyTorch` (custom Autoencoder, `nn.Linear`, `Adam`, `MSELoss`)
- **Visualisation:** `matplotlib`, `seaborn`, `Bokeh` (interactive tables and anomaly timeline)
- **Environment:** Jupyter Notebook

---

## What this project demonstrates

- End-to-end **unsupervised anomaly detection** pipeline on real, multivariate time-series.
- Sound methodological reasoning: switching from a failing PCA approach to **feature-engineered shape descriptors** based on diagnosed evidence.
- **Comparative modelling** with three different paradigms (tree-based, density-based, neural) to build robustness in the absence of ground truth.
- **PyTorch** implementation of an Autoencoder from scratch.
- Industrial-grade **interactive visualisation** with Bokeh.
- Clear written reasoning at each step — methodology, alternatives considered, and limitations.

---

## Authors

Project completed as part of the *Anomaly Detection* module of the **Advanced Master in Artificial Intelligence & Business Transformation (AIBT)** at **ISAE-SUPAERO**.

- **William Moncada** — [LinkedIn](https://www.linkedin.com/in/wmoncada/)
- El Arbi Aboulaz
- Jose Garnica-Aza

---


## Acknowledgements

Course material and dataset provided by the ISAE-SUPAERO Data Science team (SupaeroDataScience) and Airbus Commercial Aircraft