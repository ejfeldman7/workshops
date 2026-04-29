# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2: Anomaly Detection for Sign-In Data
# MAGIC ## Notebook 4 — PyOD Ensemble (Multi-Algorithm Comparison)
# MAGIC
# MAGIC No single algorithm catches all anomaly types. **PyOD** provides 40+ outlier detection algorithms
# MAGIC under a unified API, making it easy to run multiple detectors and combine their results.
# MAGIC
# MAGIC ### Why Ensemble?
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────┐
# MAGIC │              Multi-Algorithm Ensemble                            │
# MAGIC │                                                                  │
# MAGIC │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
# MAGIC │  │Isolation │  │  ECOD   │  │  LOF    │  │  KNN    │             │
# MAGIC │  │ Forest   │  │(Empir.) │  │(Local)  │  │(Dist.)  │             │
# MAGIC │  └────┬─────┘  └────┬────┘  └────┬────┘  └────┬────┘             │
# MAGIC │       │              │            │             │                │
# MAGIC │       └──────────────┴────────────┴─────────────┘                │
# MAGIC │                          │                                       │
# MAGIC │                 Average / Vote                                   │
# MAGIC │                          │                                       │
# MAGIC │                ┌─────────▼─────────┐                             │
# MAGIC │                │  Ensemble Score   │                             │
# MAGIC │                │  (more robust     │                             │
# MAGIC │                │   than any one)   │                             │
# MAGIC │                └───────────────────┘                             │
# MAGIC └──────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Docs:**
# MAGIC - [PyOD Documentation](https://pyod.readthedocs.io/)
# MAGIC - [Blog: Unsupervised Outlier Detection on Databricks](https://www.databricks.com/blog/2023/03/07/unsupervised-outlier-detection-databricks.html)
# MAGIC - [Solution Accelerator: Insider Threat Detection](https://github.com/databricks-industry-solutions/insider-threat)
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites

# COMMAND ----------

# MAGIC %pip install pyod --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("database", "login_anomaly", "Database")
from datetime import datetime
_user_email = spark.sql("SELECT current_user()").first()[0]
_name_parts = _user_email.split('@')[0].replace('_', '.').split('.')
_initials = (_name_parts[0][0] + _name_parts[-1][0]).lower() if len(_name_parts) >= 2 else _user_email[:2].lower()
_default_suffix = _initials + datetime.now().strftime('%d%m%y')
dbutils.widgets.text("table_suffix", _default_suffix, "Table Suffix (your initials)")

DATABASE = dbutils.widgets.get("database")
SUFFIX = dbutils.widgets.get("table_suffix").strip()
SUFFIX_TAG = f"_{SUFFIX}" if SUFFIX else ""
print(f"Database: {DATABASE}, Table suffix: {SUFFIX_TAG or '(none)'}")

# COMMAND ----------

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

import os
import re
_user = spark.sql("SELECT current_user()").first()[0]
USER_ID = re.sub(r'[^a-zA-Z0-9]', '_', _user.split('@')[0])
artifact_path = f"/dbfs/tmp/workshops/{DATABASE}/{USER_ID}"
os.makedirs(artifact_path, exist_ok=True)
pdf = spark.read.parquet(f"{artifact_path}/signins_with_iforest.parquet").toPandas()

with open(f"{artifact_path}/model_features.pkl", "rb") as f:
    MODEL_FEATURES = pickle.load(f)
with open(f"{artifact_path}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X = pdf[MODEL_FEATURES].fillna(0).copy()
X["geo_velocity_kmh"] = X["geo_velocity_kmh"].clip(upper=50000)
X_scaled = scaler.transform(X)

print(f"Loaded {len(pdf)} events, {len(MODEL_FEATURES)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Run Multiple PyOD Detectors
# MAGIC
# MAGIC We'll run 4 different algorithms, each with different strengths:
# MAGIC
# MAGIC | Algorithm | Approach | Best For |
# MAGIC |-----------|----------|----------|
# MAGIC | **ECOD** | Empirical cumulative distribution | Tail anomalies (extreme values) |
# MAGIC | **LOF** | Local density comparison | Anomalies in dense regions |
# MAGIC | **KNN** | Distance to K nearest neighbors | Global isolation |
# MAGIC | **COPOD** | Copula-based | Multivariate dependency anomalies |

# COMMAND ----------

from pyod.models.ecod import ECOD
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.copod import COPOD

import sys
import os

# Clear stale mlflow module state if present (prevents circular import error)
mlflow_keys = [k for k in sys.modules if k == 'mlflow' or k.startswith('mlflow.')]
for k in mlflow_keys:
    del sys.modules[k]

import mlflow
import mlflow.sklearn
# Register sklearn integration so DBR's MLflow autologging shim doesn't KeyError on fit_predict
mlflow.sklearn.autolog(disable=True)

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
mlflow.set_experiment(f"{os.path.dirname(notebook_path)}/login_anomaly_pyod")

detectors = {
    "ECOD": ECOD(contamination=0.05),
    "LOF": LOF(n_neighbors=20, contamination=0.05),
    "KNN": KNN(n_neighbors=10, contamination=0.05, method="mean"),
    "COPOD": COPOD(contamination=0.05),
}

detector_results = {}

for name, detector in detectors.items():
    with mlflow.start_run(run_name=f"pyod_{name.lower()}"):
        start = time.time()

        detector.fit(X_scaled)
        labels = detector.labels_         # 0 = normal, 1 = anomaly
        scores = detector.decision_scores_  # higher = more anomalous

        elapsed = time.time() - start
        n_anomalies = labels.sum()

        mlflow.log_param("algorithm", name)
        mlflow.log_param("contamination", 0.05)
        mlflow.log_metric("n_anomalies", int(n_anomalies))
        mlflow.log_metric("fit_time_seconds", elapsed)

        detector_results[name] = {
            "labels": labels,
            "scores": scores,
            "n_anomalies": n_anomalies,
            "time": elapsed,
        }

        pdf[f"{name}_label"] = labels
        pdf[f"{name}_score"] = scores

        print(f"{name:8s} | Anomalies: {n_anomalies:6d} ({n_anomalies/len(pdf)*100:5.1f}%) | Time: {elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Compare Detection Results
# MAGIC
# MAGIC How much do the algorithms agree? When multiple algorithms flag the same event, confidence is high.

# COMMAND ----------

# Agreement analysis
algo_labels = np.column_stack([detector_results[name]["labels"] for name in detectors])
agreement_count = algo_labels.sum(axis=1)  # How many algorithms flagged each event

pdf["n_detectors_flagged"] = agreement_count

print("Events flagged by N detectors:")
for n in range(5):
    count = (agreement_count == n).sum()
    pct = count / len(pdf) * 100
    bar = "█" * int(pct)
    print(f"  {n} detectors: {count:6d} ({pct:5.1f}%) {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Detection Overlap Heatmap

# COMMAND ----------

import seaborn as sns

# Pairwise agreement between detectors
algo_names = list(detectors.keys())
overlap = np.zeros((len(algo_names), len(algo_names)))

for i, name_i in enumerate(algo_names):
    for j, name_j in enumerate(algo_names):
        flags_i = detector_results[name_i]["labels"]
        flags_j = detector_results[name_j]["labels"]
        # Jaccard similarity: intersection / union
        intersection = ((flags_i == 1) & (flags_j == 1)).sum()
        union = ((flags_i == 1) | (flags_j == 1)).sum()
        overlap[i, j] = intersection / union if union > 0 else 0

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(overlap, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=algo_names, yticklabels=algo_names, ax=ax)
ax.set_title("Detector Agreement (Jaccard Similarity)", fontsize=14)
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Build Ensemble Score
# MAGIC
# MAGIC Combine all detector scores via **normalized averaging**. This is more robust than any single detector.

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler

# Normalize each detector's scores to [0, 1]
norm_scores = {}
mm = MinMaxScaler()

for name in detectors:
    raw = detector_results[name]["scores"].reshape(-1, 1)
    normalized = mm.fit_transform(raw).flatten()
    norm_scores[name] = normalized

# Average ensemble score
ensemble_scores = np.mean([norm_scores[name] for name in detectors], axis=0)
pdf["ensemble_score"] = ensemble_scores

# Also include Isolation Forest
iforest_norm = mm.fit_transform((-pdf["iforest_score"].values).reshape(-1, 1)).flatten()  # negate: lower IF score = more anomalous
pdf["ensemble_with_iforest"] = np.mean(
    [ensemble_scores, iforest_norm], axis=0
)

print("Ensemble score statistics:")
print(f"  Mean: {ensemble_scores.mean():.4f}")
print(f"  Std:  {ensemble_scores.std():.4f}")
print(f"  Min:  {ensemble_scores.min():.4f}")
print(f"  Max:  {ensemble_scores.max():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Compare All Methods

# COMMAND ----------

# Detection rates by anomaly type for each method
methods = list(detectors.keys()) + ["IsolationForest", "Ensemble"]

# Create binary labels for ensemble at 95th percentile threshold
threshold_95 = np.percentile(pdf["ensemble_with_iforest"], 95)
pdf["ensemble_anomaly"] = (pdf["ensemble_with_iforest"] > threshold_95).astype(int)

print(f"{'Method':15s} | {'impossible_travel':18s} | {'off_hours':12s} | {'brute_force':12s} | {'new_device_loc':15s} | {'normal FP':10s}")
print("-" * 95)

for method in methods:
    if method == "IsolationForest":
        col = "is_anomaly"
    elif method == "Ensemble":
        col = "ensemble_anomaly"
    else:
        col = f"{method}_label"

    rates = []
    for anom_type in ["impossible_travel", "off_hours", "brute_force", "new_device_location", "normal"]:
        subset = pdf[pdf["_anomaly_type"] == anom_type]
        rate = subset[col].mean() * 100
        rates.append(f"{rate:5.1f}%")

    print(f"{method:15s} | {rates[0]:18s} | {rates[1]:12s} | {rates[2]:12s} | {rates[3]:15s} | {rates[4]:10s}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Score Distribution Comparison

# COMMAND ----------

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, name in enumerate(list(detectors.keys()) + ["IsolationForest", "Ensemble"]):
    ax = axes[idx]

    if name == "IsolationForest":
        scores = -pdf["iforest_score"]  # negate for consistent direction
    elif name == "Ensemble":
        scores = pdf["ensemble_with_iforest"]
    else:
        scores = pdf[f"{name}_score"]

    for anom_type, color in [("normal", "steelblue"), ("impossible_travel", "red"),
                              ("brute_force", "orange"), ("off_hours", "purple")]:
        mask = pdf["_anomaly_type"] == anom_type
        ax.hist(scores[mask], bins=50, alpha=0.5, label=anom_type, color=color, density=True)

    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Anomaly Score")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("Score Distributions by Anomaly Type", fontsize=14, y=1.02)
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Save Ensemble Results

# COMMAND ----------

spark.createDataFrame(pdf).write.mode("overwrite").parquet(f"{artifact_path}/signins_with_ensemble.parquet")

# Save ensemble scores to Delta
score_cols = ["login_id", "iforest_score", "is_anomaly", "ensemble_score",
              "ensemble_with_iforest", "ensemble_anomaly", "n_detectors_flagged"]
for name in detectors:
    score_cols.extend([f"{name}_label", f"{name}_score"])

df_scores = spark.createDataFrame(pdf[score_cols])
df_scores.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE}.signins_ensemble{SUFFIX_TAG}")

print(f"✓ Saved ensemble results to {DATABASE}.signins_ensemble{SUFFIX_TAG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC | Insight | Detail |
# MAGIC |---------|--------|
# MAGIC | No single detector is best | Each algorithm catches different anomaly types |
# MAGIC | Ensemble is most robust | Averaging normalized scores reduces false positives |
# MAGIC | Agreement count matters | Events flagged by 3+ detectors are high-confidence anomalies |
# MAGIC | PyOD makes comparison easy | Unified API, same 3 lines per algorithm |
# MAGIC
# MAGIC **Next →** Open `05_per_user_models` to train per-user anomaly detection models with SHAP explainability.