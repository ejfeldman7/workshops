# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2: Anomaly Detection for Sign-In Data
# MAGIC ## Notebook 3 — Isolation Forest
# MAGIC
# MAGIC **Isolation Forest** is our primary anomaly detection model. It works by randomly partitioning
# MAGIC the feature space — anomalies are easier to isolate (require fewer splits) than normal points.
# MAGIC
# MAGIC ### How Isolation Forest Works
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────┐
# MAGIC │              Isolation Forest Intuition                             │
# MAGIC │                                                                     │
# MAGIC │  Normal points (dense)          Anomaly (isolated)                  │
# MAGIC │                                                                     │
# MAGIC │    ●●●●●●                           ◆                               │
# MAGIC │    ●●●●●●●                                                          │
# MAGIC │    ●●●●●●           Needs many splits           Needs few splits    │
# MAGIC │    ●●●●●            to isolate one ●            to isolate ◆        │
# MAGIC │    ●●●                                                              │
# MAGIC │                                                                     │
# MAGIC │  Path length: LONG              Path length: SHORT                  │
# MAGIC │  → Normal                       → Anomaly                           │
# MAGIC │                                                                     │
# MAGIC │  Score ≈ average path length across all trees in the forest         │
# MAGIC └─────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Docs:**
# MAGIC - [scikit-learn Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
# MAGIC - [MLflow on Databricks](https://docs.databricks.com/en/mlflow/index.html)
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
import matplotlib.pyplot as plt

import os
import re
_user = spark.sql("SELECT current_user()").first()[0]
USER_ID = re.sub(r'[^a-zA-Z0-9]', '_', _user.split('@')[0])
artifact_path = f"/dbfs/tmp/workshops/{DATABASE}/{USER_ID}"
os.makedirs(artifact_path, exist_ok=True)
pdf = spark.read.parquet(f"{artifact_path}/signins_features.parquet").toPandas()
print(f"Loaded {len(pdf)} sign-in events")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Prepare Feature Matrix
# MAGIC
# MAGIC Select numeric features for the model. Handle missing values (first logins have no "previous" data).

# COMMAND ----------

MODEL_FEATURES = [
    "hour_sin", "hour_cos",
    "is_weekend", "is_off_hours",
    "minutes_since_last_login",
    "distance_from_prev_km",
    "geo_velocity_kmh",
    "is_unknown_location",
    "failed_attempts_before",
    "mfa_numeric",
    "session_duration_min",
    "session_duration_zscore",
    "is_new_device",
    "logins_last_hour",
]

# Fill NaN with 0 for first-login cases
X = pdf[MODEL_FEATURES].fillna(0).copy()

# Cap extreme geo_velocity values to avoid scale issues
X["geo_velocity_kmh"] = X["geo_velocity_kmh"].clip(upper=50000)

print(f"Feature matrix: {X.shape}")
print(f"Features: {MODEL_FEATURES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Train Isolation Forest
# MAGIC
# MAGIC | Parameter | Value | Why |
# MAGIC |-----------|-------|-----|
# MAGIC | `contamination` | 0.05 | Expect ~5% anomalies (start conservative) |
# MAGIC | `n_estimators` | 200 | More trees = more stable scores |
# MAGIC | `max_samples` | "auto" | Subsample for efficiency |
# MAGIC | `max_features` | 1.0 | Use all features |
# MAGIC
# MAGIC > **Tuning `contamination`:** Start at 5% and adjust based on analyst feedback.
# MAGIC > Too low = miss real threats. Too high = false positive overload.

# COMMAND ----------

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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
import time

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
mlflow.set_experiment(f"{os.path.dirname(notebook_path)}/login_anomaly_iforest")

# Scale features (important for distance-based comparisons later)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sweep contamination values
contamination_values = [0.02, 0.05, 0.08, 0.10]
results = {}

for contamination in contamination_values:
    with mlflow.start_run(run_name=f"iforest_c{contamination}"):
        start = time.time()

        iforest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples="auto",
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        )

        predictions = iforest.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
        scores = iforest.decision_function(X_scaled)  # lower = more anomalous

        elapsed = time.time() - start
        n_anomalies = (predictions == -1).sum()
        pct_anomalies = n_anomalies / len(predictions) * 100

        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("features", MODEL_FEATURES)
        mlflow.log_metric("n_anomalies", n_anomalies)
        mlflow.log_metric("pct_anomalies", pct_anomalies)
        mlflow.log_metric("fit_time_seconds", elapsed)

        results[contamination] = {
            "model": iforest,
            "predictions": predictions,
            "scores": scores,
            "n_anomalies": n_anomalies,
        }

        print(f"Contamination={contamination:.2f} | Anomalies={n_anomalies:6d} ({pct_anomalies:5.1f}%) | Time={elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Analyze Results (contamination=0.05)

# COMMAND ----------

# Use 0.05 as default — adjust based on analysis below
C = 0.05
best = results[C]
pdf["iforest_prediction"] = best["predictions"]
pdf["iforest_score"] = best["scores"]
pdf["is_anomaly"] = (best["predictions"] == -1).astype(int)

print(f"Anomalies detected: {best['n_anomalies']} ({best['n_anomalies']/len(pdf)*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score Distribution
# MAGIC
# MAGIC Lower scores = more anomalous. The decision boundary is at 0.

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 5))

normal_scores = pdf[pdf["is_anomaly"] == 0]["iforest_score"]
anomaly_scores = pdf[pdf["is_anomaly"] == 1]["iforest_score"]

ax.hist(normal_scores, bins=100, alpha=0.6, label="Normal", color="steelblue")
ax.hist(anomaly_scores, bins=100, alpha=0.6, label="Anomaly", color="red")
ax.axvline(x=0, color="black", linestyle="--", label="Decision boundary")
ax.set_xlabel("Isolation Forest Score (lower = more anomalous)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Isolation Forest Score Distribution", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Feature Importance
# MAGIC
# MAGIC Isolation Forest doesn't have built-in feature importance, but we can approximate it by
# MAGIC comparing feature distributions between normal and anomalous points.

# COMMAND ----------

normal_means = X[pdf["is_anomaly"] == 0].mean()
anomaly_means = X[pdf["is_anomaly"] == 1].mean()

importance = abs(anomaly_means - normal_means) / X.std()
importance = importance.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(importance)), importance.values, color="steelblue")
ax.set_yticks(range(len(importance)))
ax.set_yticklabels(importance.index)
ax.set_xlabel("Feature Divergence (|anomaly_mean - normal_mean| / std)", fontsize=12)
ax.set_title("Feature Importance for Anomaly Detection", fontsize=14)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Validate Against Known Anomaly Types
# MAGIC
# MAGIC Using the synthetic labels (`_anomaly_type`), let's see how well Isolation Forest catches each type.
# MAGIC In production you wouldn't have these labels — this is for workshop validation only.

# COMMAND ----------

detection_rates = pdf.groupby("_anomaly_type").agg(
    total=("is_anomaly", "count"),
    detected=("is_anomaly", "sum"),
).assign(
    detection_rate=lambda x: x["detected"] / x["total"] * 100
).sort_values("detection_rate", ascending=False)

print("Detection rates by anomaly type:\n")
for anom_type, row in detection_rates.iterrows():
    bar = "█" * int(row["detection_rate"] / 2)
    print(f"  {anom_type:25s}: {row['detected']:5.0f}/{row['total']:5.0f} ({row['detection_rate']:5.1f}%) {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Examine Top Anomalies
# MAGIC
# MAGIC Let's look at the most anomalous sign-in events and understand why the model flagged them.

# COMMAND ----------

top_anomalies = pdf[pdf["is_anomaly"] == 1].nsmallest(15, "iforest_score")

for _, row in top_anomalies.iterrows():
    print(f"Score: {row['iforest_score']:.4f} | User: {row['user_id']} | Type: {row['_anomaly_type']}")
    print(f"  Time: {row['timestamp']} | Location: {row['location_name']} | Device: {row['device']}")
    print(f"  Geo-velocity: {row['geo_velocity_kmh']:.0f} km/hr | Failed attempts: {row['failed_attempts_before']}")
    print(f"  MFA: {row['mfa_numeric']} | Off-hours: {row['is_off_hours']} | New device: {row['is_new_device']}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Save Model and Results

# COMMAND ----------

import pickle

with open(f"{artifact_path}/iforest_model.pkl", "wb") as f:
    pickle.dump(best["model"], f)
with open(f"{artifact_path}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{artifact_path}/model_features.pkl", "wb") as f:
    pickle.dump(MODEL_FEATURES, f)

spark.createDataFrame(pdf).write.mode("overwrite").parquet(f"{artifact_path}/signins_with_iforest.parquet")

# Save to Delta
df_results = spark.createDataFrame(
    pdf[["login_id", "iforest_score", "is_anomaly"]]
)
df_results.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE}.signins_iforest{SUFFIX_TAG}")

print(f"✓ Saved Isolation Forest model, scaler, and results")
print(f"✓ Delta table: {DATABASE}.signins_iforest{SUFFIX_TAG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC | Finding | Detail |
# MAGIC |---------|--------|
# MAGIC | Contamination matters | Start at 5%, adjust with analyst feedback |
# MAGIC | Geo-velocity is strongest signal | Impossible travel is the clearest anomaly |
# MAGIC | Failed attempts + no MFA | Strong brute-force indicator |
# MAGIC | Feature scaling helps | StandardScaler ensures all features contribute equally |
# MAGIC
# MAGIC **Next →** Open `04_pyod_ensemble` to compare multiple anomaly detection algorithms.