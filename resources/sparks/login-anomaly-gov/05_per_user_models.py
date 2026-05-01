# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2: Anomaly Detection for Sign-In Data
# MAGIC ## Notebook 5 — Per-User Models with SHAP Explainability
# MAGIC
# MAGIC The global Isolation Forest (notebook 03) and PyOD ensemble (notebook 04) treat all users the same.
# MAGIC But a 2 AM login is normal for a night-shift worker and anomalous for a 9-5 employee.
# MAGIC
# MAGIC This notebook trains a **separate anomaly detection model per user** using Spark's `applyInPandas`,
# MAGIC then adds **SHAP explanations** so analysts know *why* each login was flagged.
# MAGIC
# MAGIC ### Per-User vs. Global Models
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────────┐
# MAGIC │  Global Model (notebooks 03-04)    Per-User Models (this notebook)   │
# MAGIC │                                                                      │
# MAGIC │  ┌──────────────┐                  ┌──────────────┐                  │
# MAGIC │  │ ALL users    │                  │   User A     │                  │
# MAGIC │  │ one model    │                  │  own model   │                  │
# MAGIC │  │              │                  ├──────────────┤                  │
# MAGIC │  │  Same        │                  │   User B     │                  │
# MAGIC │  │  threshold   │                  │  own model   │                  │
# MAGIC │  │  for         │                  ├──────────────┤                  │
# MAGIC │  │  everyone    │                  │   User C     │                  │
# MAGIC │  │              │                  │  own model   │                  │
# MAGIC │  └──────────────┘                  └──────────────┘                  │
# MAGIC │                                                                      │
# MAGIC │  "2 AM login is always             "2 AM is normal for User A        │
# MAGIC │   suspicious"                       but anomalous for User B"        │
# MAGIC │                                                                      │
# MAGIC │  Scales: 1 model                   Scales: 1 model per user          │
# MAGIC │  Explains: nothing                 Explains: SHAP per prediction     │
# MAGIC └──────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Key Reference:** [Training 10,000 Anomaly Detection Models on One Billion Records](https://www.databricks.com/blog/training-10000-anomaly-detection-models-one-billion-records-explainable-predictions)
# MAGIC
# MAGIC **Azure Gov Cloud:** ✅ This uses Pandas UDFs on classic compute — no serverless, Model Serving, or UC required.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites

# COMMAND ----------

# MAGIC %pip install pyod shap --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime

dbutils.widgets.text("database", "login_anomaly", "Database")
_user_email = spark.sql("SELECT current_user()").first()[0]
_name_parts = _user_email.split('@')[0].replace('_', '.').split('.')
_initials = (_name_parts[0][0] + _name_parts[-1][0]).lower() if len(_name_parts) >= 2 else _user_email[:2].lower()
_default_suffix = _initials + datetime.now().strftime('%d%m%y')
dbutils.widgets.text("table_suffix", _default_suffix, "Table Suffix (your initials)")

DATABASE = dbutils.widgets.get("database")
SUFFIX = dbutils.widgets.get("table_suffix").strip()
SUFFIX_TAG = f"_{SUFFIX}" if SUFFIX else ""
spark.sql(f"USE {DATABASE}")
print(f"Database: {DATABASE}, Table suffix: {SUFFIX_TAG or '(none)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Feature Data

# COMMAND ----------

import pandas as pd
import numpy as np

import os
import re

_user = spark.sql("SELECT current_user()").first()[0]
USER_ID = re.sub(r'[^a-zA-Z0-9]', '_', _user.split('@')[0])
artifact_path = f"/dbfs/tmp/workshops/{DATABASE}/{USER_ID}"
os.makedirs(artifact_path, exist_ok=True)

df_features = spark.read.table(f"{DATABASE}.signins_features{SUFFIX_TAG}")
print(f"Loaded {df_features.count()} sign-in events")
print(f"Unique users: {df_features.select('user_id').distinct().count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Define Per-User Training Function
# MAGIC
# MAGIC The core pattern: define a function that takes a **single user's data** (as a pandas DataFrame),
# MAGIC trains an Isolation Forest on that user, scores all their logins, and returns the results.
# MAGIC
# MAGIC Spark's `applyInPandas` calls this function once per user, distributing the work across the cluster.
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────┐
# MAGIC │              applyInPandas Execution Pattern                    │
# MAGIC │                                                                 │
# MAGIC │  Spark DataFrame                                                │
# MAGIC │  ┌──────────────────────────┐                                   │
# MAGIC │  │ user001 | login1 | ...   │──┐                                │
# MAGIC │  │ user001 | login2 | ...   │  │   Worker 1: train_user_model() │
# MAGIC │  │ user001 | login3 | ...   │──┘   → scores for user001         │
# MAGIC │  │ user002 | login1 | ...   │──┐                                │
# MAGIC │  │ user002 | login2 | ...   │  │   Worker 2: train_user_model() │
# MAGIC │  │ user002 | login3 | ...   │──┘   → scores for user002         │
# MAGIC │  │ ...                      │                                   │
# MAGIC │  │ user100 | login1 | ...   │──┐                                │
# MAGIC │  │ user100 | login2 | ...   │  │   Worker N: train_user_model() │
# MAGIC │  │ user100 | login3 | ...   │──┘   → scores for user100         │
# MAGIC │  └──────────────────────────┘                                   │
# MAGIC │                                                                 │
# MAGIC │  All models train in PARALLEL across cluster workers            │
# MAGIC └─────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

from pyspark.sql.types import *

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

# Output schema: original login_id + user_id + per-user anomaly score
output_schema = StructType([
    StructField("login_id", StringType()),
    StructField("user_id", StringType()),
    StructField("per_user_score", DoubleType()),
    StructField("per_user_anomaly", IntegerType()),
    StructField("user_model_n_samples", IntegerType()),
])

def train_user_model(user_pdf: pd.DataFrame) -> pd.DataFrame:
    """Train an Isolation Forest on a single user's data and score all their logins."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    user_id = user_pdf["user_id"].iloc[0]
    n_samples = len(user_pdf)

    # Need minimum samples for meaningful model
    if n_samples < 20:
        return pd.DataFrame({
            "login_id": user_pdf["login_id"],
            "user_id": user_pdf["user_id"],
            "per_user_score": [0.0] * n_samples,
            "per_user_anomaly": [0] * n_samples,
            "user_model_n_samples": [n_samples] * n_samples,
        })

    # Prepare features
    X = user_pdf[MODEL_FEATURES].fillna(0).copy()
    X["geo_velocity_kmh"] = X["geo_velocity_kmh"].clip(upper=50000)

    # Scale per-user (each user has their own baseline)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train per-user Isolation Forest
    contamination = min(0.1, max(0.01, 5 / n_samples))  # adaptive: at least 1%, at most 10%
    iforest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    iforest.fit(X_scaled)

    scores = iforest.decision_function(X_scaled)
    predictions = iforest.predict(X_scaled)

    return pd.DataFrame({
        "login_id": user_pdf["login_id"],
        "user_id": user_pdf["user_id"],
        "per_user_score": scores.tolist(),
        "per_user_anomaly": [1 if p == -1 else 0 for p in predictions],
        "user_model_n_samples": [n_samples] * n_samples,
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Train All Per-User Models in Parallel
# MAGIC
# MAGIC This single call distributes model training across all cluster workers.
# MAGIC With 100 users, we train 100 independent Isolation Forests in parallel.

# COMMAND ----------

import time

# Select only needed columns to minimize data shuffle
columns_needed = ["login_id", "user_id"] + MODEL_FEATURES
df_input = df_features.select(*columns_needed)

start = time.time()

df_per_user = df_input.groupBy("user_id").applyInPandas(
    train_user_model,
    schema=output_schema,
)

# Materialize
df_per_user = df_per_user.cache()
total_scored = df_per_user.count()
elapsed = time.time() - start

n_users = df_per_user.select("user_id").distinct().count()
n_anomalies = df_per_user.filter("per_user_anomaly = 1").count()

print(f"✓ Trained {n_users} per-user models in {elapsed:.1f}s")
print(f"  Total logins scored: {total_scored:,}")
print(f"  Anomalies detected: {n_anomalies:,} ({n_anomalies/total_scored*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Compare Per-User vs. Global Detection
# MAGIC
# MAGIC Join per-user scores with the global ensemble scores from notebook 04 to see where they differ.

# COMMAND ----------

from pyspark.sql import functions as F

df_global = spark.read.table(f"{DATABASE}.signins_ensemble{SUFFIX_TAG}") \
    .select("login_id", "is_anomaly", "ensemble_with_iforest")

df_comparison = df_per_user.join(df_global, on="login_id") \
    .join(df_features.select("login_id", "_anomaly_type"), on="login_id")

df_comparison.cache()

display(
    df_comparison.groupBy("_anomaly_type").agg(
        F.count("*").alias("total"),
        F.sum("is_anomaly").alias("global_detected"),
        F.sum("per_user_anomaly").alias("per_user_detected"),
        F.round(F.avg("is_anomaly") * 100, 1).alias("global_rate_pct"),
        F.round(F.avg("per_user_anomaly") * 100, 1).alias("per_user_rate_pct"),
    ).orderBy("_anomaly_type")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Disagreement Analysis
# MAGIC
# MAGIC The most interesting cases are where per-user and global models **disagree**:
# MAGIC - **Per-user flags, global misses**: The user's personal baseline was violated but it looked normal globally
# MAGIC - **Global flags, per-user misses**: Looks weird globally but is normal for this specific user

# COMMAND ----------

display(
    df_comparison.withColumn(
        "detection_category",
        F.when((F.col("per_user_anomaly") == 1) & (F.col("is_anomaly") == 1), "Both flag")
         .when((F.col("per_user_anomaly") == 1) & (F.col("is_anomaly") == 0), "Per-user only")
         .when((F.col("per_user_anomaly") == 0) & (F.col("is_anomaly") == 1), "Global only")
         .otherwise("Both normal")
    ).groupBy("detection_category", "_anomaly_type")
     .count()
     .orderBy("detection_category", "_anomaly_type")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: SHAP Explainability
# MAGIC
# MAGIC **SHAP (SHapley Additive exPlanations)** tells analysts *which features* drove the anomaly score
# MAGIC for each specific login. Instead of "this login has anomaly score -0.12", the analyst sees
# MAGIC "this login was flagged because geo_velocity was 15x above this user's norm and MFA was skipped."
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────┐
# MAGIC │              SHAP Explanation for Login Event                    │
# MAGIC │                                                                  │
# MAGIC │  Feature                     SHAP Value (contribution)           │
# MAGIC │  ─────────────────────────   ───────────────────────             │
# MAGIC │  geo_velocity_kmh            ████████████████  +0.32  ← TOP      │
# MAGIC │  mfa_numeric (skipped)       ██████████        +0.18             │
# MAGIC │  is_unknown_location         ████████          +0.15             │
# MAGIC │  is_off_hours                ███               +0.06             │
# MAGIC │  session_duration_min        ██                +0.04             │
# MAGIC │  failed_attempts_before      █                 +0.02             │
# MAGIC │  ...                                                             │
# MAGIC │                                                                  │
# MAGIC │  "This login was flagged primarily because the user              │
# MAGIC │   logged in from 8,000 km away within 20 minutes                 │
# MAGIC │   of their last login, without MFA."                             │
# MAGIC └──────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Docs:** [SHAP documentation](https://shap.readthedocs.io/)
# MAGIC
# MAGIC **Azure Gov Cloud:** ✅ SHAP runs locally — pure Python, no external API calls.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train a Per-User Model with SHAP for a Sample User
# MAGIC
# MAGIC We demonstrate SHAP on one user first, then scale it with `applyInPandas`.

# COMMAND ----------

import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Pick a user with anomalies
pdf_all = df_features.toPandas()
user_with_anomalies = pdf_all[pdf_all["_anomaly_type"] != "normal"]["user_id"].value_counts().index[0]
user_data = pdf_all[pdf_all["user_id"] == user_with_anomalies].copy()

print(f"User: {user_with_anomalies}")
print(f"  Total logins: {len(user_data)}")
print(f"  Anomalous: {(user_data['_anomaly_type'] != 'normal').sum()}")

# Prepare features
X_user = user_data[MODEL_FEATURES].fillna(0).copy()
X_user["geo_velocity_kmh"] = X_user["geo_velocity_kmh"].clip(upper=50000)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_user)

# Train per-user model
iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iforest.fit(X_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute SHAP Values
# MAGIC
# MAGIC We use `shap.TreeExplainer` which is optimized for tree-based models like Isolation Forest.

# COMMAND ----------

explainer = shap.TreeExplainer(iforest)
shap_values = explainer.shap_values(X_scaled)

# Create a DataFrame with feature names for readable plots
X_display = pd.DataFrame(X_scaled, columns=MODEL_FEATURES)

print(f"SHAP values shape: {shap_values.shape}")
print(f"  {shap_values.shape[0]} login events × {shap_values.shape[1]} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ### SHAP Summary Plot — Which Features Matter Most for This User?

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_display, plot_type="bar", show=False)
plt.title(f"Feature Importance for {user_with_anomalies}", fontsize=14)
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SHAP Beeswarm — Feature Values vs. Impact

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_display, show=False)
plt.title(f"SHAP Beeswarm for {user_with_anomalies}", fontsize=14)
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explain Individual Anomalous Logins
# MAGIC
# MAGIC For each flagged login, show exactly which features drove the anomaly detection.

# COMMAND ----------

predictions = iforest.predict(X_scaled)
anomaly_indices = np.where(predictions == -1)[0]

print(f"Explaining {len(anomaly_indices)} anomalous logins for {user_with_anomalies}:\n")

for idx in anomaly_indices[:5]:  # Show up to 5
    row = user_data.iloc[idx]
    print(f"Login at {row['timestamp']} from {row['location_name']} ({row['device']})")
    print(f"  Actual type: {row['_anomaly_type']}")
    print(f"  Top contributing features:")

    # Sort SHAP values for this prediction
    shap_for_event = shap_values[idx]
    sorted_idx = np.argsort(np.abs(shap_for_event))[::-1]

    for feat_idx in sorted_idx[:5]:
        feat_name = MODEL_FEATURES[feat_idx]
        feat_val = X_user.iloc[idx][feat_name]
        shap_val = shap_for_event[feat_idx]
        direction = "↑ anomalous" if shap_val < 0 else "↓ normal"
        print(f"    {feat_name:30s} value={feat_val:10.2f}  SHAP={shap_val:+.4f} {direction}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Scale SHAP to All Users with applyInPandas
# MAGIC
# MAGIC Now we combine per-user model training + SHAP into a single `applyInPandas` call
# MAGIC that produces both anomaly scores and top-3 SHAP explanations per login.

# COMMAND ----------

shap_output_schema = StructType([
    StructField("login_id", StringType()),
    StructField("user_id", StringType()),
    StructField("per_user_score", DoubleType()),
    StructField("per_user_anomaly", IntegerType()),
    StructField("shap_top1_feature", StringType()),
    StructField("shap_top1_value", DoubleType()),
    StructField("shap_top2_feature", StringType()),
    StructField("shap_top2_value", DoubleType()),
    StructField("shap_top3_feature", StringType()),
    StructField("shap_top3_value", DoubleType()),
])

def train_and_explain(user_pdf: pd.DataFrame) -> pd.DataFrame:
    """Train per-user Isolation Forest, score, and compute top-3 SHAP features."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import shap
    import numpy as np

    user_id = user_pdf["user_id"].iloc[0]
    n = len(user_pdf)
    features = [
        "hour_sin", "hour_cos", "is_weekend", "is_off_hours",
        "minutes_since_last_login", "distance_from_prev_km", "geo_velocity_kmh",
        "is_unknown_location", "failed_attempts_before", "mfa_numeric",
        "session_duration_min", "session_duration_zscore", "is_new_device", "logins_last_hour",
    ]

    empty_result = pd.DataFrame({
        "login_id": user_pdf["login_id"], "user_id": user_pdf["user_id"],
        "per_user_score": [0.0]*n, "per_user_anomaly": [0]*n,
        "shap_top1_feature": [""]*n, "shap_top1_value": [0.0]*n,
        "shap_top2_feature": [""]*n, "shap_top2_value": [0.0]*n,
        "shap_top3_feature": [""]*n, "shap_top3_value": [0.0]*n,
    })

    if n < 20:
        return empty_result

    X = user_pdf[features].fillna(0).copy()
    X["geo_velocity_kmh"] = X["geo_velocity_kmh"].clip(upper=50000)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = min(0.1, max(0.01, 5 / n))
    iforest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iforest.fit(X_scaled)

    scores = iforest.decision_function(X_scaled)
    predictions = iforest.predict(X_scaled)

    # SHAP explanations
    try:
        explainer = shap.TreeExplainer(iforest)
        shap_vals = explainer.shap_values(X_scaled)
    except Exception:
        return pd.DataFrame({
            "login_id": user_pdf["login_id"], "user_id": user_pdf["user_id"],
            "per_user_score": scores.tolist(),
            "per_user_anomaly": [1 if p == -1 else 0 for p in predictions],
            "shap_top1_feature": [""]*n, "shap_top1_value": [0.0]*n,
            "shap_top2_feature": [""]*n, "shap_top2_value": [0.0]*n,
            "shap_top3_feature": [""]*n, "shap_top3_value": [0.0]*n,
        })

    # Extract top-3 SHAP features per login
    top1_feat, top1_val = [], []
    top2_feat, top2_val = [], []
    top3_feat, top3_val = [], []

    for i in range(n):
        sorted_idx = np.argsort(np.abs(shap_vals[i]))[::-1]
        top1_feat.append(features[sorted_idx[0]])
        top1_val.append(float(shap_vals[i][sorted_idx[0]]))
        top2_feat.append(features[sorted_idx[1]] if len(sorted_idx) > 1 else "")
        top2_val.append(float(shap_vals[i][sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0)
        top3_feat.append(features[sorted_idx[2]] if len(sorted_idx) > 2 else "")
        top3_val.append(float(shap_vals[i][sorted_idx[2]]) if len(sorted_idx) > 2 else 0.0)

    return pd.DataFrame({
        "login_id": user_pdf["login_id"],
        "user_id": user_pdf["user_id"],
        "per_user_score": scores.tolist(),
        "per_user_anomaly": [1 if p == -1 else 0 for p in predictions],
        "shap_top1_feature": top1_feat, "shap_top1_value": top1_val,
        "shap_top2_feature": top2_feat, "shap_top2_value": top2_val,
        "shap_top3_feature": top3_feat, "shap_top3_value": top3_val,
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Per-User Training + SHAP Across All Users

# COMMAND ----------

start = time.time()

df_explained = df_input.groupBy("user_id").applyInPandas(
    train_and_explain,
    schema=shap_output_schema,
)

df_explained = df_explained.cache()
count = df_explained.count()
elapsed = time.time() - start

print(f"✓ Trained + explained {df_explained.select('user_id').distinct().count()} user models in {elapsed:.1f}s")
print(f"  Total logins scored with SHAP: {count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Analyze SHAP Explanations at Scale
# MAGIC
# MAGIC What features are most commonly the top driver of anomalies across all users?

# COMMAND ----------

display(
    df_explained.filter("per_user_anomaly = 1")
        .groupBy("shap_top1_feature")
        .count()
        .orderBy("count", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Explained Anomalies

# COMMAND ----------

display(
    df_explained.filter("per_user_anomaly = 1")
        .join(df_features.select("login_id", "timestamp", "location_name", "device", "_anomaly_type"), on="login_id")
        .select(
            "user_id", "timestamp", "location_name", "device", "_anomaly_type",
            "per_user_score",
            "shap_top1_feature", "shap_top1_value",
            "shap_top2_feature", "shap_top2_value",
            "shap_top3_feature", "shap_top3_value",
        )
        .orderBy("per_user_score")
        .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Save Results

# COMMAND ----------

df_explained.write.format("delta").mode("overwrite") \
    .saveAsTable(f"{DATABASE}.signins_per_user_explained{SUFFIX_TAG}")

print(f"✓ Saved to {DATABASE}.signins_per_user_explained{SUFFIX_TAG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC | Insight | Detail |
# MAGIC |---------|--------|
# MAGIC | Per-user models capture individual baselines | Night-shift workers, frequent travelers, etc. each get their own "normal" |
# MAGIC | `applyInPandas` scales linearly | 100 users = 100 models in parallel. 10,000 users = same pattern, bigger cluster |
# MAGIC | SHAP explanations are actionable | "Flagged because geo_velocity was 15x above this user's norm" vs. "anomaly score: -0.12" |
# MAGIC | Combine with global models | Per-user catches personalized anomalies; global catches cross-user patterns. Use both. |
# MAGIC | Everything runs on classic compute | No serverless, Model Serving, or UC required — works in Azure Gov Cloud today |
# MAGIC
# MAGIC **Next →** Open `06_evaluation_thresholds` to set risk tiers and tune thresholds.

# COMMAND ----------

