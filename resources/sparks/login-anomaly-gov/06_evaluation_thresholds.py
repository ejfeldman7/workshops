# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2: Anomaly Detection for Sign-In Data
# MAGIC ## Notebook 5 — Evaluation & Threshold Setting
# MAGIC
# MAGIC Set meaningful risk tiers, analyze false positive/negative tradeoffs,
# MAGIC and design the feedback loop for continuous improvement.
# MAGIC
# MAGIC ### Risk Tier Framework
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────────┐
# MAGIC │                     Risk Tier Assignment                             │
# MAGIC │                                                                      │
# MAGIC │  Ensemble Score    Risk Tier     Action                              │
# MAGIC │  ─────────────    ──────────    ────────────────────────────         │
# MAGIC │  0.0 ─── 0.3      Low           Log only, no alert                   │
# MAGIC │  0.3 ─── 0.6      Medium        Flag for weekly review               │
# MAGIC │  0.6 ─── 0.8      High          Alert SOC, investigate < 24hr        │
# MAGIC │  0.8 ─── 1.0      Critical      Page on-call, investigate < 1hr      │
# MAGIC │                                                                      │
# MAGIC └──────────────────────────────────────────────────────────────────────┘
# MAGIC ```
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
pdf = spark.read.parquet(f"{artifact_path}/signins_with_ensemble.parquet").toPandas()
print(f"Loaded {len(pdf)} events")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Define Risk Tiers
# MAGIC
# MAGIC Thresholds should be set based on operational capacity — how many alerts can your SOC handle per day?

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler

# Normalize ensemble score to [0, 1]
scaler = MinMaxScaler()
pdf["risk_score"] = scaler.fit_transform(pdf[["ensemble_with_iforest"]]).ravel()

# Define tiers
def assign_tier(score):
    if score >= 0.8:
        return "Critical"
    elif score >= 0.6:
        return "High"
    elif score >= 0.3:
        return "Medium"
    else:
        return "Low"

pdf["risk_tier"] = pdf["risk_score"].apply(assign_tier)

tier_counts = pdf["risk_tier"].value_counts()
print("Risk tier distribution:\n")
for tier in ["Critical", "High", "Medium", "Low"]:
    count = tier_counts.get(tier, 0)
    pct = count / len(pdf) * 100
    bar = "█" * int(pct)
    print(f"  {tier:10s}: {count:6d} ({pct:5.1f}%) {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Detection Quality by Tier
# MAGIC
# MAGIC For each risk tier, what types of events end up there?

# COMMAND ----------

ct = pd.crosstab(pdf["risk_tier"], pdf["_anomaly_type"], normalize="index") * 100

# Reorder
tier_order = ["Critical", "High", "Medium", "Low"]
ct = ct.reindex(tier_order)

print("Anomaly type composition per risk tier (%):\n")
print(ct.round(1).to_string())

# COMMAND ----------

import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 5))
ct.plot(kind="barh", stacked=True, ax=ax, colormap="tab10")
ax.set_xlabel("Percentage", fontsize=12)
ax.set_title("Anomaly Type Composition by Risk Tier", fontsize=14)
ax.legend(title="Anomaly Type", bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Threshold Sensitivity Analysis
# MAGIC
# MAGIC How do detection metrics change as we adjust the threshold?
# MAGIC This helps calibrate the tradeoff between catching threats (recall) and analyst fatigue (precision).

# COMMAND ----------

# Binary "true anomaly" based on synthetic labels
pdf["true_anomaly"] = (pdf["_anomaly_type"] != "normal").astype(int)

thresholds = np.arange(0.0, 1.01, 0.05)
metrics = []

for t in thresholds:
    predicted = (pdf["risk_score"] >= t).astype(int)
    tp = ((predicted == 1) & (pdf["true_anomaly"] == 1)).sum()
    fp = ((predicted == 1) & (pdf["true_anomaly"] == 0)).sum()
    fn = ((predicted == 0) & (pdf["true_anomaly"] == 1)).sum()
    tn = ((predicted == 0) & (pdf["true_anomaly"] == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    alerts_per_day = (tp + fp) / 90  # 90 days of data

    metrics.append({
        "threshold": t,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alerts_per_day": alerts_per_day,
        "true_positives": tp,
        "false_positives": fp,
    })

mdf = pd.DataFrame(metrics)

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Precision-Recall
ax = axes[0]
ax.plot(mdf["threshold"], mdf["precision"], "b-", linewidth=2, label="Precision")
ax.plot(mdf["threshold"], mdf["recall"], "r-", linewidth=2, label="Recall")
ax.plot(mdf["threshold"], mdf["f1"], "g--", linewidth=2, label="F1")
ax.set_xlabel("Threshold", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Precision / Recall / F1 vs Threshold", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Alerts per day
ax = axes[1]
ax.plot(mdf["threshold"], mdf["alerts_per_day"], "o-", color="steelblue", linewidth=2)
ax.set_xlabel("Threshold", fontsize=12)
ax.set_ylabel("Alerts per Day", fontsize=12)
ax.set_title("Alert Volume vs Threshold", fontsize=14)
ax.axhline(y=50, color="red", linestyle="--", label="SOC capacity (~50/day)")
ax.legend()
ax.grid(True, alpha=0.3)

# TP vs FP
ax = axes[2]
ax.plot(mdf["false_positives"], mdf["true_positives"], "o-", color="steelblue", linewidth=2)
ax.set_xlabel("False Positives", fontsize=12)
ax.set_ylabel("True Positives", fontsize=12)
ax.set_title("True Positives vs False Positives", fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Recommended Threshold
# MAGIC
# MAGIC Choose the threshold that best balances your operational constraints:

# COMMAND ----------

# Find threshold with best F1
best_row = mdf.loc[mdf["f1"].idxmax()]
print(f"Optimal threshold (max F1): {best_row['threshold']:.2f}")
print(f"  Precision: {best_row['precision']:.3f}")
print(f"  Recall:    {best_row['recall']:.3f}")
print(f"  F1:        {best_row['f1']:.3f}")
print(f"  Alerts/day: {best_row['alerts_per_day']:.0f}")
print()

# Threshold for ~50 alerts/day (SOC capacity example)
capacity_row = mdf[mdf["alerts_per_day"] <= 50].iloc[0] if len(mdf[mdf["alerts_per_day"] <= 50]) > 0 else mdf.iloc[-1]
print(f"SOC-capacity threshold (~50 alerts/day): {capacity_row['threshold']:.2f}")
print(f"  Precision: {capacity_row['precision']:.3f}")
print(f"  Recall:    {capacity_row['recall']:.3f}")
print(f"  F1:        {capacity_row['f1']:.3f}")
print(f"  Alerts/day: {capacity_row['alerts_per_day']:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Analyst Feedback Loop Design
# MAGIC
# MAGIC The model improves over time when analysts provide feedback on flagged events.
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────┐
# MAGIC │                   Feedback Loop                                  │
# MAGIC │                                                                  │
# MAGIC │  ┌───────────┐    ┌──────────────┐    ┌──────────────┐           │
# MAGIC │  │  Model     │───▶│  Flagged     │───▶│  Analyst     │          │
# MAGIC │  │  Scores    │    │  Events      │    │  Reviews     │          │
# MAGIC │  └───────────┘    └──────────────┘    └──────┬───────┘           │
# MAGIC │                                              │                   │
# MAGIC │                                     ┌────────▼────────┐          │
# MAGIC │                                     │ True Positive?  │          │
# MAGIC │                                     │ False Positive? │          │
# MAGIC │                                     └────────┬────────┘          │
# MAGIC │                                              │                   │
# MAGIC │  ┌───────────┐    ┌──────────────┐    ┌──────▼───────┐           │
# MAGIC │  │  Retrain   │◀──│  Labeled     │◀──│  Feedback    │            │
# MAGIC │  │  Model     │    │  Dataset     │    │  Table       │          │
# MAGIC │  └───────────┘    └──────────────┘    └──────────────┘           │
# MAGIC └──────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# Create feedback table structure
from pyspark.sql.types import *

feedback_schema = StructType([
    StructField("login_id", StringType(), False),
    StructField("analyst_id", StringType(), False),
    StructField("review_timestamp", TimestampType(), False),
    StructField("is_true_positive", BooleanType(), False),
    StructField("threat_category", StringType(), True),
    StructField("notes", StringType(), True),
])

# Create empty feedback table
spark.createDataFrame([], feedback_schema) \
    .write.format("delta").mode("overwrite") \
    .saveAsTable(f"{DATABASE}.analyst_feedback{SUFFIX_TAG}")

print(f"✓ Created feedback table: {DATABASE}.analyst_feedback{SUFFIX_TAG}")
print("  Analysts record their review of flagged events here.")
print("  Over time, this becomes training data for semi-supervised models.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Save Final Risk Scores

# COMMAND ----------

risk_cols = [
    "login_id", "user_id", "timestamp", "location_name", "device",
    "risk_score", "risk_tier", "iforest_score", "ensemble_score",
    "n_detectors_flagged", "geo_velocity_kmh", "failed_attempts_before",
    "is_off_hours", "is_new_device", "is_unknown_location",
    "_anomaly_type",
]

df_risk = spark.createDataFrame(pdf[risk_cols])
df_risk.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE}.signins_risk_scores{SUFFIX_TAG}")

spark.createDataFrame(pdf).write.mode("overwrite").parquet(f"{artifact_path}/signins_with_risk_tiers.parquet")

print(f"✓ Saved risk scores to {DATABASE}.signins_risk_scores{SUFFIX_TAG}")
display(spark.sql(f"""
    SELECT risk_tier, COUNT(*) as count,
           ROUND(AVG(risk_score), 3) as avg_score,
           ROUND(AVG(n_detectors_flagged), 1) as avg_detectors
    FROM {DATABASE}.signins_risk_scores{SUFFIX_TAG}
    GROUP BY risk_tier
    ORDER BY avg_score DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC | Decision | Recommendation |
# MAGIC |----------|---------------|
# MAGIC | Starting threshold | Use max-F1 threshold, then adjust based on SOC capacity |
# MAGIC | Risk tiers | 4 tiers with escalating response times |
# MAGIC | Feedback loop | Critical for reducing false positives over time |
# MAGIC | Threshold tuning | Re-evaluate monthly as analyst feedback accumulates |
# MAGIC
# MAGIC **Next →** Open `07_batch_scoring_pipeline` to build the production batch scoring pipeline.