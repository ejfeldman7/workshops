# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2: Anomaly Detection for Sign-In Data
# MAGIC ## Notebook 8 — Bonus: System-Level Temporal Anomaly Detection
# MAGIC
# MAGIC Notebooks 03–07 ask **"is this _user_ behaving anomalously?"** — they score
# MAGIC individual login events against per-user or per-population baselines.
# MAGIC This notebook asks a different question:
# MAGIC
# MAGIC > **"Is this _moment in time_ behaving anomalously across the whole system?"**
# MAGIC
# MAGIC Some attack patterns — credential-stuffing campaigns, automated account-takeover
# MAGIC waves, mass brute-force from a botnet — are hard to spot one-at-a-time but
# MAGIC obvious when you look at the **aggregate timeline**: a spike in failed-login
# MAGIC volume, a sudden bloom of distinct IPs, an off-hours surge in attempts.
# MAGIC
# MAGIC ### Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────────┐
# MAGIC │              System-Level Temporal Anomaly Pipeline                 │
# MAGIC │                                                                      │
# MAGIC │  ┌───────────────┐                                                   │
# MAGIC │  │ signins_bronze│  raw login events                                 │
# MAGIC │  └──────┬────────┘                                                   │
# MAGIC │         │                                                            │
# MAGIC │         ▼                                                            │
# MAGIC │  ┌────────────────────┐   per 5-minute bucket:                       │
# MAGIC │  │  Time-Bucket       │     • total_logins                           │
# MAGIC │  │  Aggregation       │     • distinct_users / distinct_ips          │
# MAGIC │  │  (system-level)    │     • avg_failed_attempts_before             │
# MAGIC │  │                    │     • mfa_rate, off_hours_share              │
# MAGIC │  └─────────┬──────────┘                                              │
# MAGIC │            │                                                         │
# MAGIC │            ▼                                                         │
# MAGIC │  ┌────────────────────┐   for each metric:                           │
# MAGIC │  │  STL Decomposition │     • remove daily seasonality               │
# MAGIC │  │  + Rolling Z-Score │     • score residuals against rolling stats  │
# MAGIC │  └─────────┬──────────┘                                              │
# MAGIC │            │                                                         │
# MAGIC │            ▼                                                         │
# MAGIC │  ┌────────────────────┐                                              │
# MAGIC │  │  Composite Score   │  max-Z across metrics → tier                 │
# MAGIC │  │  + Tier (low/.../  │                                              │
# MAGIC │  │  critical)         │                                              │
# MAGIC │  └─────────┬──────────┘                                              │
# MAGIC │            │                                                         │
# MAGIC │            ▼                                                         │
# MAGIC │  ┌────────────────────┐                                              │
# MAGIC │  │ signins_system_    │  Delta table — joinable to bronze for       │
# MAGIC │  │ temporal_scores    │  triage                                      │
# MAGIC │  └────────────────────┘                                              │
# MAGIC └──────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Why this is complementary, not competing:**
# MAGIC
# MAGIC | Question | Notebooks 03–07 (user-level) | This notebook (system-level) |
# MAGIC |---|---|---|
# MAGIC | Unit of analysis | one login event | one 5-minute window |
# MAGIC | Catches | impossible travel, off-hours, brute force per user | volume bursts, coordinated campaigns, system surges |
# MAGIC | Output | scored events | scored time buckets |
# MAGIC | Best for | "investigate this account" | "investigate this hour" |
# MAGIC
# MAGIC The two are joinable: a high-risk bucket from this notebook plus the user-level
# MAGIC scores within it gives a triage analyst the "what happened, when, and to whom"
# MAGIC in one query.
# MAGIC
# MAGIC **Azure Gov Cloud Compatibility:** ✅ Uses statsmodels (already in DBR ML); no
# MAGIC extra installs, no Model Serving, no internet egress.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites

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

from pyspark.sql import functions as F

bronze = spark.read.table(f"{DATABASE}.signins_bronze")
print(f"Loaded {bronze.count()} sign-in events from signins_bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Aggregate to a System-Level Timeline
# MAGIC
# MAGIC We bucket every login into a **5-minute window** and compute system-wide
# MAGIC metrics per bucket. The choice of 5 minutes is a tradeoff:
# MAGIC
# MAGIC - Smaller buckets (1 min) → catch sharp bursts but very noisy baseline
# MAGIC - Larger buckets (15 min) → smooth, stable baseline but coordinated 2-min
# MAGIC   bursts get diluted into a window of mostly-normal traffic
# MAGIC
# MAGIC 5 minutes is wide enough to give a stable count and narrow enough that a
# MAGIC credential-stuffing burst (50 attempts in ~5 min) fully fits inside one bucket.

# COMMAND ----------

bucket_seconds = 5 * 60

agg = (
    bronze
    .withColumn("bucket_ts", (F.unix_timestamp("timestamp") / bucket_seconds).cast("long") * bucket_seconds)
    .withColumn("bucket_ts", F.to_timestamp("bucket_ts"))
    .withColumn("hour", F.hour("timestamp"))
    .withColumn("is_off_hours", ((F.col("hour") < 7) | (F.col("hour") > 19)).cast("int"))
    .withColumn("failed_flag", (F.col("failed_attempts_before") > 0).cast("int"))
    .withColumn("mfa_flag", F.col("mfa_used").cast("int"))
    .groupBy("bucket_ts")
    .agg(
        F.count("*").alias("total_logins"),
        F.countDistinct("user_id").alias("distinct_users"),
        F.countDistinct("ip_address").alias("distinct_ips"),
        F.avg("failed_attempts_before").alias("avg_failed_attempts_before"),
        F.avg("failed_flag").alias("failure_rate"),
        F.avg("mfa_flag").alias("mfa_rate"),
        F.avg("is_off_hours").alias("off_hours_share"),
        # Track success rate for additional signal
        F.avg(F.col("success").cast("int")).alias("success_rate"),
    )
    .orderBy("bucket_ts")
)

system_pdf = agg.toPandas().sort_values("bucket_ts").reset_index(drop=True)
system_pdf["bucket_ts"] = system_pdf["bucket_ts"].astype("datetime64[ns]")
print(f"Aggregated into {len(system_pdf)} 5-minute buckets")
display(system_pdf.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Visualize the Raw System Timeline
# MAGIC
# MAGIC Before any modeling, eyeball the raw signal. Healthy systems show a strong
# MAGIC daily cycle (work-hours peak, overnight valley) with a 7-day weekly modulation
# MAGIC (weekends quieter). Anything that breaks the cycle is a candidate anomaly.

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
metrics = [
    ("total_logins", "Total logins per 5-min bucket"),
    ("distinct_ips", "Distinct IPs per bucket"),
    ("avg_failed_attempts_before", "Avg failed attempts before success"),
    ("failure_rate", "Share of events with any failure"),
]
for ax, (col, title) in zip(axes, metrics):
    ax.plot(system_pdf["bucket_ts"], system_pdf[col], linewidth=0.5)
    ax.set_title(title)
    ax.grid(alpha=0.3)
fig.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: STL Decomposition + Rolling Z-Score
# MAGIC
# MAGIC The raw signal contains three components:
# MAGIC
# MAGIC 1. **Trend** — slow drift (more users joining, holiday week, etc.)
# MAGIC 2. **Seasonality** — daily cycle (work hours), weekly cycle (weekends)
# MAGIC 3. **Residual** — what's left after removing 1 and 2
# MAGIC
# MAGIC Anomalies live in the residual. We use **STL** (Seasonal-Trend decomposition
# MAGIC using LOESS) to peel off the first two, then score residuals using a
# MAGIC **rolling z-score** so the threshold adapts to recent variability rather than
# MAGIC being a fixed global value.
# MAGIC
# MAGIC > **Why not a fixed z?** A volume of 200 logins/bucket might be normal at noon
# MAGIC > and a 10-sigma anomaly at 3 AM. STL + rolling stats handles both contexts
# MAGIC > automatically.

# COMMAND ----------

import numpy as np
from statsmodels.tsa.seasonal import STL

# Daily seasonality: 24h × 12 buckets/hour = 288 buckets/day
SEASONAL_PERIOD = 288
ROLLING_WINDOW = SEASONAL_PERIOD * 3  # 3-day rolling baseline for z-score

# Reindex to a regular 5-min grid so STL has no missing buckets
full_index = pd.date_range(
    system_pdf["bucket_ts"].min(),
    system_pdf["bucket_ts"].max(),
    freq=f"{bucket_seconds}s",
)
system_pdf = (
    system_pdf.set_index("bucket_ts")
    .reindex(full_index)
    .fillna({
        "total_logins": 0,
        "distinct_users": 0,
        "distinct_ips": 0,
        "avg_failed_attempts_before": 0.0,
        "failure_rate": 0.0,
        "mfa_rate": 0.0,
        "off_hours_share": 0.0,
        "success_rate": 1.0,
    })
    .rename_axis("bucket_ts")
    .reset_index()
)
print(f"After reindex: {len(system_pdf)} buckets")

# COMMAND ----------

def stl_zscore(series: pd.Series, period: int = SEASONAL_PERIOD,
               rolling: int = ROLLING_WINDOW) -> pd.Series:
    """Decompose a series with STL, then return rolling-z of the residual.

    NaN at the front of the series (where the rolling window hasn't filled
    yet) is replaced with 0 so those buckets aren't accidentally flagged.
    """
    if series.std() < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    stl = STL(series, period=period, robust=True).fit()
    resid = stl.resid
    rolling_mean = resid.rolling(rolling, min_periods=period).mean()
    rolling_std = resid.rolling(rolling, min_periods=period).std()
    z = (resid - rolling_mean) / rolling_std.replace(0, np.nan)
    return z.fillna(0.0)

scored_metrics = [
    "total_logins",
    "distinct_ips",
    "avg_failed_attempts_before",
    "failure_rate",
]

for m in scored_metrics:
    system_pdf[f"{m}_z"] = stl_zscore(system_pdf[m])

# Composite anomaly score: max absolute z across the metrics. We use max
# rather than mean so a single sharp signal (e.g. a 10-sigma failure_rate
# spike) isn't averaged away by quiet metrics.
z_cols = [f"{m}_z" for m in scored_metrics]
system_pdf["anomaly_score"] = system_pdf[z_cols].abs().max(axis=1)
system_pdf["top_signal"] = system_pdf[z_cols].abs().idxmax(axis=1).str.removesuffix("_z")

display(
    system_pdf[["bucket_ts", "total_logins", "anomaly_score", "top_signal"] + z_cols]
    .sort_values("anomaly_score", ascending=False)
    .head(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Risk Tiers
# MAGIC
# MAGIC Convert the continuous score into actionable tiers. The thresholds below are
# MAGIC the same shape as notebook 06's per-event tiers but tuned to STL-z residuals
# MAGIC rather than isolation-forest scores:
# MAGIC
# MAGIC | Tier | Threshold (max abs z) | Action |
# MAGIC |---|---|---|
# MAGIC | low | < 3 | log only |
# MAGIC | medium | 3–5 | review next-day |
# MAGIC | high | 5–8 | page on-call |
# MAGIC | critical | ≥ 8 | wake someone up |
# MAGIC
# MAGIC In production these are starting points — calibrate against your real
# MAGIC false-positive budget after a soak period.

# COMMAND ----------

def tier(z):
    if z < 3:
        return "low"
    if z < 5:
        return "medium"
    if z < 8:
        return "high"
    return "critical"

system_pdf["tier"] = system_pdf["anomaly_score"].apply(tier)
print("Bucket count by tier:")
print(system_pdf["tier"].value_counts().reindex(["low", "medium", "high", "critical"], fill_value=0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Validate — Did We Catch the Credential-Stuffing Bursts?
# MAGIC
# MAGIC Notebook 00 injects 12 coordinated credential-stuffing bursts (50 attempts
# MAGIC each in 5-minute windows). These are exactly the pattern this notebook is
# MAGIC built to find. Let's check.

# COMMAND ----------

# Pull the buckets that contain credential-stuffing events from bronze
stuffing_buckets = (
    bronze
    .filter(F.col("_anomaly_type") == "credential_stuffing")
    .withColumn("bucket_ts", (F.unix_timestamp("timestamp") / bucket_seconds).cast("long") * bucket_seconds)
    .withColumn("bucket_ts", F.to_timestamp("bucket_ts"))
    .select("bucket_ts")
    .distinct()
    .toPandas()
)
stuffing_buckets["bucket_ts"] = stuffing_buckets["bucket_ts"].astype("datetime64[ns]")

# Join against scored buckets
flagged = system_pdf.merge(stuffing_buckets, on="bucket_ts", how="inner")
print(f"Stuffing bursts spread across {len(flagged)} 5-min buckets")
print(f"  Tier breakdown for those buckets:")
print(flagged["tier"].value_counts().reindex(["low", "medium", "high", "critical"], fill_value=0))
print(f"  Mean anomaly score: {flagged['anomaly_score'].mean():.2f}")
print(f"  Top signal driving the score:")
print(flagged["top_signal"].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ### What user-level models miss that we caught
# MAGIC
# MAGIC Compare side-by-side: how the user-level isolation forest (notebook 03)
# MAGIC scores **individual events** inside a stuffing burst, vs how this notebook
# MAGIC scores the **bucket they live in**. Stuffing events look only mildly
# MAGIC anomalous per-event (similar feature shape to brute force) — but the bucket
# MAGIC containing them is screaming.

# COMMAND ----------

iforest_table = f"{DATABASE}.signins_iforest{SUFFIX_TAG}"
try:
    # signins_iforest has only (login_id, iforest_score, is_anomaly) — join
    # back to bronze for the _anomaly_type label.
    stuffing_iforest = (
        spark.read.table(iforest_table)
        .join(
            bronze.select("login_id", "_anomaly_type"),
            on="login_id",
            how="inner",
        )
        .filter(F.col("_anomaly_type") == "credential_stuffing")
        .toPandas()
    )
except Exception as exc:
    print(f"(notebook 03 output not found at {iforest_table} — skipping comparison; {exc})")
    stuffing_iforest = None

if stuffing_iforest is not None and len(stuffing_iforest):
    print(f"User-level (notebook 03) on the {len(stuffing_iforest)} individual stuffing events:")
    print(f"  mean iforest_score:        {stuffing_iforest['iforest_score'].mean():.4f}  "
          f"(more negative = more anomalous)")
    print(f"  share flagged as anomaly:  {stuffing_iforest['is_anomaly'].mean():.1%}")
    print()
    print(f"System-level (this notebook) on the {len(flagged)} buckets containing them:")
    print(f"  mean max-|z| score:        {flagged['anomaly_score'].mean():.2f}")
    print(f"  share in high/critical:    {flagged['tier'].isin(['high','critical']).mean():.1%}")
    print()
    print("Takeaway: stuffing events look only modestly anomalous one-at-a-time —")
    print("notebook 03's per-event score barely separates them from normal — but")
    print("the buckets they cluster into are unambiguously flagged here.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Visualize the Top-Anomalous Bucket
# MAGIC
# MAGIC Pick the single highest-score bucket and plot a 24-hour window around it,
# MAGIC with the anomaly threshold overlaid. This is the kind of view you'd put on
# MAGIC an SOC dashboard.

# COMMAND ----------

top = system_pdf.sort_values("anomaly_score", ascending=False).iloc[0]
center = top["bucket_ts"]
window = pd.Timedelta(hours=12)
slice_ = system_pdf[
    (system_pdf["bucket_ts"] >= center - window)
    & (system_pdf["bucket_ts"] <= center + window)
]

fig, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(slice_["bucket_ts"], slice_["total_logins"], color="steelblue",
         linewidth=0.7, label="total_logins")
ax1.set_ylabel("total logins / 5-min bucket", color="steelblue")
ax2 = ax1.twinx()
ax2.plot(slice_["bucket_ts"], slice_["anomaly_score"], color="crimson",
         linewidth=1.0, label="max |z|")
ax2.axhline(8, color="crimson", linestyle="--", alpha=0.4, label="critical threshold")
ax2.set_ylabel("anomaly score (max |z|)", color="crimson")
ax1.axvline(center, color="black", linestyle=":", alpha=0.5)
ax1.set_title(f"Top anomalous bucket: {center}  (top signal: {top['top_signal']})")
fig.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Persist Scores
# MAGIC
# MAGIC Write the scored timeline to a Delta table so it can be joined against
# MAGIC `signins_bronze` for triage queries like _"show me every login in the
# MAGIC critical buckets last week"_.

# COMMAND ----------

out_table = f"{DATABASE}.signins_system_temporal_scores{SUFFIX_TAG}"
out = system_pdf[
    ["bucket_ts", "total_logins", "distinct_users", "distinct_ips",
     "avg_failed_attempts_before", "failure_rate", "mfa_rate",
     "off_hours_share", "success_rate",
     "total_logins_z", "distinct_ips_z",
     "avg_failed_attempts_before_z", "failure_rate_z",
     "anomaly_score", "top_signal", "tier"]
]
spark.createDataFrame(out).write.format("delta").mode("overwrite").saveAsTable(out_table)
print(f"✓ Wrote {len(out)} bucket scores to {out_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example triage query

# COMMAND ----------

display(spark.sql(f"""
SELECT
  s.bucket_ts,
  s.tier,
  s.anomaly_score,
  s.top_signal,
  COUNT(*) AS n_events,
  COUNT(DISTINCT b.user_id) AS n_users,
  COUNT(DISTINCT b.ip_address) AS n_ips,
  ROUND(AVG(b.failed_attempts_before), 1) AS avg_failed,
  COLLECT_SET(b._anomaly_type) AS anomaly_types_in_bucket
FROM {out_table} s
JOIN {DATABASE}.signins_bronze b
  ON b.timestamp >= s.bucket_ts
 AND b.timestamp <  s.bucket_ts + INTERVAL 5 MINUTES
WHERE s.tier IN ('high', 'critical')
GROUP BY s.bucket_ts, s.tier, s.anomaly_score, s.top_signal
ORDER BY s.anomaly_score DESC
LIMIT 20
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap-up
# MAGIC
# MAGIC You now have two complementary anomaly views over the same data:
# MAGIC
# MAGIC - **`signins_iforest` / `signins_per_user_iforest` / `signins_pyod`** — per-event scores
# MAGIC   answering _"is this login weird for this user?"_
# MAGIC - **`signins_system_temporal_scores`** — per-bucket scores answering
# MAGIC   _"is this 5-minute window weird across the whole system?"_
# MAGIC
# MAGIC In production you'd typically alert on the **union** of the two — high-risk
# MAGIC events in any tier, plus everything inside a high/critical bucket — and let
# MAGIC analysts triage from there.
# MAGIC
# MAGIC ### Where to take this further
# MAGIC
# MAGIC - **Sub-bucket bursts**: try [STUMPY](https://stumpy.readthedocs.io/) matrix
# MAGIC   profile to find sub-sequence anomalies that don't align to bucket boundaries.
# MAGIC - **Multi-seasonality**: if your real data has both daily and weekly cycles,
# MAGIC   `MSTL` (multi-seasonal STL) handles them jointly.
# MAGIC - **Per-segment baselines**: aggregate by `(geo_region, app)` instead of
# MAGIC   globally, so a quiet region's anomaly isn't drowned by busy regions.
# MAGIC - **Streaming**: this notebook is batch; the same logic ports to
# MAGIC   structured streaming with stateful aggregation by 5-min watermark.
