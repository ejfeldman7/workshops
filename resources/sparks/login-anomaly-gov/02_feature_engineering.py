# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2: Anomaly Detection for Sign-In Data
# MAGIC ## Notebook 2 — Feature Engineering
# MAGIC
# MAGIC Transform raw sign-in logs into features that anomaly detection models can use.
# MAGIC Good features are the difference between catching real threats and drowning in false positives.
# MAGIC
# MAGIC ### Feature Categories
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────────┐
# MAGIC │                    Feature Engineering Pipeline                      │
# MAGIC │                                                                      │
# MAGIC │  Raw Login Event                                                     │
# MAGIC │       │                                                              │
# MAGIC │       ├──▶ Temporal Features                                         │
# MAGIC │       │      • Hour of day, day of week, is_weekend                  │
# MAGIC │       │      • Minutes since last login                              │
# MAGIC │       │      • Is off-hours for this user                            │
# MAGIC │       │                                                              │
# MAGIC │       ├──▶ Geographic Features                                       │
# MAGIC │       │      • Distance from user's home location                    │
# MAGIC │       │      • Geo-velocity (km/hr since last login)                 │
# MAGIC │       │      • Is known vs unknown location                          │
# MAGIC │       │                                                              │
# MAGIC │       ├──▶ Authentication Features                                   │
# MAGIC │       │      • Failed attempts before success                        │
# MAGIC │       │      • MFA used (yes/no)                                     │
# MAGIC │       │      • Session duration vs user average                      │
# MAGIC │       │                                                              │
# MAGIC │       └──▶ Behavioral Features                                       │
# MAGIC │              • New device flag                                       │
# MAGIC │              • New IP flag                                           │
# MAGIC │              • Login frequency vs baseline                           │
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
spark.sql(f"USE {DATABASE}")
print(f"Database: {DATABASE}, Table suffix: {SUFFIX_TAG or '(none)'}")

# COMMAND ----------

from pyspark.sql import functions as F, Window
import math

df = spark.read.table(f"{DATABASE}.signins_bronze")
print(f"Loaded {df.count()} sign-in events")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Temporal Features

# COMMAND ----------

df_feat = df.withColumn("hour", F.hour("timestamp")) \
            .withColumn("day_of_week", F.dayofweek("timestamp")) \
            .withColumn("is_weekend", F.when(F.dayofweek("timestamp").isin(1, 7), 1).otherwise(0)) \
            .withColumn("is_off_hours", F.when(
                (F.hour("timestamp") < 7) | (F.hour("timestamp") > 19), 1
            ).otherwise(0)) \
            .withColumn("epoch", F.unix_timestamp("timestamp"))

# Time since last login per user
user_window = Window.partitionBy("user_id").orderBy("timestamp")

df_feat = df_feat.withColumn(
    "prev_timestamp", F.lag("epoch").over(user_window)
).withColumn(
    "minutes_since_last_login",
    F.when(F.col("prev_timestamp").isNotNull(),
           (F.col("epoch") - F.col("prev_timestamp")) / 60.0)
    .otherwise(None)
)

# Hour encoded as cyclical features (so 23:00 is close to 01:00)
df_feat = df_feat.withColumn("hour_sin", F.sin(2 * math.pi * F.col("hour") / 24)) \
                 .withColumn("hour_cos", F.cos(2 * math.pi * F.col("hour") / 24))

print("✓ Temporal features added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Geographic Features
# MAGIC
# MAGIC **Geo-velocity** is one of the strongest anomaly signals: if a user logs in from New York,
# MAGIC and 30 minutes later logs in from Moscow, no airplane can cover that distance.
# MAGIC
# MAGIC ```
# MAGIC                     Geo-velocity = distance / time
# MAGIC
# MAGIC   New York ────────────── Moscow
# MAGIC   Login: 10:00 AM        Login: 10:30 AM
# MAGIC   Distance: 7,510 km     Time: 30 min
# MAGIC   Velocity: 15,020 km/hr ← IMPOSSIBLE (plane max ~900 km/hr)
# MAGIC ```

# COMMAND ----------

# Previous login location for geo-velocity
df_feat = df_feat.withColumn("prev_lat", F.lag("latitude").over(user_window)) \
                 .withColumn("prev_lon", F.lag("longitude").over(user_window))

# Haversine distance UDF
from pyspark.sql.types import DoubleType

@F.udf(DoubleType())
def haversine(lat1, lon1, lat2, lon2):
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return None
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(min(1.0, a)))

df_feat = df_feat.withColumn(
    "distance_from_prev_km",
    haversine("prev_lat", "prev_lon", "latitude", "longitude")
)

# Geo-velocity (km/hr)
df_feat = df_feat.withColumn(
    "geo_velocity_kmh",
    F.when(
        (F.col("minutes_since_last_login").isNotNull()) & (F.col("minutes_since_last_login") > 0),
        F.col("distance_from_prev_km") / (F.col("minutes_since_last_login") / 60.0)
    ).otherwise(None)
)

# Known vs unknown location
known_locations = list(spark.read.table(f"{DATABASE}.signins_bronze")
                       .filter("_anomaly_type = 'normal'")
                       .select("location_name").distinct()
                       .toPandas()["location_name"])

df_feat = df_feat.withColumn(
    "is_unknown_location",
    F.when(F.col("location_name").isin(known_locations), 0).otherwise(1)
)

print("✓ Geographic features added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Authentication Features

# COMMAND ----------

# User baseline stats (computed over "training" period - first 60 days)
training_cutoff = "2025-02-28"

user_baselines = df_feat.filter(F.col("timestamp") < training_cutoff).groupBy("user_id").agg(
    F.avg("session_duration_min").alias("baseline_session_avg"),
    F.stddev("session_duration_min").alias("baseline_session_std"),
    F.avg("failed_attempts_before").alias("baseline_failed_avg"),
    F.avg("hour").alias("baseline_hour_avg"),
    F.stddev("hour").alias("baseline_hour_std"),
    F.count("*").alias("baseline_login_count"),
    F.countDistinct("device").alias("baseline_device_count"),
    F.countDistinct("ip_address").alias("baseline_ip_count"),
)

df_feat = df_feat.join(user_baselines, on="user_id", how="left")

# Session duration z-score (how unusual is this session length for this user?)
df_feat = df_feat.withColumn(
    "session_duration_zscore",
    F.when(
        F.col("baseline_session_std") > 0,
        (F.col("session_duration_min") - F.col("baseline_session_avg")) / F.col("baseline_session_std")
    ).otherwise(0)
)

# MFA as numeric
df_feat = df_feat.withColumn("mfa_numeric", F.when(F.col("mfa_used") == True, 1).otherwise(0))

print("✓ Authentication features added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Behavioral Features

# COMMAND ----------

# Collect known devices per user (from training period)
from pyspark.sql.types import ArrayType, StringType

user_devices = df_feat.filter(F.col("timestamp") < training_cutoff) \
    .groupBy("user_id") \
    .agg(F.collect_set("device").alias("known_devices"))

df_feat = df_feat.join(user_devices, on="user_id", how="left")

df_feat = df_feat.withColumn(
    "is_new_device",
    F.when(F.array_contains("known_devices", F.col("device")), 0).otherwise(1)
)

# Logins in the last hour (burst detection)
hour_window = Window.partitionBy("user_id").orderBy("epoch").rangeBetween(-3600, 0)
df_feat = df_feat.withColumn("logins_last_hour", F.count("*").over(hour_window))

# Clean up intermediate columns
feature_cols = [
    "login_id", "user_id", "timestamp",
    # Temporal
    "hour", "day_of_week", "is_weekend", "is_off_hours", "hour_sin", "hour_cos",
    "minutes_since_last_login",
    # Geographic
    "distance_from_prev_km", "geo_velocity_kmh", "is_unknown_location",
    "latitude", "longitude",
    # Authentication
    "failed_attempts_before", "mfa_numeric", "session_duration_min",
    "session_duration_zscore",
    # Behavioral
    "is_new_device", "logins_last_hour",
    # Metadata
    "location_name", "device", "ip_address", "_anomaly_type",
]

df_final = df_feat.select(*feature_cols)

print(f"✓ Final feature set: {len(feature_cols)} columns")
display(df_final.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Feature Distributions

# COMMAND ----------

# Key feature statistics
display(
    df_final.select(
        "hour", "is_weekend", "is_off_hours", "minutes_since_last_login",
        "distance_from_prev_km", "geo_velocity_kmh", "is_unknown_location",
        "failed_attempts_before", "mfa_numeric", "session_duration_min",
        "session_duration_zscore", "is_new_device", "logins_last_hour"
    ).summary()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Geo-Velocity Distribution
# MAGIC
# MAGIC Most logins should have low geo-velocity (same location or nearby). Extreme values = impossible travel.

# COMMAND ----------

import matplotlib.pyplot as plt

gv = df_final.filter("geo_velocity_kmh IS NOT NULL").select("geo_velocity_kmh").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full distribution (log scale)
ax = axes[0]
ax.hist(gv["geo_velocity_kmh"].clip(upper=20000), bins=100, color="steelblue", alpha=0.7)
ax.set_xlabel("Geo-velocity (km/hr)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Geo-velocity Distribution", fontsize=14)
ax.axvline(x=900, color="red", linestyle="--", label="Max airplane speed (~900 km/hr)")
ax.legend()
ax.grid(True, alpha=0.3)

# Zoom on tail
ax = axes[1]
tail = gv[gv["geo_velocity_kmh"] > 500]
ax.hist(tail["geo_velocity_kmh"], bins=50, color="red", alpha=0.7)
ax.set_xlabel("Geo-velocity (km/hr)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Geo-velocity > 500 km/hr (Impossible Travel)", fontsize=14)
ax.axvline(x=900, color="darkred", linestyle="--", label="Max airplane speed")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Save Feature Table

# COMMAND ----------

df_final.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE}.signins_features{SUFFIX_TAG}")
print(f"✓ Saved feature table: {DATABASE}.signins_features{SUFFIX_TAG}")
print(f"  Rows: {df_final.count()}")

# Also save as parquet for pandas-based modeling
import os
import re
_user = spark.sql("SELECT current_user()").first()[0]
USER_ID = re.sub(r'[^a-zA-Z0-9]', '_', _user.split('@')[0])
artifact_path = f"/dbfs/tmp/workshops/{DATABASE}/{USER_ID}"
os.makedirs(artifact_path, exist_ok=True)

pdf = df_final.toPandas()
spark.createDataFrame(pdf).write.mode("overwrite").parquet(f"{artifact_path}/signins_features.parquet")
print(f"✓ Saved pandas DataFrame to {artifact_path}/signins_features.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Summary
# MAGIC
# MAGIC | Feature | Type | Anomaly Signal |
# MAGIC |---------|------|---------------|
# MAGIC | `hour_sin`, `hour_cos` | Temporal | Off-hours activity |
# MAGIC | `is_weekend`, `is_off_hours` | Temporal | Unusual work patterns |
# MAGIC | `minutes_since_last_login` | Temporal | Rapid re-authentication |
# MAGIC | `geo_velocity_kmh` | Geographic | **Impossible travel** (strongest signal) |
# MAGIC | `distance_from_prev_km` | Geographic | Location jumps |
# MAGIC | `is_unknown_location` | Geographic | Foreign/unexpected locations |
# MAGIC | `failed_attempts_before` | Auth | Brute force attacks |
# MAGIC | `mfa_numeric` | Auth | MFA bypass |
# MAGIC | `session_duration_zscore` | Auth | Abnormal session length |
# MAGIC | `is_new_device` | Behavioral | Compromised credentials used on new device |
# MAGIC | `logins_last_hour` | Behavioral | Credential stuffing / burst activity |
# MAGIC
# MAGIC **Next →** Open `03_isolation_forest` to train the primary anomaly detection model.