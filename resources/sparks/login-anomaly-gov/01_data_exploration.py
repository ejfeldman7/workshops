# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2: Anomaly Detection for Sign-In Data
# MAGIC ## Notebook 1 — Data Exploration
# MAGIC
# MAGIC Explore the sign-in log dataset to understand normal patterns and identify potential anomaly signals.
# MAGIC
# MAGIC **What you'll learn:**
# MAGIC - How to profile sign-in data at scale
# MAGIC - Identifying baseline user behavior patterns
# MAGIC - Spotting anomaly indicators in raw data
# MAGIC
# MAGIC **Docs:**
# MAGIC - [Delta Lake on Databricks](https://docs.databricks.com/en/delta/index.html)
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

TABLE = f"{DATABASE}.signins_bronze"
print(f"Reading from: {TABLE}")
print(f"Database: {DATABASE}, Table suffix: {SUFFIX_TAG or '(none)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load and Profile

# COMMAND ----------

df = spark.read.table(TABLE)
print(f"Total sign-in events: {df.count()}")
print(f"Columns: {df.columns}")
print(f"Unique users: {df.select('user_id').distinct().count()}")

# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Temporal Patterns
# MAGIC
# MAGIC Understanding **when** logins happen is critical for anomaly detection.
# MAGIC Most organizations have clear business-hour patterns; deviations are suspicious.
# MAGIC
# MAGIC ```
# MAGIC Normal Pattern:                Anomaly Signal:
# MAGIC   ████████░░░░░░░░████████       ░░░░░░░░████░░░░░░░░░░░░░░
# MAGIC   6am        noon       6pm      2am  3am  4am
# MAGIC ```

# COMMAND ----------

from pyspark.sql import functions as F

df_time = df.withColumn("hour", F.hour("timestamp")) \
            .withColumn("day_of_week", F.dayofweek("timestamp")) \
            .withColumn("date", F.to_date("timestamp"))

# Hourly distribution
display(
    df_time.groupBy("hour")
           .agg(F.count("*").alias("login_count"))
           .orderBy("hour")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Day of Week Pattern

# COMMAND ----------

display(
    df_time.groupBy("day_of_week")
           .agg(F.count("*").alias("login_count"))
           .orderBy("day_of_week")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Heatmap: Hour × Day of Week

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

heatmap_df = df_time.groupBy("day_of_week", "hour") \
    .count() \
    .toPandas() \
    .pivot(index="day_of_week", columns="hour", values="count") \
    .fillna(0)

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(heatmap_df.values, cmap="YlOrRd", aspect="auto")
ax.set_xlabel("Hour of Day", fontsize=12)
ax.set_ylabel("Day of Week", fontsize=12)
ax.set_yticks(range(7))
ax.set_yticklabels(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
ax.set_xticks(range(24))
ax.set_title("Login Volume: Hour × Day of Week", fontsize=14)
plt.colorbar(im, ax=ax, label="Login Count")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Geographic Patterns
# MAGIC
# MAGIC Where do users typically log in from? Remote logins from unexpected countries are a strong anomaly signal.

# COMMAND ----------

display(
    df.groupBy("location_name")
      .agg(
          F.count("*").alias("login_count"),
          F.countDistinct("user_id").alias("unique_users"),
          F.avg("session_duration_min").alias("avg_session_min"),
      )
      .orderBy("login_count", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Geographic Scatter Plot

# COMMAND ----------

loc_df = df.select("latitude", "longitude", "location_name").toPandas()

fig, ax = plt.subplots(figsize=(12, 6))
# Normal locations in blue, anomalous in red
known_locs = ["HQ_Sparks_NV", "Office_Denver_CO", "Office_Huntsville", "Remote_LasVegas", "Remote_Phoenix"]
mask_normal = loc_df["location_name"].isin(known_locs)

ax.scatter(loc_df.loc[mask_normal, "longitude"], loc_df.loc[mask_normal, "latitude"],
           c="steelblue", alpha=0.3, s=5, label="Known Locations")
ax.scatter(loc_df.loc[~mask_normal, "longitude"], loc_df.loc[~mask_normal, "latitude"],
           c="red", alpha=0.8, s=20, label="Unusual Locations", marker="x")
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
ax.set_title("Login Locations", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Authentication Patterns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Failed Attempt Distribution
# MAGIC
# MAGIC Brute force attacks typically show a burst of failed attempts before a successful login.

# COMMAND ----------

display(
    df.groupBy("failed_attempts_before")
      .agg(F.count("*").alias("count"))
      .orderBy("failed_attempts_before")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MFA Usage

# COMMAND ----------

display(
    df.groupBy("mfa_used")
      .agg(F.count("*").alias("count"), F.countDistinct("user_id").alias("unique_users"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Device Patterns

# COMMAND ----------

display(
    df.groupBy("device")
      .agg(
          F.count("*").alias("login_count"),
          F.countDistinct("user_id").alias("unique_users"),
          F.avg("failed_attempts_before").alias("avg_failed_attempts"),
      )
      .orderBy("login_count", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Per-User Behavior Profiles
# MAGIC
# MAGIC Understanding each user's baseline behavior is key to detecting **their** anomalies.
# MAGIC A 2 AM login might be normal for a night-shift worker but anomalous for a 9-5 employee.

# COMMAND ----------

user_profiles = df_time.groupBy("user_id").agg(
    F.count("*").alias("total_logins"),
    F.avg("hour").alias("avg_login_hour"),
    F.stddev("hour").alias("std_login_hour"),
    F.countDistinct("location_name").alias("unique_locations"),
    F.countDistinct("device").alias("unique_devices"),
    F.countDistinct("ip_address").alias("unique_ips"),
    F.avg("session_duration_min").alias("avg_session_min"),
    F.avg("failed_attempts_before").alias("avg_failed_attempts"),
    F.avg(F.when(F.col("mfa_used") == True, 1).otherwise(0)).alias("mfa_rate"),
)

display(user_profiles.orderBy("total_logins", ascending=False).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Users with Most Location Diversity
# MAGIC
# MAGIC Users logging in from many different locations may be traveling or may have compromised credentials.

# COMMAND ----------

display(
    user_profiles.orderBy("unique_locations", ascending=False).limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **Key observations for feature engineering (next notebook):**
# MAGIC
# MAGIC | Signal | Normal Range | Anomaly Indicator |
# MAGIC |--------|-------------|-------------------|
# MAGIC | Login hour | 7 AM – 7 PM | 1 AM – 5 AM |
# MAGIC | Day of week | Mon – Fri | Weekend |
# MAGIC | Location | Known offices / remote cities | Foreign countries, unknown VPN |
# MAGIC | Failed attempts | 0-1 | 5+ |
# MAGIC | MFA | Used ~85% of time | Skipped |
# MAGIC | Device | Corporate devices | Tor, rooted, VM instances |
# MAGIC | Session duration | 25-65 min | Very short (<5 min) |
# MAGIC
# MAGIC **Next →** Open `02_feature_engineering` to build anomaly-detection features.

# COMMAND ----------

