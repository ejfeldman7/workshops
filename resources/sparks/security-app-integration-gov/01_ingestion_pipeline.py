# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 3: Security Data Pipeline & Application Integration
# MAGIC ## Notebook 1 — Ingestion Pipeline with Health Monitoring
# MAGIC
# MAGIC **Azure Gov Cloud Compatibility:** ✅ Everything in this notebook runs on classic compute with Databricks Runtime 13.3+.
# MAGIC
# MAGIC Build a medallion pipeline (Bronze → Silver) for security events with built-in health metrics
# MAGIC at every stage.
# MAGIC
# MAGIC ### Pipeline Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────────────┐
# MAGIC │                    Ingestion Pipeline                                    │
# MAGIC │                                                                          │
# MAGIC │  ┌───────────┐      ┌─────────────────────┐     ┌───────────────────┐    │
# MAGIC │  │  Landing   │─────▶│   email_events_     │────▶│  email_events_    │   │
# MAGIC │  │  Zone      │      │   bronze             │     │  silver           │  │
# MAGIC │  │  (JSON)    │      │   (raw, all fields)  │     │  (cleaned,        │  │
# MAGIC │  │            │      │                      │     │   validated)      │  │
# MAGIC │  │            │      ├─────────────────────┤     ├───────────────────┤   │
# MAGIC │  │            │─────▶│   signin_events_    │────▶│  signin_events_   │   │
# MAGIC │  │            │      │   bronze             │     │  silver           │  │
# MAGIC │  └───────────┘      └──────────┬──────────┘     └──────────┬────────┘    │
# MAGIC │                                │                            │            │
# MAGIC │                                └──────────┬─────────────────┘            │
# MAGIC │                                           ▼                              │
# MAGIC │                                  ┌────────────────┐                      │
# MAGIC │                                  │ pipeline_health│                      │
# MAGIC │                                  │ (metrics at    │                      │
# MAGIC │                                  │  every stage)  │                      │
# MAGIC │                                  └────────────────┘                      │
# MAGIC │                                                                          │
# MAGIC │  All tables live in <your_database> (Hive metastore)                     │
# MAGIC │  Docs: https://docs.databricks.com/en/ingestion/auto-loader/index.html   │
# MAGIC └──────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites

# COMMAND ----------

dbutils.widgets.text("database", "security_app_integration", "Database")
from datetime import datetime
_user_email = spark.sql("SELECT current_user()").first()[0]
_name_parts = _user_email.split('@')[0].replace('_', '.').split('.')
_initials = (_name_parts[0][0] + _name_parts[-1][0]).lower() if len(_name_parts) >= 2 else _user_email[:2].lower()
_default_suffix = _initials + datetime.now().strftime('%d%m%y')
dbutils.widgets.text("table_suffix", _default_suffix, "Table Suffix (your initials)")

DATABASE = dbutils.widgets.get("database")
SUFFIX = dbutils.widgets.get("table_suffix").strip()
SUFFIX_TAG = f"_{SUFFIX}" if SUFFIX else ""
ARTIFACT_PATH = f"/dbfs/tmp/workshops/{DATABASE}"
LANDING_PATH = f"/dbfs/tmp/workshops/{DATABASE}/landing_zone"

import os
import re
_user = spark.sql("SELECT current_user()").first()[0]
USER_ID = re.sub(r'[^a-zA-Z0-9]', '_', _user.split('@')[0])
os.makedirs(f"/dbfs/tmp/workshops/{DATABASE}/{USER_ID}/checkpoints", exist_ok=True)
os.makedirs(f"/dbfs/tmp/workshops/{DATABASE}/{USER_ID}/schemas", exist_ok=True)

spark.sql(f"USE {DATABASE}")
print(f"Database: {DATABASE}, Table suffix: {SUFFIX_TAG or '(none)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Auto Loader Ingestion (Landing Zone → Bronze)
# MAGIC
# MAGIC [Auto Loader](https://docs.databricks.com/en/ingestion/auto-loader/index.html) incrementally
# MAGIC processes new files as they land. We use `trigger(availableNow=True)` for a batch-style run
# MAGIC that processes all available files then stops — ideal for scheduled jobs.
# MAGIC
# MAGIC In production, this would run as a continuous stream or a scheduled job picking up
# MAGIC new exports from your email gateway and identity provider.

# COMMAND ----------

from pyspark.sql import functions as F

# --- Auto Loader: Email events ---
email_checkpoint = f"dbfs:/tmp/workshops/{DATABASE}/{USER_ID}/checkpoints/email_bronze"

email_stream = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", f"dbfs:/tmp/workshops/{DATABASE}/{USER_ID}/schemas/email_bronze")
    .option("cloudFiles.inferColumnTypes", "true")
    .load(f"dbfs:/tmp/workshops/{DATABASE}/landing_zone/email_events/")
)

(
    email_stream
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", email_checkpoint)
    .trigger(availableNow=True)
    .toTable(f"{DATABASE}.email_events_bronze_autoloader{SUFFIX_TAG}")
    .awaitTermination()
)

count = spark.table(f"{DATABASE}.email_events_bronze_autoloader{SUFFIX_TAG}").count()
print(f"✓ Auto Loader ingested {count} email events")

# COMMAND ----------

# --- Auto Loader: Sign-in events ---
signin_checkpoint = f"dbfs:/tmp/workshops/{DATABASE}/{USER_ID}/checkpoints/signin_bronze"

signin_stream = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", f"dbfs:/tmp/workshops/{DATABASE}/{USER_ID}/schemas/signin_bronze")
    .option("cloudFiles.inferColumnTypes", "true")
    .load(f"dbfs:/tmp/workshops/{DATABASE}/landing_zone/signin_events/")
)

(
    signin_stream
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", signin_checkpoint)
    .trigger(availableNow=True)
    .toTable(f"{DATABASE}.signin_events_bronze_autoloader{SUFFIX_TAG}")
    .awaitTermination()
)

count = spark.table(f"{DATABASE}.signin_events_bronze_autoloader{SUFFIX_TAG}").count()
print(f"✓ Auto Loader ingested {count} sign-in events")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Health Metrics Helper
# MAGIC
# MAGIC Every pipeline stage writes a row to `pipeline_health` so we can monitor volume, null rates,
# MAGIC latency, and freshness. This is the foundation for notebook 03 (pipeline monitoring).

# COMMAND ----------

from datetime import datetime


def log_pipeline_health(stage, event_type, df, start_time, status="success"):
    """Write a health metrics row for a pipeline stage."""
    end_time = datetime.utcnow()
    processing_seconds = (end_time - start_time).total_seconds()
    row_count = df.count()

    # Count nulls across key columns
    key_cols = ["event_id", "timestamp"]
    null_count = 0
    for col in key_cols:
        if col in df.columns:
            null_count += df.filter(F.col(col).isNull()).count()

    null_rate = null_count / max(row_count * len(key_cols), 1)

    # Event timestamp range
    ts_stats = df.agg(
        F.min("timestamp").alias("min_ts"),
        F.max("timestamp").alias("max_ts"),
    ).first()

    health_row = spark.createDataFrame([{
        "pipeline_stage": stage,
        "event_type": event_type,
        "run_timestamp": end_time,
        "row_count": row_count,
        "null_count": null_count,
        "null_rate": null_rate,
        "min_event_ts": ts_stats["min_ts"],
        "max_event_ts": ts_stats["max_ts"],
        "processing_seconds": processing_seconds,
        "status": status,
    }])

    health_row.write.format("delta").mode("append").saveAsTable(
        f"{DATABASE}.pipeline_health{SUFFIX_TAG}"
    )
    print(f"  Health logged: {stage}/{event_type} — {row_count} rows, "
          f"{processing_seconds:.1f}s, null_rate={null_rate:.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Bronze → Silver (Email Events)
# MAGIC
# MAGIC Clean and validate email events:
# MAGIC - Cast types and normalize timestamps
# MAGIC - Classify direction (internal / outbound / inbound)
# MAGIC - Flag missing required fields
# MAGIC - Compute derived fields (hour, day_of_week, is_off_hours)

# COMMAND ----------

start = datetime.utcnow()

email_bronze = spark.table(f"{DATABASE}.email_events_bronze")

email_silver = (
    email_bronze
    # Drop the synthetic label — in production this column wouldn't exist
    .drop("_synthetic_category")
    # Normalize and validate
    .withColumn("timestamp", F.to_timestamp("timestamp"))
    .filter(F.col("event_id").isNotNull())
    .filter(F.col("timestamp").isNotNull())
    # Derived time fields
    .withColumn("hour", F.hour("timestamp"))
    .withColumn("day_of_week", F.dayofweek("timestamp"))
    .withColumn("is_off_hours", F.when((F.col("hour") < 7) | (F.col("hour") > 19), True).otherwise(False))
    .withColumn("is_weekend", F.when(F.col("day_of_week").isin(1, 7), True).otherwise(False))
    # Direction classification
    .withColumn("is_external_sender",
        F.when(~F.col("sender_domain").isin("snc-internal.example.com"), True).otherwise(False))
    .withColumn("is_external_recipient",
        F.when(~F.col("recipient_domain").isin("snc-internal.example.com"), True).otherwise(False))
    # Data quality flags
    .withColumn("_dq_missing_subject", F.col("subject").isNull())
    .withColumn("_dq_missing_sender", F.col("sender").isNull())
    .withColumn("_dq_oversized", F.col("size_mb") > 100)
)

email_silver.write.format("delta").mode("overwrite").saveAsTable(
    f"{DATABASE}.email_events_silver{SUFFIX_TAG}"
)

log_pipeline_health("bronze_to_silver", "email", email_silver, start)
print(f"\n✓ {email_silver.count()} email events → {DATABASE}.email_events_silver{SUFFIX_TAG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Quality Summary

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        COUNT(*) as total_events,
        SUM(CASE WHEN _dq_missing_subject THEN 1 ELSE 0 END) as missing_subject,
        SUM(CASE WHEN _dq_missing_sender THEN 1 ELSE 0 END) as missing_sender,
        SUM(CASE WHEN _dq_oversized THEN 1 ELSE 0 END) as oversized_emails,
        SUM(CASE WHEN is_off_hours THEN 1 ELSE 0 END) as off_hours_emails,
        SUM(CASE WHEN is_external_sender THEN 1 ELSE 0 END) as external_sender,
        SUM(CASE WHEN is_external_recipient THEN 1 ELSE 0 END) as external_recipient
    FROM {DATABASE}.email_events_silver{SUFFIX_TAG}
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Bronze → Silver (Sign-in Events)
# MAGIC
# MAGIC Clean and validate sign-in events:
# MAGIC - Cast types and normalize
# MAGIC - Compute geo-velocity between consecutive logins per user
# MAGIC - Flag data quality issues
# MAGIC - Add temporal features

# COMMAND ----------

from pyspark.sql.window import Window
import math

start = datetime.utcnow()

signin_bronze = spark.table(f"{DATABASE}.signin_events_bronze")

# Window for per-user sequential analysis
user_window = Window.partitionBy("user_id").orderBy("timestamp")

signin_silver = (
    signin_bronze
    .drop("_synthetic_category")
    .withColumn("timestamp", F.to_timestamp("timestamp"))
    .filter(F.col("event_id").isNotNull())
    .filter(F.col("timestamp").isNotNull())
    # Temporal features
    .withColumn("hour", F.hour("timestamp"))
    .withColumn("day_of_week", F.dayofweek("timestamp"))
    .withColumn("is_off_hours", F.when((F.col("hour") < 7) | (F.col("hour") > 19), True).otherwise(False))
    .withColumn("is_weekend", F.when(F.col("day_of_week").isin(1, 7), True).otherwise(False))
    # Previous login for geo-velocity calculation
    .withColumn("prev_lat", F.lag("latitude").over(user_window))
    .withColumn("prev_lon", F.lag("longitude").over(user_window))
    .withColumn("prev_ts", F.lag("timestamp").over(user_window))
    .withColumn("minutes_since_last",
        F.when(F.col("prev_ts").isNotNull(),
               (F.unix_timestamp("timestamp") - F.unix_timestamp("prev_ts")) / 60.0
        ).otherwise(None))
    # Haversine distance (km) — simplified using Spark SQL
    .withColumn("dlat", F.radians(F.col("latitude") - F.col("prev_lat")))
    .withColumn("dlon", F.radians(F.col("longitude") - F.col("prev_lon")))
    .withColumn("a",
        F.sin(F.col("dlat") / 2) ** 2 +
        F.cos(F.radians(F.col("prev_lat"))) * F.cos(F.radians(F.col("latitude"))) *
        F.sin(F.col("dlon") / 2) ** 2
    )
    .withColumn("distance_km",
        F.when(F.col("prev_lat").isNotNull(),
               2 * 6371 * F.asin(F.sqrt(F.col("a")))
        ).otherwise(0.0))
    # Geo-velocity (km/h)
    .withColumn("geo_velocity_kmh",
        F.when((F.col("minutes_since_last").isNotNull()) & (F.col("minutes_since_last") > 0),
               F.col("distance_km") / (F.col("minutes_since_last") / 60.0)
        ).otherwise(0.0))
    # Known location flag
    .withColumn("is_known_location",
        F.col("city").isin("Sparks", "Louisville", "Denver", "Huntsville", "Dayton"))
    # Data quality
    .withColumn("_dq_missing_user", F.col("user_id").isNull())
    .withColumn("_dq_impossible_coords",
        (F.abs(F.col("latitude")) > 90) | (F.abs(F.col("longitude")) > 180))
    # Clean up intermediate columns
    .drop("dlat", "dlon", "a", "prev_lat", "prev_lon", "prev_ts")
)

signin_silver.write.format("delta").mode("overwrite").saveAsTable(
    f"{DATABASE}.signin_events_silver{SUFFIX_TAG}"
)

log_pipeline_health("bronze_to_silver", "signin", signin_silver, start)
print(f"\n✓ {signin_silver.count()} sign-in events → {DATABASE}.signin_events_silver{SUFFIX_TAG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sign-in Quality Summary

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        COUNT(*) as total_events,
        SUM(CASE WHEN login_success THEN 1 ELSE 0 END) as successful_logins,
        SUM(CASE WHEN NOT login_success THEN 1 ELSE 0 END) as failed_logins,
        SUM(CASE WHEN is_off_hours THEN 1 ELSE 0 END) as off_hours,
        SUM(CASE WHEN NOT is_known_location THEN 1 ELSE 0 END) as unknown_locations,
        SUM(CASE WHEN NOT mfa_used THEN 1 ELSE 0 END) as no_mfa,
        SUM(CASE WHEN geo_velocity_kmh > 900 THEN 1 ELSE 0 END) as impossible_travel,
        ROUND(AVG(geo_velocity_kmh), 1) as avg_geo_velocity_kmh,
        ROUND(MAX(geo_velocity_kmh), 1) as max_geo_velocity_kmh
    FROM {DATABASE}.signin_events_silver{SUFFIX_TAG}
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Review Pipeline Health
# MAGIC
# MAGIC Every stage logged its metrics. Let's confirm everything is healthy.

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        pipeline_stage,
        event_type,
        run_timestamp,
        row_count,
        null_rate,
        processing_seconds,
        status
    FROM {DATABASE}.pipeline_health{SUFFIX_TAG}
    ORDER BY run_timestamp DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Asset | Location |
# MAGIC |-------|----------|
# MAGIC | Email silver table | `{DATABASE}.email_events_silver` |
# MAGIC | Sign-in silver table | `{DATABASE}.signin_events_silver` |
# MAGIC | Auto Loader (email) | `{DATABASE}.email_events_bronze_autoloader` |
# MAGIC | Auto Loader (sign-in) | `{DATABASE}.signin_events_bronze_autoloader` |
# MAGIC | Pipeline health | `{DATABASE}.pipeline_health` |
# MAGIC | Checkpoints | `dbfs:/tmp/workshops/{DATABASE}/<user_id>/checkpoints/` |
# MAGIC
# MAGIC **Key patterns demonstrated:**
# MAGIC - Auto Loader with `trigger(availableNow=True)` for incremental file ingestion
# MAGIC - Schema inference and evolution from JSON files
# MAGIC - Medallion architecture (Bronze → Silver) with data quality flags
# MAGIC - Geo-velocity computation for sign-in anomaly detection
# MAGIC - Pipeline health metrics at every stage
# MAGIC
# MAGIC **Next →** Open `02_risk_scoring` to add rule-based risk scoring on top of the silver tables.
