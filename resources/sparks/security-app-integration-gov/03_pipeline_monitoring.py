# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 3: Security Data Pipeline & Application Integration
# MAGIC ## Notebook 3 — Pipeline Monitoring & Alerting
# MAGIC
# MAGIC **Azure Gov Cloud Compatibility:** ✅ Everything in this notebook runs on classic compute with Databricks Runtime 13.3+.
# MAGIC
# MAGIC Use the `pipeline_health` table to detect problems in the data pipeline itself:
# MAGIC volume drops, null rate spikes, processing slowdowns, and stale data.
# MAGIC
# MAGIC ### Monitoring Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌───────────────────────────────────────────────────────────────────────┐
# MAGIC │                    Pipeline Monitoring                                │
# MAGIC │                                                                       │
# MAGIC │  ┌────────────────┐                                                   │
# MAGIC │  │ pipeline_health│──▶  Volume trends                                 │
# MAGIC │  │ (metrics from  │──▶  Null rate tracking                            │
# MAGIC │  │  every stage)  │──▶  Processing latency                            │
# MAGIC │  └───────┬────────┘──▶  Data freshness                                │
# MAGIC │          │                                                            │
# MAGIC │          ▼                                                            │
# MAGIC │  ┌────────────────┐     ┌────────────────┐                            │
# MAGIC │  │ pipeline_      │     │ Webhook /       │                           │
# MAGIC │  │ anomalies      │────▶│ Slack / Email   │                           │
# MAGIC │  │ (flagged runs) │     │ alerts          │                           │
# MAGIC │  └────────────────┘     └────────────────┘                            │
# MAGIC │                                                                       │
# MAGIC │  All tables live in <your_database> (Hive metastore)                  │
# MAGIC │  This notebook doubles as a scheduled monitoring job.                 │
# MAGIC └───────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites

# COMMAND ----------

from datetime import datetime

dbutils.widgets.text("database", "security_app_integration", "Database")
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
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Pipeline Health Overview
# MAGIC
# MAGIC Review all runs logged by the ingestion and scoring pipelines.

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        pipeline_stage,
        event_type,
        run_timestamp,
        row_count,
        ROUND(null_rate, 4) as null_rate,
        ROUND(processing_seconds, 1) as seconds,
        status,
        min_event_ts,
        max_event_ts
    FROM {DATABASE}.pipeline_health{SUFFIX_TAG}
    ORDER BY run_timestamp DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Define Anomaly Thresholds
# MAGIC
# MAGIC In production, these thresholds would be tuned based on historical baselines.
# MAGIC For the workshop, we use reasonable defaults and show how to detect problems.

# COMMAND ----------

# Anomaly detection thresholds
THRESHOLDS = {
    "min_row_count": 100,          # Fewer than this = possible data loss
    "max_null_rate": 0.05,          # More than 5% nulls = data quality issue
    "max_processing_seconds": 300,  # More than 5 minutes = performance problem
    "max_staleness_hours": 24,      # Data older than 24h = freshness issue
}

print("Monitoring thresholds:")
for name, value in THRESHOLDS.items():
    print(f"  {name:30s}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Run Anomaly Detection on Pipeline Health
# MAGIC
# MAGIC Check each health record against thresholds and flag problems.

# COMMAND ----------

health_df = spark.table(f"{DATABASE}.pipeline_health{SUFFIX_TAG}")

anomalies = (
    health_df
    .withColumn("is_low_volume",
        F.col("row_count") < THRESHOLDS["min_row_count"])
    .withColumn("is_high_null_rate",
        F.col("null_rate") > THRESHOLDS["max_null_rate"])
    .withColumn("is_slow",
        F.col("processing_seconds") > THRESHOLDS["max_processing_seconds"])
    .withColumn("is_stale",
        (F.unix_timestamp(F.current_timestamp()) - F.unix_timestamp("max_event_ts")) / 3600
        > THRESHOLDS["max_staleness_hours"])
    .withColumn("has_anomaly",
        F.col("is_low_volume") | F.col("is_high_null_rate") | F.col("is_slow") | F.col("is_stale"))
    .withColumn("anomaly_reasons",
        F.concat_ws("; ",
            F.when(F.col("is_low_volume"),
                   F.concat(F.lit("low_volume:"), F.col("row_count").cast("string"))),
            F.when(F.col("is_high_null_rate"),
                   F.concat(F.lit("high_null_rate:"), F.round(F.col("null_rate"), 4).cast("string"))),
            F.when(F.col("is_slow"),
                   F.concat(F.lit("slow_processing:"), F.round(F.col("processing_seconds"), 1).cast("string"), F.lit("s"))),
            F.when(F.col("is_stale"),
                   F.lit("stale_data")),
        ))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pipeline Anomaly Results

# COMMAND ----------

display(anomalies.select(
    "pipeline_stage", "event_type", "run_timestamp",
    "row_count", "null_rate", "processing_seconds",
    "has_anomaly", "anomaly_reasons"
).orderBy(F.desc("has_anomaly"), F.desc("run_timestamp")))

# COMMAND ----------

anomaly_count = anomalies.filter("has_anomaly").count()
total_count = anomalies.count()
print(f"\nPipeline health: {anomaly_count}/{total_count} runs flagged")
if anomaly_count == 0:
    print("✓ All pipeline stages are healthy")
else:
    print("⚠ Pipeline anomalies detected — review flagged runs above")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Save Pipeline Anomalies
# MAGIC
# MAGIC Write flagged runs to a `pipeline_anomalies` table for the application layer to query.

# COMMAND ----------

anomalies.write.format("delta").mode("overwrite").saveAsTable(
    f"{DATABASE}.pipeline_anomalies{SUFFIX_TAG}"
)
print(f"✓ Pipeline anomalies → {DATABASE}.pipeline_anomalies{SUFFIX_TAG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Simulating a Problem (Interactive Exercise)
# MAGIC
# MAGIC Let's inject a bad pipeline run to see the monitoring catch it.
# MAGIC This simulates what would happen if an upstream data source went silent or sent corrupt data.

# COMMAND ----------

from datetime import datetime

# Simulate a pipeline run that processed very few rows with high null rate
bad_run = spark.createDataFrame([{
    "pipeline_stage": "bronze_to_silver",
    "event_type": "email",
    "run_timestamp": datetime.utcnow(),
    "row_count": 12,            # way below threshold of 100
    "null_count": 5,
    "null_rate": 0.42,          # way above threshold of 0.05
    "min_event_ts": datetime(2025, 3, 28),
    "max_event_ts": datetime(2025, 3, 28),
    "processing_seconds": 0.3,
    "status": "success",
}])

bad_run.write.format("delta").mode("append").saveAsTable(
    f"{DATABASE}.pipeline_health{SUFFIX_TAG}"
)
print("✓ Injected simulated bad pipeline run")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Re-run anomaly detection — the bad run should now be flagged

# COMMAND ----------

health_df = spark.table(f"{DATABASE}.pipeline_health{SUFFIX_TAG}")

recheck = (
    health_df
    .withColumn("is_low_volume", F.col("row_count") < THRESHOLDS["min_row_count"])
    .withColumn("is_high_null_rate", F.col("null_rate") > THRESHOLDS["max_null_rate"])
    .withColumn("is_slow", F.col("processing_seconds") > THRESHOLDS["max_processing_seconds"])
    .withColumn("has_anomaly",
        F.col("is_low_volume") | F.col("is_high_null_rate") | F.col("is_slow"))
    .withColumn("anomaly_reasons",
        F.concat_ws("; ",
            F.when(F.col("is_low_volume"),
                   F.concat(F.lit("low_volume:"), F.col("row_count").cast("string"))),
            F.when(F.col("is_high_null_rate"),
                   F.concat(F.lit("high_null_rate:"), F.round(F.col("null_rate"), 4).cast("string"))),
            F.when(F.col("is_slow"),
                   F.concat(F.lit("slow_processing:"), F.round(F.col("processing_seconds"), 1).cast("string"), F.lit("s"))),
        ))
)

flagged = recheck.filter("has_anomaly")
print(f"Flagged runs: {flagged.count()}")
display(flagged.select(
    "pipeline_stage", "event_type", "run_timestamp",
    "row_count", "null_rate", "anomaly_reasons"
).orderBy(F.desc("run_timestamp")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Alerting Patterns
# MAGIC
# MAGIC In production, pipeline anomalies trigger alerts. Here are patterns that work
# MAGIC with Databricks Jobs:
# MAGIC
# MAGIC ### Option A: Job Webhook Notifications
# MAGIC
# MAGIC Databricks Jobs support webhook notifications on success/failure:
# MAGIC
# MAGIC ```python
# MAGIC # When creating a Job (see notebook 04), add webhook destinations:
# MAGIC from databricks.sdk.service.jobs import WebhookNotifications, Webhook
# MAGIC
# MAGIC webhook = WebhookNotifications(
# MAGIC     on_failure=[Webhook(id="webhook_id_from_workspace_settings")]
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ### Option B: Notebook Exit with Status
# MAGIC
# MAGIC The monitoring notebook can exit with a status code that the Job interprets:
# MAGIC
# MAGIC ```python
# MAGIC anomaly_count = flagged.count()
# MAGIC if anomaly_count > 0:
# MAGIC     # This makes the Job report as "succeeded with output"
# MAGIC     # Your alerting system can parse the output
# MAGIC     dbutils.notebook.exit(json.dumps({
# MAGIC         "status": "anomalies_detected",
# MAGIC         "count": anomaly_count,
# MAGIC         "details": flagged.toPandas().to_dict(orient="records")
# MAGIC     }))
# MAGIC else:
# MAGIC     dbutils.notebook.exit(json.dumps({"status": "healthy"}))
# MAGIC ```
# MAGIC
# MAGIC ### Option C: Direct API Call from Notebook
# MAGIC
# MAGIC For Slack, Teams, or PagerDuty:
# MAGIC
# MAGIC ```python
# MAGIC import requests
# MAGIC
# MAGIC if anomaly_count > 0:
# MAGIC     requests.post(
# MAGIC         "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
# MAGIC         json={"text": f"⚠️ {anomaly_count} pipeline anomalies detected. Check pipeline_anomalies table."}
# MAGIC     )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: SQL Queries for Application Layer
# MAGIC
# MAGIC These are the queries that Defensible Suite (or any external app) would run
# MAGIC to power its pipeline monitoring dashboard. We'll use these same queries
# MAGIC via the SDK in notebook 04.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 1: Current pipeline status (latest run per stage)

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        pipeline_stage,
        event_type,
        row_count,
        ROUND(null_rate, 4) as null_rate,
        ROUND(processing_seconds, 1) as processing_seconds,
        status,
        run_timestamp,
        ROUND((unix_timestamp(current_timestamp()) - unix_timestamp(max_event_ts)) / 3600, 1) as hours_since_latest_data
    FROM (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY pipeline_stage, event_type
            ORDER BY run_timestamp DESC
        ) as rn
        FROM {DATABASE}.pipeline_health{SUFFIX_TAG}
    )
    WHERE rn = 1
    ORDER BY pipeline_stage, event_type
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 2: Risk score summary for the dashboard

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        event_type,
        risk_tier,
        alert_count,
        ROUND(avg_score, 3) as avg_score,
        latest_alert
    FROM (
        SELECT
            'email' as event_type, risk_tier,
            COUNT(*) as alert_count,
            AVG(risk_score) as avg_score,
            MAX(timestamp) as latest_alert
        FROM {DATABASE}.email_risk_scores{SUFFIX_TAG}
        GROUP BY risk_tier

        UNION ALL

        SELECT
            'signin' as event_type, risk_tier,
            COUNT(*) as alert_count,
            AVG(risk_score) as avg_score,
            MAX(timestamp) as latest_alert
        FROM {DATABASE}.signin_risk_scores{SUFFIX_TAG}
        GROUP BY risk_tier
    )
    ORDER BY event_type, avg_score DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Asset | Location |
# MAGIC |-------|----------|
# MAGIC | Pipeline anomalies | `{DATABASE}.pipeline_anomalies` |
# MAGIC | Pipeline health | `{DATABASE}.pipeline_health` (includes simulated bad run) |
# MAGIC
# MAGIC **Key patterns demonstrated:**
# MAGIC - Threshold-based anomaly detection on pipeline metrics
# MAGIC - Human-readable explanations for flagged runs
# MAGIC - Simulated problem injection and detection
# MAGIC - Alerting patterns (webhooks, notebook exit codes, direct API calls)
# MAGIC - SQL queries ready for the application layer
# MAGIC
# MAGIC **Next →** Open `04_app_integration` to connect everything to an external application via the Databricks SDK.