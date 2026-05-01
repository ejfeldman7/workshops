# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 3: Security Data Pipeline & Application Integration
# MAGIC ## Notebook 2 — Rule-Based Risk Scoring
# MAGIC
# MAGIC **Azure Gov Cloud Compatibility:** ✅ Everything in this notebook runs on classic compute with Databricks Runtime 13.3+.
# MAGIC
# MAGIC Score security events using a **deterministic rule engine** — no ML model training required.
# MAGIC This produces the same shape of output (risk scores, tiers, explanations) as the ML workshops,
# MAGIC making it a drop-in data source for the application integration in notebook 04.
# MAGIC
# MAGIC ### Scoring Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌───────────────────────────────────────────────────────────────────────┐
# MAGIC │                    Rule-Based Risk Scoring                            │
# MAGIC │                                                                       │
# MAGIC │  ┌──────────────────┐         ┌──────────────────┐                    │
# MAGIC │  │ email_events_    │         │ signin_events_   │                    │
# MAGIC │  │ silver           │         │ silver           │                    │
# MAGIC │  └────────┬─────────┘         └────────┬─────────┘                    │
# MAGIC │           │                             │                             │
# MAGIC │           ▼                             ▼                             │
# MAGIC │  ┌────────────────┐          ┌─────────────────────┐                  │
# MAGIC │  │ Keyword match  │          │ Geo-velocity check  │                  │
# MAGIC │  │ Size threshold │          │ Failed attempt count│                  │
# MAGIC │  │ Direction flag │          │ Off-hours + MFA     │                  │
# MAGIC │  │ Temporal signal│          │ Unknown device/loc  │                  │
# MAGIC │  └────────┬───────┘          └─────────┬───────────┘                  │
# MAGIC │           │                             │                             │
# MAGIC │           ▼                             ▼                             │
# MAGIC │  ┌────────────────┐          ┌──────────────────┐                     │
# MAGIC │  │ email_risk_    │          │ signin_risk_     │                     │
# MAGIC │  │ scores         │          │ scores           │                     │
# MAGIC │  │ (score + tier  │          │ (score + tier    │                     │
# MAGIC │  │  + reasons)    │          │  + reasons)      │                     │
# MAGIC │  └────────────────┘          └──────────────────┘                     │
# MAGIC │                                                                       │
# MAGIC │  All tables live in <your_database> (Hive metastore)                  │
# MAGIC │  No ML training — pure SQL/PySpark rule engine                        │
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
from pyspark.sql.types import DoubleType, StringType, ArrayType
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Email Risk Scoring
# MAGIC
# MAGIC Score each email on four signals, then combine into a composite risk score:
# MAGIC
# MAGIC | Signal | Weight | What it catches |
# MAGIC |--------|--------|-----------------|
# MAGIC | **Keyword** | 0.35 | Risky language in subject line (exfiltration, phishing, credential sharing) |
# MAGIC | **Attachment** | 0.25 | Large attachments, suspicious file types (.exe, .sql, .tar.gz) |
# MAGIC | **Direction** | 0.20 | Outbound emails to external domains, especially with attachments |
# MAGIC | **Temporal** | 0.20 | Off-hours activity, weekend sends |

# COMMAND ----------

# --- Define risk keyword lists ---
EXFIL_KEYWORDS = [
    "personal email", "usb drive", "dropbox", "forwarded to gmail",
    "send me the database", "copy to external", "download before",
    "zip up", "exported all records", "transfer to my account",
]

PHISHING_KEYWORDS = [
    "urgent", "verify your account", "click here immediately",
    "password expires", "unusual sign-in", "confirm your identity",
    "gift cards", "payment overdue", "action required", "account deactivated",
]

POLICY_KEYWORDS = [
    "admin password", "disabled the firewall", "use my credentials",
    "installed without it approval", "shared the api key", "bypassed",
    "exception for our department", "gave access to production",
]

SUSPICIOUS_EXTENSIONS = ["exe", "html", "sql", "tar.gz", "zip", "env"]

# Build keyword matching expressions
def build_keyword_match(col_name, keywords):
    """Build a Spark Column that returns True if any keyword is found."""
    conditions = [F.lower(F.col(col_name)).contains(kw.lower()) for kw in keywords]
    return conditions[0] if len(conditions) == 1 else conditions[0]
    # Chain with OR

# COMMAND ----------

email_silver = spark.table(f"{DATABASE}.email_events_silver{SUFFIX_TAG}")

# --- Signal 1: Keyword risk ---
# Build OR chain for each keyword category
exfil_conds = F.lit(False)
for kw in EXFIL_KEYWORDS:
    exfil_conds = exfil_conds | F.lower(F.col("subject")).contains(kw.lower())

phishing_conds = F.lit(False)
for kw in PHISHING_KEYWORDS:
    phishing_conds = phishing_conds | F.lower(F.col("subject")).contains(kw.lower())

policy_conds = F.lit(False)
for kw in POLICY_KEYWORDS:
    policy_conds = policy_conds | F.lower(F.col("subject")).contains(kw.lower())

email_scored = (
    email_silver
    # Keyword signal (0-1)
    .withColumn("_exfil_match", exfil_conds)
    .withColumn("_phishing_match", phishing_conds)
    .withColumn("_policy_match", policy_conds)
    .withColumn("keyword_signal",
        F.when(F.col("_exfil_match"), 0.9)
         .when(F.col("_phishing_match"), 0.85)
         .when(F.col("_policy_match"), 0.8)
         .otherwise(0.0))
    .withColumn("keyword_category",
        F.when(F.col("_exfil_match"), "data_exfiltration")
         .when(F.col("_phishing_match"), "phishing")
         .when(F.col("_policy_match"), "policy_violation")
         .otherwise("none"))
)

# --- Signal 2: Attachment risk ---
# Suspicious file types or unusually large emails
sus_ext_conds = F.lit(False)
for ext in SUSPICIOUS_EXTENSIONS:
    sus_ext_conds = sus_ext_conds | F.col("attachment_types").contains(ext)

email_scored = (
    email_scored
    .withColumn("attachment_signal",
        F.when(
            F.col("has_attachment") & sus_ext_conds & (F.col("size_mb") > 10), 0.9
        ).when(
            F.col("has_attachment") & (F.col("size_mb") > 50), 0.8
        ).when(
            F.col("has_attachment") & sus_ext_conds, 0.6
        ).when(
            F.col("has_attachment") & (F.col("size_mb") > 10), 0.4
        ).otherwise(0.0))
)

# --- Signal 3: Direction risk ---
email_scored = (
    email_scored
    .withColumn("direction_signal",
        F.when(
            (F.col("direction") == "outbound") & F.col("has_attachment") & (F.col("size_mb") > 5), 0.8
        ).when(
            (F.col("direction") == "outbound") & F.col("has_attachment"), 0.5
        ).when(
            (F.col("direction") == "inbound") & F.col("is_external_sender"), 0.3
        ).otherwise(0.0))
)

# --- Signal 4: Temporal risk ---
email_scored = (
    email_scored
    .withColumn("temporal_signal",
        F.when(F.col("is_off_hours") & F.col("is_weekend"), 0.7)
         .when(F.col("is_off_hours"), 0.4)
         .when(F.col("is_weekend"), 0.3)
         .otherwise(0.0))
)

# --- Composite score ---
email_scored = (
    email_scored
    .withColumn("risk_score",
        0.35 * F.col("keyword_signal") +
        0.25 * F.col("attachment_signal") +
        0.20 * F.col("direction_signal") +
        0.20 * F.col("temporal_signal"))
    .withColumn("risk_tier",
        F.when(F.col("risk_score") >= 0.6, "Critical")
         .when(F.col("risk_score") >= 0.4, "High")
         .when(F.col("risk_score") >= 0.2, "Medium")
         .otherwise("Low"))
    # Build human-readable explanation
    .withColumn("risk_reasons",
        F.concat_ws("; ",
            F.when(F.col("keyword_signal") > 0,
                   F.concat(F.lit("keyword_match:"), F.col("keyword_category"))),
            F.when(F.col("attachment_signal") > 0.5,
                   F.concat(F.lit("suspicious_attachment:"), F.round(F.col("size_mb"), 1).cast("string"), F.lit("MB"))),
            F.when(F.col("direction_signal") > 0.3,
                   F.concat(F.lit("external_"), F.col("direction"))),
            F.when(F.col("temporal_signal") > 0,
                   F.lit("off_hours_activity")),
        ))
    .drop("_exfil_match", "_phishing_match", "_policy_match")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Email Risk Scores

# COMMAND ----------

start = datetime.utcnow()

email_risk_cols = [
    "event_id", "timestamp", "sender", "recipient", "subject", "direction",
    "has_attachment", "attachment_count", "attachment_types", "size_mb",
    "is_off_hours", "is_weekend",
    "keyword_signal", "keyword_category", "attachment_signal",
    "direction_signal", "temporal_signal",
    "risk_score", "risk_tier", "risk_reasons",
]

email_scored.select(email_risk_cols).write.format("delta").mode("overwrite").saveAsTable(
    f"{DATABASE}.email_risk_scores{SUFFIX_TAG}"
)

# Log health
scored_df = spark.table(f"{DATABASE}.email_risk_scores{SUFFIX_TAG}")

end = datetime.utcnow()
health_row = spark.createDataFrame([{
    "pipeline_stage": "risk_scoring",
    "event_type": "email",
    "run_timestamp": end,
    "row_count": scored_df.count(),
    "null_count": 0,
    "null_rate": 0.0,
    "min_event_ts": scored_df.agg(F.min("timestamp")).first()[0],
    "max_event_ts": scored_df.agg(F.max("timestamp")).first()[0],
    "processing_seconds": (end - start).total_seconds(),
    "status": "success",
}])
health_row.write.format("delta").mode("append").saveAsTable(f"{DATABASE}.pipeline_health{SUFFIX_TAG}")

print(f"✓ Email risk scores → {DATABASE}.email_risk_scores{SUFFIX_TAG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Email Risk Distribution

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        risk_tier,
        COUNT(*) as count,
        ROUND(AVG(risk_score), 3) as avg_score,
        ROUND(MIN(risk_score), 3) as min_score,
        ROUND(MAX(risk_score), 3) as max_score
    FROM {DATABASE}.email_risk_scores{SUFFIX_TAG}
    GROUP BY risk_tier
    ORDER BY avg_score DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top 10 Riskiest Emails

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        risk_tier, ROUND(risk_score, 3) as risk_score,
        subject, sender, recipient, direction,
        ROUND(size_mb, 1) as size_mb,
        risk_reasons
    FROM {DATABASE}.email_risk_scores{SUFFIX_TAG}
    WHERE risk_tier IN ('Critical', 'High')
    ORDER BY risk_score DESC
    LIMIT 10
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Sign-in Risk Scoring
# MAGIC
# MAGIC Score each sign-in event on four signals:
# MAGIC
# MAGIC | Signal | Weight | What it catches |
# MAGIC |--------|--------|-----------------|
# MAGIC | **Geo-velocity** | 0.30 | Impossible travel (>900 km/h between consecutive logins) |
# MAGIC | **Auth failure** | 0.25 | Brute force patterns (multiple failed attempts) |
# MAGIC | **Location/Device** | 0.25 | Unknown locations or anomalous devices |
# MAGIC | **Temporal + MFA** | 0.20 | Off-hours access without MFA |

# COMMAND ----------

signin_silver = spark.table(f"{DATABASE}.signin_events_silver{SUFFIX_TAG}")

signin_scored = (
    signin_silver
    # --- Signal 1: Geo-velocity ---
    .withColumn("geo_signal",
        F.when(F.col("geo_velocity_kmh") > 5000, 1.0)    # physically impossible
         .when(F.col("geo_velocity_kmh") > 900, 0.9)      # faster than commercial flight
         .when(F.col("geo_velocity_kmh") > 500, 0.6)      # suspicious
         .otherwise(0.0))
    # --- Signal 2: Auth failure pattern ---
    .withColumn("auth_signal",
        F.when(F.col("failed_attempts") >= 10, 1.0)       # definite brute force
         .when(F.col("failed_attempts") >= 5, 0.8)         # likely brute force
         .when((F.col("failed_attempts") >= 3) & (~F.col("login_success")), 0.6)
         .when(F.col("failed_attempts") >= 2, 0.3)
         .otherwise(0.0))
    # --- Signal 3: Location + device ---
    .withColumn("location_device_signal",
        F.when(~F.col("is_known_location") & F.col("device_type").isin(
            "Unknown_Linux", "Tor_Browser", "VPS_Cloud", "Public_Kiosk"), 1.0)
         .when(~F.col("is_known_location"), 0.6)
         .when(F.col("device_type").isin(
            "Unknown_Linux", "Tor_Browser", "VPS_Cloud", "Public_Kiosk"), 0.5)
         .otherwise(0.0))
    # --- Signal 4: Temporal + MFA ---
    .withColumn("temporal_mfa_signal",
        F.when(F.col("is_off_hours") & (~F.col("mfa_used")), 0.8)
         .when(F.col("is_off_hours"), 0.4)
         .when(~F.col("mfa_used"), 0.3)
         .otherwise(0.0))
    # --- Composite score ---
    .withColumn("risk_score",
        0.30 * F.col("geo_signal") +
        0.25 * F.col("auth_signal") +
        0.25 * F.col("location_device_signal") +
        0.20 * F.col("temporal_mfa_signal"))
    .withColumn("risk_tier",
        F.when(F.col("risk_score") >= 0.6, "Critical")
         .when(F.col("risk_score") >= 0.4, "High")
         .when(F.col("risk_score") >= 0.2, "Medium")
         .otherwise("Low"))
    # Build explanation
    .withColumn("risk_reasons",
        F.concat_ws("; ",
            F.when(F.col("geo_signal") > 0.5,
                   F.concat(F.lit("impossible_travel:"), F.round(F.col("geo_velocity_kmh"), 0).cast("string"), F.lit("km/h"))),
            F.when(F.col("auth_signal") > 0.3,
                   F.concat(F.lit("failed_attempts:"), F.col("failed_attempts").cast("string"))),
            F.when(F.col("location_device_signal") > 0.3,
                   F.concat(F.lit("unknown_location_or_device:"), F.col("city"), F.lit("/"), F.col("device_type"))),
            F.when(F.col("temporal_mfa_signal") > 0.3,
                   F.when(~F.col("mfa_used"), F.lit("off_hours_no_mfa")).otherwise(F.lit("off_hours"))),
        ))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Sign-in Risk Scores

# COMMAND ----------

start = datetime.utcnow()

signin_risk_cols = [
    "event_id", "timestamp", "user_id", "ip_address",
    "city", "country", "latitude", "longitude",
    "device_type", "device_id", "mfa_used", "login_success", "failed_attempts",
    "application", "geo_velocity_kmh", "distance_km",
    "is_off_hours", "is_weekend", "is_known_location",
    "geo_signal", "auth_signal", "location_device_signal", "temporal_mfa_signal",
    "risk_score", "risk_tier", "risk_reasons",
]

signin_scored.select(signin_risk_cols).write.format("delta").mode("overwrite").saveAsTable(
    f"{DATABASE}.signin_risk_scores{SUFFIX_TAG}"
)

# Log health
scored_signin = spark.table(f"{DATABASE}.signin_risk_scores{SUFFIX_TAG}")
end = datetime.utcnow()
health_row = spark.createDataFrame([{
    "pipeline_stage": "risk_scoring",
    "event_type": "signin",
    "run_timestamp": end,
    "row_count": scored_signin.count(),
    "null_count": 0,
    "null_rate": 0.0,
    "min_event_ts": scored_signin.agg(F.min("timestamp")).first()[0],
    "max_event_ts": scored_signin.agg(F.max("timestamp")).first()[0],
    "processing_seconds": (end - start).total_seconds(),
    "status": "success",
}])
health_row.write.format("delta").mode("append").saveAsTable(f"{DATABASE}.pipeline_health{SUFFIX_TAG}")

print(f"✓ Sign-in risk scores → {DATABASE}.signin_risk_scores{SUFFIX_TAG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sign-in Risk Distribution

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        risk_tier,
        COUNT(*) as count,
        ROUND(AVG(risk_score), 3) as avg_score,
        SUM(CASE WHEN NOT login_success THEN 1 ELSE 0 END) as failed_logins,
        SUM(CASE WHEN NOT mfa_used THEN 1 ELSE 0 END) as no_mfa
    FROM {DATABASE}.signin_risk_scores{SUFFIX_TAG}
    GROUP BY risk_tier
    ORDER BY avg_score DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top 10 Riskiest Sign-ins

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        risk_tier, ROUND(risk_score, 3) as risk_score,
        user_id, city, country, device_type,
        login_success, failed_attempts, mfa_used,
        ROUND(geo_velocity_kmh, 0) as geo_velocity_kmh,
        risk_reasons
    FROM {DATABASE}.signin_risk_scores{SUFFIX_TAG}
    WHERE risk_tier IN ('Critical', 'High')
    ORDER BY risk_score DESC
    LIMIT 10
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Unified Security Summary View
# MAGIC
# MAGIC Create a view that combines both event types for the application layer to query.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE VIEW {DATABASE}.security_alerts{SUFFIX_TAG} AS
    SELECT
        event_id,
        timestamp,
        'email' as event_type,
        sender as principal,
        risk_score,
        risk_tier,
        risk_reasons,
        subject as detail
    FROM {DATABASE}.email_risk_scores{SUFFIX_TAG}
    WHERE risk_tier IN ('Critical', 'High')

    UNION ALL

    SELECT
        event_id,
        timestamp,
        'signin' as event_type,
        user_id as principal,
        risk_score,
        risk_tier,
        risk_reasons,
        CONCAT(city, '/', country, ' via ', device_type) as detail
    FROM {DATABASE}.signin_risk_scores{SUFFIX_TAG}
    WHERE risk_tier IN ('Critical', 'High')
""")

alert_count = spark.table(f"{DATABASE}.security_alerts{SUFFIX_TAG}").count()
print(f"✓ Unified security_alerts view created — {alert_count} active alerts")

# COMMAND ----------

display(spark.sql(f"""
    SELECT event_type, risk_tier, COUNT(*) as alert_count
    FROM {DATABASE}.security_alerts{SUFFIX_TAG}
    GROUP BY event_type, risk_tier
    ORDER BY event_type, risk_tier
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Asset | Location |
# MAGIC |-------|----------|
# MAGIC | Email risk scores | `{DATABASE}.email_risk_scores` |
# MAGIC | Sign-in risk scores | `{DATABASE}.signin_risk_scores` |
# MAGIC | Unified alerts view | `{DATABASE}.security_alerts` |
# MAGIC | Pipeline health | `{DATABASE}.pipeline_health` (2 new rows) |
# MAGIC
# MAGIC **Key patterns demonstrated:**
# MAGIC - Deterministic rule engine with weighted composite scoring
# MAGIC - Human-readable `risk_reasons` per event (same pattern ML models would use with SHAP)
# MAGIC - Unified alert view combining multiple event types
# MAGIC - Health metrics logged at every stage
# MAGIC
# MAGIC > **Note:** In a production environment, these rules would be tuned alongside the ML models
# MAGIC > from Workshops 1 & 2. The rule engine catches known-bad patterns; the ML models catch
# MAGIC > novel anomalies. Together they form a defense-in-depth scoring layer.
# MAGIC
# MAGIC **Next →** Open `03_pipeline_monitoring` to build the monitoring dashboard on top of `pipeline_health`.
