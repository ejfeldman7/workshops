# Databricks notebook: /Shared/security/query_proxy

dbutils.widgets.text("op", "")
dbutils.widgets.text("database", "")
dbutils.widgets.text("table_suffix", "")
dbutils.widgets.text("limit", "20")
dbutils.widgets.text("user_id", "")

import json

op = dbutils.widgets.get("op")
database = dbutils.widgets.get("database")
table_suffix = dbutils.widgets.get("table_suffix").strip()
limit = int(dbutils.widgets.get("limit") or "20")
user_id = dbutils.widgets.get("user_id")

suffix_tag = f"_{table_suffix}" if table_suffix else ""

def rows(sql_text: str):
    return [r.asDict(recursive=True) for r in spark.sql(sql_text).collect()]

if op == "get_alert_summary":
    result = rows(f"""
        SELECT event_type, risk_tier, COUNT(*) AS count, ROUND(AVG(risk_score), 3) AS avg_score
        FROM (
            SELECT 'email' AS event_type, risk_tier, risk_score
            FROM {database}.email_risk_scores{suffix_tag}
            UNION ALL
            SELECT 'signin' AS event_type, risk_tier, risk_score
            FROM {database}.signin_risk_scores{suffix_tag}
        )
        GROUP BY event_type, risk_tier
        ORDER BY event_type, avg_score DESC
    """)

elif op == "get_top_alerts":
    result = rows(f"""
        SELECT event_type, event_id, timestamp, principal, ROUND(risk_score, 3) AS risk_score,
               risk_tier, risk_reasons, detail
        FROM {database}.security_alerts{suffix_tag}
        ORDER BY risk_score DESC
        LIMIT {limit}
    """)

elif op == "get_user_profile":
    result = rows(f"""
        SELECT user_id,
               COUNT(*) AS total_signins,
               SUM(CASE WHEN risk_tier IN ('Critical','High') THEN 1 ELSE 0 END) AS high_risk_count,
               ROUND(AVG(risk_score), 3) AS avg_risk,
               ROUND(MAX(risk_score), 3) AS max_risk,
               SUM(CASE WHEN NOT mfa_used THEN 1 ELSE 0 END) AS no_mfa_count,
               SUM(failed_attempts) AS total_failed
        FROM {database}.signin_risk_scores{suffix_tag}
        WHERE user_id = '{user_id}'
        GROUP BY user_id
    """)

elif op == "get_pipeline_status":
    result = rows(f"""
        SELECT pipeline_stage, event_type, row_count,
               ROUND(null_rate, 4) AS null_rate,
               ROUND(processing_seconds, 1) AS seconds,
               status, run_timestamp
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY pipeline_stage, event_type
                       ORDER BY run_timestamp DESC
                   ) AS rn
            FROM {database}.pipeline_health{suffix_tag}
        )
        WHERE rn = 1
        ORDER BY pipeline_stage, event_type
    """)

elif op == "get_pipeline_anomalies":
    result = rows(f"""
        SELECT pipeline_stage, event_type, run_timestamp, row_count, null_rate, anomaly_reasons
        FROM {database}.pipeline_anomalies{suffix_tag}
        WHERE has_anomaly = true
        ORDER BY run_timestamp DESC
    """)

else:
    raise ValueError(f"unsupported op: {op}")

# Keep this for small payloads only.
dbutils.notebook.exit(json.dumps(result, default=str))
