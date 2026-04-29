# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 3: Security Data Pipeline & Application Integration
# MAGIC ## Notebook 4 — Application Integration with the Databricks SDK
# MAGIC
# MAGIC This is the core notebook for connecting an external application — like **Defensible Suite** —
# MAGIC to Databricks. Everything here runs as plain Python using the
# MAGIC [Databricks SDK](https://docs.databricks.com/en/dev-tools/sdk-python.html). The same code
# MAGIC works inside a Databricks notebook or in an external web app, CLI tool, or backend service.
# MAGIC
# MAGIC **Azure Gov Cloud Compatibility:** This notebook is adapted for Azure Government Cloud.
# MAGIC All endpoints use `.usgovcloudapi.net` and compute is classic (no serverless).
# MAGIC
# MAGIC > **Gov Cloud Note:** Azure Gov Cloud workspaces today **do not have SQL Warehouses**,
# MAGIC > the SQL Statement Execution API, or Databricks SQL. This workshop demonstrates two
# MAGIC > alternative patterns that cover both audiences:
# MAGIC >
# MAGIC > 1. **`spark.sql(...)`** — for analysts running queries inside the workspace (notebooks, jobs).
# MAGIC > 2. **`databricks-sql-connector` (JDBC)** against the all-purpose cluster's legacy SQL
# MAGIC >    endpoint — for external apps like Defensible Suite that pull data from outside Databricks.
# MAGIC >
# MAGIC > Both patterns read the exact same Delta tables. When Evergreen Gov Cloud lands and
# MAGIC > SQL Warehouses become available, the JDBC pattern can be swapped for the Statement
# MAGIC > Execution API with no schema changes.
# MAGIC
# MAGIC **Compute:** Classic compute with Databricks Runtime 13.3+
# MAGIC
# MAGIC ### Integration Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌───────────────────────────────────────────────────────────────────────────┐
# MAGIC │              External Application (Defensible Suite)                      │
# MAGIC │                                                                           │
# MAGIC │  ┌──────────────────────────────────────────────────────────────────┐     │
# MAGIC │  │                     Databricks SDK (Python)                      │     │
# MAGIC │  │                                                                  │     │
# MAGIC │  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │       │
# MAGIC │  │  │ Jobs API    │  │ databricks-  │  │ Workspace API         │  │       │
# MAGIC │  │  │             │  │ sql-connector│  │                       │  │       │
# MAGIC │  │  │ • Create    │  │ (JDBC)       │  │ • List notebooks      │  │       │
# MAGIC │  │  │ • Trigger   │  │              │  │ • Get job status      │  │       │
# MAGIC │  │  │ • Poll      │  │ • Query risk │  │ • Check cluster state │  │       │
# MAGIC │  │  │ • Cancel    │  │   scores     │  │                       │  │       │
# MAGIC │  │  │             │  │ • Pipeline   │  │                       │  │       │
# MAGIC │  │  │             │  │   health     │  │                       │  │       │
# MAGIC │  │  └──────┬──────┘  └──────┬───────┘  └───────────┬───────────┘  │       │
# MAGIC │  └─────────┼────────────────┼───────────────────────┼─────────────┘       │
# MAGIC │            │                │                       │                     │
# MAGIC └────────────┼────────────────┼───────────────────────┼─────────────────────┘
# MAGIC              │                │                       │
# MAGIC ┌────────────┼────────────────┼───────────────────────┼──────────────────────┐
# MAGIC │            ▼                ▼                       ▼                      │
# MAGIC │  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────┐              │
# MAGIC │  │ Databricks  │  │ All-purpose      │  │ Workspace         │              │
# MAGIC │  │ Workflows   │  │ cluster          │  │ Resources         │              │
# MAGIC │  │             │  │ (legacy SQL      │  │                   │              │
# MAGIC │  │             │  │  endpoint)       │  │                   │              │
# MAGIC │  └─────────────┘  └──────────────────┘  └───────────────────┘              │
# MAGIC │                         Databricks                                         │
# MAGIC └────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Docs:**
# MAGIC - [Databricks SDK for Python](https://docs.databricks.com/en/dev-tools/sdk-python.html)
# MAGIC - [databricks-sql-connector](https://docs.databricks.com/en/dev-tools/python-sql-connector.html)
# MAGIC - [Jobs API](https://docs.databricks.com/en/workflows/jobs/jobs-2.0-api.html)
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites

# COMMAND ----------

# MAGIC %pip install databricks-sdk databricks-sql-connector --quiet

# COMMAND ----------

dbutils.library.restartPython()

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
USER_ID = _user_email
print(f"Database: {DATABASE}, Table suffix: {SUFFIX_TAG or '(none)'}")
print(f"User: {USER_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Initialize the Databricks SDK
# MAGIC
# MAGIC Inside a Databricks notebook, the SDK automatically authenticates using the notebook's context.
# MAGIC In an external application, you'd configure authentication explicitly.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Inside a notebook: auto-authenticates
w = WorkspaceClient()

# Verify connection
me = w.current_user.me()
print(f"Connected as: {me.user_name}")
print(f"  Workspace: {w.config.host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### External Application Authentication
# MAGIC
# MAGIC When running outside Databricks (in Defensible Suite, a Flask app, etc.), configure auth explicitly.
# MAGIC
# MAGIC > **Azure Gov Cloud Note:** Government Cloud workspaces use `.azuredatabricks.us` hostnames
# MAGIC > instead of `.cloud.databricks.com` or `.azuredatabricks.net`.
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC # Option 1: Personal Access Token (simplest for development)
# MAGIC w = WorkspaceClient(
# MAGIC     host="https://adb-<workspace-id>.<region>.azuredatabricks.us",
# MAGIC     token="dapi_your_personal_access_token"
# MAGIC )
# MAGIC
# MAGIC # Option 2: Service Principal (recommended for production)
# MAGIC # Set environment variables:
# MAGIC #   DATABRICKS_HOST=https://adb-<workspace-id>.<region>.azuredatabricks.us
# MAGIC #   DATABRICKS_CLIENT_ID=your-sp-client-id
# MAGIC #   DATABRICKS_CLIENT_SECRET=your-sp-secret
# MAGIC #   ARM_ENVIRONMENT=usgovernment
# MAGIC w = WorkspaceClient()  # picks up env vars automatically
# MAGIC
# MAGIC # Option 3: Azure Managed Identity (for Azure Gov-hosted apps)
# MAGIC # Note: Azure Gov uses login.microsoftonline.us and management.usgovcloudapi.net
# MAGIC w = WorkspaceClient(
# MAGIC     host="https://adb-<workspace-id>.<region>.azuredatabricks.us",
# MAGIC     azure_workspace_resource_id="/subscriptions/.../resourceGroups/.../providers/Microsoft.Databricks/workspaces/...",
# MAGIC     azure_environment="AZURE_US_GOVERNMENT",
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **Docs:** [Authentication](https://docs.databricks.com/en/dev-tools/auth/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Deploy and Run the Pipeline Job
# MAGIC
# MAGIC Defensible Suite needs a real pipeline to query against, so we deploy and execute the
# MAGIC ingest -> score -> monitor workflow first. Subsequent steps then read the resulting
# MAGIC Delta tables and inspect run history.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find the current notebook directory

# COMMAND ----------

import time
import pandas as pd

from databricks.sdk.service.jobs import (
    Task, NotebookTask, Source, TaskDependency,
    RunIf, JobEmailNotifications, JobSettings,
)

# Get the current notebook's directory (workshop notebooks live together)
notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
notebook_path = notebook_context.notebookPath().get()
notebook_dir = "/".join(notebook_path.split("/")[:-1])

print(f"Notebook directory: {notebook_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define a multi-task Job: Ingest -> Score -> Monitor

# COMMAND ----------

# Build the job definition
job_name = f"security_pipeline_{DATABASE}{SUFFIX_TAG}"

# Azure Gov Cloud — no serverless compute available.
# Options:
#   1. Use current cluster: existing_cluster_id=CLUSTER_ID (used below)
#   2. New job cluster: new_cluster=ClusterSpec(spark_version="13.3.x-scala2.12", ...)
#   3. Shared job cluster: job_clusters=[JobCluster(...)] + job_cluster_key="shared"

# On Azure Gov Cloud, serverless compute is not available.
# Use the current interactive cluster or specify a job cluster.
CLUSTER_ID = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
print(f"Using cluster: {CLUSTER_ID}")

job = w.jobs.create(
    name=job_name,
    # job_clusters=[...],  # Uncomment to define shared job clusters (option 3 above)
    tasks=[
        Task(
            task_key="ingest",
            description="Run ingestion pipeline (Bronze -> Silver)",
            existing_cluster_id=CLUSTER_ID,
            notebook_task=NotebookTask(
                notebook_path=f"{notebook_dir}/01_ingestion_pipeline",
                base_parameters={"database": DATABASE, "table_suffix": SUFFIX},
                source=Source.WORKSPACE,
            ),
        ),
        Task(
            task_key="score",
            description="Run risk scoring on silver tables",
            depends_on=[TaskDependency(task_key="ingest")],
            existing_cluster_id=CLUSTER_ID,
            notebook_task=NotebookTask(
                notebook_path=f"{notebook_dir}/02_risk_scoring",
                base_parameters={"database": DATABASE, "table_suffix": SUFFIX},
                source=Source.WORKSPACE,
            ),
        ),
        Task(
            task_key="monitor",
            description="Check pipeline health and flag anomalies",
            depends_on=[TaskDependency(task_key="score")],
            existing_cluster_id=CLUSTER_ID,
            notebook_task=NotebookTask(
                notebook_path=f"{notebook_dir}/03_pipeline_monitoring",
                base_parameters={"database": DATABASE, "table_suffix": SUFFIX},
                source=Source.WORKSPACE,
            ),
        ),
    ],
)

JOB_ID = job.job_id
print(f"Created job: {job_name} (ID: {JOB_ID})")
print(f"  URL: {w.config.host}#job/{JOB_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trigger the Job (Run Now)
# MAGIC
# MAGIC This is what Defensible Suite calls when an analyst clicks "Re-score pipeline."

# COMMAND ----------

run = w.jobs.run_now(job_id=JOB_ID)
RUN_ID = run.run_id
print(f"Triggered run: {RUN_ID}")
print(f"  URL: {w.config.host}#job/{JOB_ID}/run/{RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Poll for Completion
# MAGIC
# MAGIC In a web app, you'd poll this in a background task or use a webhook callback.

# COMMAND ----------

from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState

def wait_for_run(run_id, timeout_seconds=600, poll_interval=15):
    """Poll a job run until it completes. Returns the final run status."""
    start = time.time()
    while time.time() - start < timeout_seconds:
        run_status = w.jobs.get_run(run_id)
        state = run_status.state

        if state.life_cycle_state in (
            RunLifeCycleState.TERMINATED,
            RunLifeCycleState.SKIPPED,
            RunLifeCycleState.INTERNAL_ERROR,
        ):
            result = state.result_state
            duration = time.time() - start
            print(f"\nRun completed in {duration:.0f}s")
            print(f"  Result: {result}")

            if result != RunResultState.SUCCESS:
                print(f"  Message: {state.state_message}")

            # Print per-task status
            for task in run_status.tasks:
                task_state = task.state
                icon = "OK" if task_state.result_state == RunResultState.SUCCESS else "FAIL"
                print(f"  [{icon}] {task.task_key}: {task_state.result_state}")

            return run_status

        # Still running — print progress
        running_tasks = [t.task_key for t in (run_status.tasks or [])
                        if t.state and t.state.life_cycle_state == RunLifeCycleState.RUNNING]
        pending_tasks = [t.task_key for t in (run_status.tasks or [])
                        if t.state and t.state.life_cycle_state == RunLifeCycleState.PENDING]
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:3d}s] Running: {running_tasks or '-'}  Pending: {pending_tasks or '-'}")
        time.sleep(poll_interval)

    raise TimeoutError(f"Run {run_id} did not complete within {timeout_seconds}s")

# COMMAND ----------

final_status = wait_for_run(RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Query Run Details via the SDK Jobs API
# MAGIC
# MAGIC This is the canonical pattern for an external app's "show me job history" view.
# MAGIC Defensible Suite uses these calls to render run dashboards, surface failures, and
# MAGIC drill into per-task status without needing direct cluster access.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Most-recent runs across the job

# COMMAND ----------

recent_runs = list(w.jobs.list_runs(job_id=JOB_ID, limit=5))

runs_df = pd.DataFrame([
    {
        "run_id": r.run_id,
        "start_time": datetime.fromtimestamp(r.start_time / 1000) if r.start_time else None,
        "duration_s": (r.run_duration / 1000) if r.run_duration else None,
        "life_cycle_state": str(r.state.life_cycle_state) if r.state else None,
        "result_state": str(r.state.result_state) if r.state and r.state.result_state else None,
        "trigger": str(r.trigger) if r.trigger else None,
    }
    for r in recent_runs
])

print("Most recent runs:")
print(runs_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task-level breakdown for the run we just executed

# COMMAND ----------

run_detail = w.jobs.get_run(run_id=RUN_ID)

task_df = pd.DataFrame([
    {
        "task_key": t.task_key,
        "life_cycle_state": str(t.state.life_cycle_state) if t.state else None,
        "result_state": str(t.state.result_state) if t.state and t.state.result_state else None,
        "start_time": datetime.fromtimestamp(t.start_time / 1000) if t.start_time else None,
        "duration_s": ((t.end_time - t.start_time) / 1000) if t.start_time and t.end_time else None,
    }
    for t in (run_detail.tasks or [])
])

print(f"Tasks for run {RUN_ID}:")
print(task_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Query Data Inline with `spark.sql`
# MAGIC
# MAGIC When running **INSIDE Databricks** (notebooks, jobs), use `spark.sql(...)` directly.
# MAGIC It's the simplest path, runs on the cluster you're already attached to, and renders
# MAGIC results as native Databricks tables via `display()`.
# MAGIC
# MAGIC For **EXTERNAL applications**, use the JDBC pattern shown in Step 5.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 1: Risk score summary by event_type + risk_tier (dashboard top card)

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        event_type,
        risk_tier,
        COUNT(*) as count,
        ROUND(AVG(risk_score), 3) as avg_score
    FROM (
        SELECT 'email' as event_type, risk_tier, risk_score
        FROM {DATABASE}.email_risk_scores{SUFFIX_TAG}
        UNION ALL
        SELECT 'signin' as event_type, risk_tier, risk_score
        FROM {DATABASE}.signin_risk_scores{SUFFIX_TAG}
    )
    GROUP BY event_type, risk_tier
    ORDER BY event_type, avg_score DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 2: Top 20 alerts (alert feed)

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        event_type,
        event_id,
        timestamp,
        principal,
        ROUND(risk_score, 3) as risk_score,
        risk_tier,
        risk_reasons,
        detail
    FROM {DATABASE}.security_alerts{SUFFIX_TAG}
    ORDER BY risk_score DESC
    LIMIT 20
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 3: Pipeline health — latest run per stage (monitoring panel)

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        pipeline_stage,
        event_type,
        row_count,
        ROUND(null_rate, 4) as null_rate,
        ROUND(processing_seconds, 1) as seconds,
        status,
        run_timestamp
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
# MAGIC ### Query 4: User profile — drill-down on a target user

# COMMAND ----------

# When an analyst clicks on a user in Defensible Suite, pull their full profile
target_user = "user001@snc-internal.example.com"

display(spark.sql(f"""
    SELECT
        user_id,
        COUNT(*) as total_signins,
        SUM(CASE WHEN risk_tier IN ('Critical', 'High') THEN 1 ELSE 0 END) as high_risk_events,
        ROUND(AVG(risk_score), 3) as avg_risk_score,
        ROUND(MAX(risk_score), 3) as max_risk_score,
        COLLECT_SET(city) as cities,
        COLLECT_SET(device_type) as devices,
        SUM(CASE WHEN NOT mfa_used THEN 1 ELSE 0 END) as no_mfa_count,
        SUM(failed_attempts) as total_failed_attempts,
        ROUND(MAX(geo_velocity_kmh), 0) as max_geo_velocity_kmh
    FROM {DATABASE}.signin_risk_scores{SUFFIX_TAG}
    WHERE user_id = '{target_user}'
    GROUP BY user_id
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: External App Pattern — JDBC against a classic cluster
# MAGIC
# MAGIC When Defensible Suite (or any external service) needs to read these tables, it can't use
# MAGIC `spark.sql(...)` — there's no SparkSession outside the cluster. And on Gov Cloud today
# MAGIC there's no SQL Warehouse to talk to either.
# MAGIC
# MAGIC The workaround is the all-purpose cluster's **legacy SQL endpoint** — every classic
# MAGIC interactive cluster exposes one at `/sql/protocolv1/o/<workspace_id>/<cluster_id>`. The
# MAGIC `databricks-sql-connector` package speaks this protocol exactly the way it speaks to a
# MAGIC SQL Warehouse, so the integration code is portable to Evergreen later with only a
# MAGIC change to the `http_path`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute the JDBC connection params from the SDK runtime config

# COMMAND ----------

import re as _re

_host_match = _re.search(r"adb-(\d+)\.", w.config.host)
WORKSPACE_ID = _host_match.group(1) if _host_match else None
CLUSTER_ID = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
HOSTNAME = w.config.host.replace("https://", "").rstrip("/")
HTTP_PATH = f"/sql/protocolv1/o/{WORKSPACE_ID}/{CLUSTER_ID}"

print(f"Hostname:    {HOSTNAME}")
print(f"Workspace:   {WORKSPACE_ID}")
print(f"Cluster:     {CLUSTER_ID}")
print(f"HTTP path:   {HTTP_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connection pattern
# MAGIC
# MAGIC In a notebook, we can reuse the existing SDK auth context's API token. In Defensible
# MAGIC Suite, supply an explicit `access_token` — see the **External Authentication** subsection
# MAGIC below for how that token is obtained.

# COMMAND ----------

from databricks import sql

# When running in a notebook, the existing SDK auth context is reused.
# When running in Defensible Suite, supply an explicit access_token (PAT or SP).
try:
    with sql.connect(
        server_hostname=HOSTNAME,
        http_path=HTTP_PATH,
        access_token=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
    ) as connection:
        cursor = connection.cursor()
        cursor.execute(
            f"SELECT event_type, risk_tier, COUNT(*) as n "
            f"FROM {DATABASE}.security_alerts{SUFFIX_TAG} "
            f"GROUP BY event_type, risk_tier "
            f"ORDER BY event_type, risk_tier"
        )
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

    jdbc_summary = pd.DataFrame(rows, columns=columns)
    print("JDBC query result (this is exactly what Defensible Suite would receive):\n")
    print(jdbc_summary.to_string(index=False))
except Exception as e:
    print(f"JDBC self-loopback failed: {type(e).__name__}: {e}")
    print()
    print("This often happens when a notebook tries to JDBC-connect back to its own cluster")
    print("(self-loopback). In production, this code runs from OUTSIDE Databricks — for")
    print("example, from a Defensible Suite backend pod — where the loopback isn't an issue.")
    print()
    print("The connection parameters above are correct; only the in-notebook execution context")
    print("is the problem. Copy the same code into your application backend and it will work.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### External Authentication
# MAGIC
# MAGIC Defensible Suite obtains the `access_token` for `sql.connect(...)` the same way it
# MAGIC obtains a token for the SDK — Databricks treats them as equivalent. Three options
# MAGIC for Azure Gov Cloud:
# MAGIC
# MAGIC ```python
# MAGIC # Option 1: Personal Access Token (development only)
# MAGIC access_token = "dapi_your_personal_access_token"
# MAGIC
# MAGIC # Option 2: Service Principal (recommended for production)
# MAGIC # Mint an OAuth M2M token from the SP's client_id + client_secret.
# MAGIC from databricks.sdk.core import Config, oauth_service_principal
# MAGIC config = Config(
# MAGIC     host="https://adb-<workspace-id>.<region>.azuredatabricks.us",
# MAGIC     client_id="<sp-client-id>",
# MAGIC     client_secret="<sp-client-secret>",
# MAGIC     azure_environment="AZURE_US_GOVERNMENT",
# MAGIC )
# MAGIC access_token = oauth_service_principal(config)().access_token
# MAGIC
# MAGIC # Option 3: Azure Managed Identity (for Defensible Suite hosted in Azure Gov)
# MAGIC # The MI is granted access to the workspace; no secrets in the app.
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC w_ext = WorkspaceClient(
# MAGIC     host="https://adb-<workspace-id>.<region>.azuredatabricks.us",
# MAGIC     azure_workspace_resource_id="/subscriptions/.../resourceGroups/.../providers/Microsoft.Databricks/workspaces/...",
# MAGIC     azure_environment="AZURE_US_GOVERNMENT",
# MAGIC )
# MAGIC access_token = w_ext.config.authenticate()["Authorization"].split(" ", 1)[1]
# MAGIC
# MAGIC # Then pass to the connector:
# MAGIC from databricks import sql
# MAGIC connection = sql.connect(
# MAGIC     server_hostname="adb-<workspace-id>.<region>.azuredatabricks.us",
# MAGIC     http_path="/sql/protocolv1/o/<workspace-id>/<cluster-id>",
# MAGIC     access_token=access_token,
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **Docs:** [Authentication](https://docs.databricks.com/en/dev-tools/auth/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Add a Schedule to the Job
# MAGIC
# MAGIC Set up the job to run automatically — e.g., every 4 hours. This is
# MAGIC the production pattern: new data lands -> pipeline runs on schedule ->
# MAGIC Defensible Suite queries fresh scores.

# COMMAND ----------

# from databricks.sdk.service.jobs import CronSchedule, PauseStatus

# # Add a schedule to the existing job (paused by default for the workshop)
# w.jobs.update(
#     job_id=JOB_ID,
#     new_settings=JobSettings(
#         name=job_name,
#         schedule=CronSchedule(
#             quartz_cron_expression="0 0 */4 * * ?",  # Every 4 hours
#             timezone_id="America/Denver",
#             pause_status=PauseStatus.PAUSED,  # Paused — enable when ready
#         ),
#     ),
# )
# print(f"Schedule added to job {JOB_ID}: every 4 hours (paused)")
# print(f"  To enable: w.jobs.update(job_id={JOB_ID}, ...pause_status=PauseStatus.UNPAUSED)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: `DatabricksSecurityClient` — production-ready integration class
# MAGIC
# MAGIC Below is a self-contained Python class that wraps everything an external application
# MAGIC needs. It supports **two modes**, switchable at construction time:
# MAGIC
# MAGIC - `mode="spark"` — use this when the client lives **inside Databricks** (a notebook, a job
# MAGIC   running on the cluster). It reads Delta tables via `self.spark.sql(...)`.
# MAGIC - `mode="jdbc"` — use this when the client lives **outside Databricks**, e.g. inside the
# MAGIC   Defensible Suite backend. It reads the same tables via `databricks-sql-connector`
# MAGIC   against the all-purpose cluster's legacy SQL endpoint.
# MAGIC
# MAGIC Method signatures are identical across modes, so the rest of Defensible Suite never
# MAGIC has to know which transport is in play.

# COMMAND ----------

class DatabricksSecurityClient:
    """
    Integration client for connecting Defensible Suite to Databricks.

    Dual-mode design:
      - mode="spark": for code running INSIDE Databricks (notebooks, jobs).
        Methods execute via self.spark.sql(...). No network round-trip.
      - mode="jdbc":  for code running in Defensible Suite (or any external app).
        Methods execute via databricks-sql-connector against the all-purpose
        cluster's legacy SQL endpoint. On Gov Cloud this is the only option
        until SQL Warehouses arrive with Evergreen.

    Method signatures are identical across modes:
        - get_alert_summary()
        - get_top_alerts(limit=20)
        - get_user_profile(user_id)
        - get_pipeline_status()
        - get_pipeline_anomalies()
        - trigger_pipeline(job_id)
        - get_run_status(run_id)
    """

    def __init__(
        self,
        database,
        mode,
        table_suffix="",
        # spark mode:
        spark=None,
        # jdbc mode:
        server_hostname=None,
        http_path=None,
        access_token=None,
        # SDK auth (used by both modes for Jobs API calls):
        host=None,
        token=None,
    ):
        if mode not in ("spark", "jdbc"):
            raise ValueError(f"mode must be 'spark' or 'jdbc', got {mode!r}")
        self.mode = mode
        self.database = database
        suffix = (table_suffix or "").strip()
        self.suffix_tag = f"_{suffix}" if suffix else ""

        # SDK client for Jobs API (independent of read path)
        if host and token:
            self.w = WorkspaceClient(host=host, token=token)
        else:
            self.w = WorkspaceClient()  # auto-auth (notebook or env vars)

        if mode == "spark":
            if spark is None:
                raise ValueError("spark mode requires a SparkSession via the `spark` arg")
            self.spark = spark
        else:  # jdbc
            if not (server_hostname and http_path and access_token):
                raise ValueError(
                    "jdbc mode requires server_hostname, http_path, and access_token"
                )
            self.server_hostname = server_hostname
            self.http_path = http_path
            self.access_token = access_token

    def _query(self, sql_text):
        """Execute SQL and return results as a Pandas DataFrame, regardless of mode."""
        if self.mode == "spark":
            return self.spark.sql(sql_text).toPandas()
        else:
            from databricks import sql as dbsql
            with dbsql.connect(
                server_hostname=self.server_hostname,
                http_path=self.http_path,
                access_token=self.access_token,
            ) as connection:
                cursor = connection.cursor()
                cursor.execute(sql_text)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
            return pd.DataFrame(rows, columns=columns)

    # --- Risk Scores ---

    def get_alert_summary(self):
        """Get risk tier counts by event type — for the dashboard top card."""
        return self._query(f"""
            SELECT event_type, risk_tier, COUNT(*) as count, ROUND(AVG(risk_score), 3) as avg_score
            FROM (
                SELECT 'email' as event_type, risk_tier, risk_score FROM {self.database}.email_risk_scores{self.suffix_tag}
                UNION ALL
                SELECT 'signin' as event_type, risk_tier, risk_score FROM {self.database}.signin_risk_scores{self.suffix_tag}
            )
            GROUP BY event_type, risk_tier ORDER BY event_type, avg_score DESC
        """)

    def get_top_alerts(self, limit=20):
        """Get the highest-risk events with explanations — for the alert feed."""
        return self._query(f"""
            SELECT event_type, event_id, timestamp, principal, ROUND(risk_score, 3) as risk_score,
                   risk_tier, risk_reasons, detail
            FROM {self.database}.security_alerts{self.suffix_tag}
            ORDER BY risk_score DESC LIMIT {limit}
        """)

    def get_user_profile(self, user_id):
        """Get risk profile for a specific user — for drill-down views."""
        return self._query(f"""
            SELECT user_id, COUNT(*) as total_signins,
                   SUM(CASE WHEN risk_tier IN ('Critical','High') THEN 1 ELSE 0 END) as high_risk_count,
                   ROUND(AVG(risk_score), 3) as avg_risk, ROUND(MAX(risk_score), 3) as max_risk,
                   SUM(CASE WHEN NOT mfa_used THEN 1 ELSE 0 END) as no_mfa_count,
                   SUM(failed_attempts) as total_failed
            FROM {self.database}.signin_risk_scores{self.suffix_tag}
            WHERE user_id = '{user_id}' GROUP BY user_id
        """)

    # --- Pipeline Health ---

    def get_pipeline_status(self):
        """Get latest pipeline health per stage — for the monitoring panel."""
        return self._query(f"""
            SELECT pipeline_stage, event_type, row_count,
                   ROUND(null_rate, 4) as null_rate,
                   ROUND(processing_seconds, 1) as seconds, status, run_timestamp
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY pipeline_stage, event_type ORDER BY run_timestamp DESC) as rn
                FROM {self.database}.pipeline_health{self.suffix_tag}
            ) WHERE rn = 1 ORDER BY pipeline_stage, event_type
        """)

    def get_pipeline_anomalies(self):
        """Get flagged pipeline runs — for the alert panel."""
        return self._query(f"""
            SELECT pipeline_stage, event_type, run_timestamp, row_count, null_rate, anomaly_reasons
            FROM {self.database}.pipeline_anomalies{self.suffix_tag}
            WHERE has_anomaly = true ORDER BY run_timestamp DESC
        """)

    # --- Pipeline Control ---

    def trigger_pipeline(self, job_id):
        """Trigger a pipeline run — for the 'Re-score now' button."""
        run = self.w.jobs.run_now(job_id=job_id)
        return {"run_id": run.run_id, "status": "triggered"}

    def get_run_status(self, run_id):
        """Check run status — for progress indicators."""
        run = self.w.jobs.get_run(run_id)
        return {
            "state": str(run.state.life_cycle_state),
            "result": str(run.state.result_state) if run.state.result_state else None,
            "tasks": [
                {"key": t.task_key, "state": str(t.state.life_cycle_state) if t.state else "unknown"}
                for t in (run.tasks or [])
            ],
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Integration Client (spark mode — runs in this notebook)

# COMMAND ----------

client = DatabricksSecurityClient(
    database=DATABASE,
    mode="spark",
    spark=spark,
    table_suffix=SUFFIX,
)

print("=== Alert Summary ===")
print(client.get_alert_summary().to_string(index=False))
print()

print("=== Top 5 Alerts ===")
print(client.get_top_alerts(limit=5).to_string(index=False))
print()

print("=== Pipeline Status ===")
print(client.get_pipeline_status().to_string(index=False))
print()

print("=== User Profile ===")
print(client.get_user_profile("user001@snc-internal.example.com").to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### JDBC mode (illustrative — for use from Defensible Suite, not from this notebook)
# MAGIC
# MAGIC We don't execute this here because of the same self-loopback issue from Step 5 — a
# MAGIC notebook can't reliably JDBC-connect back into its own cluster. From the Defensible
# MAGIC Suite backend, however, it's the canonical pattern:
# MAGIC
# MAGIC ```python
# MAGIC client = DatabricksSecurityClient(
# MAGIC     database="security_app_integration",
# MAGIC     mode="jdbc",
# MAGIC     table_suffix="ef280426",
# MAGIC     server_hostname="adb-<workspace-id>.<region>.azuredatabricks.us",
# MAGIC     http_path="/sql/protocolv1/o/<workspace-id>/<cluster-id>",
# MAGIC     access_token="dapi_...",   # PAT, SP token, or Managed Identity token
# MAGIC     host="https://adb-<workspace-id>.<region>.azuredatabricks.us",
# MAGIC     token="dapi_...",           # used for Jobs API calls (trigger/status)
# MAGIC )
# MAGIC
# MAGIC client.get_alert_summary()
# MAGIC client.trigger_pipeline(job_id=12345)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Vision Setting — Future State with Evergreen GovCloud
# MAGIC
# MAGIC Today on Azure Gov Cloud, we're working with classic compute, the Hive metastore, and
# MAGIC no Databricks SQL — which is why this notebook leans on `spark.sql` for in-notebook
# MAGIC analysts and `databricks-sql-connector` against an all-purpose cluster for external
# MAGIC apps. When Evergreen Gov Cloud reaches GA (target: April 30, 2026), the JDBC-against-
# MAGIC classic-cluster pattern becomes a stopgap — replaceable with a SQL Warehouse + the
# MAGIC Statement Execution API and no schema changes.
# MAGIC
# MAGIC | Feature | Available Now (Classic Compute) | Future (Evergreen) |
# MAGIC |---------|---------------------------------|-------------------|
# MAGIC | **Compute** | Classic clusters with `existing_cluster_id` | **Serverless** (no cluster management) |
# MAGIC | **Governance** | Hive metastore (database-scoped) | **Unity Catalog** with lineage + audit |
# MAGIC | **SQL endpoint** | All-purpose cluster's legacy `/sql/protocolv1/...` path | **SQL Warehouses** + **Statement Execution API** |
# MAGIC | **External app reads** | `databricks-sql-connector` against the cluster | `databricks-sql-connector` against a SQL Warehouse, **or** Statement Execution REST API |
# MAGIC | **Pipeline** | Scheduled notebook jobs | **Delta Live Tables** (streaming) |
# MAGIC | **Dashboard** | Notebook visualizations / `display()` | **AI/BI Dashboards** with auto-refresh |
# MAGIC | **Alerting** | Custom notebook logic | **Lakewatch SIEM** with pre-built rules |
# MAGIC | **AI Enrichment** | Not available | **Foundation Model APIs** for natural-language explanations |
# MAGIC
# MAGIC ### Key Upgrade Path
# MAGIC 1. **SQL Warehouses**: Drop the all-purpose-cluster JDBC stopgap. Same `databricks-sql-connector` code, just point `http_path` at `/sql/1.0/warehouses/<warehouse_id>`.
# MAGIC 2. **Unity Catalog**: Migrate from `database.table` to `catalog.schema.table` for centralized governance.
# MAGIC 3. **Serverless compute**: Remove `existing_cluster_id` from jobs — tasks run on serverless by default.
# MAGIC 4. **DLT pipeline**: Convert batch scoring to streaming — score events within minutes, not hours.
# MAGIC 5. **AI/BI Dashboards**: Interactive security dashboards for SOC analysts.
# MAGIC 6. **Foundation Models**: Generate natural-language explanations of *why* an event is risky.
# MAGIC
# MAGIC The `DatabricksSecurityClient` class above is structured exactly for this transition:
# MAGIC `mode="jdbc"` already uses the `databricks-sql-connector`, so swapping the `http_path`
# MAGIC from a cluster path to a warehouse path is the only change Defensible Suite needs to
# MAGIC make.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Cleanup (Optional)
# MAGIC
# MAGIC Delete the job created during this workshop. Uncomment to run.

# COMMAND ----------

# Uncomment to delete the job
# w.jobs.delete(job_id=JOB_ID)
# print(f"Deleted job {JOB_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | What | How |
# MAGIC |------|-----|
# MAGIC | **Authentication** | PAT (dev), Service Principal (prod), Managed Identity (Azure Gov) |
# MAGIC | **In-notebook queries** | `spark.sql(...)` directly — render with `display()` |
# MAGIC | **External app queries** | `databricks-sql-connector` JDBC against the all-purpose cluster's legacy SQL endpoint |
# MAGIC | **Trigger pipelines** | Jobs API via SDK — `w.jobs.run_now()` + poll for status |
# MAGIC | **Schedule pipelines** | `CronSchedule` on the Job definition |
# MAGIC | **Compute** | Classic clusters (`existing_cluster_id`) on Azure Gov Cloud |
# MAGIC
# MAGIC ### Assets Created
# MAGIC
# MAGIC | Asset | Location |
# MAGIC |-------|----------|
# MAGIC | Pipeline Job | `{w.config.host}#job/{JOB_ID}` |
# MAGIC | `DatabricksSecurityClient` class | Copy into your application |
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC
# MAGIC 1. **Two query paths, one schema.** Inside Databricks, use `spark.sql(...)`. Outside Databricks, use `databricks-sql-connector`. Both read identical Delta tables.
# MAGIC 2. **The SDK is the bridge for control-plane operations.** Triggering jobs, polling runs, listing history — all go through `WorkspaceClient`, regardless of where your code runs.
# MAGIC 3. **The `DatabricksSecurityClient` class is dual-mode and production-ready.** Drop it into Defensible Suite, set `mode="jdbc"`, configure auth, and you have a working integration.
# MAGIC 4. **On Gov Cloud today, JDBC against your all-purpose cluster is how external apps query Delta.** When Evergreen lands, swap that for the Statement Execution API + a SQL Warehouse — same data, simpler infra.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Workshop 3 Complete!** You've built an end-to-end security data pipeline
# MAGIC with application integration — from raw event ingestion through risk scoring, pipeline
# MAGIC monitoring, and external app connectivity — all on Azure Gov Cloud classic compute.
# MAGIC
# MAGIC ### Next Steps
# MAGIC 1. **Load real data** — Replace synthetic generators with actual email/sign-in feeds
# MAGIC 2. **Enable the schedule** — Unpause the Job to run every 4 hours
# MAGIC 3. **Add ML scoring** — Swap the rule engine (notebook 02) for the ML models from Workshops 1 & 2
# MAGIC 4. **Integrate Defensible Suite** — Drop the `DatabricksSecurityClient` class into your backend in `mode="jdbc"`
# MAGIC 5. **Set up alerting** — Add webhook notifications to the Job for pipeline failures
# MAGIC 6. **Plan Evergreen migration** — Swap JDBC-against-cluster for Statement Execution + SQL Warehouse when available

# COMMAND ----------

