# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2: Anomaly Detection for Sign-In Data
# MAGIC ## Notebook 7 — Composite Batch Scoring Pipeline
# MAGIC
# MAGIC Combine **all three detection layers** — global Isolation Forest, PyOD ensemble, and per-user models —
# MAGIC into a single production scoring pipeline with SHAP explanations.
# MAGIC
# MAGIC ### Production Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌───────────────────────────────────────────────────────────────────────┐
# MAGIC │            Composite Batch Scoring Pipeline (Azure Gov Cloud)         │
# MAGIC │                                                                       │
# MAGIC │  ┌──────────────┐                                                     │
# MAGIC │  │ Sign-In Logs │  (new events since last run)                        │
# MAGIC │  │ Delta Table   │                                                    │
# MAGIC │  └──────┬───────┘                                                     │
# MAGIC │         │                                                             │
# MAGIC │         ▼                                                             │
# MAGIC │  ┌──────────────┐                                                     │
# MAGIC │  │   Feature    │                                                     │
# MAGIC │  │  Engineering │                                                     │
# MAGIC │  └──────┬───────┘                                                     │
# MAGIC │         │                                                             │
# MAGIC │         ├───────────────┬───────────────┐                             │
# MAGIC │         ▼               ▼               ▼                             │
# MAGIC │  ┌────────────┐  ┌────────────┐  ┌─────────────────┐                  │
# MAGIC │  │  Global    │  │  PyOD      │  │  Per-User       │                  │
# MAGIC │  │  IForest   │  │  Ensemble  │  │  IForest + SHAP │                  │
# MAGIC │  │  (nb 03)   │  │  (nb 04)   │  │  (nb 05)        │                  │
# MAGIC │  └─────┬──────┘  └─────┬──────┘  └────────┬────────┘                  │
# MAGIC │        │               │                   │                          │
# MAGIC │        └───────────────┴───────────────────┘                          │
# MAGIC │                        │                                              │
# MAGIC │                ┌───────▼───────┐                                      │
# MAGIC │                │  Composite    │                                      │
# MAGIC │                │  Risk Score   │                                      │
# MAGIC │                │  + SHAP Why   │                                      │
# MAGIC │                └───────┬───────┘                                      │
# MAGIC │                        │                                              │
# MAGIC │         ┌──────────────┼──────────────┐                               │
# MAGIC │         ▼              ▼              ▼                               │
# MAGIC │  ┌────────────┐ ┌──────────┐  ┌──────────────┐                        │
# MAGIC │  │ Delta Table│ │ Risk     │  │ Alert on     │                        │
# MAGIC │  │ (scores)   │ │ Dashboard│  │ High/Crit    │                        │
# MAGIC │  └────────────┘ └──────────┘  └──────────────┘                        │
# MAGIC │                                                                       │
# MAGIC │  Scheduled: every 1-4 hours via Databricks Jobs                       │
# MAGIC │  No Model Serving required — pure batch on classic compute            │
# MAGIC └───────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Docs:**
# MAGIC - [MLflow Model Registry](https://docs.databricks.com/en/mlflow/index.html)
# MAGIC - [Databricks Jobs](https://docs.databricks.com/en/workflows/index.html)
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
print(f"Database: {DATABASE}, Table suffix: {SUFFIX_TAG or '(none)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load All Model Artifacts
# MAGIC
# MAGIC We load the global Isolation Forest (notebook 03), the PyOD detectors will be re-instantiated,
# MAGIC and per-user models will be trained inline via `applyInPandas`.

# COMMAND ----------

import pickle
import pandas as pd
import numpy as np

import os
import re
_user = spark.sql("SELECT current_user()").first()[0]
USER_ID = re.sub(r'[^a-zA-Z0-9]', '_', _user.split('@')[0])
artifact_path = f"/dbfs/tmp/workshops/{DATABASE}/{USER_ID}"
os.makedirs(artifact_path, exist_ok=True)

with open(f"{artifact_path}/model_features.pkl", "rb") as f:
    MODEL_FEATURES = pickle.load(f)
with open(f"{artifact_path}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(f"{artifact_path}/iforest_model.pkl", "rb") as f:
    iforest = pickle.load(f)

print(f"Loaded model artifacts. Features: {MODEL_FEATURES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Package Composite Scorer as MLflow PyFunc
# MAGIC
# MAGIC This model wraps all three detection layers:
# MAGIC 1. **Global Isolation Forest** — cross-user anomaly patterns
# MAGIC 2. **PyOD ECOD** — empirical cumulative distribution outlier detection (fast, no hyperparams)
# MAGIC 3. **Per-user Isolation Forest** — personalized baselines (via grouped scoring at call time)
# MAGIC
# MAGIC The composite score is a weighted average of all three, with SHAP explanations from the global model.

# COMMAND ----------

import sys
import os

# Clear stale mlflow module state if present (prevents circular import error)
mlflow_keys = [k for k in sys.modules if k == 'mlflow' or k.startswith('mlflow.')]
for k in mlflow_keys:
    del sys.modules[k]

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
# Register sklearn integration so DBR's MLflow autologging shim doesn't KeyError on fit_predict
mlflow.sklearn.autolog(disable=True)
from mlflow.models import infer_signature

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
mlflow.set_experiment(f"{os.path.dirname(notebook_path)}/login_anomaly_pipeline")

class CompositeAnomalyDetector(mlflow.pyfunc.PythonModel):
    """
    Composite anomaly detection combining global IForest, PyOD ECOD, and per-user scoring.

    Input: DataFrame with feature columns + user_id
    Output: DataFrame with composite risk_score, risk_tier, component scores, and SHAP top features
    """

    def load_context(self, context):
        import pickle
        with open(context.artifacts["iforest_model"], "rb") as f:
            self.iforest = pickle.load(f)
        with open(context.artifacts["scaler"], "rb") as f:
            self.scaler = pickle.load(f)
        with open(context.artifacts["model_features"], "rb") as f:
            self.model_features = pickle.load(f)
        with open(context.artifacts["known_locations"], "rb") as f:
            self.known_locations = pickle.load(f)
        with open(context.artifacts["known_devices"], "rb") as f:
            self.known_devices = pickle.load(f)

    def _engineer_features(self, df):
        import math
        import numpy as np

        result = df.copy()

        # Temporal features
        result["hour_sin"] = np.sin(2 * math.pi * result["hour"] / 24)
        result["hour_cos"] = np.cos(2 * math.pi * result["hour"] / 24)
        result["is_weekend"] = result["day_of_week"].isin([1, 7]).astype(int)
        result["is_off_hours"] = ((result["hour"] < 7) | (result["hour"] > 19)).astype(int)

        # Geographic
        result["is_unknown_location"] = (~result["location_name"].isin(self.known_locations)).astype(int)

        # Auth
        result["mfa_numeric"] = result.get("mfa_used", 1).astype(int)
        result["session_duration_zscore"] = 0  # Simplified for batch

        # Behavioral
        result["is_new_device"] = (~result["device"].isin(self.known_devices)).astype(int)
        result["logins_last_hour"] = result.get("logins_last_hour", 1)

        # Fill missing
        for col in self.model_features:
            if col not in result.columns:
                result[col] = 0
            result[col] = result[col].fillna(0)

        # Clip extremes
        if "geo_velocity_kmh" in result.columns:
            result["geo_velocity_kmh"] = result["geo_velocity_kmh"].clip(upper=50000)

        return result

    def predict(self, context, model_input):
        import numpy as np
        import shap
        from sklearn.preprocessing import MinMaxScaler
        from pyod.models.ecod import ECOD

        features = self._engineer_features(model_input)
        X = features[self.model_features].values
        X_scaled = self.scaler.transform(X)
        mm = MinMaxScaler()

        # --- Layer 1: Global Isolation Forest ---
        iforest_raw = self.iforest.decision_function(X_scaled)
        iforest_signal = mm.fit_transform((-iforest_raw).reshape(-1, 1)).flatten()

        # --- Layer 2: PyOD ECOD (fast, parameter-free) ---
        ecod = ECOD(contamination=0.05)
        ecod.fit(X_scaled)
        ecod_raw = ecod.decision_scores_
        ecod_signal = mm.fit_transform(ecod_raw.reshape(-1, 1)).flatten()

        # --- Layer 3: Per-user scoring ---
        # Train a mini IForest per user and score against their personal baseline
        from sklearn.ensemble import IsolationForest as IF

        per_user_signal = np.zeros(len(model_input))
        user_col = model_input.get("user_id", None)

        if user_col is not None:
            for user_id in user_col.unique():
                mask = (user_col == user_id).values
                X_user = X_scaled[mask]
                if len(X_user) >= 20:
                    contamination = min(0.1, max(0.01, 5 / len(X_user)))
                    user_if = IF(n_estimators=100, contamination=contamination, random_state=42)
                    user_if.fit(X_user)
                    user_raw = user_if.decision_function(X_user)
                    user_norm = mm.fit_transform((-user_raw).reshape(-1, 1)).flatten()
                    per_user_signal[mask] = user_norm

        # --- Composite score: weighted average ---
        composite = (
            0.35 * iforest_signal +
            0.30 * ecod_signal +
            0.35 * per_user_signal
        )
        composite = mm.fit_transform(composite.reshape(-1, 1)).flatten()

        # --- SHAP explanations (from global IForest) ---
        try:
            explainer = shap.TreeExplainer(self.iforest)
            shap_vals = explainer.shap_values(X_scaled)

            top1_feat, top1_val = [], []
            top2_feat, top2_val = [], []
            for i in range(len(model_input)):
                sorted_idx = np.argsort(np.abs(shap_vals[i]))[::-1]
                top1_feat.append(self.model_features[sorted_idx[0]])
                top1_val.append(float(shap_vals[i][sorted_idx[0]]))
                top2_feat.append(self.model_features[sorted_idx[1]] if len(sorted_idx) > 1 else "")
                top2_val.append(float(shap_vals[i][sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0)
        except Exception:
            top1_feat = [""] * len(model_input)
            top1_val = [0.0] * len(model_input)
            top2_feat = [""] * len(model_input)
            top2_val = [0.0] * len(model_input)

        # --- Risk tiers ---
        tiers = []
        for score in composite:
            if score >= 0.8:
                tiers.append("Critical")
            elif score >= 0.6:
                tiers.append("High")
            elif score >= 0.3:
                tiers.append("Medium")
            else:
                tiers.append("Low")

        return pd.DataFrame({
            "risk_score": composite,
            "risk_tier": tiers,
            "is_anomaly": (composite >= 0.6).astype(int),
            "global_iforest_signal": iforest_signal,
            "ecod_signal": ecod_signal,
            "per_user_signal": per_user_signal,
            "shap_top1_feature": top1_feat,
            "shap_top1_value": top1_val,
            "shap_top2_feature": top2_feat,
            "shap_top2_value": top2_val,
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Save Supporting Artifacts & Register Model

# COMMAND ----------

# Save known locations and devices for the model
pdf = spark.read.parquet(f"{artifact_path}/signins_features.parquet").toPandas()
known_locations = pdf[pdf.get("_anomaly_type", "normal") == "normal"]["location_name"].unique().tolist()
known_devices_list = pdf[pdf.get("_anomaly_type", "normal") == "normal"]["device"].unique().tolist()

# Fallback if _anomaly_type not present
if not known_locations:
    known_locations = pdf["location_name"].value_counts().head(10).index.tolist()
if not known_devices_list:
    known_devices_list = pdf["device"].value_counts().head(10).index.tolist()

with open(f"{artifact_path}/known_locations.pkl", "wb") as f:
    pickle.dump(known_locations, f)
with open(f"{artifact_path}/known_devices.pkl", "wb") as f:
    pickle.dump(known_devices_list, f)

# Define model signature and input example
input_example = pd.DataFrame({
    "user_id": ["user001@snc-internal.example.com"],
    "hour": [14], "day_of_week": [3], "latitude": [39.53], "longitude": [-119.75],
    "failed_attempts_before": [0], "mfa_used": [True], "session_duration_min": [45.0],
    "device": ["Windows_Laptop_Corp"], "location_name": ["HQ_Sparks_NV"],
    "minutes_since_last_login": [120.0], "distance_from_prev_km": [0.0],
    "geo_velocity_kmh": [0.0], "logins_last_hour": [1],
})
output_example = pd.DataFrame({
    "risk_score": [0.1], "risk_tier": ["Low"], "is_anomaly": [0],
    "global_iforest_signal": [0.1], "ecod_signal": [0.08], "per_user_signal": [0.05],
    "shap_top1_feature": ["geo_velocity_kmh"], "shap_top1_value": [-0.02],
    "shap_top2_feature": ["is_off_hours"], "shap_top2_value": [-0.01],
})
signature = infer_signature(input_example, output_example)

# Log and register
with mlflow.start_run(run_name="composite_anomaly_detector_v1") as run:
    mlflow.log_params({
        "model_type": "Composite(IForest+ECOD+PerUser+SHAP)",
        "n_features": len(MODEL_FEATURES),
        "features": str(MODEL_FEATURES),
        "weights": "iforest=0.35, ecod=0.30, per_user=0.35",
    })

    artifacts = {
        "iforest_model": f"{artifact_path}/iforest_model.pkl",
        "scaler": f"{artifact_path}/scaler.pkl",
        "model_features": f"{artifact_path}/model_features.pkl",
        "known_locations": f"{artifact_path}/known_locations.pkl",
        "known_devices": f"{artifact_path}/known_devices.pkl",
    }

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CompositeAnomalyDetector(),
        artifacts=artifacts,
        signature=signature,
        input_example=input_example,
        pip_requirements=["scikit-learn", "pyod", "shap", "numpy", "pandas"],
    )

    model_uri = f"runs:/{run.info.run_id}/model"
    print(f"✓ Model logged: {model_uri}")

# Azure Gov Cloud: UC Model Registry is not available. Use the workspace registry.
mlflow.set_registry_uri("databricks")
result = mlflow.register_model(model_uri, f"composite_anomaly_detector{SUFFIX_TAG}")
print(f"✓ Registered: composite_anomaly_detector{SUFFIX_TAG} version {result.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Test Batch Inference
# MAGIC
# MAGIC Simulate scoring a batch of new sign-in events using the composite model.

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(model_uri)

# Sample 20 events as if they're new
sample = pdf.sample(20, random_state=42)

# Reconstruct mfa_used from mfa_numeric (model expects raw input, not engineered features)
sample["mfa_used"] = sample["mfa_numeric"].astype(bool)

# Select only the columns the model signature expects
model_input_cols = signature.inputs.input_names()
predictions = loaded_model.predict(sample[model_input_cols])

sample_result = sample[["login_id", "user_id", "timestamp", "location_name", "device"]].reset_index(drop=True)
sample_result = pd.concat([sample_result, predictions.reset_index(drop=True)], axis=1)

print("Composite batch inference results:\n")
for _, row in sample_result.iterrows():
    marker = "🔴" if row["risk_tier"] in ["Critical", "High"] else "🟡" if row["risk_tier"] == "Medium" else "🟢"
    print(f"  {marker} {row['risk_tier']:10s} (score={row['risk_score']:.3f}) | "
          f"User: {row['user_id']}")
    print(f"    Signals: Global={row['global_iforest_signal']:.2f}  ECOD={row['ecod_signal']:.2f}  "
          f"Per-user={row['per_user_signal']:.2f}")
    print(f"    Why: {row['shap_top1_feature']} ({row['shap_top1_value']:+.3f}), "
          f"{row['shap_top2_feature']} ({row['shap_top2_value']:+.3f})")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Production Batch Scoring Pattern
# MAGIC
# MAGIC The cell below shows the pattern you'd use in a **scheduled notebook job**.
# MAGIC It reads new events since the last run, scores them with the composite model, and writes results.

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC # === PRODUCTION COMPOSITE BATCH SCORING NOTEBOOK ===
# MAGIC # Schedule this as a Databricks Job (e.g., every 1-4 hours)
# MAGIC
# MAGIC import mlflow
# MAGIC from pyspark.sql import functions as F
# MAGIC from datetime import datetime, timedelta
# MAGIC
# MAGIC
# MAGIC DATABASE = dbutils.widgets.get("database")
# MAGIC SUFFIX = dbutils.widgets.get("table_suffix").strip()
# MAGIC SUFFIX_TAG = f"_{SUFFIX}" if SUFFIX else ""
# MAGIC
# MAGIC # 1. Load composite model from registry
# MAGIC model = mlflow.pyfunc.load_model(f"models:/composite_anomaly_detector{SUFFIX_TAG}/Production")
# MAGIC
# MAGIC # 2. Read new events since last scored timestamp
# MAGIC last_scored = spark.sql(f"""
# MAGIC     SELECT COALESCE(MAX(timestamp), '2000-01-01') as last_ts
# MAGIC     FROM {DATABASE}.signins_risk_scores{SUFFIX_TAG}
# MAGIC """).first()["last_ts"]
# MAGIC
# MAGIC new_events = spark.read.table(f"{DATABASE}.signins_bronze") \
# MAGIC     .filter(F.col("timestamp") > last_scored)
# MAGIC
# MAGIC if new_events.count() == 0:
# MAGIC     print("No new events to score")
# MAGIC     dbutils.notebook.exit("no_new_events")
# MAGIC
# MAGIC # 3. Feature engineering (same as notebook 02)
# MAGIC # ... (apply feature engineering pipeline)
# MAGIC
# MAGIC # 4. Score batch — composite model handles all three layers + SHAP
# MAGIC pdf_new = new_events.toPandas()
# MAGIC predictions = model.predict(pdf_new)
# MAGIC
# MAGIC # 5. Write scores with component breakdowns and explanations
# MAGIC result = pd.concat([
# MAGIC     pdf_new[["login_id", "user_id", "timestamp", "location_name", "device"]],
# MAGIC     predictions
# MAGIC ], axis=1)
# MAGIC result["scored_at"] = datetime.utcnow()
# MAGIC
# MAGIC spark.createDataFrame(result).write \
# MAGIC     .format("delta").mode("append") \
# MAGIC     .saveAsTable(f"{DATABASE}.signins_risk_scores{SUFFIX_TAG}")
# MAGIC
# MAGIC # 6. Alert on Critical/High — now includes SHAP explanations
# MAGIC critical = result[result["risk_tier"].isin(["Critical", "High"])]
# MAGIC if len(critical) > 0:
# MAGIC     print(f"⚠️ {len(critical)} high-risk events detected!")
# MAGIC     for _, row in critical.iterrows():
# MAGIC         print(f"  User: {row['user_id']} | Score: {row['risk_score']:.3f}")
# MAGIC         print(f"  Why: {row['shap_top1_feature']}, {row['shap_top2_feature']}")
# MAGIC     # In production: send to SIEM, Slack, PagerDuty, etc.
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Summary Dashboard Query
# MAGIC
# MAGIC Once risk scores are in Delta, analysts can query them with SQL.
# MAGIC Today this runs in notebooks; post-Evergreen, use Databricks SQL dashboards.

# COMMAND ----------

spark.sql(f"USE {DATABASE}")

# COMMAND ----------

# Daily anomaly summary
display(spark.sql(f"""
    SELECT
      DATE(timestamp) as date,
      risk_tier,
      COUNT(*) as event_count,
      COUNT(DISTINCT user_id) as unique_users,
      ROUND(AVG(risk_score), 3) as avg_score
    FROM {DATABASE}.signins_risk_scores{SUFFIX_TAG}
    GROUP BY DATE(timestamp), risk_tier
    ORDER BY date DESC, risk_tier
"""))

# COMMAND ----------

# Top 20 highest-risk events with explanations
display(spark.sql(f"""
    SELECT
      user_id,
      timestamp,
      location_name,
      device,
      ROUND(risk_score, 3) as risk_score,
      risk_tier,
      ROUND(iforest_score, 3) as global_signal,
      ROUND(ensemble_score, 3) as ensemble_signal,
      n_detectors_flagged,
      is_off_hours,
      is_new_device,
      is_unknown_location
    FROM {DATABASE}.signins_risk_scores{SUFFIX_TAG}
    WHERE risk_tier IN ('Critical', 'High')
    ORDER BY risk_score DESC
    LIMIT 20
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vision Setting — Future State with Evergreen GovCloud
# MAGIC
# MAGIC When Evergreen Azure GovCloud reaches GA (target: April 30, 2026), the following upgrades become available:
# MAGIC
# MAGIC | Feature | Current (Classic Compute) | Future (Evergreen) |
# MAGIC |---------|--------------------------|-------------------|
# MAGIC | **Inference** | Batch via notebook jobs | **Real-time Model Serving endpoints** |
# MAGIC | **Pipeline** | Scheduled notebooks | **Delta Live Tables** (streaming) |
# MAGIC | **Dashboard** | Notebook visualizations | **AI/BI Dashboards** with auto-refresh |
# MAGIC | **Governance** | Workspace-level | **Unity Catalog** with lineage + audit |
# MAGIC | **Alerting** | Custom notebook logic | **Lakewatch SIEM** with pre-built rules |
# MAGIC | **Compute** | Classic clusters | **Serverless** (no cluster management) |
# MAGIC | **AI Enrichment** | Not available | **Foundation Model APIs** for natural language explanations |
# MAGIC
# MAGIC ### Key Upgrade Path
# MAGIC 1. **DLT pipeline**: Convert batch scoring to streaming — score events within minutes, not hours
# MAGIC 2. **Model Serving**: Real-time REST API — score individual events as they happen
# MAGIC 3. **Unity Catalog**: Centralized governance over models, features, and scores
# MAGIC 4. **AI/BI Dashboards**: Interactive anomaly dashboards for SOC analysts
# MAGIC 5. **Foundation Models**: Generate natural-language explanations of *why* a login is anomalous
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🎉 Workshop 2 Complete!** You've built a full anomaly detection pipeline combining
# MAGIC global Isolation Forest, PyOD ensemble, per-user models, and SHAP explainability —
# MAGIC all running on classic compute in Azure Gov Cloud.
# MAGIC
# MAGIC ### Next Steps
# MAGIC 1. **Tune composite weights** based on analyst feedback (currently: IForest 35%, ECOD 30%, Per-user 35%)
# MAGIC 2. **Schedule batch scoring** as a Databricks Job (every 1-4 hours)
# MAGIC 3. **Build analyst feedback loop** to improve detection over time
# MAGIC 4. **Expand features** — incorporate HR data (RDS → Databricks) and travel data for geo-velocity validation
# MAGIC 5. **Plan Evergreen migration** — convert to DLT streaming when available
# MAGIC 6. **Connect to SIEM/SOC (or Lakewatch in the future)** — export scores to existing security tools