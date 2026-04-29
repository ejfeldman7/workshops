
# Workshops — Onsite

Three hands-on workshops for planned future on-sites. Each workshop is a sequence of numbered Databricks notebooks that build on each other, generating synthetic data and walking through a complete pipeline.

Workshops 1 & 2 are Data Science focused (~90 min each). Workshop 3 is Data Engineering focused (~90 min), covering pipeline architecture and application integration. Workshop 3 is fully self-contained and does not depend on Workshops 1 & 2, so it works for a separate audience.

The `-gov` variants use Hive metastore and DBFS for Azure Government Cloud; the standard variants use Unity Catalog and Volumes.

## Workshop 1: Email Risk Classification with NLP (`nlp-email-risk/`)

Unsupervised NLP pipeline that discovers risk categories in email data and produces a composite risk score — without any labeled training data.

### Notebooks

| # | Notebook | Duration | What It Does |
|---|----------|----------|-------------|
| 00 | `setup_and_config` | 5 min | Creates database/schema, installs libraries, generates 5,000 synthetic emails across 6 risk categories with realistic attachment metadata and sizes |
| 01 | `data_exploration` | 10 min | Text profiling (length distributions, sender activity, temporal patterns), attachment/size analysis by risk type, text cleaning, TF-IDF matrix construction, word cloud |
| 02 | `nmf_topic_discovery` | 15 min | NMF sweep K=2→15, reconstruction error elbow plot, top words per topic inspection, dominant topic assignment with confidence scores. K=12 is the sweet spot — surfaces 6 normal + 4–5 risk categories on the rebalanced data. |
| 03 | `lda_comparison` | 10 min | LDA with matching K for comparison. Soft probabilistic topic assignment, entropy analysis for multi-topic emails, pyLDAvis interactive visualization, NMF vs LDA cross-tabulation |
| 04 | `embeddings_clustering` | 10 min | Sentence-transformer embeddings (all-MiniLM-L6-v2), UMAP dimensionality reduction, HDBSCAN clustering (`min_cluster_size=80, min_samples=15`), 3-panel UMAP comparison (HDBSCAN vs NMF vs ground truth), noise point analysis |
| 05 | `risk_scoring` | 15 min | Composite risk score combining 5 signals (NMF topic 30%, embedding anomaly 30%, attachment/size 15%, temporal 15%, LDA entropy 10%). MLflow model registration, batch inference pattern, risk tier assignment |
| 06 | `educational_supervised` | 5 min | **Educational only** — TF-IDF+XGBoost, DistilBERT fine-tuning (code shown but not run), zero-shot classification demo. Assume no labeled data so this is future reference for if they do. |

### Key Technical Choices

- **NMF before LDA**: NMF produces more interpretable topics for the "sweep 1→N and inspect top words" workflow. LDA follows as a comparison for soft/probabilistic assignment.
- **No labeled data**: The entire actionable pipeline is unsupervised. Supervised approaches are educational only.
- **Attachment/size as risk signal**: Data exfiltration emails tend to have large attachments (database dumps, code archives). This is a feature in the composite score.
- **Batch inference only**: No Model Serving in Azure Gov Cloud. Scoring runs via scheduled notebook jobs.
### Synthetic Data

The setup notebook generates emails across 6 categories with realistic distributions:

| Category | % of Data | Templates | Attachment Rate | Avg Size |
|----------|-----------|-----------|----------------|----------|
| Normal business | 30% | 6 | 35% | Small (meeting notes, presentations) |
| HR/personnel | 14% | 8 | 20% | Small |
| Financial irregularity | 14% | 8 | 55% | Medium (receipts, invoices) |
| Phishing indicators | 14% | 8 | 40% | Small (suspicious .html, .exe) |
| Policy violation | 14% | 8 | 30% | Small (credentials, configs) |
| Data exfiltration | 14% | 8 | 75% | Large (database dumps up to 200 MB) |

> **Distribution skew:** the data intentionally over-weights risk (70% of the corpus) so that unsupervised topic models can surface risk categories at low K. On real-world email corpora — where risk is a small minority — you'd need much higher K or supervised methods to pull out the same signal.

---

## Workshop 2: Anomaly Detection for Sign-In Data (`login-anomaly/`)

Multi-layered anomaly detection pipeline for sign-in logs, combining global models, per-user models, and SHAP explainability.

### Notebooks

| # | Notebook | Duration | What It Does |
|---|----------|----------|-------------|
| 00 | `setup_and_config` | 5 min | Creates database/schema, installs libraries, generates ~40K+ synthetic sign-in events with embedded anomalies (impossible travel, off-hours, brute force, new device+location) |
| 01 | `data_exploration` | 10 min | Temporal heatmaps (hour x day-of-week), geographic scatter plots, device/MFA analysis, per-user behavior profiles, failed attempt distributions |
| 02 | `feature_engineering` | 10 min | Geo-velocity calculation (distance/time between consecutive logins), cyclical hour encoding, session duration z-scores, new device/IP flags, login burst detection |
| 03 | `isolation_forest` | 15 min | Global Isolation Forest with contamination sweep (2%–10%), score distributions, feature importance approximation, detection rate validation by anomaly type |
| 04 | `pyod_ensemble` | 10 min | ECOD, LOF, KNN, COPOD via PyOD unified API. Detector agreement analysis (Jaccard similarity), normalized ensemble score averaging, per-algorithm detection comparison |
| 05 | `per_user_models` | 15 min | Per-user Isolation Forest via `applyInPandas` — trains one model per user in parallel. SHAP explainability (TreeExplainer) showing which features drove each anomaly. Scaled SHAP across all users with top-3 feature output per login. |
| 06 | `evaluation_thresholds` | 10 min | Risk tier assignment (Critical/High/Medium/Low), precision-recall-F1 vs threshold curves, alert volume analysis (alerts/day vs SOC capacity), analyst feedback table creation |
| 07 | `batch_scoring_pipeline` | 15 min | Composite MLflow PyFunc model combining all three layers: global IForest (35%), PyOD ECOD (30%), per-user IForest (35%). Includes SHAP top-2 features per prediction. Model registration, batch inference test, production scheduling pattern, SQL dashboard queries. |

### Key Technical Choices

- **Three detection layers**: No single algorithm catches everything. Global models catch cross-user patterns, per-user models catch individual baseline violations, PyOD ECOD catches distributional tail anomalies.
- **Per-user models via `applyInPandas`**: A 2 AM login is normal for a night-shift worker but anomalous for a 9-5 employee. Each user gets their own trained Isolation Forest with adaptive contamination.
- **SHAP explainability**: Analysts see "flagged because geo_velocity was 15x above this user's norm" instead of "anomaly score: -0.12". Top-2 SHAP features are included in the batch scoring output.
- **Composite scoring in production**: The final MLflow model wraps all three layers so production deployment is a single `model.predict()` call.
- **Geo-velocity is the strongest signal**: If a user logs in from New York and 30 minutes later from Moscow, no airplane covers that distance. The feature engineering notebook includes a haversine distance calculation and km/hr velocity.

### Synthetic Data

The setup notebook generates sign-in events for 100 users across 90 days:

| Anomaly Type | % of Data | Key Signals |
|-------------|-----------|-------------|
| Normal | ~90% | Business hours, known locations, corporate devices |
| Off-hours access | ~3% | 1-5 AM logins, unusual IPs |
| Impossible travel | ~2% | Moscow/Shanghai/Lagos logins within minutes of domestic login |
| Brute force | ~2% | 5-20 failed attempts, anomalous devices, no MFA |
| New device + location | ~2% | Never-seen device AND foreign location simultaneously |

---

## Workshop 3: Security Data Pipeline & Application Integration (`security-app-integration/`)

End-to-end data engineering pipeline for security events with external application integration via the Databricks SDK. Builds the infrastructure layer that connects Databricks to an external web application (Defensible Suite).

### Notebooks

| # | Notebook | Duration | What It Does |
|---|----------|----------|-------------|
| 00 | `setup_and_config` | 5 min | Creates catalog/schema/volumes, generates 10,000 synthetic security events (5K email + 5K sign-in) with embedded anomalies, writes raw JSON to a landing zone and bronze Delta tables |
| 01 | `ingestion_pipeline` | 15 min | Auto Loader ingestion from landing zone, Bronze → Silver medallion with schema enforcement, geo-velocity computation for sign-ins, data quality flags, pipeline health metrics logged at every stage |
| 02 | `risk_scoring` | 15 min | Rule-based composite risk scoring (no ML training). Email: keyword matching, attachment risk, direction, temporal signals. Sign-in: geo-velocity, failed attempts, unknown location/device, MFA status. Human-readable explanations per event. Unified `security_alerts` view. |
| 03 | `pipeline_monitoring` | 10 min | Threshold-based anomaly detection on pipeline health metrics (volume drops, null rate spikes, latency). Simulated problem injection and detection. Alerting patterns (webhooks, Slack, notebook exit codes). |
| 04 | `app_integration` | 25 min | **The main event.** Databricks SDK for Python — Jobs API for creating/triggering/scheduling the pipeline, run polling, run-history queries via `list_runs`/`get_run`. Two query patterns side-by-side: `spark.sql(...)` for in-notebook analysts and `databricks-sql-connector` JDBC against the all-purpose cluster for external apps. Includes a dual-mode (`spark` / `jdbc`) `DatabricksSecurityClient` class. |

### Key Technical Choices

- **Self-contained**: Generates its own data and doesn't depend on Workshops 1 & 2. The rule-based scorer produces the same output shape (risk score, tier, reasons) as the ML models, so the app integration code works with either.
- **Delta tables as the contract**: The pipeline writes scored results and health metrics to Delta; the external app reads them via JDBC. Clean separation of concerns.
- **Two query patterns, one workshop**: `spark.sql(...)` covers in-notebook analyst use; `databricks-sql-connector` JDBC against the all-purpose cluster covers external-app use (Defensible Suite). Same Delta tables underneath.
- **No SQL Warehouses on Gov Cloud (yet)**: The Statement Execution API requires a SQL Warehouse, which Azure Gov Cloud doesn't offer today. The notebook uses the legacy `/sql/protocolv1/o/{workspace_id}/{cluster_id}` JDBC endpoint as a stopgap. When Evergreen lands, swap the `http_path` to point at a warehouse — same connector, same client code.
- **No Databricks Apps required**: The architecture keeps the web app external. Databricks is the compute and data layer, not the hosting layer. This works in environments where Databricks Apps aren't available.
- **Pipeline health as a first-class table**: Every stage writes metrics (row counts, null rates, processing time) to `pipeline_health`, enabling the monitoring notebook to detect ingestion anomalies independently of the security event scoring.

### Synthetic Data

The setup notebook generates two event types:

| Event Type | Count | Normal % | Anomaly Types |
|-----------|-------|----------|---------------|
| Email | 5,000 | ~90% | Data exfiltration (4%), phishing (3%), policy violation (3%) |
| Sign-in | 5,000 | ~90% | Impossible travel (3%), brute force (2%), off-hours+unknown device (3%), failed bursts (3%) |

---

## Environment Variants

| Variant | Namespace | Artifact Storage | Compute | `.cache()` |
|---------|-----------|-----------------|---------|-----------|
| Standard (`nlp-email-risk/`, `login-anomaly/`, `security-app-integration/`) | Unity Catalog (catalog.schema.table) | Volumes | Serverless or classic | Commented out |
| Gov Cloud (`*-gov/`) | Hive metastore (database.table) | DBFS (`/dbfs/tmp/workshops/`) | Classic with ML Runtime | Enabled |

All variants contain identical logic, visualizations, and model/pipeline architectures. The only differences are storage paths and namespace conventions.

## Prerequisites

- **Cluster**: Databricks Runtime 13.3+ ML for the `-gov` variants (GPU recommended for NLP notebook 04 embeddings, CPU sufficient for everything else). Standard variants run on serverless or any DBR 14.3+.
- **Libraries**: Installed automatically via `%pip install` in each notebook (sentence-transformers, pyod, shap, umap-learn, hdbscan, databricks-sdk, databricks-sql-connector, etc.)
- **SQL Warehouse**: Standard Workshop 3 uses the Statement Execution API and benefits from a SQL warehouse. The `-gov` variant uses JDBC against the all-purpose cluster instead — no warehouse needed.
- **Data**: All synthetic — no external data needed. Each workshop's `00_setup_and_config` generates everything.

## Multi-User Workshop Delivery

These workshops are designed to be run by multiple attendees in the same workspace at the same time without collisions. Two mechanisms keep each user's work isolated:

| Asset type | Isolation mechanism |
|---|---|
| Delta tables, MLflow models, Job names, Auto Loader stream targets | `table_suffix` widget (defaults to `<initials><DDMMYY>` like `ef280426`) |
| DBFS artifact files (pickles, parquets, Auto Loader checkpoints/schemas) | `USER_ID` derived from `current_user()` — fully automatic |
| Shared bronze tables (`signins_bronze`, `emails_bronze`, `email_events_bronze`, `signin_events_bronze`) | **Not isolated** — everyone reads the same source data 00 generates |

The `table_suffix` widget appears at the top of every notebook (except 00) with an auto-computed default. Users can override it if needed (e.g., to share tables with a colleague), or just accept the default and run.

## Pre-Workshop Data Loading (Production)

For the actual onsite, real data may replace the synthetic generators:

| Data Source | Status | Action |
|-------------|--------|--------|
| Email data | Not yet in Databricks | One-time bulk load to Azure Blob/ADLS before workshop |
| RDS (Travel & HR) | Needs pipeline | JDBC connector for one-time load; Lakeflow Connect post-Evergreen |
| Sign-in logs | Already in Databricks | Ready to use |

## Reference Documentation

- [Hugging Face on Databricks](https://docs.databricks.com/en/machine-learning/train-model/huggingface/index.html)
- [NLP Reference Solutions](https://docs.databricks.com/en/machine-learning/reference-solutions/nlp.html)
- [MLflow on Databricks](https://docs.databricks.com/en/mlflow/index.html)
- [PyOD Documentation](https://pyod.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [scikit-learn NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
- [Auto Loader](https://docs.databricks.com/en/ingestion/auto-loader/index.html)
- [Databricks SDK for Python](https://docs.databricks.com/en/dev-tools/sdk-python.html)
- [SQL Statement Execution API](https://docs.databricks.com/en/sql/admin/sql-execution-tutorial.html)
- [Databricks Jobs API](https://docs.databricks.com/en/workflows/jobs/jobs-2.0-api.html)
- [Training 10,000 Anomaly Detection Models (Databricks Blog)](https://www.databricks.com/blog/training-10000-anomaly-detection-models-one-billion-records-explainable-predictions)
- [Insider Threat Detection Accelerator](https://github.com/databricks-industry-solutions/insider-threat)
- [Unsupervised Outlier Detection on Databricks](https://www.databricks.com/blog/2023/03/07/unsupervised-outlier-detection-databricks.html)
