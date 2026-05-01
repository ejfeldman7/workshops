# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 1: Email Risk Classification with NLP
# MAGIC ## Notebook 5 — Composite Risk Scoring & Batch Pipeline
# MAGIC
# MAGIC Combine the outputs of NMF, LDA, and embedding clusters into a single **composite risk score**,
# MAGIC register it in MLflow, and set up a batch inference pattern.
# MAGIC
# MAGIC ### Risk Score Design
# MAGIC
# MAGIC ```
# MAGIC ┌───────────────────────────────────────────────────────────────────────┐
# MAGIC │                    Composite Risk Score                               │
# MAGIC │                                                                       │
# MAGIC │   ┌────────┐  ┌────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐     │
# MAGIC │   │  NMF   │  │  LDA   │  │Embedding│  │Attachment│  │ Temporal  │    │
# MAGIC │   │ Topic  │ +│Entropy │ +│ Cluster │ +│ & Size   │ +│  Signal   │    │
# MAGIC │   │ Signal │  │ Signal │  │ Signal  │  │ Signal   │  │           │    │
# MAGIC │   └───┬────┘  └───┬────┘  └────┬────┘  └────┬─────┘  └─────┬────┘     │
# MAGIC │       │            │            │            │              │         │
# MAGIC │       └────────────┴────────────┴────────────┴──────────────┘         │
# MAGIC │                            │                                          │
# MAGIC │                   Weighted Combination                                │
# MAGIC │                            │                                          │
# MAGIC │                   ┌────────▼────────┐                                 │
# MAGIC │                   │   Risk Score    │                                 │
# MAGIC │                   │   0.0 — 1.0     │                                 │
# MAGIC │                   │                 │                                 │
# MAGIC │                   │  Low | Med | Hi │                                 │
# MAGIC │                   └─────────────────┘                                 │
# MAGIC └───────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites

# COMMAND ----------

# MAGIC %pip install sentence-transformers umap-learn hdbscan pyLDAvis wordcloud --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime

dbutils.widgets.text("database", "nlp_email_risk", "Database")
dbutils.widgets.text("best_k", "6", "Number of Topics")
_user_email = spark.sql("SELECT current_user()").first()[0]
_name_parts = _user_email.split('@')[0].replace('_', '.').split('.')
_initials = (_name_parts[0][0] + _name_parts[-1][0]).lower() if len(_name_parts) >= 2 else _user_email[:2].lower()
_default_suffix = _initials + datetime.now().strftime('%d%m%y')
dbutils.widgets.text("table_suffix", _default_suffix, "Table Suffix (your initials)")

DATABASE = dbutils.widgets.get("database")
BEST_K = int(dbutils.widgets.get("best_k"))
SUFFIX = dbutils.widgets.get("table_suffix").strip()
SUFFIX_TAG = f"_{SUFFIX}" if SUFFIX else ""
print(f"Database: {DATABASE}, Table suffix: {SUFFIX_TAG or '(none)'}")

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

# Read via Spark instead of pd.read_parquet (avoids pyarrow version mismatch)
pdf = spark.read.parquet(f"{artifact_path}/emails_full_analysis.parquet").toPandas()
print(f"Loaded {len(pdf)} emails with NMF, LDA, and embedding analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Define Topic Risk Mapping
# MAGIC
# MAGIC Based on our topic exploration in notebooks 02–04, we assign a **risk weight** to each NMF topic.
# MAGIC This is where analyst judgment meets ML output.
# MAGIC
# MAGIC > **Action:** Review the NMF topic top words from notebook 02 and adjust the risk mapping below.
# MAGIC > Topics with security-relevant words (credentials, password, exfiltration) get higher weights.

# COMMAND ----------

# Default risk mapping — ADJUST based on your NMF topic inspection
# Higher weight = higher risk signal from that topic
# The indices here correspond to NMF topic indices from notebook 02

TOPIC_RISK_WEIGHTS = {
    0: 0.2,  # Adjust: likely normal business
    1: 0.8,  # Adjust: likely data exfiltration / policy violation
    2: 0.9,  # Adjust: likely phishing indicators
    3: 0.5,  # Adjust: likely HR / personnel
    4: 0.6,  # Adjust: likely financial irregularity
    5: 0.1,  # Adjust: likely normal business
}

# Fill any missing topics
for i in range(BEST_K):
    if i not in TOPIC_RISK_WEIGHTS:
        TOPIC_RISK_WEIGHTS[i] = 0.3  # default moderate

print("Topic risk weight mapping:")
for t, w in sorted(TOPIC_RISK_WEIGHTS.items()):
    bar = "█" * int(w * 30)
    print(f"  Topic {t}: {w:.1f} {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Compute Component Scores
# MAGIC
# MAGIC Each method contributes a signal between 0 and 1:

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler

# --- Signal 1: NMF topic risk (based on dominant topic + confidence) ---
pdf["nmf_risk_signal"] = pdf["dominant_topic"].map(TOPIC_RISK_WEIGHTS) * pdf["topic_confidence"]

# --- Signal 2: LDA entropy (high entropy = spans multiple topics = potentially suspicious) ---
scaler = MinMaxScaler()
pdf["lda_entropy_signal"] = scaler.fit_transform(pdf[["lda_entropy"]]).ravel()

# --- Signal 3: Embedding cluster anomaly ---
# Noise points (-1) get highest anomaly score, low-probability cluster members also score higher
pdf["embedding_anomaly_signal"] = np.where(
    pdf["embedding_cluster"] == -1,
    1.0,
    1.0 - pdf["cluster_probability"]
)

# --- Signal 4: Temporal anomaly (emails sent outside business hours) ---
pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
pdf["hour"] = pdf["timestamp"].dt.hour
pdf["is_off_hours"] = ((pdf["hour"] < 7) | (pdf["hour"] > 19)).astype(float) * 0.5
pdf["is_weekend"] = (pdf["timestamp"].dt.dayofweek >= 5).astype(float) * 0.3
pdf["temporal_signal"] = pdf["is_off_hours"] + pdf["is_weekend"]

# --- Signal 5: Attachment & size anomaly ---
# Large emails with attachments are riskier (especially data exfiltration patterns)
pdf["has_attachment"] = pdf["has_attachment"].fillna(False).astype(float)
pdf["size_mb"] = pdf["size_mb"].fillna(0.0)
pdf["attachment_count"] = pdf["attachment_count"].fillna(0)

# Size signal: log-scaled, normalized. Larger emails get higher scores.
pdf["log_size"] = np.log1p(pdf["size_mb"])
pdf["attachment_signal"] = scaler.fit_transform(pdf[["log_size"]]).ravel() * 0.7 + pdf["has_attachment"] * 0.3

print("Signal distributions:")
for signal in ["nmf_risk_signal", "lda_entropy_signal", "embedding_anomaly_signal", "temporal_signal", "attachment_signal"]:
    print(f"  {signal:30s}: mean={pdf[signal].mean():.3f}, std={pdf[signal].std():.3f}, "
          f"min={pdf[signal].min():.3f}, max={pdf[signal].max():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Weighted Composite Score

# COMMAND ----------

# Weights for each signal component
SIGNAL_WEIGHTS = {
    "nmf_risk_signal": 0.30,         # NMF topic assignment is primary
    "lda_entropy_signal": 0.10,      # LDA entropy as secondary
    "embedding_anomaly_signal": 0.30, # Embedding anomaly is strong signal
    "attachment_signal": 0.15,        # Attachment presence + email size
    "temporal_signal": 0.15,          # Time-based signal is supplementary
}

print("Signal weights:")
for s, w in SIGNAL_WEIGHTS.items():
    print(f"  {s:30s}: {w:.2f}")
print(f"  {'Total':30s}: {sum(SIGNAL_WEIGHTS.values()):.2f}")

# Compute composite score
pdf["risk_score"] = sum(
    pdf[signal] * weight
    for signal, weight in SIGNAL_WEIGHTS.items()
)

# Normalize to 0-1
pdf["risk_score"] = scaler.fit_transform(pdf[["risk_score"]])

# Risk tiers
pdf["risk_tier"] = pd.cut(
    pdf["risk_score"],
    bins=[0, 0.3, 0.6, 0.8, 1.0],
    labels=["Low", "Medium", "High", "Critical"],
    include_lowest=True,
)

print(f"\nRisk tier distribution:")
tier_counts = pdf["risk_tier"].value_counts().sort_index()
for tier, count in tier_counts.items():
    pct = count / len(pdf) * 100
    bar = "█" * int(pct)
    print(f"  {tier:10s}: {count:5d} ({pct:5.1f}%) {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Risk Score Distribution

# COMMAND ----------

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
colors = {"Low": "green", "Medium": "orange", "High": "red", "Critical": "darkred"}
for tier in ["Low", "Medium", "High", "Critical"]:
    subset = pdf[pdf["risk_tier"] == tier]["risk_score"]
    if len(subset) > 0:
        ax.hist(subset, bins=30, alpha=0.6, label=tier, color=colors[tier])
ax.set_xlabel("Risk Score", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Risk Score Distribution by Tier", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# UMAP colored by risk score
ax = axes[1]
scatter = ax.scatter(pdf["umap_x"], pdf["umap_y"], c=pdf["risk_score"],
                     cmap="RdYlGn_r", alpha=0.5, s=8)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title("Risk Score in Embedding Space", fontsize=14)
plt.colorbar(scatter, ax=ax, label="Risk Score")
ax.grid(True, alpha=0.2)

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Review High-Risk Emails

# COMMAND ----------

high_risk = pdf[pdf["risk_tier"].isin(["High", "Critical"])].nlargest(15, "risk_score")

print(f"Top 15 highest-risk emails:\n")
for _, row in high_risk.iterrows():
    print(f"  Score: {row['risk_score']:.3f} | Tier: {row['risk_tier']} | NMF: T{row['dominant_topic']} | Cluster: {row['embedding_cluster']}")
    print(f"  Signals: NMF={row['nmf_risk_signal']:.2f}, LDA_ent={row['lda_entropy_signal']:.2f}, "
          f"Emb={row['embedding_anomaly_signal']:.2f}, Attach={row['attachment_signal']:.2f}, Time={row['temporal_signal']:.2f}")
    att_info = f"{row['attachment_count']} attachment(s), {row['size_mb']:.2f} MB" if row['has_attachment'] else "no attachments"
    print(f"  Attachments: {att_info}")
    print(f"  Body: {row['body'][:150]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Register Model with MLflow
# MAGIC
# MAGIC We package the entire scoring pipeline as an MLflow model so it can be loaded and applied
# MAGIC to new emails via batch inference.

# COMMAND ----------

from mlflow.models import infer_signature
import os
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
# Register sklearn integration so DBR's MLflow autologging shim doesn't KeyError on fit
mlflow.sklearn.autolog(disable=True)

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
mlflow.set_experiment(f"{os.path.dirname(notebook_path)}/nlp_email_risk_scoring")

class EmailRiskScorer(mlflow.pyfunc.PythonModel):
    """Composite email risk scorer combining NMF, LDA, and embeddings."""

    def load_context(self, context):
        import pickle
        with open(context.artifacts["tfidf_vectorizer"], "rb") as f:
            self.tfidf = pickle.load(f)
        with open(context.artifacts["nmf_model"], "rb") as f:
            self.nmf = pickle.load(f)
        with open(context.artifacts["lda_model"], "rb") as f:
            self.lda = pickle.load(f)
        with open(context.artifacts["count_vectorizer"], "rb") as f:
            self.count_vec = pickle.load(f)
        with open(context.artifacts["config"], "rb") as f:
            self.config = pickle.load(f)

    def predict(self, context, model_input):
        import re
        import numpy as np
        from scipy.stats import entropy
        from sklearn.preprocessing import MinMaxScaler

        texts = model_input["body"].tolist()

        # Clean text
        cleaned = []
        for text in texts:
            text = text.lower()
            text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
            text = re.sub(r'\S+@\S+\.\S+', ' EMAIL ', text)
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            cleaned.append(text)

        # NMF scoring
        tfidf_matrix = self.tfidf.transform(cleaned)
        W_nmf = self.nmf.transform(tfidf_matrix)
        topic_weights = self.config["topic_risk_weights"]
        nmf_signal = np.array([topic_weights.get(t, 0.3) for t in W_nmf.argmax(axis=1)]) * W_nmf.max(axis=1)

        # LDA entropy
        count_matrix = self.count_vec.transform(cleaned)
        W_lda = self.lda.transform(count_matrix)
        lda_probs = W_lda / W_lda.sum(axis=1, keepdims=True)
        lda_ent = np.array([entropy(row) for row in lda_probs])
        scaler = MinMaxScaler()
        lda_signal = scaler.fit_transform(lda_ent.reshape(-1, 1)).flatten()

        # Composite score (without embedding signal for batch — embeddings are expensive)
        risk_score = (
            0.50 * nmf_signal +
            0.25 * lda_signal +
            0.25 * 0.3  # default temporal placeholder
        )
        risk_score = scaler.fit_transform(risk_score.reshape(-1, 1)).flatten()

        return risk_score

# Save config
config = {
    "topic_risk_weights": TOPIC_RISK_WEIGHTS,
    "signal_weights": SIGNAL_WEIGHTS,
    "best_k": BEST_K,
}
with open(f"{artifact_path}/scoring_config.pkl", "wb") as f:
    pickle.dump(config, f)

# Define model signature and input example
input_example = pd.DataFrame({"body": ["Sample email text for risk scoring"]})
output_example = np.array([0.5])
signature = infer_signature(input_example, output_example)

# Log and register model
with mlflow.start_run(run_name="email_risk_scorer_v1") as run:
    mlflow.log_params({
        "n_topics": BEST_K,
        "scoring_version": "v1",
        "components": "nmf+lda+embedding+temporal",
    })

    artifacts = {
        "tfidf_vectorizer": f"{artifact_path}/tfidf_vectorizer.pkl",
        "nmf_model": f"{artifact_path}/nmf_model_k{BEST_K}.pkl",
        "lda_model": f"{artifact_path}/lda_model_k{BEST_K}.pkl",
        "count_vectorizer": f"{artifact_path}/count_vectorizer.pkl",
        "config": f"{artifact_path}/scoring_config.pkl",
    }

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=EmailRiskScorer(),
        artifacts=artifacts,
        signature=signature,
        input_example=input_example,
        pip_requirements=["scikit-learn", "scipy", "numpy", "pandas"],
    )

    model_uri = f"runs:/{run.info.run_id}/model"
    print(f"\u2713 Model logged: {model_uri}")

# Register in workspace model registry
# Azure Gov Cloud: UC Model Registry is not available. Force the workspace registry.
mlflow.set_registry_uri("databricks")
result = mlflow.register_model(model_uri, f"email_risk_scorer{SUFFIX_TAG}")
print(f"\u2713 Model registered: email_risk_scorer{SUFFIX_TAG} version {result.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Test Batch Inference
# MAGIC
# MAGIC Simulate how the model would score new incoming emails.

# COMMAND ----------

# Load model back and score a sample
loaded_model = mlflow.pyfunc.load_model(model_uri)

sample = pdf[["body"]].sample(10, random_state=42)
scores = loaded_model.predict(sample)

sample["predicted_risk_score"] = scores
print("Batch inference test:\n")
for _, row in sample.iterrows():
    print(f"  Score: {row['predicted_risk_score']:.3f} | Body: {row['body'][:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Save Final Risk Scores to Delta

# COMMAND ----------

risk_cols = ["email_id", "risk_score", "risk_tier", "nmf_risk_signal",
             "lda_entropy_signal", "embedding_anomaly_signal", "attachment_signal",
             "temporal_signal", "has_attachment", "attachment_count", "size_mb",
             "dominant_topic", "lda_dominant_topic", "embedding_cluster"]

df_risk = spark.createDataFrame(pdf[risk_cols].astype({"risk_tier": str}))
df_risk.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE}.email_risk_scores{SUFFIX_TAG}")

print(f"✓ Risk scores saved to {DATABASE}.email_risk_scores{SUFFIX_TAG}")
display(spark.sql(f"SELECT risk_tier, COUNT(*) as count FROM {DATABASE}.email_risk_scores{SUFFIX_TAG} GROUP BY risk_tier ORDER BY risk_tier"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Batch Inference Pattern for Production
# MAGIC
# MAGIC In production on Azure Gov Cloud, you'd schedule this notebook (or a slimmed-down version) as a **Job**:
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
# MAGIC │  New Emails  │───▶│ Load MLflow  │───▶│ Score Batch  │───▶│ Write Risk   │
# MAGIC │  (Delta)     │    │ Model        │    │ (predict)    │    │ Scores       │
# MAGIC │              │    │              │    │              │    │ (Delta)      │
# MAGIC └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
# MAGIC                                                                     │
# MAGIC                                                              ┌──────▼──────┐
# MAGIC                                                              │  Alert on   │
# MAGIC                                                              │  High/Crit  │
# MAGIC                                                              └─────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Schedule:** Run every N hours via Databricks Jobs (Workflows)
# MAGIC **No Model Serving required** — pure batch pattern works in Azure Gov Cloud today.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Asset | Location |
# MAGIC |-------|----------|
# MAGIC | Risk scores table | `{DATABASE}.email_risk_scores{SUFFIX_TAG}` |
# MAGIC | MLflow model | `email_risk_scorer{SUFFIX_TAG}` (workspace registry) |
# MAGIC | Scoring config | `{ARTIFACT_PATH}/scoring_config.pkl` |
# MAGIC
# MAGIC **Next →** Open `06_educational_supervised` for a brief educational overview of supervised approaches (for future reference).

# COMMAND ----------

