# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 1: Email Risk Classification with NLP
# MAGIC ## Notebook 2 — Topic Discovery with NMF
# MAGIC
# MAGIC **Non-Negative Matrix Factorization (NMF)** is the primary exploration tool for this workshop.
# MAGIC We'll sweep from 2 to N topics, inspect the top words at each K, and find the natural topic structure in the email data.
# MAGIC
# MAGIC ### Why NMF?
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────┐
# MAGIC │                     NMF Factorization                            │
# MAGIC │                                                                  │
# MAGIC │   TF-IDF Matrix          ≈      W          ×       H             │
# MAGIC │  (docs × terms)            (docs × K)        (K × terms)         │
# MAGIC │                                                                  │
# MAGIC │  ┌─────────────┐      ┌──────────┐    ┌──────────────┐           │
# MAGIC │  │ 5000 × 5000 │  ≈   │ 5000 × K │  × │ K × 5000    │            │
# MAGIC │  │  (sparse)   │      │(doc-topic)│    │(topic-terms) │          │
# MAGIC │  └─────────────┘      └──────────┘    └──────────────┘           │
# MAGIC │                                                                  │
# MAGIC │  • Non-negativity → additive, interpretable parts                │
# MAGIC │  • Topics are non-overlapping → crisp, clean separation          │
# MAGIC │  • Fast to compute → sweep many K values quickly                 │
# MAGIC │  • Top words per topic are meaningful and actionable             │
# MAGIC └──────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Reference:** [scikit-learn NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
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

dbutils.widgets.text("database", "nlp_email_risk", "Database")
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
# MAGIC ## Step 1: Load TF-IDF Artifacts

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

with open(f"{artifact_path}/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open(f"{artifact_path}/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

# Read from Delta table instead of parquet (avoids pyarrow version mismatch)
pdf = spark.read.table(f"{DATABASE}.emails_cleaned{SUFFIX_TAG}").toPandas()

feature_names = tfidf.get_feature_names_out()
print(f"Loaded TF-IDF matrix: {tfidf_matrix.shape}")
print(f"Loaded {len(pdf)} email records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: NMF Topic Sweep (K = 2 to 15)
# MAGIC
# MAGIC The goal is to find the **optimal number of topics** by:
# MAGIC 1. Running NMF for each K from 2 to 15
# MAGIC 2. Recording the **reconstruction error** (how well K topics approximate the original matrix)
# MAGIC 3. Looking for an **elbow point** — where adding more topics yields diminishing returns
# MAGIC 4. Inspecting the **top words** at each K to see if topics make semantic sense
# MAGIC
# MAGIC This is the core analytical workflow: sweep, inspect, decide.

# COMMAND ----------

from sklearn.decomposition import NMF
import sys
import os

# Clear stale mlflow module state if present (prevents circular import error)
mlflow_keys = [k for k in sys.modules if k == 'mlflow' or k.startswith('mlflow.')]
for k in mlflow_keys:
    del sys.modules[k]

import mlflow
import mlflow.sklearn
# Register sklearn integration so DBR's MLflow autologging shim doesn't KeyError on fit
mlflow.sklearn.autolog(disable=True)
import time

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
mlflow.set_experiment(f"{os.path.dirname(notebook_path)}/nlp_email_risk_nmf")

k_range = range(2, 16)
results = []

for k in k_range:
    with mlflow.start_run(run_name=f"nmf_k{k}"):
        start = time.time()

        nmf_model = NMF(
            n_components=k,
            random_state=42,
            max_iter=500,
            init="nndsvda",       # Deterministic initialization
        )

        W = nmf_model.fit_transform(tfidf_matrix)  # doc-topic matrix
        H = nmf_model.components_                    # topic-term matrix

        elapsed = time.time() - start
        error = nmf_model.reconstruction_err_

        # Get top words per topic
        top_words = {}
        for topic_idx in range(k):
            top_indices = H[topic_idx].argsort()[-10:][::-1]
            top_words[f"topic_{topic_idx}"] = ", ".join(feature_names[top_indices])

        mlflow.log_param("n_topics", k)
        mlflow.log_metric("reconstruction_error", error)
        mlflow.log_metric("fit_time_seconds", elapsed)
        mlflow.log_dict(top_words, "top_words.json")

        results.append({
            "k": k,
            "reconstruction_error": error,
            "fit_time": elapsed,
            "top_words": top_words,
            "W": W,
            "H": H,
            "model": nmf_model,
        })

        print(f"K={k:2d} | Error={error:.2f} | Time={elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Elbow Plot — Finding Optimal K
# MAGIC
# MAGIC The reconstruction error should decrease as K increases. We look for the **elbow** — the point where
# MAGIC the rate of improvement slows significantly. Beyond this point, additional topics tend to split
# MAGIC meaningful topics rather than discovering new ones.

# COMMAND ----------

import matplotlib.pyplot as plt

errors = [r["reconstruction_error"] for r in results]
ks = [r["k"] for r in results]

# Calculate rate of change for elbow detection
error_diffs = [errors[i] - errors[i+1] for i in range(len(errors)-1)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Reconstruction error
ax1.plot(ks, errors, "bo-", linewidth=2, markersize=8)
ax1.set_xlabel("Number of Topics (K)", fontsize=12)
ax1.set_ylabel("Reconstruction Error", fontsize=12)
ax1.set_title("NMF Reconstruction Error vs. K", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(ks)

# Rate of change (helps identify elbow)
ax2.bar(ks[:-1], error_diffs, color="steelblue", alpha=0.7)
ax2.set_xlabel("K → K+1", fontsize=12)
ax2.set_ylabel("Error Reduction", fontsize=12)
ax2.set_title("Marginal Error Reduction per Additional Topic", fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Inspect Top Words at Each K
# MAGIC
# MAGIC Numbers alone don't tell the full story. Let's look at the **top 10 words per topic** for a range of K values.
# MAGIC The question at each K: **Do these topics represent distinct, meaningful risk categories?**

# COMMAND ----------

# Show top words for K=4, 6, 8, 10 side by side
for target_k in [4, 6, 8, 10]:
    result = next(r for r in results if r["k"] == target_k)
    print(f"\n{'='*80}")
    print(f"  K = {target_k} Topics (Error = {result['reconstruction_error']:.2f})")
    print(f"{'='*80}")
    for topic_name, words in result["top_words"].items():
        print(f"  {topic_name}: {words}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Deep Dive into Best K
# MAGIC
# MAGIC Based on the elbow plot and topic interpretability above, select the best K.
# MAGIC We'll use K=6 as a starting point (matching our 6 synthetic risk categories), but adjust based on what you see.

# COMMAND ----------

dbutils.widgets.text("best_k", "6", "Best K")
BEST_K = int(dbutils.widgets.get("best_k"))

best_result = next(r for r in results if r["k"] == BEST_K)
W_best = best_result["W"]
H_best = best_result["H"]

print(f"Selected K = {BEST_K}")
print(f"Reconstruction error: {best_result['reconstruction_error']:.2f}")
print()

for topic_idx in range(BEST_K):
    top_idx = H_best[topic_idx].argsort()[-15:][::-1]
    words = [(feature_names[i], H_best[topic_idx][i]) for i in top_idx]
    print(f"\nTopic {topic_idx}:")
    for word, score in words:
        bar = "█" * int(score * 50)
        print(f"  {word:25s} {score:.4f} {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Document-Topic Distribution
# MAGIC
# MAGIC For each email, NMF gives us a vector of topic weights (the W matrix).
# MAGIC The **dominant topic** is the one with the highest weight.

# COMMAND ----------

# Assign dominant topic to each email
pdf["dominant_topic"] = W_best.argmax(axis=1)
pdf["topic_confidence"] = W_best.max(axis=1)

# Add all topic weights
for i in range(BEST_K):
    pdf[f"topic_{i}_weight"] = W_best[:, i]

# Distribution of dominant topics
topic_dist = pdf["dominant_topic"].value_counts().sort_index()
print("Emails per dominant topic:")
for topic, count in topic_dist.items():
    pct = count / len(pdf) * 100
    bar = "█" * int(pct)
    print(f"  Topic {topic}: {count:5d} ({pct:5.1f}%) {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Topic Confidence Distribution
# MAGIC
# MAGIC How confident is the model in its dominant topic assignment?
# MAGIC High confidence = email clearly belongs to one topic.
# MAGIC Low confidence = email spans multiple topics (could be interesting for risk analysis).

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 5))
for topic in range(BEST_K):
    subset = pdf[pdf["dominant_topic"] == topic]["topic_confidence"]
    ax.hist(subset, bins=30, alpha=0.5, label=f"Topic {topic}")
ax.set_xlabel("Topic Confidence (max weight)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Confidence Distribution by Dominant Topic", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Sample Emails per Topic
# MAGIC
# MAGIC Let's look at example emails for each topic to validate that the NMF topics are semantically meaningful.

# COMMAND ----------

for topic in range(BEST_K):
    topic_emails = pdf[pdf["dominant_topic"] == topic].nlargest(3, "topic_confidence")
    print(f"\n{'='*80}")
    print(f"  TOPIC {topic} — Top 3 most confident emails")
    print(f"{'='*80}")
    for _, row in topic_emails.iterrows():
        print(f"  Confidence: {row['topic_confidence']:.3f}")
        print(f"  Body: {row['body'][:200]}...")
        print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Save NMF Results
# MAGIC
# MAGIC Save the best model and topic assignments for use in subsequent notebooks.

# COMMAND ----------

import pickle

# Save NMF model and results
with open(f"{artifact_path}/nmf_model_k{BEST_K}.pkl", "wb") as f:
    pickle.dump(best_result["model"], f)

with open(f"{artifact_path}/nmf_W_k{BEST_K}.pkl", "wb") as f:
    pickle.dump(W_best, f)

with open(f"{artifact_path}/nmf_H_k{BEST_K}.pkl", "wb") as f:
    pickle.dump(H_best, f)

# Save via Spark instead of pandas to_parquet (avoids pyarrow version mismatch)
spark.createDataFrame(pdf).write.mode("overwrite").parquet(f"{artifact_path}/emails_with_nmf_topics.parquet")

# Also save as Delta table
df_topics = spark.createDataFrame(
    pdf[["email_id", "dominant_topic", "topic_confidence"] + [f"topic_{i}_weight" for i in range(BEST_K)]]
)
df_topics.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE}.email_nmf_topics{SUFFIX_TAG}")

print(f"✓ Saved NMF model (K={BEST_K}) and topic assignments")
print(f"✓ Delta table: {DATABASE}.email_nmf_topics{SUFFIX_TAG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC | Observation | Implication |
# MAGIC |-------------|-------------|
# MAGIC | NMF produces crisp, non-overlapping topics | Each topic maps to a clear risk category |
# MAGIC | Reconstruction error elbow helps choose K | Systematic way to find the right number of risk categories |
# MAGIC | Top words per topic are immediately interpretable | Analysts can name and act on topics without ML expertise |
# MAGIC | Low-confidence emails span multiple topics | These may warrant additional human review |
# MAGIC
# MAGIC **Next →** Open `03_lda_comparison` to run LDA and compare with NMF results.