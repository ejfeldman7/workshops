# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 1: Email Risk Classification with NLP
# MAGIC ## Notebook 3 — LDA Comparison & Soft Topic Assignment
# MAGIC
# MAGIC **Latent Dirichlet Allocation (LDA)** takes a different approach from NMF:
# MAGIC - LDA is **probabilistic** — each document has a probability distribution over topics
# MAGIC - Documents can **belong to multiple topics** (soft clustering)
# MAGIC - LDA models topics as distributions over words, and documents as distributions over topics
# MAGIC
# MAGIC ### NMF vs. LDA — When to Use Which
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────┐     ┌───────────────────────────┐
# MAGIC │         NMF              │     │          LDA              │
# MAGIC │                          │     │                           │
# MAGIC │  • Deterministic         │     │  • Probabilistic          │
# MAGIC │  • Non-negative parts    │     │  • Generative model       │
# MAGIC │  • Crisp topic sep.      │     │  • Soft topic assign.     │
# MAGIC │  • Fast to compute       │     │  • Documents span topics  │
# MAGIC │  • Great for sweep &     │     │  • Better for mixed-      │
# MAGIC │    inspect workflow       │     │    topic documents       │
# MAGIC │                          │     │                           │
# MAGIC │  Use when: you want      │     │  Use when: emails may     │
# MAGIC │  clean, distinct risk    │     │  legitimately belong to   │
# MAGIC │  categories              │     │  multiple risk types      │
# MAGIC └──────────────────────────┘     └───────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Docs:**
# MAGIC - [Spark MLlib LDA](https://spark.apache.org/docs/latest/ml-clustering.html#latent-dirichlet-allocation-lda)
# MAGIC - [Blog: Topic Extraction from Text with PySpark](https://www.databricks.com/blog/topic-extraction-text-pyspark)
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
dbutils.widgets.text("best_k", "6", "Number of Topics (match NMF)")
from datetime import datetime
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

# MAGIC %md
# MAGIC ## Step 1: Load Artifacts

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

# Read via Spark instead of pd.read_parquet (avoids pyarrow version mismatch)
pdf = spark.read.parquet(f"{artifact_path}/emails_with_nmf_topics.parquet").toPandas()
feature_names = tfidf.get_feature_names_out()

print(f"Loaded {len(pdf)} emails with NMF topics")
print(f"TF-IDF matrix: {tfidf_matrix.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Run LDA with scikit-learn
# MAGIC
# MAGIC We use scikit-learn's LDA (which uses variational Bayes) to match the NMF K value
# MAGIC for direct comparison. We also need a count-based matrix (not TF-IDF) for LDA —
# MAGIC technically LDA expects raw counts, but TF-IDF works reasonably well in practice.
# MAGIC
# MAGIC For a more principled approach, we'll build a CountVectorizer with the same vocabulary.

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import mlflow
import mlflow.sklearn
# Register sklearn integration so DBR's MLflow autologging shim doesn't KeyError on fit
mlflow.sklearn.autolog(disable=True)
import time
import os

# Build count matrix with same params as TF-IDF
count_vec = CountVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.85,
    stop_words="english",
    ngram_range=(1, 2),
)
count_matrix = count_vec.fit_transform(pdf["clean_body"])
count_features = count_vec.get_feature_names_out()

print(f"Count matrix shape: {count_matrix.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train LDA Model

# COMMAND ----------

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
mlflow.set_experiment(f"{os.path.dirname(notebook_path)}/nlp_email_risk_lda")

with mlflow.start_run(run_name=f"lda_k{BEST_K}"):
    start = time.time()

    lda_model = LatentDirichletAllocation(
        n_components=BEST_K,
        random_state=42,
        max_iter=50,
        learning_method="online",
        learning_offset=50.0,
        doc_topic_prior=0.1,       # Alpha — sparse document-topic distribution
        topic_word_prior=0.01,     # Beta — sparse topic-word distribution
    )

    lda_W = lda_model.fit_transform(count_matrix)  # doc-topic distribution

    elapsed = time.time() - start
    perplexity = lda_model.perplexity(count_matrix)
    log_likelihood = lda_model.score(count_matrix)

    mlflow.log_param("n_topics", BEST_K)
    mlflow.log_metric("perplexity", perplexity)
    mlflow.log_metric("log_likelihood", log_likelihood)
    mlflow.log_metric("fit_time_seconds", elapsed)

    print(f"LDA K={BEST_K} | Perplexity={perplexity:.2f} | Log-likelihood={log_likelihood:.2f} | Time={elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Inspect LDA Topics

# COMMAND ----------

print(f"LDA Topics (K={BEST_K}) — Top 15 Words\n")

lda_top_words = {}
for topic_idx in range(BEST_K):
    top_idx = lda_model.components_[topic_idx].argsort()[-15:][::-1]
    words = [(count_features[i], lda_model.components_[topic_idx][i]) for i in top_idx]
    lda_top_words[topic_idx] = words

    print(f"Topic {topic_idx}:")
    for word, score in words:
        bar = "█" * int(score / max(s for _, s in words) * 30)
        print(f"  {word:25s} {score:8.1f} {bar}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Compare NMF vs LDA Topics
# MAGIC
# MAGIC Do the two methods discover the same underlying topic structure?
# MAGIC Let's compare them side by side.

# COMMAND ----------

# Load NMF results
with open(f"{artifact_path}/nmf_H_k{BEST_K}.pkl", "rb") as f:
    nmf_H = pickle.load(f)

print(f"{'NMF Topics':^45s} | {'LDA Topics':^45s}")
print(f"{'-'*45}-+-{'-'*45}")

for topic_idx in range(BEST_K):
    # NMF top words
    nmf_top_idx = nmf_H[topic_idx].argsort()[-8:][::-1]
    nmf_words = ", ".join(feature_names[i] for i in nmf_top_idx)

    # LDA top words
    lda_top_idx = lda_model.components_[topic_idx].argsort()[-8:][::-1]
    lda_words = ", ".join(count_features[i] for i in lda_top_idx)

    print(f"T{topic_idx}: {nmf_words[:43]:43s} | T{topic_idx}: {lda_words[:43]:43s}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Soft Topic Assignment
# MAGIC
# MAGIC The key advantage of LDA: each email gets a **probability distribution** over topics.
# MAGIC This reveals emails that span multiple risk categories.

# COMMAND ----------

# Normalize LDA weights to probabilities
lda_probs = lda_W / lda_W.sum(axis=1, keepdims=True)

pdf["lda_dominant_topic"] = lda_probs.argmax(axis=1)
pdf["lda_max_prob"] = lda_probs.max(axis=1)

# Entropy: high entropy = email spans many topics
from scipy.stats import entropy
pdf["lda_entropy"] = [entropy(row) for row in lda_probs]

for i in range(BEST_K):
    pdf[f"lda_topic_{i}_prob"] = lda_probs[:, i]

print("LDA topic distribution:")
print(pdf["lda_dominant_topic"].value_counts().sort_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ### High-Entropy Emails (Span Multiple Topics)
# MAGIC
# MAGIC These emails don't fit neatly into one topic — they may reference multiple risk categories
# MAGIC and could warrant additional analyst review.

# COMMAND ----------

multi_topic = pdf.nlargest(10, "lda_entropy")
for _, row in multi_topic.iterrows():
    print(f"Entropy: {row['lda_entropy']:.3f} | NMF Topic: {row['dominant_topic']} | LDA Topic: {row['lda_dominant_topic']}")
    probs = [f"T{i}:{row[f'lda_topic_{i}_prob']:.2f}" for i in range(BEST_K)]
    print(f"  LDA probs: {', '.join(probs)}")
    print(f"  Body: {row['body'][:150]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: NMF vs LDA Agreement
# MAGIC
# MAGIC How often do NMF and LDA assign the same dominant topic?
# MAGIC High agreement = robust topic structure. Low agreement = topics may need reinterpretation.

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Note: topic indices may not align between NMF and LDA, so this is a rough comparison
agreement = (pdf["dominant_topic"] == pdf["lda_dominant_topic"]).mean()
print(f"Exact dominant topic agreement: {agreement:.1%}")

# Cross-tabulation
ct = pd.crosstab(pdf["dominant_topic"], pdf["lda_dominant_topic"],
                 rownames=["NMF Topic"], colnames=["LDA Topic"])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("NMF vs LDA Dominant Topic Cross-Tabulation", fontsize=14)
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Interactive Topic Visualization with pyLDAvis
# MAGIC
# MAGIC pyLDAvis creates an interactive visualization showing:
# MAGIC - **Left panel**: Topics as circles — distance = topic dissimilarity, size = prevalence
# MAGIC - **Right panel**: Top terms for selected topic — blue bars = overall frequency, red = topic-specific
# MAGIC
# MAGIC > **Note:** pyLDAvis rendering may vary by Databricks Runtime version. If the interactive widget
# MAGIC > doesn't render inline, the HTML is saved to the Volume and can be downloaded.

# COMMAND ----------

import pyLDAvis
import pyLDAvis.lda_model

vis_data = pyLDAvis.lda_model.prepare(lda_model, count_matrix, count_vec)

# Save as HTML for portability
html_path = f"{artifact_path}/lda_vis.html"
pyLDAvis.save_html(vis_data, html_path)
print(f"✓ Saved interactive LDA visualization to {html_path}")

# Display inline
pyLDAvis.display(vis_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Save LDA Results

# COMMAND ----------

spark.createDataFrame(pdf).write.mode("overwrite").parquet(f"{artifact_path}/emails_with_all_topics.parquet")

with open(f"{artifact_path}/lda_model_k{BEST_K}.pkl", "wb") as f:
    pickle.dump(lda_model, f)
with open(f"{artifact_path}/count_vectorizer.pkl", "wb") as f:
    pickle.dump(count_vec, f)

# Save combined topic table
df_combined = spark.createDataFrame(
    pdf[["email_id", "dominant_topic", "topic_confidence", "lda_dominant_topic", "lda_max_prob", "lda_entropy"]]
)
df_combined.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE}.email_all_topics{SUFFIX_TAG}")

print(f"✓ Saved LDA model and combined topic assignments")
print(f"✓ Delta table: {DATABASE}.email_all_topics{SUFFIX_TAG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC | Method | Strength | Best For |
# MAGIC |--------|----------|----------|
# MAGIC | **NMF** | Crisp, non-overlapping topics; fast sweep & inspect | Primary risk categorization, elbow analysis |
# MAGIC | **LDA** | Soft probabilistic assignment; multi-topic emails | Finding emails that span risk categories |
# MAGIC | **Both together** | Cross-validation of topic structure | Robust risk taxonomy when methods agree |
# MAGIC
# MAGIC **Next →** Open `04_embeddings_clustering` for semantic embedding-based clustering with sentence-transformers.

# COMMAND ----------

