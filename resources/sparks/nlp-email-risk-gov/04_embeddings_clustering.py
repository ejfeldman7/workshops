# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 1: Email Risk Classification with NLP
# MAGIC ## Notebook 4 — Embeddings & Clustering
# MAGIC
# MAGIC Topic models (NMF, LDA) work on word frequencies. **Sentence embeddings** capture deeper semantic meaning —
# MAGIC two emails can use completely different words but have similar meaning, and embeddings will recognize that.
# MAGIC
# MAGIC ### Approach
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
# MAGIC │  Email Text  │───▶│  Sentence    │───▶│    UMAP      │───▶│   HDBSCAN    │
# MAGIC │              │    │  Transformer │    │  (384D→2D)   │    │  (Clusters)  │
# MAGIC │              │    │  (384D emb.) │    │              │    │              │
# MAGIC └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Why this matters for risk classification:**
# MAGIC - "I forwarded files to my personal email" and "I uploaded data to my Dropbox" → **same risk cluster** despite different words
# MAGIC - Embeddings capture intent and context, not just vocabulary
# MAGIC
# MAGIC **Docs:**
# MAGIC - [Hugging Face on Databricks](https://docs.databricks.com/en/machine-learning/train-model/huggingface/index.html)
# MAGIC - [sentence-transformers](https://www.sbert.net/)
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

import pickle
import pandas as pd
import numpy as np

import os
import re
_user = spark.sql("SELECT current_user()").first()[0]
USER_ID = re.sub(r'[^a-zA-Z0-9]', '_', _user.split('@')[0])
artifact_path = f"/dbfs/tmp/workshops/{DATABASE}/{USER_ID}"
os.makedirs(artifact_path, exist_ok=True)

pdf = spark.read.parquet(f"{artifact_path}/emails_with_all_topics.parquet").toPandas()

# Ensure the hidden ground-truth label is present for the visualization below.
# Skip the join if it's already carried through from upstream — otherwise we'd get _x/_y suffix collision.
if "_synthetic_label" not in pdf.columns:
    labels_pdf = (
        spark.table(f"{DATABASE}.emails_bronze")
        .select("email_id", "_synthetic_label")
        .toPandas()
    )
    pdf = pdf.merge(labels_pdf, on="email_id", how="left")

print(f"Loaded {len(pdf)} emails with NMF + LDA topics")
print(f"Ground-truth label coverage: {pdf['_synthetic_label'].notna().sum()} / {len(pdf)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Generate Sentence Embeddings
# MAGIC
# MAGIC We use **all-MiniLM-L6-v2** — a small, fast model that produces 384-dimensional embeddings.
# MAGIC It's pre-trained on 1B+ sentence pairs and works well for semantic similarity tasks.
# MAGIC
# MAGIC > **Azure Gov Cloud Note:** This model runs entirely locally on the cluster — no external API calls required.
# MAGIC > The model weights are downloaded once and cached. For air-gapped environments, pre-download to a Volume.

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import time

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
start = time.time()
embeddings = model.encode(
    pdf["body"].tolist(),
    show_progress_bar=True,
    batch_size=64,
)
elapsed = time.time() - start

print(f"✓ Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} in {elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: UMAP Dimensionality Reduction
# MAGIC
# MAGIC 384 dimensions is too many to visualize or cluster efficiently. **UMAP** (Uniform Manifold
# MAGIC Approximation and Projection) reduces to 2D while preserving the local neighborhood structure.
# MAGIC
# MAGIC | Parameter | Value | Why |
# MAGIC |-----------|-------|-----|
# MAGIC | `n_neighbors` | 15 | Balance local vs global structure |
# MAGIC | `min_dist` | 0.1 | Allow tight clusters |
# MAGIC | `metric` | cosine | Standard for text embeddings |
# MAGIC
# MAGIC **Reference:** [UMAP documentation](https://umap-learn.readthedocs.io/)

# COMMAND ----------

import umap

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)

print("Running UMAP...")
start = time.time()
umap_embeddings = reducer.fit_transform(embeddings)
elapsed = time.time() - start
print(f"✓ UMAP reduction complete in {elapsed:.1f}s")

pdf["umap_x"] = umap_embeddings[:, 0]
pdf["umap_y"] = umap_embeddings[:, 1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Visualize Embedding Space
# MAGIC
# MAGIC Let's see how emails cluster in the 2D UMAP space, colored by NMF topic assignment.

# COMMAND ----------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(
    pdf["umap_x"], pdf["umap_y"],
    c=pdf["dominant_topic"],
    cmap="tab10",
    alpha=0.5,
    s=8,
)
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title("Email Embeddings (UMAP 2D) — Colored by NMF Topic", fontsize=14)
plt.colorbar(scatter, ax=ax, label="NMF Topic")
ax.grid(True, alpha=0.2)
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: HDBSCAN Clustering
# MAGIC
# MAGIC **HDBSCAN** (Hierarchical Density-Based Spatial Clustering) is ideal here because:
# MAGIC - It **automatically determines the number of clusters** (no need to specify K)
# MAGIC - It identifies **noise points** — emails that don't fit any cluster
# MAGIC - It's robust to clusters of different sizes and densities
# MAGIC
# MAGIC Noise points (cluster = -1) are particularly interesting for risk: they're outlier emails
# MAGIC that don't match normal patterns.
# MAGIC
# MAGIC **Reference:** [HDBSCAN documentation](https://hdbscan.readthedocs.io/)

# COMMAND ----------

import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=80,       # Minimum cluster size (tuned for templated synthetic data)
    min_samples=15,            # Core point threshold
    metric="euclidean",        # On UMAP output
    cluster_selection_method="eom",  # Excess of mass
)

print("Running HDBSCAN...")
cluster_labels = clusterer.fit_predict(umap_embeddings)

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = (cluster_labels == -1).sum()

pdf["embedding_cluster"] = cluster_labels
pdf["cluster_probability"] = clusterer.probabilities_

print(f"✓ Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(pdf):.1%})")
print(f"\nCluster distribution:")
for c in sorted(pdf["embedding_cluster"].unique()):
    count = (pdf["embedding_cluster"] == c).sum()
    label = "NOISE" if c == -1 else f"Cluster {c}"
    print(f"  {label}: {count} emails ({count/len(pdf):.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Embedding Clusters Visualization

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# --- Left: HDBSCAN clusters ---
ax = axes[0]
unique_clusters = sorted(pdf["embedding_cluster"].unique())
real_clusters = [c for c in unique_clusters if c != -1]
palette = plt.cm.tab20(np.linspace(0, 1, max(len(real_clusters), 1)))
for i, cluster in enumerate(real_clusters):
    mask = pdf["embedding_cluster"] == cluster
    ax.scatter(pdf.loc[mask, "umap_x"], pdf.loc[mask, "umap_y"],
               c=[palette[i]], alpha=0.6, s=8, label=f"Cluster {cluster}")
# Plot noise underneath
noise_mask = pdf["embedding_cluster"] == -1
if noise_mask.any():
    ax.scatter(pdf.loc[noise_mask, "umap_x"], pdf.loc[noise_mask, "umap_y"],
               c="lightgray", alpha=0.2, s=6, label="Noise", zorder=0)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title(f"HDBSCAN ({len(real_clusters)} clusters, {noise_mask.sum()} noise)", fontsize=13)
ax.legend(fontsize=7, markerscale=2, loc="best", ncol=2)
ax.grid(True, alpha=0.2)

# --- Middle: NMF topics ---
ax = axes[1]
scatter = ax.scatter(pdf["umap_x"], pdf["umap_y"],
                     c=pdf["dominant_topic"], cmap="tab20", alpha=0.5, s=8)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_title(f"NMF Topics (K={pdf['dominant_topic'].nunique()})", fontsize=13)
plt.colorbar(scatter, ax=ax, label="NMF Topic")
ax.grid(True, alpha=0.2)

# --- Right: Ground truth (if available) ---
ax = axes[2]
if "_synthetic_label" in pdf.columns:
    labels = sorted(pdf["_synthetic_label"].unique())
    label_palette = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for i, lab in enumerate(labels):
        mask = pdf["_synthetic_label"] == lab
        ax.scatter(pdf.loc[mask, "umap_x"], pdf.loc[mask, "umap_y"],
                   c=[label_palette[i]], alpha=0.6, s=8, label=lab)
    ax.legend(fontsize=8, markerscale=2, loc="best")
    ax.set_title("Ground Truth (synthetic_label)", fontsize=13)
else:
    ax.text(0.5, 0.5, "_synthetic_label not in pdf",
            ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Ground Truth (not available)", fontsize=13)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.grid(True, alpha=0.2)

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Compare Embeddings Clusters with NMF/LDA Topics
# MAGIC
# MAGIC Do the embedding-based clusters align with the topic model results?
# MAGIC Agreement across methods gives us confidence in the risk categories.

# COMMAND ----------

# Cross-tabulation: embedding clusters vs NMF topics
ct = pd.crosstab(pdf["embedding_cluster"], pdf["dominant_topic"],
                 rownames=["Embedding Cluster"], colnames=["NMF Topic"])

import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
ax.set_title("Embedding Clusters vs NMF Topics", fontsize=14)
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Noise Points Analysis
# MAGIC
# MAGIC Emails classified as noise by HDBSCAN don't fit any cluster pattern.
# MAGIC These are worth investigating — they may represent unusual/novel risk types.

# COMMAND ----------

noise_emails = pdf[pdf["embedding_cluster"] == -1].sample(min(10, n_noise), random_state=42)
print(f"Sample noise emails ({n_noise} total):\n")
for _, row in noise_emails.iterrows():
    print(f"  NMF Topic: {row['dominant_topic']} | LDA Topic: {row['lda_dominant_topic']}")
    print(f"  Body: {row['body'][:200]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Save Embedding Results

# COMMAND ----------

# Save embeddings
np.save(f"{artifact_path}/embeddings_384d.npy", embeddings)
np.save(f"{artifact_path}/umap_2d.npy", umap_embeddings)
# Write via pandas (FUSE path) to avoid Spark/dbfs path-translation issues —
# 05 and 06 both read this back with pd.read_parquet anyway.
pdf.to_parquet(f"{artifact_path}/emails_full_analysis.parquet", index=False)

# Save to Delta
df_clusters = spark.createDataFrame(
    pdf[["email_id", "dominant_topic", "lda_dominant_topic", "embedding_cluster",
         "cluster_probability", "umap_x", "umap_y"]]
)
df_clusters.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE}.email_clusters{SUFFIX_TAG}")

print(f"✓ Saved embeddings, UMAP coordinates, and cluster assignments")
print(f"✓ Delta table: {DATABASE}.email_clusters{SUFFIX_TAG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC | Method | What It Captures | Strength |
# MAGIC |--------|-----------------|----------|
# MAGIC | **NMF** | Word-level topic patterns | Interpretable, sweepable |
# MAGIC | **LDA** | Probabilistic topic mixtures | Multi-topic documents |
# MAGIC | **Embeddings + HDBSCAN** | Semantic meaning | Catches synonyms, paraphrases; finds outliers |
# MAGIC
# MAGIC **When all three agree on a risk category, you can be highly confident in the assignment.**
# MAGIC When they disagree, it's a signal for human review.
# MAGIC
# MAGIC **Next →** Open `05_risk_scoring` to build a composite risk score from all three methods.

# COMMAND ----------

