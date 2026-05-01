# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 1: Email Risk Classification with NLP
# MAGIC ## Notebook 1 — Data Exploration & Preprocessing
# MAGIC
# MAGIC Explore the email dataset, understand text distributions, and prepare the data for NLP modeling.
# MAGIC
# MAGIC **What you'll learn:**
# MAGIC - How to profile text data at scale with Spark
# MAGIC - Text cleaning and tokenization strategies
# MAGIC - Building a TF-IDF matrix for topic modeling
# MAGIC
# MAGIC **Docs:**
# MAGIC - [Hugging Face Tokenizers on Databricks](https://docs.databricks.com/en/machine-learning/train-model/huggingface/index.html)
# MAGIC - [MLflow Experiment Tracking](https://docs.databricks.com/en/mlflow/index.html)
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC Run this cell if you're starting from this notebook directly (skipped notebook 00).

# COMMAND ----------

# MAGIC %pip install sentence-transformers umap-learn hdbscan pyLDAvis wordcloud --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime

dbutils.widgets.text("database", "nlp_email_risk", "Database")
_user_email = spark.sql("SELECT current_user()").first()[0]
_name_parts = _user_email.split('@')[0].replace('_', '.').split('.')
_initials = (_name_parts[0][0] + _name_parts[-1][0]).lower() if len(_name_parts) >= 2 else _user_email[:2].lower()
_default_suffix = _initials + datetime.now().strftime('%d%m%y')
dbutils.widgets.text("table_suffix", _default_suffix, "Table Suffix (your initials)")

DATABASE = dbutils.widgets.get("database")
SUFFIX = dbutils.widgets.get("table_suffix").strip()
SUFFIX_TAG = f"_{SUFFIX}" if SUFFIX else ""
print(f"Database: {DATABASE}, Table suffix: {SUFFIX_TAG or '(none)'}")
spark.sql(f"USE {DATABASE}")

TABLE = f"{DATABASE}.emails_bronze"
print(f"Reading from: {TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load and Profile the Data

# COMMAND ----------

df = spark.read.table(TABLE)
print(f"Total emails: {df.count()}")
print(f"Columns: {df.columns}")
print(f"Date range: {df.selectExpr('min(timestamp)', 'max(timestamp)').first()}")

# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Text Length Distribution
# MAGIC
# MAGIC Understanding the distribution of email body lengths helps us choose appropriate model parameters
# MAGIC (e.g., max sequence length for transformers, min_df/max_df for TF-IDF).

# COMMAND ----------

from pyspark.sql import functions as F

df_stats = df.withColumn("body_length", F.length("body")) \
             .withColumn("word_count", F.size(F.split("body", r"\s+"))) \
             .withColumn("subject_length", F.length("subject"))

display(
    df_stats.select("body_length", "word_count", "subject_length")
            .summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Body Length Histogram

# COMMAND ----------

display(df_stats.select("word_count"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Attachment & Size Analysis
# MAGIC
# MAGIC Attachment presence and email size are important risk signals:
# MAGIC - **Data exfiltration** emails tend to have large attachments (database exports, code archives)
# MAGIC - **Phishing** emails often include small, suspicious files (.html, .exe)
# MAGIC - **Normal business** emails have moderate sizes (presentations, meeting notes)
# MAGIC
# MAGIC ```
# MAGIC  Risk Signal:
# MAGIC  ┌─────────────────────────────────────────────────────────────────┐
# MAGIC  │  Size (MB)    0.01    0.1     1      10      50     200+      │
# MAGIC  │               │───────│───────│───────│───────│───────│        │
# MAGIC  │  Normal       ████████████                                     │
# MAGIC  │  Phishing     ████                                             │
# MAGIC  │  Exfiltration              ████████████████████████████        │
# MAGIC  └─────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# Attachment rate and size by category (using synthetic labels for validation)
display(
    df.groupBy("_synthetic_label")
      .agg(
          F.count("*").alias("email_count"),
          F.avg(F.col("has_attachment").cast("int")).alias("attachment_rate"),
          F.avg("attachment_count").alias("avg_attachments"),
          F.avg("size_mb").alias("avg_size_mb"),
          F.max("size_mb").alias("max_size_mb"),
      )
      .orderBy("avg_size_mb", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Email Size Distribution

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

size_pdf = df.select("size_mb", "has_attachment").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Size distribution (log scale)
ax = axes[0]
ax.hist(size_pdf["size_mb"], bins=100, color="steelblue", alpha=0.7, log=True)
ax.set_xlabel("Email Size (MB)", fontsize=12)
ax.set_ylabel("Count (log scale)", fontsize=12)
ax.set_title("Email Size Distribution", fontsize=14)
ax.axvline(x=10, color="red", linestyle="--", alpha=0.7, label="10 MB threshold")
ax.legend()
ax.grid(True, alpha=0.3)

# Size by attachment presence
ax = axes[1]
with_att = size_pdf[size_pdf["has_attachment"] == True]["size_mb"]
without_att = size_pdf[size_pdf["has_attachment"] == False]["size_mb"]
ax.hist(without_att, bins=50, alpha=0.6, label=f"No attachment (n={len(without_att)})", color="steelblue")
ax.hist(with_att, bins=50, alpha=0.6, label=f"Has attachment (n={len(with_att)})", color="coral")
ax.set_xlabel("Email Size (MB)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Size by Attachment Presence", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Attachment Type Frequency
# MAGIC
# MAGIC What types of files are being attached? Certain extensions (.exe, .html, .sql, .tar.gz) are higher-risk.

# COMMAND ----------

from pyspark.sql.functions import explode, split

attachment_types = df.filter("attachment_names IS NOT NULL") \
    .select(explode(split("attachment_names", "\\|")).alias("filename")) \
    .withColumn("extension", F.regexp_extract("filename", r"\.([^.]+)$", 1))

display(
    attachment_types.groupBy("extension")
        .agg(F.count("*").alias("count"))
        .orderBy("count", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sender Activity Distribution
# MAGIC
# MAGIC Who sends the most emails? Unusual volume from a single sender could itself be a risk signal.

# COMMAND ----------

display(
    df_stats.groupBy("sender")
            .agg(F.count("*").alias("email_count"), F.avg("word_count").alias("avg_word_count"))
            .orderBy("email_count", ascending=False)
            .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Temporal Patterns
# MAGIC
# MAGIC Emails sent at unusual hours (late night, weekends) may carry different risk profiles.

# COMMAND ----------

df_temporal = df_stats.withColumn("hour", F.hour("timestamp")) \
                       .withColumn("day_of_week", F.dayofweek("timestamp"))

display(
    df_temporal.groupBy("hour")
               .agg(F.count("*").alias("count"))
               .orderBy("hour")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Text Cleaning
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────┐     ┌──────────────┐     ┌───────────────┐
# MAGIC │  Raw Email   │────▶│  Lowercase,  │────▶│  Clean Text   │
# MAGIC │  Body Text   │     │  Remove URLs, │     │  Ready for   │
# MAGIC │              │     │  Punctuation  │     │  TF-IDF      │
# MAGIC └──────────────┘     └──────────────┘     └───────────────┘
# MAGIC ```
# MAGIC
# MAGIC We apply minimal cleaning — enough to remove noise without losing semantic meaning:
# MAGIC 1. Lowercase
# MAGIC 2. Remove URLs and email addresses
# MAGIC 3. Remove special characters (keep alphanumeric and spaces)
# MAGIC 4. Collapse whitespace

# COMMAND ----------

import re
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(StringType())
def clean_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' EMAIL ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_clean = df.withColumn("clean_body", clean_text("body"))
display(df_clean.select("body", "clean_body").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Build TF-IDF Matrix
# MAGIC
# MAGIC **TF-IDF (Term Frequency–Inverse Document Frequency)** converts text into numerical features:
# MAGIC - **TF**: How often a word appears in a document
# MAGIC - **IDF**: How rare a word is across all documents
# MAGIC - **TF-IDF = TF × IDF**: Words that are frequent in a document but rare globally get high scores
# MAGIC
# MAGIC This matrix is the input for both **NMF** and **LDA** topic models in the next notebooks.
# MAGIC
# MAGIC **Reference:** [scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

# COMMAND ----------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Collect to pandas for scikit-learn processing
pdf = df_clean.select("email_id", "clean_body", "body", "sender", "recipient", "timestamp", "subject",
                       "has_attachment", "attachment_count", "attachment_names", "size_mb").toPandas()

# Build TF-IDF matrix
tfidf = TfidfVectorizer(
    max_features=5000,       # Keep top 5000 terms
    min_df=5,                # Term must appear in at least 5 documents
    max_df=0.85,             # Ignore terms appearing in >85% of documents
    stop_words="english",    # Remove common English stop words
    ngram_range=(1, 2),      # Include unigrams and bigrams
)

tfidf_matrix = tfidf.fit_transform(pdf["clean_body"])

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"  - {tfidf_matrix.shape[0]} documents")
print(f"  - {tfidf_matrix.shape[1]} features (terms)")
print(f"  - Sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top Terms by TF-IDF Score
# MAGIC
# MAGIC Let's see which terms have the highest average TF-IDF scores across the corpus.

# COMMAND ----------

import numpy as np

feature_names = tfidf.get_feature_names_out()
mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()

top_indices = mean_tfidf.argsort()[-30:][::-1]
top_terms = [(feature_names[i], float(mean_tfidf[i])) for i in top_indices]

display(spark.createDataFrame(top_terms, ["term", "mean_tfidf_score"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Save Processed Data
# MAGIC
# MAGIC We save both the cleaned DataFrame and the TF-IDF artifacts so subsequent notebooks can pick up without re-processing.

# COMMAND ----------

import os
import re
import pickle

# Ensure the volume exists

# Save cleaned dataframe as Delta table
df_clean.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE}.emails_cleaned{SUFFIX_TAG}")
print(f"✓ Saved cleaned emails to {DATABASE}.emails_cleaned{SUFFIX_TAG}")

# Save TF-IDF artifacts to Volume for cross-notebook use
_user = spark.sql("SELECT current_user()").first()[0]
USER_ID = re.sub(r'[^a-zA-Z0-9]', '_', _user.split('@')[0])
artifact_path = f"/dbfs/tmp/workshops/{DATABASE}/{USER_ID}"
os.makedirs(artifact_path, exist_ok=True)

with open(f"{artifact_path}/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open(f"{artifact_path}/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

# Save pandas DataFrame via Spark (avoids pyarrow version issues)
spark.createDataFrame(pdf).write.mode("overwrite").parquet(f"{artifact_path}/emails_pandas.parquet")

print(f"✓ Saved TF-IDF vectorizer, matrix, and pandas DataFrame to {artifact_path}/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Word Cloud Preview
# MAGIC
# MAGIC A quick visual of the most prominent terms in the corpus.

# COMMAND ----------

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(
    width=1000, height=400,
    background_color="white",
    max_words=100,
    colormap="viridis",
).fit_words(dict(zip(feature_names, mean_tfidf)))

fig, ax = plt.subplots(figsize=(14, 5))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
ax.set_title("Top Terms by Mean TF-IDF Score", fontsize=14)
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Asset | Location |
# MAGIC |-------|----------|
# MAGIC | Cleaned emails table | `{DATABASE}.emails_cleaned{SUFFIX_TAG}` |
# MAGIC | TF-IDF vectorizer | `{ARTIFACT_PATH}/tfidf_vectorizer.pkl` |
# MAGIC | TF-IDF matrix | `{ARTIFACT_PATH}/tfidf_matrix.pkl` |
# MAGIC | Pandas DataFrame | `{ARTIFACT_PATH}/emails_pandas.parquet` |
# MAGIC
# MAGIC **Next →** Open `02_nmf_topic_discovery` to run NMF topic sweep and find risk categories.
