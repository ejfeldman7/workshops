# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 1: Email Risk Classification with NLP
# MAGIC ## Notebook 6 — Educational: Supervised Approaches (Future Reference)
# MAGIC
# MAGIC > ⚠️ **This notebook is educational only.** SNC does not have labeled email data today.
# MAGIC > The code below demonstrates supervised approaches for when labeled data becomes available.
# MAGIC > One path to labels: have analysts review and label the unsupervised clusters from notebooks 02–04.
# MAGIC
# MAGIC ### Supervised Learning Roadmap
# MAGIC
# MAGIC ```
# MAGIC ┌────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                   Path from Unsupervised → Supervised                      │
# MAGIC │                                                                            │
# MAGIC │  TODAY (No Labels)              NEAR-TERM                FUTURE            │
# MAGIC │  ┌──────────────┐    ┌───────────────────────┐    ┌──────────────────┐     │
# MAGIC │  │ Unsupervised │───▶│ Analysts review NMF/  │───▶│ Fine-tune        │     │
# MAGIC │  │ NMF/LDA/Emb  │    │ LDA clusters, label   │    │ DistilBERT on    │     │
# MAGIC │  │ Risk Scoring  │    │ 500-1000 emails       │    │ labeled data     │    │
# MAGIC │  └──────────────┘    └───────────────────────┘    └──────────────────┘     │
# MAGIC │                                                                            │
# MAGIC │  ✅ Available now    ✅ Available now (human)     ✅ Available on Gov Cloud   │
# MAGIC └────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Docs:**
# MAGIC - [Fine-tune Hugging Face on Databricks](https://docs.databricks.com/en/machine-learning/train-model/huggingface/fine-tune-model.html)
# MAGIC - [scikit-learn on Databricks](https://docs.databricks.com/en/machine-learning/train-model/scikit-learn.html)
# MAGIC - [MLflow Model Registry](https://docs.databricks.com/en/mlflow/index.html)
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

import pandas as pd
import numpy as np

import os
import re
_user = spark.sql("SELECT current_user()").first()[0]
USER_ID = re.sub(r'[^a-zA-Z0-9]', '_', _user.split('@')[0])
artifact_path = f"/dbfs/tmp/workshops/{DATABASE}/{USER_ID}"
os.makedirs(artifact_path, exist_ok=True)
pdf = pd.read_parquet(f"{artifact_path}/emails_full_analysis.parquet")

# Ensure the hidden ground-truth label is present (used as supervised target below).
# If notebook 04 already joined it into the parquet, this is a no-op; otherwise we join here.
if "_synthetic_label" not in pdf.columns:
    labels_pdf = (
        spark.table(f"{DATABASE}.emails_bronze")
        .select("email_id", "_synthetic_label")
        .toPandas()
    )
    pdf = pdf.merge(labels_pdf, on="email_id", how="left")

print(f"Loaded {len(pdf)} emails (label coverage: {pdf['_synthetic_label'].notna().sum()}/{len(pdf)})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 1: TF-IDF + XGBoost (Fast Baseline)
# MAGIC
# MAGIC When you have labels, the fastest supervised baseline is:
# MAGIC 1. TF-IDF features (already built in notebook 01)
# MAGIC 2. XGBoost classifier
# MAGIC 3. 5-fold cross-validation
# MAGIC
# MAGIC **Why XGBoost first?** It trains in seconds, handles class imbalance, and provides feature importance.
# MAGIC This tells you immediately whether supervised learning will work and which words are most predictive.
# MAGIC
# MAGIC > **Note:** We use the `_synthetic_label` column as a stand-in for real labels.
# MAGIC > This column exists only in our synthetic data and would NOT exist in production.

# COMMAND ----------

import pickle
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load TF-IDF
with open(f"{artifact_path}/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open(f"{artifact_path}/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

# Encode labels (using synthetic labels for demo)
le = LabelEncoder()
y = le.fit_transform(pdf["_synthetic_label"])

print(f"Classes: {le.classes_}")
print(f"Distribution: {np.bincount(y)}")

# COMMAND ----------

# Cross-validated XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
)

scores = cross_val_score(xgb, tfidf_matrix, y, cv=5, scoring="f1_weighted")
print(f"5-Fold CV F1 (weighted): {scores.mean():.3f} ± {scores.std():.3f}")

# Train final model and show classification report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42, stratify=y)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Importance — Which Words Predict Risk?

# COMMAND ----------

import matplotlib.pyplot as plt

feature_names = tfidf.get_feature_names_out()
importance = xgb.feature_importances_
top_idx = importance.argsort()[-25:][::-1]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(25), importance[top_idx][::-1], color="steelblue")
ax.set_yticks(range(25))
ax.set_yticklabels([feature_names[i] for i in top_idx][::-1])
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_title("Top 25 Most Predictive Terms (XGBoost)", fontsize=14)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 2: Fine-Tuning DistilBERT
# MAGIC
# MAGIC For higher accuracy (especially on nuanced emails), fine-tune a pre-trained transformer.
# MAGIC **DistilBERT** is a good choice: 40% smaller than BERT, 60% faster, retains 97% of performance.
# MAGIC
# MAGIC ### Requirements
# MAGIC - GPU cluster (e.g., Standard_NC4as_T4_v3 on Azure Gov Cloud)
# MAGIC - At least 500 labeled examples per class
# MAGIC - Databricks Runtime 13.3+ ML
# MAGIC
# MAGIC **Azure Gov Cloud:** ✅ This runs on classic GPU compute — no Model Serving needed.
# MAGIC Fine-tuning happens on the cluster; inference is batch via notebook job.
# MAGIC
# MAGIC > The code below is a complete working example. Uncomment and run when labeled data is available.

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC # === UNCOMMENT WHEN LABELED DATA IS AVAILABLE ===
# MAGIC
# MAGIC from transformers import (
# MAGIC     AutoTokenizer, AutoModelForSequenceClassification,
# MAGIC     TrainingArguments, Trainer
# MAGIC )
# MAGIC from datasets import Dataset
# MAGIC import torch
# MAGIC
# MAGIC # Prepare dataset
# MAGIC tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# MAGIC
# MAGIC def tokenize(batch):
# MAGIC     return tokenizer(batch["body"], padding="max_length", truncation=True, max_length=256)
# MAGIC
# MAGIC train_ds = Dataset.from_pandas(train_pdf[["body", "label"]])
# MAGIC test_ds = Dataset.from_pandas(test_pdf[["body", "label"]])
# MAGIC
# MAGIC train_ds = train_ds.map(tokenize, batched=True)
# MAGIC test_ds = test_ds.map(tokenize, batched=True)
# MAGIC
# MAGIC # Load model
# MAGIC model = AutoModelForSequenceClassification.from_pretrained(
# MAGIC     "distilbert-base-uncased",
# MAGIC     num_labels=len(label_classes),
# MAGIC )
# MAGIC
# MAGIC # Training
# MAGIC training_args = TrainingArguments(
# MAGIC     output_dir="/tmp/distilbert-email-risk",
# MAGIC     num_train_epochs=3,
# MAGIC     per_device_train_batch_size=16,
# MAGIC     per_device_eval_batch_size=32,
# MAGIC     evaluation_strategy="epoch",
# MAGIC     save_strategy="epoch",
# MAGIC     learning_rate=2e-5,
# MAGIC     weight_decay=0.01,
# MAGIC     load_best_model_at_end=True,
# MAGIC     metric_for_best_model="f1",
# MAGIC )
# MAGIC
# MAGIC trainer = Trainer(
# MAGIC     model=model,
# MAGIC     args=training_args,
# MAGIC     train_dataset=train_ds,
# MAGIC     eval_dataset=test_ds,
# MAGIC     tokenizer=tokenizer,
# MAGIC )
# MAGIC
# MAGIC # Train and log to MLflow
# MAGIC with mlflow.start_run(run_name="distilbert_email_risk"):
# MAGIC     trainer.train()
# MAGIC     mlflow.log_metrics(trainer.evaluate())
# MAGIC     mlflow.transformers.log_model(
# MAGIC         transformers_model={"model": model, "tokenizer": tokenizer},
# MAGIC         artifact_path="model",
# MAGIC     )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 3: Zero-Shot Classification
# MAGIC
# MAGIC If you have a GPU but no labeled data, **zero-shot classification** uses a pre-trained NLI model
# MAGIC to classify text against arbitrary candidate labels — no training required.
# MAGIC
# MAGIC > **Azure Gov Cloud:** ✅ Runs locally on GPU cluster. Model downloaded once and cached.

# COMMAND ----------

from transformers import pipeline

# Load zero-shot classifier
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if __import__("torch").cuda.is_available() else -1,
)

candidate_labels = [
    "data exfiltration or unauthorized data transfer",
    "security policy violation or credential sharing",
    "phishing or social engineering attempt",
    "HR complaint or personnel issue",
    "financial irregularity or expense fraud",
    "normal business communication",
]

# Score a sample
sample_emails = pdf.sample(5, random_state=42)

for _, row in sample_emails.iterrows():
    result = classifier(row["body"][:512], candidate_labels, multi_label=True)
    print(f"Email: {row['body'][:100]}...")
    for label, score in zip(result["labels"][:3], result["scores"][:3]):
        bar = "█" * int(score * 30)
        print(f"  {score:.3f} {bar} {label}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: When to Use Each Approach
# MAGIC
# MAGIC | Approach | Data Needed | Training Time | Accuracy | Azure Gov Cloud |
# MAGIC |----------|-------------|--------------|----------|-----------------|
# MAGIC | **NMF/LDA** (unsupervised) | None | Seconds | Good for discovery | ✅ Works today |
# MAGIC | **Zero-shot** (no training) | None | None | Moderate | ✅ Works today (GPU) |
# MAGIC | **TF-IDF + XGBoost** | 500+ labels | Seconds | Good | ✅ Works today |
# MAGIC | **DistilBERT fine-tune** | 500+ labels/class | Minutes (GPU) | Excellent | ✅ Works today (GPU) |
# MAGIC | **Foundation Model APIs** | None | None | Excellent | ❌ Not in Gov Cloud yet |
# MAGIC
# MAGIC ### Recommended Path for SNC
# MAGIC 1. **Now:** Use unsupervised pipeline (notebooks 02–05) to score emails and discover risk categories
# MAGIC 2. **Short-term:** Analysts review clusters and label 500–1000 emails
# MAGIC 3. **With labels:** Train XGBoost baseline, then fine-tune DistilBERT for production quality
# MAGIC 4. **Post-Evergreen:** Consider Foundation Model APIs and Model Serving for real-time scoring
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🎉 Workshop 1 Complete!** You've built a full NLP risk classification pipeline using entirely
# MAGIC unsupervised methods that work in Azure Gov Cloud today.