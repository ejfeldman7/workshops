# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 1: Email Risk Classification with NLP
# MAGIC ## Notebook 0 — Setup & Configuration
# MAGIC
# MAGIC This notebook configures the workshop environment and generates synthetic email data for the hands-on exercises.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Sets your catalog and schema for all workshop assets
# MAGIC 2. Installs required libraries
# MAGIC 3. Generates realistic synthetic email data (since we don't have real email data loaded yet)
# MAGIC 4. Writes the data to a Delta table
# MAGIC
# MAGIC **Azure Gov Cloud Compatibility:** ✅ Everything in this notebook runs on classic compute with Databricks Runtime ML.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Architecture Overview
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────┐
# MAGIC │                    Workshop Data Flow                           │
# MAGIC │                                                                 │
# MAGIC │  [Synthetic Email Data]                                         │
# MAGIC │         │                                                       │
# MAGIC │         ▼                                                       │
# MAGIC │  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    │
# MAGIC │  │ Bronze Table │───▶│ TF-IDF / Emb │───▶│ NMF / LDA / Cls │    │
# MAGIC │  │ (Raw Email)  │    │ (Features)   │    │ (Risk Scores)   │    │
# MAGIC │  └─────────────┘    └──────────────┘    └──────────────────┘    │
# MAGIC │                                                                 │
# MAGIC │  All assets created in: <your_catalog>.<your_schema>            │
# MAGIC └─────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure Your Database
# MAGIC
# MAGIC Update the widgets below to point to your target catalog and schema. All tables, models, and experiments will be created there.

# COMMAND ----------

dbutils.widgets.text("database", "nlp_email_risk", "Database")

DATABASE = dbutils.widgets.get("database")

print(f"Workshop assets will be created in: {DATABASE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Install Required Libraries
# MAGIC
# MAGIC These libraries are used across all workshop notebooks. On Databricks Runtime ML 13.3+, most are pre-installed.
# MAGIC The additional installs below cover topic modeling visualization and embedding-based clustering.

# COMMAND ----------

# MAGIC %pip install sentence-transformers umap-learn hdbscan pyLDAvis wordcloud --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Database

# COMMAND ----------

DATABASE = dbutils.widgets.get("database")

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
spark.sql(f"USE {DATABASE}")

import os
ARTIFACT_PATH = f"/dbfs/tmp/workshops/{DATABASE}"
os.makedirs(ARTIFACT_PATH, exist_ok=True)
print(f"Using database: {DATABASE} (Hive metastore)")
print(f"Artifacts path: {ARTIFACT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Generate Synthetic Email Data
# MAGIC
# MAGIC Since SNC does not yet have email data loaded into Databricks, we generate a realistic synthetic dataset.
# MAGIC The data includes emails across several risk categories that the NLP models will discover:
# MAGIC
# MAGIC | Category | Description | Risk Level |
# MAGIC |----------|-------------|------------|
# MAGIC | **Data Exfiltration** | Emails discussing sending files to external accounts, USB transfers | High |
# MAGIC | **Policy Violation** | Discussions about bypassing security controls, sharing credentials | High |
# MAGIC | **Phishing Indicators** | Emails with suspicious links, urgency language, spoofed senders | High |
# MAGIC | **HR / Personnel** | Complaints, resignation discussions, performance issues | Medium |
# MAGIC | **Financial Irregularity** | Unusual expense requests, invoice discrepancies | Medium |
# MAGIC | **Normal Business** | Meeting requests, project updates, routine communications | Low |
# MAGIC
# MAGIC
# MAGIC Each email also includes **attachment metadata** and **overall size (MB)**:
# MAGIC - `has_attachment` — boolean flag
# MAGIC - `attachment_count` — number of attachments (0 if none)
# MAGIC - `attachment_names` — pipe-delimited filenames (e.g., `export.csv|backup.zip`)
# MAGIC - `size_mb` — total email size in MB (body + attachments)
# MAGIC
# MAGIC Data exfiltration emails tend to have larger attachments (database dumps, code archives).
# MAGIC Phishing emails often have small, suspicious attachments (.html, .exe).
# MAGIC
# MAGIC > **Note:** In production, this table would be populated from your email system via a one-time bulk load or streaming pipeline.

# COMMAND ----------

import random
import uuid
from datetime import datetime, timedelta
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

random.seed(42)

# --- Email templates by risk category ---

TEMPLATES = {
    "data_exfiltration": [
        "Can you send me the {doc_type} to my personal email {personal_email}? I need to review it over the weekend and VPN is too slow.",
        "I've uploaded the {doc_type} to {cloud_service}. Here's the link so you can access it from home.",
        "Is there a way to copy the {doc_type} to a USB drive? The file is too large for email and I need it for the {event} presentation.",
        "I forwarded the {doc_type} with the {data_type} data to my {personal_email} account. Easier to work from there.",
        "Attached is the {doc_type} with all {data_type} records. Please download before I remove it from the shared drive.",
        "I'm going to move the {data_type} files to my personal {cloud_service} tonight so I can work on the analysis from home.",
        "Can you zip up the entire {doc_type} directory and send it to {personal_email}? Need the full dataset.",
        "I exported all the {data_type} records to CSV — about 50k rows. Sending via {cloud_service} since email has a size limit.",
    ],
    "policy_violation": [
        "Here's the admin password for the {system} system: {password}. Don't share it but you'll need it to run the {task}.",
        "I disabled the {security_control} on my machine because it was blocking {software}. Can you do the same?",
        "Just use my credentials to log into {system} — username: {username}, password: {password}. I'll be out of office.",
        "I installed {software} on my workstation without going through IT. It's way faster for {task} than the approved tool.",
        "I shared the {system} API key in the {channel} channel. Everyone on the team needs access for the {task} sprint.",
        "The {security_control} keeps flagging our scripts as malicious. I added an exception for our entire department.",
        "FYI I set up a {software} instance on my personal AWS account for the {task} project. Faster than waiting for IT approval.",
        "I gave {name} access to the {system} production database. They needed it urgently and the approval process takes too long.",
    ],
    "phishing_indicators": [
        "URGENT: Your {system} account will be deactivated in 24 hours. Click here to verify: {suspicious_url}",
        "ACTION REQUIRED: Unusual sign-in detected on your account. Confirm your identity immediately at {suspicious_url}",
        "Hi {name}, I'm the new IT admin. I need you to reset your {system} password using this secure link: {suspicious_url}",
        "IMPORTANT: Your {doc_type} access expires today. Renew now to avoid losing all your files: {suspicious_url}",
        "Dear employee, payroll has been updated. Review your new compensation package here: {suspicious_url}",
        "Your {system} storage is 98% full. Click to upgrade immediately or risk losing data: {suspicious_url}",
        "From: CEO Office — Please purchase {amount} in gift cards for a client meeting today. This is urgent and confidential.",
        "Invoice #{invoice_num} is past due. Please process payment immediately to avoid service interruption: {suspicious_url}",
    ],
    "hr_personnel": [
        "I wanted to flag a concern about {name}'s behavior in yesterday's meeting. They were dismissive of the entire {team} team's input.",
        "I'm considering putting in my two weeks. The workload since the {event} has been unsustainable and management isn't listening.",
        "Can we schedule a private meeting? I need to discuss a situation with {name} that's affecting the whole {team} team's morale.",
        "{name} has been consistently missing deadlines on the {project} project. I think we need to have a formal conversation.",
        "I'd like to request a transfer to a different {team}. The current management style isn't aligned with my career goals.",
        "There's been tension between {name} and the rest of the {team} team since the {event}. Productivity has dropped noticeably.",
        "I have concerns about how the {event} situation was handled. Several people on {team} team are updating their resumes.",
        "Can HR review the overtime records for {team} team? We've been working 60+ hour weeks since {event} with no additional support.",
    ],
    "financial_irregularity": [
        "I need to expense {amount} for the {event} dinner. I know it's over the limit but the client expected a certain level of hospitality.",
        "Can we process this invoice from {vendor}? I know they're not an approved vendor but they gave us a 40% discount on {task} services.",
        "The {project} project is {amount} over budget. I've been splitting charges across multiple cost centers to avoid triggering the review.",
        "I approved {name}'s travel to {location} — they booked first class but said it was the only option. Total was {amount}.",
        "{vendor} is offering us a personal referral bonus if we sign the contract this quarter. Should we factor that into the {project} decision?",
        "Please reimburse {amount} to my personal card. I fronted the cost for {task} supplies because the PO process was taking too long.",
        "I've been invoicing {vendor} monthly but the actual deliverables are quarterly. It smooths out our {project} budget reporting.",
        "Can we reclassify the {amount} {task} expense as training? It's easier to get approval under the education budget.",
    ],
    "normal_business": [
        "Hi team, the {project} sprint planning meeting is scheduled for {day} at {time}. Please review the backlog beforehand.",
        "Attached are the meeting notes from today's {project} standup. Action items are highlighted in yellow.",
        "Please review the {doc_type} I shared in the {channel} channel. Need feedback by {day} so we can finalize before {event}.",
        "FYI: {name} will be out next week for PTO. {name2} is covering their {task} responsibilities.",
        "Sharing the {project} Q3 status report. All metrics are green except {task} which needs additional resources.",
        "Hi {name}, welcome to the {team} team! Your onboarding schedule is attached. First day is {day}.",
    ],
}

# --- Fill-in values ---
FILL = {
    "doc_type": ["customer database", "source code repo", "financial report", "security audit", "employee records", "contract files", "classified document", "project roadmap"],
    "personal_email": ["jsmith.home@gmail.com", "analyst99@yahoo.com", "work.backup@protonmail.com", "myfiles@outlook.com"],
    "cloud_service": ["Google Drive", "Dropbox", "OneDrive personal", "WeTransfer", "Box personal"],
    "data_type": ["customer PII", "employee SSN", "financial", "classified", "proprietary", "source code", "credentials"],
    "system": ["Active Directory", "SharePoint", "JIRA", "AWS Console", "Azure Portal", "Salesforce", "SAP", "ServiceNow"],
    "password": ["Welcome1!", "Pass@word123", "Admin2024!", "CompanyName1"],
    "security_control": ["endpoint protection", "DLP agent", "firewall rule", "MFA requirement", "proxy filter", "USB lockdown"],
    "software": ["Wireshark", "TeamViewer", "ngrok", "Tor Browser", "uTorrent", "ChatGPT desktop"],
    "username": ["admin_jdoe", "svc_account", "root_user", "sa_prod"],
    "suspicious_url": ["http://acc0unt-verify.security-check.com/login", "https://portal.micros0ft-secure.com/verify", "http://it-helpdesk.company-support.net/reset"],
    "name": ["Alex", "Jordan", "Casey", "Morgan", "Taylor", "Riley", "Sam", "Drew"],
    "name2": ["Pat", "Quinn", "Reese", "Skyler", "Jamie", "Avery"],
    "team": ["Engineering", "Data Science", "Security", "Platform", "Analytics", "Infrastructure"],
    "project": ["Phoenix", "Atlas", "Mercury", "Orion", "Falcon", "Titan"],
    "event": ["reorg", "audit", "Q4 review", "product launch", "security incident", "migration"],
    "task": ["ETL pipeline", "data migration", "model training", "dashboard", "API integration", "testing"],
    "channel": ["#project-updates", "#team-general", "#data-eng", "#security-alerts"],
    "vendor": ["Acme Corp", "TechServ LLC", "DataFlow Inc", "CloudPrime"],
    "amount": ["$4,500", "$12,000", "$8,750", "$23,000", "$2,800", "$15,500"],
    "location": ["Las Vegas", "Miami", "New York", "San Francisco", "London"],
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "time": ["10am", "2pm", "9am", "3pm", "11am"],
    "invoice_num": ["INV-2024-8831", "INV-2024-7742", "INV-2025-0019"],
}

SENDERS = [f"user{i}@snc-internal.example.com" for i in range(1, 51)]
RECIPIENTS = SENDERS.copy()

# --- Attachment profiles by risk category ---
ATTACHMENT_TYPES = {
    "data_exfiltration": {
        "has_attachment_prob": 0.75,
        "types": [
            ("customer_export.csv", 2.0, 25.0),
            ("database_dump.sql", 5.0, 50.0),
            ("employee_records.xlsx", 1.0, 15.0),
            ("source_code.zip", 3.0, 80.0),
            ("financial_data.csv", 1.5, 20.0),
            ("classified_report.pdf", 0.5, 8.0),
            ("full_backup.tar.gz", 10.0, 200.0),
        ],
    },
    "policy_violation": {
        "has_attachment_prob": 0.30,
        "types": [
            ("credentials.txt", 0.001, 0.01),
            ("api_keys.env", 0.001, 0.005),
            ("setup_guide.pdf", 0.2, 2.0),
            ("config_backup.json", 0.01, 0.1),
        ],
    },
    "phishing_indicators": {
        "has_attachment_prob": 0.40,
        "types": [
            ("invoice_details.pdf", 0.05, 0.3),
            ("urgent_notice.html", 0.01, 0.05),
            ("payment_form.docx", 0.08, 0.2),
            ("verify_account.html", 0.01, 0.04),
            ("update_required.exe", 0.5, 3.0),
        ],
    },
    "hr_personnel": {
        "has_attachment_prob": 0.20,
        "types": [
            ("performance_review.pdf", 0.1, 0.5),
            ("overtime_log.xlsx", 0.05, 0.3),
            ("transfer_request.docx", 0.08, 0.2),
        ],
    },
    "financial_irregularity": {
        "has_attachment_prob": 0.55,
        "types": [
            ("expense_report.pdf", 0.1, 1.5),
            ("invoice_acme.pdf", 0.05, 0.8),
            ("receipt_scan.jpg", 0.5, 3.0),
            ("budget_override.xlsx", 0.1, 0.6),
            ("travel_itinerary.pdf", 0.08, 0.4),
        ],
    },
    "normal_business": {
        "has_attachment_prob": 0.35,
        "types": [
            ("meeting_notes.docx", 0.02, 0.15),
            ("sprint_backlog.xlsx", 0.05, 0.3),
            ("status_report.pdf", 0.1, 0.8),
            ("presentation.pptx", 1.0, 15.0),
            ("architecture_diagram.png", 0.3, 2.0),
            ("onboarding_checklist.pdf", 0.05, 0.2),
        ],
    },
}

def generate_attachments(category):
    """Generate attachment metadata for an email based on its risk category."""
    profile = ATTACHMENT_TYPES[category]
    if random.random() > profile["has_attachment_prob"]:
        return False, 0, None, round(random.uniform(0.001, 0.05), 4)  # no attachment, body-only size

    # Pick 1-3 attachments
    n_attachments = random.choices([1, 2, 3], weights=[70, 20, 10])[0]
    chosen = random.choices(profile["types"], k=n_attachments)
    filenames = []
    total_size_mb = 0.0
    for fname, min_mb, max_mb in chosen:
        filenames.append(fname)
        total_size_mb += random.uniform(min_mb, max_mb)

    # Add body size (small)
    total_size_mb += random.uniform(0.001, 0.05)

    return True, n_attachments, "|".join(filenames), round(total_size_mb, 4)

def fill_template(template):
    """Replace placeholders with random values."""
    import re
    def replacer(match):
        key = match.group(1)
        if key in FILL:
            return random.choice(FILL[key])
        return match.group(0)
    return re.sub(r'\{(\w+)\}', replacer, template)

def generate_emails(n=5000):
    """Generate n synthetic emails with rebalanced distribution: ~30% normal, ~70% risk.
    Skewed toward risk so unsupervised topic modeling can surface risk categories at low K.
    """
    rows = []
    weights = {
        "normal_business": 0.30,
        "hr_personnel": 0.14,
        "financial_irregularity": 0.14,
        "phishing_indicators": 0.14,
        "policy_violation": 0.14,
        "data_exfiltration": 0.14,
    }
    categories = list(weights.keys())
    probs = list(weights.values())
    base_date = datetime(2025, 1, 1)

    for i in range(n):
        category = random.choices(categories, probs)[0]
        template = random.choice(TEMPLATES[category])
        body = fill_template(template)
        subject_words = body.split()[:6]
        subject = " ".join(subject_words) + "..."
        ts = base_date + timedelta(
            days=random.randint(0, 120),
            hours=random.randint(6, 22),
            minutes=random.randint(0, 59),
        )
        has_attachment, attachment_count, attachment_names, size_mb = generate_attachments(category)
        rows.append(Row(
            email_id=str(uuid.uuid4()),
            timestamp=ts,
            sender=random.choice(SENDERS),
            recipient=random.choice(RECIPIENTS),
            subject=subject,
            body=body,
            has_attachment=has_attachment,
            attachment_count=attachment_count,
            attachment_names=attachment_names,
            size_mb=float(size_mb),
            _synthetic_label=category,  # hidden label for validation only
        ))
    return rows

emails = generate_emails(5000)
df = spark.createDataFrame(emails)
print(f"Generated {df.count()} synthetic emails")
df.groupBy("_synthetic_label").count().orderBy("count", ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Write to Delta Table
# MAGIC
# MAGIC The `_synthetic_label` column is included for **validation purposes only** — the workshop treats this as unlabeled data.
# MAGIC In production, you would not have this column.

# COMMAND ----------

DATABASE = dbutils.widgets.get("database")

table_name = f"{DATABASE}.emails_bronze"

df.write.format("delta").mode("overwrite").saveAsTable(table_name)

print(f"✓ Wrote {df.count()} emails to {table_name}")
display(spark.sql(f"SELECT * FROM {table_name} LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Complete ✓
# MAGIC
# MAGIC You now have:
# MAGIC - **Database**: `{DATABASE}` created
# MAGIC - **Libraries**: sentence-transformers, umap-learn, hdbscan, pyLDAvis, wordcloud installed
# MAGIC - **Data**: 5,000 synthetic emails in `emails_bronze` table
# MAGIC
# MAGIC **Next →** Open `01_data_exploration` to explore the dataset.
