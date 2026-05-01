# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 3: Security Data Pipeline & Application Integration
# MAGIC ## Notebook 0 — Setup & Configuration
# MAGIC
# MAGIC This notebook configures the workshop environment and generates synthetic security event data.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Sets your database for all workshop assets
# MAGIC 2. Installs required libraries
# MAGIC 3. Generates realistic synthetic security events (email alerts + sign-in logs)
# MAGIC 4. Writes raw data to a DBFS landing zone (for ingestion pipeline) and to bronze Delta tables
# MAGIC
# MAGIC **Azure Gov Cloud Compatibility:** ✅ Everything in this notebook runs on classic compute with Databricks Runtime 13.3+.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Workshop Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────────────┐
# MAGIC │             Security Data Pipeline & App Integration                     │
# MAGIC │                                                                          │
# MAGIC │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────────────┐      │
# MAGIC │  │ Landing  │──▶│  Bronze  │──▶│  Silver  │──▶│  Risk Scores      │      │
# MAGIC │  │ Zone     │   │  (Raw)   │   │ (Clean)  │   │  (Scored + Why)   │      │
# MAGIC │  │ (JSON)   │   │          │   │          │   │                   │      │
# MAGIC │  └──────────┘   └──────────┘   └──────────┘   └────────┬──────────┘      │
# MAGIC │       │                │              │                  │               │
# MAGIC │       │         ┌──────▼──────┐      │           ┌──────▼──────┐         │
# MAGIC │       │         │  Pipeline   │      │           │  External   │         │
# MAGIC │       │         │  Health     │◀─────┘           │  App (SDK)  │         │
# MAGIC │       │         │  Metrics    │                  │             │         │
# MAGIC │       │         └─────────────┘                  └─────────────┘         │
# MAGIC │       │                                                                  │
# MAGIC │  All assets in: <your_database> (Hive metastore)                         │
# MAGIC └──────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure Your Database
# MAGIC
# MAGIC Update the widget below to set your target database name.

# COMMAND ----------

dbutils.widgets.text("database", "security_app_integration", "Database")

DATABASE = dbutils.widgets.get("database")

print(f"Workshop assets will be created in: {DATABASE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Install Required Libraries
# MAGIC
# MAGIC The `databricks-sdk` is the core dependency for notebook 04 (application integration).

# COMMAND ----------

# MAGIC %pip install databricks-sdk --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Database and Artifact Directory

# COMMAND ----------

import os

DATABASE = dbutils.widgets.get("database")

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
spark.sql(f"USE {DATABASE}")

ARTIFACT_PATH = f"/dbfs/tmp/workshops/{DATABASE}"
LANDING_PATH = f"/dbfs/tmp/workshops/{DATABASE}/landing_zone"
os.makedirs(ARTIFACT_PATH, exist_ok=True)
os.makedirs(f"{LANDING_PATH}/email_events", exist_ok=True)
os.makedirs(f"{LANDING_PATH}/signin_events", exist_ok=True)

print(f"Using database: {DATABASE} (Hive metastore)")
print(f"Artifacts:    {ARTIFACT_PATH}")
print(f"Landing zone: {LANDING_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Generate Synthetic Security Events
# MAGIC
# MAGIC We generate two types of events that a security operations center would ingest:
# MAGIC
# MAGIC | Event Type | Count | Description |
# MAGIC |-----------|-------|-------------|
# MAGIC | **Email** | 5,000 | Email metadata with subject, sender, attachments, size |
# MAGIC | **Sign-in** | 5,000 | Authentication logs with location, device, MFA status |
# MAGIC
# MAGIC Each event type includes a mix of normal activity (~90%) and suspicious patterns (~10%).
# MAGIC
# MAGIC > **Note:** In production, these events would come from your email gateway (e.g., Exchange, M365)
# MAGIC > and identity provider (e.g., Azure AD, Okta) via streaming connectors or batch exports.

# COMMAND ----------

import random
import uuid
import json
from datetime import datetime, timedelta

random.seed(42)

# --- Email event generation ---

RISKY_KEYWORDS = {
    "data_exfiltration": [
        "personal email", "USB drive", "upload to Dropbox", "forwarded to gmail",
        "send me the database", "copy to external", "download before I remove",
        "zip up the directory", "exported all records", "transfer to my account",
    ],
    "phishing": [
        "URGENT: verify your account", "click here immediately", "password expires today",
        "unusual sign-in detected", "confirm your identity", "gift cards",
        "payment overdue", "ACTION REQUIRED", "account deactivated",
    ],
    "policy_violation": [
        "here's the admin password", "disabled the firewall", "use my credentials",
        "installed without IT approval", "shared the API key", "bypassed the proxy",
        "exception for our department", "gave access to production",
    ],
}

NORMAL_SUBJECTS = [
    "Q4 Sprint Planning — {day}", "Re: {project} status update",
    "{project} deployment successful", "Meeting notes — {day} standup",
    "Team lunch moved to {day}", "PTO request — next week",
    "FYI: {project} roadmap shared", "Welcome to the team!",
    "Feedback on {project} design doc", "Updated {project} timeline",
    "Office supplies order", "Parking pass renewal",
    "Monthly all-hands agenda", "Training session reminder",
    "Code review: PR #{num}", "CI/CD pipeline green — ready to merge",
]

FILL_VALS = {
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "project": ["Phoenix", "Atlas", "Mercury", "Orion", "Falcon", "Titan"],
    "num": [str(i) for i in range(100, 999)],
}

INTERNAL_DOMAINS = ["snc-internal.example.com"]
EXTERNAL_DOMAINS = ["gmail.com", "yahoo.com", "protonmail.com", "outlook.com", "contractor.example.com"]

ATTACHMENT_PROFILES = {
    "normal":             {"prob": 0.30, "types": ["pdf", "docx", "pptx", "xlsx", "png"], "size": (0.01, 5.0)},
    "data_exfiltration":  {"prob": 0.80, "types": ["csv", "sql", "zip", "tar.gz", "xlsx"], "size": (5.0, 200.0)},
    "phishing":           {"prob": 0.35, "types": ["html", "exe", "docx", "pdf"],          "size": (0.01, 0.5)},
    "policy_violation":   {"prob": 0.25, "types": ["txt", "env", "json", "pdf"],            "size": (0.001, 0.5)},
}


def fill_template(template):
    import re
    def replacer(match):
        key = match.group(1)
        return random.choice(FILL_VALS.get(key, [match.group(0)]))
    return re.sub(r'\{(\w+)\}', replacer, template)


def generate_email_events(n=5000):
    events = []
    base_date = datetime(2025, 1, 1)

    for _ in range(n):
        roll = random.random()
        if roll < 0.04:
            category = "data_exfiltration"
            subject = random.choice(RISKY_KEYWORDS["data_exfiltration"])
        elif roll < 0.07:
            category = "phishing"
            subject = random.choice(RISKY_KEYWORDS["phishing"])
        elif roll < 0.10:
            category = "policy_violation"
            subject = random.choice(RISKY_KEYWORDS["policy_violation"])
        else:
            category = "normal"
            subject = fill_template(random.choice(NORMAL_SUBJECTS))

        sender_domain = random.choice(INTERNAL_DOMAINS)
        if category == "phishing":
            sender_domain = random.choice(EXTERNAL_DOMAINS + INTERNAL_DOMAINS)
        sender = f"user{random.randint(1, 50)}@{sender_domain}"

        recipient_domain = random.choice(INTERNAL_DOMAINS)
        if category == "data_exfiltration" and random.random() < 0.6:
            recipient_domain = random.choice(EXTERNAL_DOMAINS)
        recipient = f"user{random.randint(1, 50)}@{recipient_domain}"

        direction = "internal"
        if sender_domain not in INTERNAL_DOMAINS:
            direction = "inbound"
        elif recipient_domain not in INTERNAL_DOMAINS:
            direction = "outbound"

        ts = base_date + timedelta(days=random.randint(0, 90))
        if category in ("data_exfiltration", "policy_violation") and random.random() < 0.5:
            ts += timedelta(hours=random.choice([1, 2, 3, 4, 22, 23]))
        else:
            ts += timedelta(hours=random.randint(7, 18), minutes=random.randint(0, 59))

        profile = ATTACHMENT_PROFILES.get(category, ATTACHMENT_PROFILES["normal"])
        has_attachment = random.random() < profile["prob"]
        if has_attachment:
            att_count = random.choices([1, 2, 3], weights=[70, 20, 10])[0]
            att_types = "|".join(random.choices(profile["types"], k=att_count))
            size_mb = round(sum(random.uniform(*profile["size"]) for _ in range(att_count)), 3)
        else:
            att_count = 0
            att_types = None
            size_mb = round(random.uniform(0.001, 0.05), 4)

        events.append({
            "event_type": "email",
            "event_id": str(uuid.uuid4()),
            "timestamp": ts.isoformat(),
            "sender": sender,
            "sender_domain": sender_domain,
            "recipient": recipient,
            "recipient_domain": recipient_domain,
            "subject": subject,
            "body_length": random.randint(50, 5000),
            "has_attachment": has_attachment,
            "attachment_count": att_count,
            "attachment_types": att_types,
            "size_mb": size_mb,
            "direction": direction,
            "_synthetic_category": category,
        })
    return events


# --- Sign-in event generation ---

KNOWN_LOCATIONS = [
    ("Sparks", "US", 39.53, -119.81),
    ("Louisville", "US", 38.25, -85.76),
    ("Denver", "US", 39.74, -104.99),
    ("Huntsville", "US", 34.73, -86.59),
    ("Dayton", "US", 39.76, -84.19),
]

ANOMALOUS_LOCATIONS = [
    ("Moscow", "RU", 55.76, 37.62),
    ("Shanghai", "CN", 31.23, 121.47),
    ("Lagos", "NG", 6.52, 3.38),
    ("Pyongyang", "KP", 39.02, 125.75),
    ("Tehran", "IR", 35.69, 51.39),
]

DEVICE_TYPES = ["Windows_Laptop_Corp", "MacOS_Corp", "iPhone_Corp", "Android_Corp", "Linux_Workstation"]
ANOMALOUS_DEVICES = ["Unknown_Linux", "Tor_Browser", "VPS_Cloud", "Public_Kiosk"]
APPLICATIONS = ["SharePoint", "Azure Portal", "Email", "VPN", "JIRA", "SAP", "Internal Wiki"]


def generate_signin_events(n=5000):
    events = []
    base_date = datetime(2025, 1, 1)
    users = [f"user{i:03d}@snc-internal.example.com" for i in range(1, 101)]

    for _ in range(n):
        user = random.choice(users)
        ts = base_date + timedelta(days=random.randint(0, 90))

        roll = random.random()
        if roll < 0.03:
            category = "impossible_travel"
            loc = random.choice(ANOMALOUS_LOCATIONS)
            device = random.choice(DEVICE_TYPES + ANOMALOUS_DEVICES)
            ts += timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
            mfa_used = random.random() < 0.3
            login_success = True
            failed_attempts = 0
        elif roll < 0.05:
            category = "brute_force"
            loc = random.choice(KNOWN_LOCATIONS + ANOMALOUS_LOCATIONS)
            device = random.choice(ANOMALOUS_DEVICES)
            ts += timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
            mfa_used = False
            login_success = True
            failed_attempts = random.randint(5, 25)
        elif roll < 0.07:
            category = "off_hours_unknown_device"
            loc = random.choice(KNOWN_LOCATIONS)
            device = random.choice(ANOMALOUS_DEVICES)
            ts += timedelta(hours=random.choice([1, 2, 3, 4, 23]))
            mfa_used = random.random() < 0.4
            login_success = True
            failed_attempts = random.randint(0, 3)
        elif roll < 0.10:
            category = "failed_burst"
            loc = random.choice(KNOWN_LOCATIONS + ANOMALOUS_LOCATIONS)
            device = random.choice(DEVICE_TYPES + ANOMALOUS_DEVICES)
            ts += timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
            mfa_used = False
            login_success = False
            failed_attempts = random.randint(3, 15)
        else:
            category = "normal"
            loc = random.choice(KNOWN_LOCATIONS)
            device = random.choice(DEVICE_TYPES)
            ts += timedelta(hours=random.randint(7, 18), minutes=random.randint(0, 59))
            mfa_used = True
            login_success = True
            failed_attempts = 0

        city, country, lat, lon = loc
        lat += random.uniform(-0.05, 0.05)
        lon += random.uniform(-0.05, 0.05)

        events.append({
            "event_type": "signin",
            "event_id": str(uuid.uuid4()),
            "timestamp": ts.isoformat(),
            "user_id": user,
            "ip_address": f"{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(0,254)}.{random.randint(1,254)}",
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "city": city,
            "country": country,
            "device_type": device,
            "device_id": f"{device}-{random.randint(1000, 9999)}",
            "mfa_used": mfa_used,
            "login_success": login_success,
            "failed_attempts": failed_attempts,
            "application": random.choice(APPLICATIONS),
            "_synthetic_category": category,
        })
    return events

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Events

# COMMAND ----------

from collections import Counter

email_events = generate_email_events(5000)
signin_events = generate_signin_events(5000)
all_events = email_events + signin_events
random.shuffle(all_events)

print(f"Generated {len(email_events)} email events and {len(signin_events)} sign-in events")

email_cats = Counter(e["_synthetic_category"] for e in email_events)
print("\nEmail categories:")
for cat, count in email_cats.most_common():
    print(f"  {cat:25s}: {count:5d} ({count/len(email_events)*100:.1f}%)")

signin_cats = Counter(e["_synthetic_category"] for e in signin_events)
print("\nSign-in categories:")
for cat, count in signin_cats.most_common():
    print(f"  {cat:25s}: {count:5d} ({count/len(signin_events)*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Write to Landing Zone (JSON Files)
# MAGIC
# MAGIC Write events as batched JSON files to the DBFS landing zone.
# MAGIC This simulates how data would arrive from an email gateway or identity provider —
# MAGIC one file per batch, dropped into a directory for Auto Loader to pick up.

# COMMAND ----------

batch_size = 500
for i in range(0, len(email_events), batch_size):
    batch = email_events[i:i + batch_size]
    file_path = f"{LANDING_PATH}/email_events/batch_{i // batch_size:04d}.json"
    with open(file_path, "w") as f:
        for e in batch:
            f.write(json.dumps(e) + "\n")

print(f"✓ Wrote {len(email_events)} email events in {len(email_events) // batch_size} files to {LANDING_PATH}/email_events/")

for i in range(0, len(signin_events), batch_size):
    batch = signin_events[i:i + batch_size]
    file_path = f"{LANDING_PATH}/signin_events/batch_{i // batch_size:04d}.json"
    with open(file_path, "w") as f:
        for e in batch:
            f.write(json.dumps(e) + "\n")

print(f"✓ Wrote {len(signin_events)} sign-in events in {len(signin_events) // batch_size} files to {LANDING_PATH}/signin_events/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Write Bronze Delta Tables
# MAGIC
# MAGIC Also write directly to Delta for notebooks that don't need the Auto Loader demo.

# COMMAND ----------

from pyspark.sql import functions as F

DATABASE = dbutils.widgets.get("database")
spark.sql(f"USE {DATABASE}")

# Email bronze
email_df = spark.createDataFrame(email_events)
email_df = email_df.withColumn("timestamp", F.to_timestamp("timestamp"))
email_df.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE}.email_events_bronze")
print(f"✓ {email_df.count()} email events → {DATABASE}.email_events_bronze")

# Sign-in bronze
signin_df = spark.createDataFrame(signin_events)
signin_df = signin_df.withColumn("timestamp", F.to_timestamp("timestamp"))
signin_df.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE}.signin_events_bronze")
print(f"✓ {signin_df.count()} sign-in events → {DATABASE}.signin_events_bronze")

# Pipeline health table (empty — will be populated by notebook 01)
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {DATABASE}.pipeline_health (
        pipeline_stage STRING,
        event_type STRING,
        run_timestamp TIMESTAMP,
        row_count LONG,
        null_count LONG,
        null_rate DOUBLE,
        min_event_ts TIMESTAMP,
        max_event_ts TIMESTAMP,
        processing_seconds DOUBLE,
        status STRING
    )
""")
print(f"✓ Pipeline health table created: {DATABASE}.pipeline_health")

display(spark.sql(f"SELECT * FROM {DATABASE}.email_events_bronze LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Complete ✓
# MAGIC
# MAGIC You now have:
# MAGIC - **Database:** `{DATABASE}` created (Hive metastore)
# MAGIC - **Landing zone:** JSON files in `{LANDING_PATH}/email_events/` and `signin_events/`
# MAGIC - **Bronze tables:** `email_events_bronze` and `signin_events_bronze`
# MAGIC - **Health table:** `pipeline_health` (empty, populated by notebook 01)
# MAGIC - **Libraries:** `databricks-sdk` installed
# MAGIC
# MAGIC **Next →** Open `01_ingestion_pipeline` to build the medallion pipeline with health monitoring.
