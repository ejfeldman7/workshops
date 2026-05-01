# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop 2: Anomaly Detection for Sign-In Data
# MAGIC ## Notebook 0 — Setup & Configuration
# MAGIC
# MAGIC This notebook configures the workshop environment and generates synthetic sign-in log data.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Sets your catalog and schema for all workshop assets
# MAGIC 2. Installs required libraries
# MAGIC 3. Generates realistic synthetic sign-in data with embedded anomalies
# MAGIC 4. Writes the data to a Delta table
# MAGIC
# MAGIC **Azure Gov Cloud Compatibility:** ✅ Everything in this notebook runs on classic compute with Databricks Runtime ML.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Architecture Overview
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────────┐
# MAGIC │                  Workshop Data Flow                                  │
# MAGIC │                                                                      │
# MAGIC │  [Synthetic Sign-In Logs]                                            │
# MAGIC │         │                                                            │
# MAGIC │         ▼                                                            │
# MAGIC │  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐    │
# MAGIC │  │ Bronze Table  │──▶│   Feature    │──▶│ Isolation Forest /    │    │
# MAGIC │  │ (Raw Logs)    │   │  Engineering │   │ PyOD Ensemble         │    │
# MAGIC │  └──────────────┘   └──────────────┘   └────────────────────────┘    │
# MAGIC │                                                  │                   │
# MAGIC │                                         ┌────────▼────────┐          │
# MAGIC │                                         │  Anomaly Scores │          │
# MAGIC │                                         │  + Risk Tiers   │          │
# MAGIC │                                         └─────────────────┘          │
# MAGIC │                                                                      │
# MAGIC │  All assets created in: <your_catalog>.<your_schema>                 │
# MAGIC └──────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure Your Database

# COMMAND ----------

dbutils.widgets.text("database", "login_anomaly", "Database")

DATABASE = dbutils.widgets.get("database")

print(f"Workshop assets will be created in: {DATABASE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Install Required Libraries

# COMMAND ----------

# MAGIC %pip install pyod --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Database

# COMMAND ----------

import os

DATABASE = dbutils.widgets.get("database")

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
spark.sql(f"USE {DATABASE}")

ARTIFACT_PATH = f"/dbfs/tmp/workshops/{DATABASE}"
os.makedirs(ARTIFACT_PATH, exist_ok=True)
print(f"Using database: {DATABASE} (Hive metastore)")
print(f"Artifacts path: {ARTIFACT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Generate Synthetic Sign-In Data
# MAGIC
# MAGIC We generate realistic sign-in logs with several types of embedded anomalies:
# MAGIC
# MAGIC | Anomaly Type | Description | % of Data |
# MAGIC |-------------|-------------|-----------|
# MAGIC | **Impossible Travel** | Logins from geographically distant locations within short time windows | ~2% |
# MAGIC | **Off-Hours Access** | Logins at 2-5 AM on weekdays or any time on weekends from unusual IPs | ~3% |
# MAGIC | **Brute Force** | Multiple failed login attempts followed by success from same IP | ~2% |
# MAGIC | **New Device + Location** | Login from never-before-seen device AND location simultaneously | ~2% |
# MAGIC | **Credential Stuffing** | Same IP attempting logins across many different user accounts | ~1% |
# MAGIC | **Normal** | Regular business-hours logins from known devices and locations | ~90% |
# MAGIC
# MAGIC > **Note:** In production, this table would be your actual sign-in log data (already in Databricks).

# COMMAND ----------

import random
import uuid
import math
from datetime import datetime, timedelta
from pyspark.sql import Row
from pyspark.sql.types import *

random.seed(42)

# --- User profiles (normal behavior patterns) ---
LOCATIONS = {
    "HQ_Sparks_NV":      (39.5349, -119.7527),
    "Office_Denver_CO":   (39.7392, -104.9903),
    "Office_Huntsville":  (34.7304, -86.5861),
    "Remote_LasVegas":    (36.1699, -115.1398),
    "Remote_Phoenix":     (33.4484, -112.0740),
}

ANOMALOUS_LOCATIONS = {
    "Moscow_RU":          (55.7558, 37.6173),
    "Shanghai_CN":        (31.2304, 121.4737),
    "Sao_Paulo_BR":       (-23.5505, -46.6333),
    "Lagos_NG":           (6.5244, 3.3792),
    "Unknown_VPN":        (0.0, 0.0),
}

DEVICES = ["Windows_Laptop_Corp", "MacBook_Corp", "iPhone_Corp", "Android_Corp", "Linux_Workstation"]
ANOMALOUS_DEVICES = ["Unknown_Linux", "Tor_Browser", "Rooted_Android", "VM_Instance", "Headless_Chrome"]

USERS = [f"user{i:03d}@snc-internal.example.com" for i in range(1, 101)]

# Assign each user a "home" location and primary devices
USER_PROFILES = {}
for user in USERS:
    home = random.choice(list(LOCATIONS.keys()))
    devices = random.sample(DEVICES, random.randint(1, 3))
    work_start = random.randint(7, 9)
    work_end = random.randint(16, 19)
    USER_PROFILES[user] = {
        "home_location": home,
        "devices": devices,
        "work_start": work_start,
        "work_end": work_end,
        "avg_sessions_per_day": random.uniform(2, 8),
    }

def haversine_km(lat1, lon1, lat2, lon2):
    """Distance between two lat/lon points in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def generate_normal_login(user, ts):
    """Generate a normal login event."""
    profile = USER_PROFILES[user]
    loc_name = profile["home_location"]
    # Occasionally log in from another office
    if random.random() < 0.1:
        loc_name = random.choice(list(LOCATIONS.keys()))
    lat, lon = LOCATIONS[loc_name]
    # Add small jitter
    lat += random.gauss(0, 0.01)
    lon += random.gauss(0, 0.01)

    return {
        "login_id": str(uuid.uuid4()),
        "user_id": user,
        "timestamp": ts,
        "location_name": loc_name,
        "latitude": lat,
        "longitude": lon,
        "device": random.choice(profile["devices"]),
        "ip_address": f"10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
        "success": True,
        "failed_attempts_before": random.choices([0, 0, 0, 0, 1], weights=[80, 5, 5, 5, 5])[0],
        "session_duration_min": float(max(5, random.gauss(45, 20))),
        "mfa_used": random.random() < 0.85,
        "_anomaly_type": "normal",
    }

def generate_impossible_travel(user, ts):
    """Login from distant location within 30 min of last login."""
    profile = USER_PROFILES[user]
    loc_name = random.choice(list(ANOMALOUS_LOCATIONS.keys()))
    lat, lon = ANOMALOUS_LOCATIONS[loc_name]
    return {
        "login_id": str(uuid.uuid4()),
        "user_id": user,
        "timestamp": ts,
        "location_name": loc_name,
        "latitude": lat,
        "longitude": lon,
        "device": random.choice(profile["devices"]),
        "ip_address": f"185.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
        "success": True,
        "failed_attempts_before": random.randint(0, 2),
        "session_duration_min": float(max(1, random.gauss(10, 5))),
        "mfa_used": random.random() < 0.3,
        "_anomaly_type": "impossible_travel",
    }

def generate_off_hours(user, ts):
    """Login at 2-5 AM from unusual IP."""
    profile = USER_PROFILES[user]
    off_hour = random.randint(1, 5)
    ts = ts.replace(hour=off_hour, minute=random.randint(0, 59))
    loc = profile["home_location"]
    lat, lon = LOCATIONS[loc]
    return {
        "login_id": str(uuid.uuid4()),
        "user_id": user,
        "timestamp": ts,
        "location_name": loc,
        "latitude": lat + random.gauss(0, 0.01),
        "longitude": lon + random.gauss(0, 0.01),
        "device": random.choice(profile["devices"] + ANOMALOUS_DEVICES[:1]),
        "ip_address": f"72.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
        "success": True,
        "failed_attempts_before": random.randint(0, 3),
        "session_duration_min": float(max(2, random.gauss(15, 10))),
        "mfa_used": random.random() < 0.5,
        "_anomaly_type": "off_hours",
    }

def generate_brute_force(user, ts):
    """Multiple failed attempts then success."""
    profile = USER_PROFILES[user]
    loc = profile["home_location"]
    lat, lon = LOCATIONS[loc]
    return {
        "login_id": str(uuid.uuid4()),
        "user_id": user,
        "timestamp": ts,
        "location_name": loc,
        "latitude": lat,
        "longitude": lon,
        "device": random.choice(ANOMALOUS_DEVICES),
        "ip_address": f"91.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
        "success": True,
        "failed_attempts_before": random.randint(5, 20),
        "session_duration_min": float(max(1, random.gauss(5, 3))),
        "mfa_used": False,
        "_anomaly_type": "brute_force",
    }

def generate_new_device_location(user, ts):
    """New device AND new location simultaneously."""
    anom_loc = random.choice(list(ANOMALOUS_LOCATIONS.keys()))
    lat, lon = ANOMALOUS_LOCATIONS[anom_loc]
    return {
        "login_id": str(uuid.uuid4()),
        "user_id": user,
        "timestamp": ts,
        "location_name": anom_loc,
        "latitude": lat,
        "longitude": lon,
        "device": random.choice(ANOMALOUS_DEVICES),
        "ip_address": f"178.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
        "success": True,
        "failed_attempts_before": random.randint(1, 5),
        "session_duration_min": float(max(1, random.gauss(8, 5))),
        "mfa_used": False,
        "_anomaly_type": "new_device_location",
    }

def generate_credential_stuffing_burst(start_ts, n_attempts=50):
    """Coordinated burst: ~50 login attempts in a 5-minute window from one IP,
    hitting different users. Each event individually looks like brute force,
    but the temporal clustering is what makes it a system-level anomaly.
    """
    burst_ip = f"185.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
    burst_device = random.choice(ANOMALOUS_DEVICES)
    burst_loc = random.choice(list(ANOMALOUS_LOCATIONS.keys()))
    lat, lon = ANOMALOUS_LOCATIONS[burst_loc]
    targets = random.sample(USERS, min(n_attempts, len(USERS)))

    events = []
    for i, user in enumerate(targets):
        # Spread attempts across a ~5-minute window
        ts = start_ts + timedelta(seconds=random.randint(0, 300))
        events.append({
            "login_id": str(uuid.uuid4()),
            "user_id": user,
            "timestamp": ts,
            "location_name": burst_loc,
            "latitude": lat + random.gauss(0, 0.001),
            "longitude": lon + random.gauss(0, 0.001),
            "device": burst_device,
            "ip_address": burst_ip,
            # Most attempts fail; ~10% succeed (the credential hits)
            "success": random.random() < 0.1,
            "failed_attempts_before": random.randint(3, 15),
            "session_duration_min": float(max(1, random.gauss(3, 2))),
            "mfa_used": False,
            "_anomaly_type": "credential_stuffing",
        })
    return events

# --- Generate dataset ---
rows = []
base_date = datetime(2025, 1, 1)

# Normal logins (~90%)
for day in range(90):
    date = base_date + timedelta(days=day)
    for user in USERS:
        profile = USER_PROFILES[user]
        n_sessions = max(1, int(random.gauss(profile["avg_sessions_per_day"], 1.5)))
        for _ in range(n_sessions):
            hour = random.randint(profile["work_start"], profile["work_end"])
            ts = date.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
            rows.append(generate_normal_login(user, ts))

# Impossible travel (~2%)
for _ in range(int(len(rows) * 0.022)):
    user = random.choice(USERS)
    day = random.randint(0, 89)
    ts = base_date + timedelta(days=day, hours=random.randint(8, 18), minutes=random.randint(0, 59))
    rows.append(generate_impossible_travel(user, ts))

# Off-hours (~3%)
for _ in range(int(len(rows) * 0.033)):
    user = random.choice(USERS)
    day = random.randint(0, 89)
    ts = base_date + timedelta(days=day)
    rows.append(generate_off_hours(user, ts))

# Brute force (~2%)
for _ in range(int(len(rows) * 0.022)):
    user = random.choice(USERS)
    day = random.randint(0, 89)
    ts = base_date + timedelta(days=day, hours=random.randint(0, 23), minutes=random.randint(0, 59))
    rows.append(generate_brute_force(user, ts))

# New device + location (~2%)
for _ in range(int(len(rows) * 0.022)):
    user = random.choice(USERS)
    day = random.randint(0, 89)
    ts = base_date + timedelta(days=day, hours=random.randint(8, 22), minutes=random.randint(0, 59))
    rows.append(generate_new_device_location(user, ts))

# Credential stuffing bursts: 12 coordinated attacks of ~50 attempts each.
# These are temporally clustered (5-min windows), making them a system-level
# anomaly that aggregate volume / failure-rate / distinct-IP metrics catch
# clearly, even though individual events look similar to brute force.
for _ in range(12):
    day = random.randint(0, 89)
    # Bias toward off-hours (stuffing campaigns commonly hit overnight)
    hour = random.choices(
        list(range(24)),
        weights=[3]*6 + [1]*12 + [3]*6,  # heavier 0-5 and 18-23
    )[0]
    start_ts = base_date + timedelta(
        days=day, hours=hour, minutes=random.randint(0, 55)
    )
    rows.extend(generate_credential_stuffing_burst(start_ts, n_attempts=50))

random.shuffle(rows)

df = spark.createDataFrame([Row(**r) for r in rows])
print(f"Generated {df.count()} sign-in events")
df.groupBy("_anomaly_type").count().orderBy("count", ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Write to Delta Table

# COMMAND ----------

DATABASE = dbutils.widgets.get("database")

table_name = f"{DATABASE}.signins_bronze"
df.write.format("delta").mode("overwrite").saveAsTable(table_name)

print(f"✓ Wrote {df.count()} sign-in events to {table_name}")
display(spark.sql(f"SELECT * FROM {table_name} LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Complete ✓
# MAGIC
# MAGIC You now have:
# MAGIC - **Database**: `{DATABASE}` created
# MAGIC - **Libraries**: pyod installed
# MAGIC - **Data**: ~40,000+ synthetic sign-in events in `signins_bronze` table
# MAGIC   - ~90% normal logins
# MAGIC   - ~10% anomalous: impossible travel, off-hours, brute force, new device+location,
# MAGIC     and credential-stuffing bursts (12 coordinated attacks of ~50 attempts each
# MAGIC     in 5-minute windows — the system-level temporal anomaly used in notebook 08)
# MAGIC
# MAGIC **Next →** Open `01_data_exploration` to explore the sign-in data.

# COMMAND ----------

