# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab Bootstrap: Create Bronze Tables
# MAGIC
# MAGIC Run this notebook **once** before starting the Spark Declarative Pipelines lab.
# MAGIC It creates two bronze tables that the pipeline will read:
# MAGIC
# MAGIC | Table | Description |
# MAGIC |---|---|
# MAGIC | `dim_customers_raw` | Customer master data (dimension table) |
# MAGIC | `raw_customer_orders` | Customer orders (fact table) |
# MAGIC
# MAGIC **Configure the target catalog and schema using the widgets at the top of this notebook.**
# MAGIC Defaults are `ball` / `bronze` — change them if your environment uses a different catalog or schema.
# MAGIC
# MAGIC All columns land as **STRING** — exactly as they would from a raw CSV ingest —
# MAGIC so your silver layer has something real to cast and validate.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure target catalog and schema

# COMMAND ----------

dbutils.widgets.text("catalog", "ball",   "Target Catalog")
dbutils.widgets.text("schema",  "bronze", "Target Schema")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA  = dbutils.widgets.get("schema").strip()

if not CATALOG:
    raise ValueError("'catalog' widget is empty — please enter a catalog name.")
if not SCHEMA:
    raise ValueError("'schema' widget is empty — please enter a schema name.")

BRONZE = f"{CATALOG}.{SCHEMA}"
print(f"Target location : {BRONZE}")
print(f"  dim_customers_raw    -> {BRONZE}.dim_customers_raw")
print(f"  raw_customer_orders  -> {BRONZE}.raw_customer_orders")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create catalog and schema (if they don't exist)

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS `{CATALOG}`")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
print(f"Catalog and schema ready: {BRONZE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Load customer master data
# MAGIC
# MAGIC 13 input rows across 4 regions and 4 tiers. A few rows have intentional
# MAGIC data quality issues so the silver layer has something to act on:
# MAGIC
# MAGIC | Row | Intent | What the pipeline does |
# MAGIC |---|---|---|
# MAGIC | NULL customer_id (`Mystery Customer`) | Constraint **drop** | `valid_customer_id` removes it (ON VIOLATION DROP ROW) |
# MAGIC | C011 `payment_terms_days = 999` | Constraint **warn** | `valid_payment_terms` has no ON VIOLATION clause → row is **kept** and the violation is counted in the event log |
# MAGIC | Duplicate C001 (`...(dup)`) | **CTE filter** | `ROW_NUMBER()` keeps one row per customer_id; the dup is invisibly removed before constraints run |
# MAGIC | C006 `is_active = false` | **WHERE filter** | Final `WHERE is_active = TRUE` drops it |
# MAGIC
# MAGIC Net effect: **13 input → 10 rows in `silver_customers`**.

# COMMAND ----------

customers_data = [
    # (customer_id, customer_name,          tier,       region,     contract_type,  payment_terms_days, is_active)
    ("C001", "Acme Beverages LLC",           "Gold",     "Northeast","Annual",       "30",               "true"),
    ("C002", "Blue Ridge Bottling Co",       "Silver",   "Southeast","Monthly",      "45",               "true"),
    ("C003", "Summit Canning Inc",           "Platinum", "West",     "Annual",       "15",               "true"),
    ("C004", "Prairie Pack Solutions",       "Bronze",   "Midwest",  "Quarterly",    "60",               "true"),
    ("C005", "Coastal Container Group",      "Gold",     "Southeast","Annual",       "30",               "true"),
    ("C006", "Rocky Mountain Refresh",       "Silver",   "West",     "Monthly",      "45",               "false"),  # inactive
    ("C007", "Heartland Beverages",          "Gold",     "Midwest",  "Annual",       "30",               "true"),
    ("C008", "Eastern Seaboard Drinks",      "Platinum", "Northeast","Annual",       "15",               "true"),
    ("C009", "Desert Sun Packaging",         "Bronze",   "West",     "Monthly",      "60",               "true"),
    ("C010", "Great Lakes Canning",          "Silver",   "Midwest",  "Quarterly",    "45",               "true"),
    # --- intentional DQ issues below ---
    ("C001", "Acme Beverages LLC (dup)",     "Gold",     "Northeast","Annual",       "30",               "true"),  # duplicate C001
    (None,   "Mystery Customer",             "Bronze",   "Northeast","Monthly",      "30",               "true"),  # NULL customer_id
    ("C011", "Bad Terms Corp",               "Silver",   "Southeast","Annual",       "999",              "true"),  # payment_terms 999 > 365
]

customers_schema = "customer_id STRING, customer_name STRING, tier STRING, region STRING, contract_type STRING, payment_terms_days STRING, is_active STRING"

df_customers = spark.createDataFrame(customers_data, schema=customers_schema)

(df_customers.write
    .mode("overwrite")
    .saveAsTable(f"{BRONZE}.dim_customers_raw"))

print(f"Created {BRONZE}.dim_customers_raw with {df_customers.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Load customer orders
# MAGIC
# MAGIC 31 orders spread across active customers and multiple years.
# MAGIC
# MAGIC **Important:** `unit_price` values here are ~1000x higher than realistic
# MAGIC per-can prices. The silver layer applies a `/1000` scale correction —
# MAGIC that correction and why it lives in silver is one of the teaching points
# MAGIC in `silver_orders`.
# MAGIC
# MAGIC Intentional DQ issues (all caught by ON VIOLATION DROP ROW constraints):
# MAGIC - 2 rows with NULL `order_id`
# MAGIC - 1 row with NULL `customer_id`
# MAGIC - 2 rows with `quantity_units <= 0`
# MAGIC
# MAGIC Net effect: **31 input → 26 rows in `silver_orders`**. After the `/1000`
# MAGIC scale correction, all prices land in [0, 1] so `plausible_price` records
# MAGIC 0 violations.

# COMMAND ----------

orders_data = [
    # (order_id, order_date,   customer_id, product_id, plant_id, quantity_units, unit_price, freight_cost_per_unit, requested_delivery_date, status)
    ("O1001", "2023-01-15", "C001", "P-CAN-12OZ", "PLT-ATL", "500",  "108.50", "0.012", "2023-01-22", "delivered"),
    ("O1002", "2023-02-03", "C002", "P-CAN-16OZ", "PLT-ATL", "750",  "124.00", "0.015", "2023-02-10", "delivered"),
    ("O1003", "2023-02-18", "C003", "P-CAN-12OZ", "PLT-DEN", "1200", "108.50", "0.010", "2023-02-25", "delivered"),
    ("O1004", "2023-03-07", "C001", "P-CAN-16OZ", "PLT-ATL", "300",  "124.00", "0.012", "2023-03-14", "delivered"),
    ("O1005", "2023-04-12", "C004", "P-CAN-12OZ", "PLT-CHI", "900",  "108.50", "0.011", "2023-04-19", "delivered"),
    ("O1006", "2023-05-01", "C005", "P-CAN-16OZ", "PLT-ATL", "600",  "124.00", "0.014", "2023-05-08", "delivered"),
    ("O1007", "2023-06-20", "C007", "P-CAN-12OZ", "PLT-CHI", "450",  "108.50", "0.011", "2023-06-27", "delivered"),
    ("O1008", "2023-07-05", "C008", "P-CAN-24OZ", "PLT-NYC", "200",  "156.75", "0.018", "2023-07-12", "delivered"),
    ("O1009", "2023-08-14", "C003", "P-CAN-16OZ", "PLT-DEN", "1000", "124.00", "0.010", "2023-08-21", "delivered"),
    ("O1010", "2023-09-22", "C010", "P-CAN-12OZ", "PLT-CHI", "700",  "108.50", "0.011", "2023-09-29", "delivered"),
    ("O1011", "2023-10-10", "C002", "P-CAN-12OZ", "PLT-ATL", "550",  "108.50", "0.015", "2023-10-17", "delivered"),
    ("O1012", "2023-11-28", "C009", "P-CAN-16OZ", "PLT-DEN", "400",  "124.00", "0.013", "2023-12-05", "delivered"),
    ("O1013", "2024-01-09", "C001", "P-CAN-12OZ", "PLT-ATL", "800",  "108.50", "0.012", "2024-01-16", "delivered"),
    ("O1014", "2024-02-14", "C005", "P-CAN-24OZ", "PLT-ATL", "250",  "156.75", "0.014", "2024-02-21", "delivered"),
    ("O1015", "2024-03-03", "C007", "P-CAN-16OZ", "PLT-CHI", "600",  "124.00", "0.011", "2024-03-10", "delivered"),
    ("O1016", "2024-04-18", "C008", "P-CAN-12OZ", "PLT-NYC", "900",  "108.50", "0.018", "2024-04-25", "delivered"),
    ("O1017", "2024-05-27", "C003", "P-CAN-24OZ", "PLT-DEN", "300",  "156.75", "0.010", "2024-06-03", "delivered"),
    ("O1018", "2024-07-11", "C010", "P-CAN-16OZ", "PLT-CHI", "850",  "124.00", "0.011", "2024-07-18", "delivered"),
    ("O1019", "2024-08-06", "C002", "P-CAN-12OZ", "PLT-ATL", "500",  "108.50", "0.015", "2024-08-13", "delivered"),
    ("O1020", "2024-09-19", "C004", "P-CAN-16OZ", "PLT-CHI", "700",  "124.00", "0.011", "2024-09-26", "delivered"),
    ("O1021", "2024-10-30", "C001", "P-CAN-12OZ", "PLT-ATL", "1000", "108.50", "0.012", "2024-11-06", "delivered"),
    ("O1022", "2024-11-14", "C009", "P-CAN-24OZ", "PLT-DEN", "150",  "156.75", "0.013", "2024-11-21", "delivered"),
    ("O1023", "2025-01-08", "C005", "P-CAN-16OZ", "PLT-ATL", "650",  "124.00", "0.014", "2025-01-15", "delivered"),
    ("O1024", "2025-02-20", "C008", "P-CAN-12OZ", "PLT-NYC", "750",  "108.50", "0.018", "2025-02-27", "delivered"),
    ("O1025", "2025-03-15", "C007", "P-CAN-24OZ", "PLT-CHI", "400",  "156.75", "0.011", "2025-03-22", "pending"),
    ("O1026", "2025-04-02", "C010", "P-CAN-12OZ", "PLT-CHI", "900",  "108.50", "0.011", "2025-04-09", "pending"),
    # --- intentional DQ issues below ---
    (None,   "2025-01-15", "C001", "P-CAN-12OZ", "PLT-ATL", "200",  "108.50", "0.012", "2025-01-22", "pending"),
    (None,   "2025-02-01", "C003", "P-CAN-16OZ", "PLT-DEN", "300",  "124.00", "0.010", "2025-02-08", "pending"),
    ("O1027","2025-03-10", None,   "P-CAN-12OZ", "PLT-ATL", "100",  "108.50", "0.012", "2025-03-17", "pending"),
    ("O1028","2025-04-05", "C002", "P-CAN-16OZ", "PLT-ATL", "0",    "124.00", "0.015", "2025-04-12", "cancelled"),
    ("O1029","2025-04-10", "C004", "P-CAN-12OZ", "PLT-CHI", "-50",  "108.50", "0.011", "2025-04-17", "cancelled"),
]

orders_schema = """
    order_id STRING, order_date STRING, customer_id STRING, product_id STRING,
    plant_id STRING, quantity_units STRING, unit_price STRING,
    freight_cost_per_unit STRING, requested_delivery_date STRING, status STRING
"""

df_orders = spark.createDataFrame(orders_data, schema=orders_schema)

(df_orders.write
    .mode("overwrite")
    .saveAsTable(f"{BRONZE}.raw_customer_orders"))

print(f"Created {BRONZE}.raw_customer_orders with {df_orders.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Verify

# COMMAND ----------

verify_sql = f"""
SELECT 'dim_customers_raw'   AS table_name, COUNT(*) AS row_count, 13 AS expected FROM `{CATALOG}`.`{SCHEMA}`.dim_customers_raw
UNION ALL
SELECT 'raw_customer_orders',               COUNT(*),              31           FROM `{CATALOG}`.`{SCHEMA}`.raw_customer_orders
"""
display(spark.sql(verify_sql))

# Hard assertion so a partial load doesn't silently break the lab.
counts = {
    "dim_customers_raw":   spark.table(f"`{CATALOG}`.`{SCHEMA}`.dim_customers_raw").count(),
    "raw_customer_orders": spark.table(f"`{CATALOG}`.`{SCHEMA}`.raw_customer_orders").count(),
}
expected = {"dim_customers_raw": 13, "raw_customer_orders": 31}
mismatches = [(k, counts[k], expected[k]) for k in expected if counts[k] != expected[k]]
if mismatches:
    raise AssertionError(f"Row count mismatch — bronze tables not loaded as expected: {mismatches}")
print(f"Bronze row counts OK: {counts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## You're ready!
# MAGIC
# MAGIC Both bronze tables are loaded. When you set up your pipeline, point the silver layer sources at:
# MAGIC
# MAGIC ```
# MAGIC <catalog>.<schema>.dim_customers_raw
# MAGIC <catalog>.<schema>.raw_customer_orders
# MAGIC ```
# MAGIC
# MAGIC **Expected DQ behavior when the pipeline runs:**
# MAGIC
# MAGIC | Layer | Mechanism | Action | Effect on row count |
# MAGIC |---|---|---|---|
# MAGIC | `silver_customers` | Constraint `valid_customer_id` | **DROP** (ON VIOLATION DROP ROW) | −1 (NULL customer_id) |
# MAGIC | `silver_customers` | Constraint `valid_payment_terms` | **WARN** (no ON VIOLATION clause) | 0 — row kept, violation counted in event log (C011, payment_terms_days = 999) |
# MAGIC | `silver_customers` | `ROW_NUMBER()` CTE | CTE filter (not a constraint) | −1 (duplicate C001) |
# MAGIC | `silver_customers` | `WHERE is_active = TRUE` | SELECT filter (not a constraint) | −1 (C006 Rocky Mountain Refresh) |
# MAGIC | `silver_customers` | **Net** | — | **13 → 10 rows** |
# MAGIC | `silver_orders` | Constraint `has_order_id` | **DROP** | −2 |
# MAGIC | `silver_orders` | Constraint `has_customer_id` | **DROP** | −1 |
# MAGIC | `silver_orders` | Constraint `has_positive_qty` | **DROP** | −2 |
# MAGIC | `silver_orders` | Constraint `plausible_price` | **WARN** | 0 violations (all prices fall in [0, 1] after `/1000` scaling) |
# MAGIC | `silver_orders` | **Net** | — | **31 → 26 rows** |
# MAGIC
# MAGIC **Constraint event log (what the pipeline UI will show):**
# MAGIC - `silver_customers`: 1 drop (`valid_customer_id`), 1 warning (`valid_payment_terms`). The dedup and `is_active` filters do not appear in the event log because they aren't constraints.
# MAGIC - `silver_orders`: 5 drops total (2 + 1 + 2). `plausible_price` shows 0.

# COMMAND ----------

print(f"Bootstrap complete!")
print(f"  {BRONZE}.dim_customers_raw")
print(f"  {BRONZE}.raw_customer_orders")
print(f"\nUpdate your pipeline SQL to reference these tables before running.")
