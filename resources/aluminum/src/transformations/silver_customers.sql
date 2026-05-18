-- ============================================================================
--  Silver: silver_customers
-- ----------------------------------------------------------------------------
--  Cleans up ball.bronze.dim_customers_raw, our customer master data. The
--  bronze table is loaded from CSV and so every column lands as a STRING.
--  This silver layer:
--    * casts each column to its real type
--    * filters out rows that are missing the primary key
--    * keeps only currently-active customers (a typical business rule)
--    * deduplicates so customer_id is truly unique
--
--  This is a MATERIALIZED VIEW (not a STREAMING TABLE) because dim tables
--  are small and we want a full refresh every run. For an append-only fact
--  source you'd use `CREATE OR REFRESH STREAMING TABLE ... AS SELECT * FROM
--  STREAM(...)` instead -- see silver_orders for that pattern.
-- ============================================================================

CREATE OR REFRESH MATERIALIZED VIEW silver_customers
(
  -- Expectations are SDP's data-quality contract. Rows that violate the
  -- condition get the action you specify:
  --   ON VIOLATION DROP ROW   -> silently drop the bad row
  --   ON VIOLATION FAIL UPDATE-> stop the pipeline (fail loudly)
  --   (no clause)             -> still count the violation, but keep the row
  CONSTRAINT valid_customer_id  EXPECT (customer_id IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valid_payment_terms EXPECT (payment_terms_days BETWEEN 0 AND 365)
)
COMMENT "Cleaned, typed, deduplicated customer master data. One row per active customer."
AS
WITH typed AS (
  -- Step 1: cast every column from STRING to its real type.
  -- Using TRY_CAST so a malformed value becomes NULL rather than killing
  -- the whole query. Bad rows will be caught by the constraints above.
  SELECT
    customer_id,
    customer_name,
    tier,
    region,
    contract_type,
    TRY_CAST(payment_terms_days AS INT)     AS payment_terms_days,
    TRY_CAST(LOWER(is_active) IN ('true','t','1','yes') AS BOOLEAN) AS is_active
  FROM ball.bronze.dim_customers_raw
),
deduped AS (
  -- Step 2: keep one row per customer_id. If the bronze layer has duplicates
  -- (common with CDC or re-ingested files) we pick an arbitrary "last" row
  -- using ROW_NUMBER(). For a real SCD-2 use AUTO CDC FROM SNAPSHOT instead.
  SELECT *
  FROM (
    SELECT
      *,
      ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY customer_name) AS rn
    FROM typed
  )
  WHERE rn = 1
)
SELECT
  customer_id,
  customer_name,
  tier,
  region,
  contract_type,
  payment_terms_days,
  is_active
FROM deduped
WHERE is_active = TRUE;   -- business rule: only surface active customers downstream
