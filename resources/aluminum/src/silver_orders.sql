-- ============================================================================
--  Silver: silver_orders
-- ----------------------------------------------------------------------------
--  Cleans raw_customer_orders, the fact table of customer orders.
--  Like the dim table, every column lands in bronze as a STRING. Here we:
--    * cast each column to its real type
--    * compute a few useful derived columns (gross_revenue, total_freight)
--    * drop rows with the worst data-quality issues
--    * keep good-but-suspicious rows so the warehouse team can investigate
--
--  This is a STREAMING TABLE because the bronze source is append-only and we
--  want incremental processing: each pipeline run picks up only the NEW rows
--  that arrived since the last run, instead of re-reading the entire table.
--  STREAM(<table>) is the SQL function that tells SDP "read this as a
--  stream of new rows".
-- ============================================================================

-- Data-quality note: raw_customer_orders has unit_price values
-- that are ~1000x larger than realistic per-can prices (avg ~$108/unit when
-- aluminum beverage cans actually sell for ~$0.05-$0.10). To keep downstream
-- KPIs interpretable we apply a one-time scale correction (/1000) here and
-- document it explicitly. In a real pipeline you'd push back on the source
-- system; this fix lives in silver so gold/metric layers stay clean.
CREATE OR REFRESH STREAMING TABLE silver_orders
(
  -- DROP ROW: severe issues. We can't reason about an order without these.
  CONSTRAINT has_order_id    EXPECT (order_id    IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT has_customer_id EXPECT (customer_id IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT has_positive_qty EXPECT (quantity_units > 0)     ON VIOLATION DROP ROW,

  -- No action: row stays, but SDP tracks the violation count in event logs.
  -- Use this for "data smell" checks where you want visibility, not blocking.
  -- (Threshold is post-scale, so $1 max is generous for a beverage can.)
  CONSTRAINT plausible_price EXPECT (unit_price BETWEEN 0 AND 1)
)
COMMENT "Typed, validated customer orders with revenue + freight derivations. unit_price is scale-corrected /1000 from bronze."
AS
SELECT
  order_id,
  TRY_CAST(order_date AS DATE)               AS order_date,
  customer_id,
  product_id,
  plant_id,
  TRY_CAST(quantity_units AS INT)            AS quantity_units,
  -- Scale correction: bronze unit_price is ~1000x too high. DECIMAL(10,4)
  -- gives us 4 decimal places so $108.6148 -> $0.1086 is preserved.
  TRY_CAST(unit_price AS DECIMAL(10,4)) / 1000      AS unit_price,
  TRY_CAST(freight_cost_per_unit AS DECIMAL(10,4))  AS freight_cost_per_unit,
  TRY_CAST(requested_delivery_date AS DATE)  AS requested_delivery_date,
  status,
  -- Derived columns use the *scaled* unit_price so downstream layers don't
  -- have to know about the correction. Freight is left at source scale.
  TRY_CAST(quantity_units AS INT) * (TRY_CAST(unit_price AS DECIMAL(10,4)) / 1000)
      AS gross_revenue,
  TRY_CAST(quantity_units AS INT) * TRY_CAST(freight_cost_per_unit AS DECIMAL(10,4))
      AS total_freight_cost
FROM STREAM(raw_customer_orders);
