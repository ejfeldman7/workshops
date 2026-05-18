-- ============================================================================
--  Gold: gold_customer_summary
-- ----------------------------------------------------------------------------
--  Joins silver_orders to silver_customers and aggregates one row per
--  customer. This is the kind of "customer 360" rollup that a BI tool or
--  AI/BI Genie space typically consumes.
--
--  Notice how we reference `silver_orders` and `silver_customers` by their
--  short names (no catalog/schema prefix). Inside an SDP pipeline, sibling
--  tables are auto-resolved -- SDP also uses this to build the DAG and
--  ensure silver runs before gold.
-- ============================================================================

CREATE OR REFRESH MATERIALIZED VIEW gold_customer_summary
COMMENT "One row per active customer with their lifetime order metrics."
AS
SELECT
  c.customer_id,
  c.customer_name,
  c.tier,
  c.region,
  c.contract_type,

  -- Volume metrics
  COUNT(DISTINCT o.order_id)              AS order_count,
  SUM(o.quantity_units)                   AS total_units_sold,

  -- Revenue metrics. Round to 2 dp so dashboards display cleanly.
  ROUND(SUM(o.gross_revenue), 2)          AS lifetime_revenue,
  ROUND(SUM(o.total_freight_cost), 2)     AS lifetime_freight_cost,
  ROUND(AVG(o.gross_revenue), 2)          AS avg_order_revenue,

  -- Recency. Useful for churn / engagement signals.
  MIN(o.order_date)                       AS first_order_date,
  MAX(o.order_date)                       AS last_order_date

FROM silver_customers AS c
-- LEFT JOIN so customers with zero orders still appear (NULL metrics).
-- Use INNER JOIN if you only want customers who have actually ordered.
LEFT JOIN silver_orders AS o
  ON c.customer_id = o.customer_id
GROUP BY ALL;   -- shorthand for "GROUP BY every non-aggregated column"
