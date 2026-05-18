-- ============================================================================
--  Gold: gold_region_summary
-- ----------------------------------------------------------------------------
--  Rolls gold_customer_summary up one more level: one row per region with
--  KPIs the sales leadership might care about.
--
--  Building this from gold_customer_summary (rather than re-joining the
--  silver tables) demonstrates how gold layers can stack: cheaper to
--  compute, and the customer-level numbers stay consistent with the
--  region-level numbers because both come from the same upstream rollup.
-- ============================================================================

CREATE OR REFRESH MATERIALIZED VIEW gold_region_summary
COMMENT "Region-level KPIs derived from gold_customer_summary."
AS
SELECT
  region,
  COUNT(*)                                AS active_customers,
  SUM(order_count)                        AS total_orders,
  ROUND(SUM(lifetime_revenue), 2)         AS total_revenue,
  ROUND(SUM(lifetime_freight_cost), 2)    AS total_freight,

  -- A simple derived KPI: revenue per active customer.
  -- NULLIF guards against division-by-zero on an empty region.
  ROUND(SUM(lifetime_revenue) / NULLIF(COUNT(*), 0), 2) AS revenue_per_customer

FROM gold_customer_summary
WHERE region IS NOT NULL
GROUP BY region;
