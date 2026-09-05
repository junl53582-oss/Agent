# PREDICTION_V2_JQDATA_FEATURE_OVERLAP_AND_INFORMATION_AUDIT_REPORT

## Outcome

Status: `PREDICTION_V2_JQDATA_INFORMATION_AUDIT_COMPLETE`

Research status: `RESEARCH_ONLY`

This audit identifies structural overlap, coverage and collection readiness only. It did not read return-label values, compute RankIC, train a model or claim predictive alpha.

## Inputs

- JQData features: 71
- JQData rows / symbols: 14044 / 300
- JQData date range: 2025-05-28 to 2026-08-25
- Frozen Gen2 features: 61
- Safe-projection Gen2 rows / symbols: 947079 / 747
- Exact overlap: 11880 rows, 262 symbols, 268 dates
- Overlap range: 2025-05-28 to 2026-08-14

## Collection Shortlist

The 20 entries below are worth continued collection or bounded residual research. They are not promoted signals. Continuous candidates are greedily de-duplicated at absolute within-JQData date-ranked correlation 0.90.

| shortlist_rank | feature | family | selection_status | active_dates | maximum_abs_gen2_rank_corr | novelty_class |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | jq_company_forecast_profit_max | company_forecast | KEEP_ACCUMULATING_EVENT | 82 |  | NOT_ESTIMABLE |
| 2 | jq_company_forecast_profit_min | company_forecast | KEEP_ACCUMULATING_EVENT | 82 |  | NOT_ESTIMABLE |
| 3 | jq_company_forecast_ratio_max | company_forecast | KEEP_ACCUMULATING_EVENT | 82 |  | NOT_ESTIMABLE |
| 4 | jq_company_forecast_ratio_min | company_forecast | KEEP_ACCUMULATING_EVENT | 82 |  | NOT_ESTIMABLE |
| 5 | jq_hkhold_share_number | hkhold | KEEP_ACCUMULATING_SNAPSHOT | 5 |  | NOT_ESTIMABLE |
| 6 | jq_hkhold_share_ratio | hkhold | KEEP_ACCUMULATING_SNAPSHOT | 5 |  | NOT_ESTIMABLE |
| 7 | jq_valuation_pcf_ratio_own_percentile | valuation | KEEP_FOR_RESIDUAL_AUDIT | 50 | 0.138049 | LOW_REDUNDANCY |
| 8 | jq_risk_kurtosis20 | risk | KEEP_FOR_RESIDUAL_AUDIT | 248 | 0.162759 | LOW_REDUNDANCY |
| 9 | jq_valuation_pcf_ratio_industry_percentile | valuation | KEEP_FOR_RESIDUAL_AUDIT | 50 | 0.214612 | LOW_REDUNDANCY |
| 10 | jq_risk_skewness20 | risk | KEEP_FOR_RESIDUAL_AUDIT | 248 | 0.239272 | LOW_REDUNDANCY |
| 11 | jq_valuation_pe_ratio_industry_percentile | valuation | KEEP_FOR_RESIDUAL_AUDIT | 50 | 0.263285 | LOW_REDUNDANCY |
| 12 | jq_valuation_pcf_ratio_percentile | valuation | KEEP_FOR_RESIDUAL_AUDIT | 50 | 0.265282 | LOW_REDUNDANCY |
| 13 | jq_valuation_pb_ratio_industry_percentile | valuation | KEEP_FOR_RESIDUAL_AUDIT | 50 | 0.280927 | LOW_REDUNDANCY |
| 14 | jq_growth_net_operate_cashflow_growth_rate | growth | KEEP_FOR_RESIDUAL_AUDIT | 248 | 0.396281 | LOW_REDUNDANCY |
| 15 | jq_valuation_pe_ratio_own_zscore | valuation | KEEP_FOR_RESIDUAL_AUDIT | 49 | 0.491532 | LOW_REDUNDANCY |
| 16 | jq_valuation_market_cap_industry_percentile | valuation | KEEP_FOR_RESIDUAL_AUDIT | 50 | 0.504160 | LOW_REDUNDANCY |
| 17 | jq_valuation_ps_ratio_industry_percentile | valuation | KEEP_FOR_RESIDUAL_AUDIT | 50 | 0.533763 | LOW_REDUNDANCY |
| 18 | jq_valuation_pb_ratio_percentile | valuation | KEEP_FOR_RESIDUAL_AUDIT | 50 | 0.540007 | LOW_REDUNDANCY |
| 19 | jq_growth_total_asset_growth_rate | growth | KEEP_FOR_RESIDUAL_AUDIT | 248 | 0.544037 | LOW_REDUNDANCY |
| 20 | jq_growth_net_profit_growth_rate | growth | KEEP_FOR_RESIDUAL_AUDIT | 248 | 0.528566 | LOW_REDUNDANCY |

## High Structural Redundancy

Threshold: absolute contemporaneous date-ranked correlation at least 0.85, including sector-conditioned comparison.

| feature | most_correlated_gen2_feature | maximum_abs_gen2_rank_corr | most_correlated_jq_feature | maximum_abs_jq_rank_corr |
| --- | --- | --- | --- | --- |
| jq_momentum_price1m | momentum | 0.875265 | jq_momentum_rank1m | 0.849920 |
| jq_momentum_price3m | momentum | 0.898173 | jq_momentum_roc20 | 0.791075 |
| jq_momentum_rank1m | momentum | 0.902626 | jq_momentum_roc20 | 0.999789 |
| jq_momentum_roc20 | momentum | 0.902672 | jq_momentum_rank1m | 0.999789 |
| jq_quality_debt_to_asset_ratio | debt_ratio_rank | 0.995478 | jq_quality_roic_ttm | 0.713223 |
| jq_quality_roe_ttm | roe_rank | 0.865177 | jq_quality_roic_ttm | 0.926022 |
| jq_quality_roic_ttm | quality | 0.888139 | jq_quality_roe_ttm | 0.926022 |
| jq_risk_variance20 | low_volatility | 0.947591 | jq_risk_variance60 | 0.864599 |
| jq_risk_variance60 | volatility_60_rank | 0.943182 | jq_risk_variance20 | 0.864599 |

## Classification Counts

```json
{
  "CONTROL_ONLY": 22,
  "EXCLUDE_PIT_RISK": 3,
  "KEEP_ACCUMULATING_EVENT": 4,
  "KEEP_ACCUMULATING_SNAPSHOT": 2,
  "KEEP_FOR_RESIDUAL_AUDIT": 31,
  "PAUSE_HIGH_REDUNDANCY": 9
}
```

## Quota Actions

Provider queries in this audit: 0.

Pause or stop consuming quota:

- `MONEYFLOW_HISTORY_DAILY`: STOP_RETRY_UNTIL_ENTITLEMENT_CHANGES
- `FINANCE_BALANCE_SHEET`: PAUSE_UNTIL_RESTATEMENT_REPLAY_IS_PROVED
- `FINANCE_INCOME_STATEMENT`: PAUSE_UNTIL_RESTATEMENT_REPLAY_IS_PROVED
- `FINANCE_CASHFLOW_STATEMENT`: PAUSE_UNTIL_RESTATEMENT_REPLAY_IS_PROVED
- `STK_PERFORMANCE_LETTERS`: PAUSE_FEATURE_ADMISSION_UNTIL_REVISION_LINEAGE_IS_PROVED
- `INDUSTRY_CATALOG`: STATIC_CACHE_REFRESH_ONLY_ON_PROVIDER_VERSION_CHANGE

The factor library should be reduced to the recorded collection shortlist. Valuation stays weekly; company forecasts, HK holdings and disclosure metadata stay incremental-only.

## Integrity

- Return labels read: FALSE
- RankIC: NOT_COMPUTED
- Model training: FALSE
- Predictive alpha claim: FALSE
- Gen2 modified: 0
- 007-012 modified: 0
- DAILY PIT modified: 0
- daily prediction modified: 0
- Production modified: 0

## Next Stage

`JQDATA_FORWARD_ONLY_RESIDUAL_CHALLENGER_RESEARCH_ONLY`

That stage must remain research-only until enough mature, cross-regime forward observations exist. Frozen Gen2 remains the production champion.
