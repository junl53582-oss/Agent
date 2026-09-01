# GEN2_ALPHA_PREDICTION_DIAGNOSTIC_REPORT

## 1. Final Status

`GEN2_ALPHA_DIAGNOSTIC_COMPLETE`

## 2. Baseline

Git baseline `63a866830efad38098ae8cc3237b4cd8340970c8`; model `GEN2-LGBM-20D-SECTOR-BALANCED-TOP20`; target is the cross-sectional rank of 20-trading-day T+1-open to T+21-open return. The 000300 PIT constituent universe uses 61 candidate features, yearly train-only selection (15–20 active), an eight-year rolling training window, one validation year, 21-session purge, and yearly 2020–2025 unseen folds.

## 3. PIT / Leakage Audit

Membership snapshots, fundamental availability dates, and industry effective dates are on or before each decision date. Training labels mature before validation/test boundaries; date-symbol duplicates are absent. No 2026 labels were read. Historical OOS folds are valid for diagnostic description, but 2020–2025 was previously used for model selection and 2026 is disqualified, so there is no untouched confirmatory holdout.

## 4. Overall Predictive Power

Pearson IC `0.025334`, Rank IC `0.049877`, ICIR `0.2653`, positive IC ratio `59.27%`. Median Rank IC is `0.041725` over `1434` dates. The Q5−Q1 mean-return spread is `0.002899`; Top20 cross-sectional proxy excess is `0.000750` with Precision@20 `52.95%`.

## 5. Ranking Monotonicity

The curve is not monotonic: Q3 has the highest mean realized return, while Q5 is only modestly above Q1. This indicates broad but weak ordering information rather than a clean calibrated return ladder.

| quantile | mean_return | excess_return | precision_above_median |
| --- | --- | --- | --- |
| 1 | 0.001915 | -0.002819 | 0.457840 |
| 2 | 0.007410 | 0.002676 | 0.496889 |
| 3 | 0.007678 | 0.002944 | 0.508037 |
| 4 | 0.005149 | 0.000415 | 0.508044 |
| 5 | 0.004814 | 0.000080 | 0.524112 |

## 6. Time Stability

All six annual mean Rank IC values are positive, but stability is poor: 2023 is strongest (`0.1078`) and 2025 is effectively flat (`0.00125`, positive-date ratio `41.9%`). Monthly and quarterly evidence is in `walk_forward_metrics.csv`.

| period | rank_ic_mean | icir | positive_ic_ratio |
| --- | --- | --- | --- |
| 2020 | 0.014266 | 0.102974 | 0.493827 |
| 2021 | 0.046501 | 0.393373 | 0.641975 |
| 2022 | 0.030089 | 0.270345 | 0.541322 |
| 2023 | 0.107825 | 0.697513 | 0.764463 |
| 2024 | 0.095473 | 0.367684 | 0.681818 |
| 2025 | 0.001249 | 0.004736 | 0.418919 |

## 7. Regime Performance

Risk-off Rank IC (`0.0187`) is far below neutral (`0.0665`). High-volatility Rank IC (`0.0768`) exceeds low-volatility (`0.0282`), so Gen2 does not generalize evenly. Volatility regimes use the current cross-section against a shifted trailing 252-session median; existing panel regimes are named `risk_on`, `risk_off`, and `neutral` rather than relabelled bull/bear/sideways.

| dimension | regime | rank_ic_mean | icir |
| --- | --- | --- | --- |
| market_regime | neutral | 0.066480 | 0.385078 |
| market_regime | risk_off | 0.018718 | 0.105839 |
| market_regime | risk_on | 0.039865 | 0.172887 |
| volatility_regime | high_vol | 0.076785 | 0.391009 |
| volatility_regime | insufficient_history | 0.080265 | 0.382606 |
| volatility_regime | low_vol | 0.028230 | 0.159328 |

## 8. Sector Performance

Finance/real-estate (`0.0905`) and cyclical manufacturing (`0.0677`) are strongest. Technology (`0.0075`) and `other` (`0.0012`) are nearly uninformative; defensive has fewer than 20 names per daily cross-section and is not evaluable for within-sector IC. Sector balancing historically changed Top20 net proxy alpha from `-0.0221` to `+0.0249` and reduced worst sector weight from `0.899` to `0.420`, at the cost of higher turnover and worse drawdown. This is development evidence, not untouched confirmation.

| sector | rows | rank_ic_mean | positive_ic_ratio |
| --- | --- | --- | --- |
| finance_real_estate | 63024 | 0.090520 | 0.626220 |
| cyclical_manufacturing | 125839 | 0.067719 | 0.627615 |
| healthcare | 33513 | 0.048087 | 0.597674 |
| consumer | 24685 | 0.012512 | 0.538071 |
| technology | 70747 | 0.007489 | 0.503487 |
| other | 21075 | 0.001161 | 0.452880 |
| defensive | 18436 | N/A | N/A |

## 9. Cap / Liquidity / Volatility

The signal is strongest in small-cap (`0.0751`) versus large-cap (`0.0235`) and high-volatility (`0.0598`) versus low-volatility (`0.0096`) buckets. It is not merely an illiquidity artifact: high-liquidity Rank IC (`0.0630`) exceeds low-liquidity (`0.0369`). Size uses PIT benchmark-weight rank.

| dimension | bucket | rank_ic_mean | icir |
| --- | --- | --- | --- |
| market_cap | small | 0.075091 | 0.381437 |
| market_cap | mid | 0.058063 | 0.280992 |
| market_cap | large | 0.023479 | 0.106041 |
| liquidity | low | 0.036857 | 0.213682 |
| liquidity | medium | 0.038464 | 0.183692 |
| liquidity | high | 0.062988 | 0.274798 |
| volatility | low | 0.009644 | 0.055526 |
| volatility | mid | 0.018376 | 0.112059 |
| volatility | high | 0.059840 | 0.362531 |

## 10. Feature Diagnostics

Risk, liquidity and fundamental changes dominate. `volatility_60_rank` has the highest mean gain but is unstable; `liquidity`, `revenue_growth_change_rank`, `profit_growth_change_rank`, and `gross_margin_change_rank` are active in every fold. Only three sampled pairs exceed `|rho|=0.90`: volatility/downside-volatility, revenue-growth/growth, and profit-growth/growth. Importance is fold-specific and absence from yearly train-only selection counts as zero activity.

| feature | feature_group | mean_gain_importance | active_fold_ratio | classification |
| --- | --- | --- | --- | --- |
| volatility_60_rank | risk | 0.141904 | 0.833333 | unstable_or_regime |
| liquidity | liquidity | 0.071851 | 1.000000 | stable |
| revenue_growth_change_rank | fundamental | 0.060632 | 1.000000 | stable |
| profit_growth_change_rank | fundamental | 0.057586 | 1.000000 | stable |
| deducted_profit_growth_rank | fundamental | 0.053813 | 0.666667 | unstable_or_regime |
| benchmark_weight_rank | price_behavior | 0.053418 | 0.500000 | unstable_or_regime |
| gross_margin_change_rank | fundamental | 0.051107 | 1.000000 | stable |
| momentum | price_behavior | 0.045756 | 0.833333 | unstable_or_regime |
| gross_margin_yoy_change_rank | fundamental | 0.044198 | 1.000000 | stable |
| operating_cash_margin_rank | fundamental | 0.041510 | 0.500000 | unstable_or_regime |
| deducted_profit_growth_change_rank | fundamental | 0.040067 | 1.000000 | stable |
| revenue_growth_rank | fundamental | 0.037958 | 0.500000 | unstable_or_regime |

## 11. Feature Group Ablation

Removing risk reduces Rank IC by `0.01169`; removing liquidity by `0.00788`; removing price behavior by `0.00384`. Removing the full fundamental group slightly raises mean Rank IC (`+0.00122`) but lowers ICIR (`-0.0278`), suggesting unstable/conditional rather than uniformly useless fundamentals.

| ablation | rank_ic_mean | rank_ic_change | icir_change |
| --- | --- | --- | --- |
| full_gen2 | 0.049877 | 0.000000 | 0.000000 |
| minus_fundamental | 0.051093 | 0.001216 | -0.027796 |
| minus_industry_technology | 0.048441 | -0.001435 | -0.006457 |
| minus_liquidity | 0.041996 | -0.007881 | -0.040863 |
| minus_price_behavior | 0.046035 | -0.003842 | -0.033319 |
| minus_risk | 0.038188 | -0.011689 | 0.009699 |

## 12. Factor Exposure

The score is strongly exposed to volatility (`rho=0.422`) and size (`0.212`), and negatively related to momentum (`-0.252`), liquidity score (`-0.208`), and sector momentum (`-0.185`). It is therefore not a disguised positive momentum ranking. After controlling sector, size, volatility, momentum and liquidity day by day, residual Rank IC is `0.0205` (t-stat `6.76`). Beta is not available.

| factor | mean_cross_sectional_rank_correlation | positive_ratio |
| --- | --- | --- |
| momentum | -0.252384 | 0.097629 |
| reversal | 0.090299 | 0.701534 |
| size | 0.212483 | 0.894700 |
| volatility | 0.421987 | 0.886332 |
| liquidity | -0.207522 | 0.011158 |
| sector_momentum | -0.185101 | 0.085774 |
| beta | N/A | N/A |

## 13. Error Analysis

False negatives concentrate in technology and cyclical manufacturing across neutral and risk-off periods; corresponding false-positive distributions are retained in `error_patterns.csv`. The 100 worst top-ranked outcomes are in `extreme_failures.csv`. Earnings/event, gap, order-book depth and crowding explanations are not supported by the current panel and are explicitly not inferred.

## 14. Turnover

Daily Top20 retention `70.28%` and turnover `29.72%`. The canonical stateful Top20 portfolio has average one-way turnover `59.69%` and annualized turnover `7.52`. Adjacent score-rank persistence is `0.867`.

## 15. Cost Sensitivity

Top20 research-proxy alpha is `0.1481` at 0 bps, `0.0311` at 10 bps, `-0.0765` at 20 bps and `-0.3495` at 50 bps. The alpha is therefore cost-fragile. These are one-way-turnover stress tests, not broker/execution claims.

| cost_bps_per_one_way_turnover | net_research_proxy_alpha | cost_drag_sum |
| --- | --- | --- |
| 0 | 0.148084 | 0.000000 |
| 10 | 0.031077 | 0.085953 |
| 20 | -0.076455 | 0.171905 |
| 50 | -0.349478 | 0.429763 |

## 16. Top-K Sensitivity

Top20 is not uniquely supported. In the stateful development replay, Top10 is approximately flat after canonical costs, Top20 is `+0.0249`, Top30 `+0.1447`, and Top50 `+0.2255`; Top5 is high-return but concentrated and more volatile. These comparisons reuse development folds and are hypotheses, not parameters to select retroactively.

| top_k | net_research_proxy_alpha | annualized_turnover | max_drawdown |
| --- | --- | --- | --- |
| 5 | 0.403186 | 8.594734 | -0.230797 |
| 10 | -0.000223 | 8.313345 | -0.213211 |
| 20 | 0.024934 | 7.520851 | -0.225204 |
| 30 | 0.144715 | 6.829187 | -0.226616 |
| 50 | 0.225525 | 5.735111 | -0.236509 |

## 17. Horizon Diagnostic

20D Rank IC (`0.0499`, ICIR `0.2653`) exceeds 5D (`0.0359`, ICIR `0.1939`) under existing PIT labels and fixed folds. 10D/40D are explicitly not evaluable because no frozen same-semantics labels exist.

| horizon | status | rank_ic_mean | icir |
| --- | --- | --- | --- |
| 5 | evaluated | 0.035855 | 0.193875 |
| 20 | evaluated | 0.049877 | 0.265347 |
| 10 | not_evaluable_no_frozen_same_semantics_label | N/A | N/A |
| 40 | not_evaluable_no_frozen_same_semantics_label | N/A | N/A |

## 18. Challenger Models

LightGBM (`0.04988`) only narrowly exceeds Ridge (`0.04828`). The fixed 50/50 rank ensemble has the highest mean Rank IC (`0.05103`) but lower ICIR than LightGBM (`0.2556` versus `0.2653`), so ensemble promotion is not justified. No new dependency or model zoo was introduced.

| model | rank_ic_mean | icir | positive_ic_ratio |
| --- | --- | --- | --- |
| lightgbm | 0.049877 | 0.265347 | 0.592748 |
| ridge | 0.048281 | 0.241256 | 0.580195 |
| ensemble | 0.051027 | 0.255645 | 0.592748 |

## 19. Overfitting / Statistical Confidence

Training Rank IC is `0.160–0.175` across folds versus OOS `0.0499`, a gap of roughly `0.110–0.125`, consistent with material overfit. The 20-session block-bootstrap 95% CI for mean daily Rank IC is `[0.0173, 0.0869]`; serial dependence remains a major uncertainty. Rows are not treated as independent observations. No untouched final holdout exists.

## 20. Key Findings

- Mean daily Rank IC is 0.0499; 20-session block-bootstrap 95% CI is [0.017311704609088104, 0.08686498031154578].
- 6/6 yearly folds have positive Rank IC; 2025 is 0.0012.
- Rank IC confidence lower bound is above zero.
- There is no untouched confirmatory holdout: all 2020–2025 folds were available during historical development, and 2026 is disqualified.
- Residual Rank IC after sector/size/volatility/momentum/liquidity controls is 0.0205.
- Ridge Rank IC is 0.0483; model complexity must earn its incremental value.
- Daily Top20 retention is 70.3%; cost fragility is reported without execution claims.
- The Q1→Q5 return curve is not monotonic: Q3 outperforms Q5, so ranking information is not cleanly calibrated.
- Risk-off, technology, large-cap, and low-volatility slices are the principal weak regions.
- Top20 is not uniquely supported: Top30/Top50 had stronger development net proxy alpha, but must not be selected on these reused folds.

## 21. Gen3 Priorities

### P0 — Create a genuinely untouched confirmation period

Hypothesis: Selection bias is the largest unresolved uncertainty.

Evidence: The frozen model was selected using 2020–2025 and 2026 is not untouched.

Experiment: Pre-register features, model, Top-K and gates, then collect prospective matured labels without retuning.

Success criterion: Positive Rank IC block-bootstrap lower bound and positive net proxy alpha across the pre-registered period.

### P0 — Target the empirically weakest regime and tail

Hypothesis: Gen2 failure is concentrated rather than uniform.

Evidence: Risk-off Rank IC is 0.0187 versus 0.0665 in neutral; technology is 0.0075 and 2025 is 0.00125.

Experiment: Develop PIT-safe regime interactions only on a new development partition.

Success criterion: Worst-regime Rank IC improves while full-period ICIR and turnover do not deteriorate.

### P1 — Compress redundant features and benchmark Ridge

Hypothesis: A smaller information set may match tree performance with less variance.

Evidence: Ridge Rank IC 0.0483 nearly matches LightGBM 0.0499; only three sampled pairs exceed |rho|=0.90.

Experiment: Pre-register cluster representatives and compare frozen Ridge/LGBM on new folds.

Success criterion: No Rank IC loss beyond 0.003 and improved fold variance or turnover.

### P1 — Treat Top-K and turnover as joint design variables

Hypothesis: Cutoff concentration and churn can erase weak ranking alpha.

Evidence: Top20 proxy alpha turns negative at 20 bps; Top30/50 outperform Top20 in reused development evidence.

Experiment: Pre-register one buffered selection challenger on development-only data.

Success criterion: Higher net proxy alpha at 20–50 bps with no worse drawdown and sector concentration.

### P2 — Test rank ensemble only if complementarity persists

Hypothesis: Ridge can diversify LightGBM fold errors.

Evidence: The fixed ensemble raises mean Rank IC to 0.0510 but lowers ICIR to 0.2556 versus LightGBM 0.2653.

Experiment: Freeze ensemble weights before a new validation period.

Success criterion: Higher ICIR and worst-regime Rank IC without higher turnover.


## 22. What NOT To Do

- Do not claim promotion or production readiness from reused 2020–2025 development OOS.
- Do not add deep learning, hundreds of features, or alternative data without a new untouched protocol.
- Do not tune Top-K, horizon, or model complexity against these same folds and relabel them holdout.
- Do not interpret the research benchmark proxy as approved official benchmark alpha.

## 23. Git / PR

Branch `codex/gen2-alpha-diagnostic`; commit and PR metadata are populated during delivery. Frozen files modified: none.

## 24. Final Assessment

`Is Gen2 genuinely predictive out-of-sample?` — `WEAK / REGIME_DEPENDENT`.

The strict annual folds show descriptive OOS ranking information, but temporal/regime variation, weak tail evidence, cost sensitivity, and absence of an untouched holdout prevent a strong YES conclusion.

The next phase should focus on evidence-driven alpha improvement rather than further core architecture expansion.
