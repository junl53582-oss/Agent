# NEXT_GENERATION_ALPHA_SIGNAL_DISCOVERY_REPORT

## 1. Final Status

`ALPHA_RESEARCH_PLATEAU`. Historical optimization stopped; no production or champion mutation occurred.

## 2. Baseline

Verified merged SHA `442a88a9fc24b9c43e62ec48f38ed7858490adfd`. Gen2 exact reproduction: Rank IC 0.049877, ICIR 0.265347; diagnostic residual IC 0.020488; Top20 20 bps proxy alpha -0.076455. Gen3 ended `GEN3_ALPHA_IMPROVEMENT_INCONCLUSIVE / NO`; Stable-Core Ridge harness reproduced Rank IC 0.064102.

## 3. Signal Families Tested

| family | hypothesis | pit_status | rank_ic_mean | residual_ic | worst_year | incremental_vs_harness | status |
|---|---|---|---|---|---|---|---|
| residual_momentum | Past-beta and sector residual momentum adds independent price information. | PIT_SAFE | 0.062333 | 0.014961 | -0.012746 | -0.001769 | REJECTED |
| liquidity_shock | Abnormal liquidity and price impact contain information beyond liquidity level. | PIT_SAFE | 0.066687 | 0.011893 | -0.015955 | 0.002585 | REJECTED |
| price_path_shape | Path smoothness and return sequencing distinguish durable from noisy trends. | PIT_SAFE | 0.061410 | 0.014066 | -0.010101 | -0.002693 | REJECTED |

Three genuinely different mechanisms were bounded in advance: past-beta/sector residual momentum, abnormal liquidity/price impact, and daily price-path shape. Existing fundamental/valuation features were not relabeled as new information; historically incomplete event/surprise sources were not evaluated.

## 4. Label Families Tested

| label_id | experiment_id | rank_ic_mean | icir | worst_year_rank_ic | train_oos_gap | status |
|---|---|---|---|---|---|---|
| L0_GEN2_CROSS_SECTIONAL_RANK | R1_STABLE_CORE_RIDGE_HARNESS | 0.064102 | 0.287004 | -0.016105 | 0.021424 | REJECTED |
| L1_RAW_FORWARD_RETURN | B1_RAW_FORWARD_RETURN | 0.011303 | 0.091689 | -0.015293 | 0.041562 | REJECTED |
| L2_SECTOR_NEUTRAL_RANK | B2_SECTOR_NEUTRAL_RANK | 0.061168 | 0.272005 | -0.019862 | 0.020909 | REJECTED |
| L4_BETA_RESIDUAL_RANK | B3_BETA_RESIDUAL_RANK | 0.067145 | 0.266540 | -0.027347 | 0.015362 | REJECTED |
| L5_VOL_ADJUSTED_RANK | B4_VOL_ADJUSTED_RANK | 0.053279 | 0.303589 | -0.009848 | 0.025625 | REJECTED |
| L6_ROBUST_WINSORIZED_RETURN | B5_ROBUST_WINSORIZED | 0.023286 | 0.162337 | -0.006127 | 0.041043 | REJECTED |

Market-neutral rank was not trained because subtracting one same-date scalar cannot change cross-sectional ranks. Multi-horizon remained `NOT_EVALUATED` because no frozen same-semantics 10D/40D label was introduced.

## 5. Best New Signals

| family | feature | rank_ic | residual_ic | positive_years | worst_year | maximum_abs_correlation_stable_core |
|---|---|---|---|---|---|---|
| residual_momentum | ng_residual_momentum_120 | 0.007842 | 0.014891 | 4 | -0.119780 | 0.394807 |
| residual_momentum | ng_residual_momentum_60 | -0.006438 | 0.012202 | 3 | -0.080162 | 0.584989 |
| residual_momentum | ng_residual_momentum_consistency | -0.006841 | 0.011350 | 3 | -0.104968 | 0.624982 |
| price_path_shape | ng_return_autocorrelation_20 | -0.002040 | 0.004778 | 3 | -0.024948 | 0.058427 |
| liquidity_shock | ng_volume_price_divergence | 0.011285 | 0.004768 | 5 | -0.011110 | 0.160995 |
| price_path_shape | ng_trend_efficiency_20 | 0.013299 | 0.004415 | 5 | -0.014451 | 0.051165 |
| price_path_shape | ng_recovery_from_low_20 | -0.029184 | 0.003918 | 2 | -0.089411 | 0.729085 |
| price_path_shape | ng_up_day_ratio_20 | -0.013755 | 0.002622 | 2 | -0.091834 | 0.564731 |
| liquidity_shock | ng_abnormal_amount_20_60 | -0.034065 | 0.002549 | 1 | -0.089883 | 0.346996 |
| liquidity_shock | ng_liquidity_dryup_5_60 | -0.024650 | 0.001648 | 1 | -0.051559 | 0.431216 |

## 6. Rejected Signals

| family | rank_ic_mean | residual_ic | incremental_vs_harness | status |
|---|---|---|---|---|
| residual_momentum | 0.062333 | 0.014961 | -0.001769 | REJECTED |
| liquidity_shock | 0.066687 | 0.011893 | 0.002585 | REJECTED |
| price_path_shape | 0.061410 | 0.014066 | -0.002693 | REJECTED |

## 7. Best Label

`L4_BETA_RESIDUAL_RANK` had the highest unified realized-return Rank IC (0.067145). It was selected only from 2020-2023 for the automatic cross experiment. Its weakness is that all label evidence still reuses development history and must map back to actual stock return/cost outcomes.

## 8. Stable-Core Ridge Results

| experiment_id | rank_ic_mean | icir | worst_year_rank_ic | train_rank_ic | train_oos_gap |
|---|---|---|---|---|---|
| R0_GEN2_EXACT_REFERENCE | 0.049877 | 0.265347 | 0.001249 | 0.169366 | 0.119489 |
| R1_STABLE_CORE_RIDGE_HARNESS | 0.064102 | 0.287004 | -0.016105 | 0.085526 | 0.021424 |

## 9. LightGBM Incremental Results

| experiment_id | rank_ic_mean | icir | worst_year_rank_ic | train_oos_gap |
|---|---|---|---|---|
| C2_DEV_SELECTED_SIGNAL_LABEL_LGBM | 0.064081 | 0.306400 | -0.017184 | 0.072338 |

## 10. Residual Alpha

| experiment_id | rank_ic_mean | icir | positive_ic_ratio |
|---|---|---|---|
| R0_GEN2_EXACT_REFERENCE | 0.022726 | 0.234333 | 0.611576 |
| B4_VOL_ADJUSTED_RANK | 0.018017 | 0.211613 | 0.576011 |
| B3_BETA_RESIDUAL_RANK | 0.017147 | 0.203096 | 0.563459 |
| B1_RAW_FORWARD_RETURN | 0.016341 | 0.195407 | 0.582985 |
| R1_STABLE_CORE_RIDGE_HARNESS | 0.016239 | 0.189677 | 0.575314 |
| B2_SECTOR_NEUTRAL_RANK | 0.016006 | 0.189991 | 0.563459 |
| B5_ROBUST_WINSORIZED | 0.015132 | 0.182428 | 0.579498 |
| A1_RESIDUAL_MOMENTUM | 0.014961 | 0.120924 | 0.566248 |
| A3_PRICE_PATH_SHAPE | 0.014066 | 0.167772 | 0.584379 |
| C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE | 0.013940 | 0.171644 | 0.572524 |
| C2_DEV_SELECTED_SIGNAL_LABEL_LGBM | 0.012328 | 0.165755 | 0.576011 |
| A2_LIQUIDITY_SHOCK | 0.011893 | 0.146163 | 0.564854 |

Scores were residualized same-date against sector, size, volatility, momentum and liquidity. Beta is separately addressed in the new signal and beta-residual label; no future beta feature was used.

## 11. Weak-Regime Performance

| experiment_id | slice | rank_ic_mean | icir | positive_ic_ratio |
|---|---|---|---|---|
| R0_GEN2_EXACT_REFERENCE | risk_off | 0.018718 | 0.105839 | 0.519520 |
| R0_GEN2_EXACT_REFERENCE | technology | 0.007489 | 0.033262 | 0.503487 |
| R0_GEN2_EXACT_REFERENCE | large_cap | 0.023565 | 0.106419 | 0.521618 |
| R0_GEN2_EXACT_REFERENCE | low_volatility | 0.009796 | 0.056438 | 0.512552 |
| R1_STABLE_CORE_RIDGE_HARNESS | risk_off | 0.023793 | 0.120133 | 0.591592 |
| R1_STABLE_CORE_RIDGE_HARNESS | technology | 0.037826 | 0.162685 | 0.589261 |
| R1_STABLE_CORE_RIDGE_HARNESS | large_cap | 0.038705 | 0.148735 | 0.582287 |
| R1_STABLE_CORE_RIDGE_HARNESS | low_volatility | 0.033911 | 0.204594 | 0.582287 |
| C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE | risk_off | 0.020165 | 0.091095 | 0.552553 |
| C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE | technology | 0.036392 | 0.147887 | 0.567643 |
| C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE | large_cap | 0.040392 | 0.139768 | 0.566946 |
| C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE | low_volatility | 0.046725 | 0.261773 | 0.625523 |

Weak-year details, including 2025, are in yearly_metrics.csv.

## 12. Quantile Monotonicity

| experiment_id | quantile | mean_return | monotonic_correlation | q5_minus_q1 | adjacent_consistency |
|---|---|---|---|---|---|
| R0_GEN2_EXACT_REFERENCE | 1 | 0.001915 | 0.100000 | 0.002899 | 0.500000 |
| R0_GEN2_EXACT_REFERENCE | 2 | 0.007410 | 0.100000 | 0.002899 | 0.500000 |
| R0_GEN2_EXACT_REFERENCE | 3 | 0.007678 | 0.100000 | 0.002899 | 0.500000 |
| R0_GEN2_EXACT_REFERENCE | 4 | 0.005149 | 0.100000 | 0.002899 | 0.500000 |
| R0_GEN2_EXACT_REFERENCE | 5 | 0.004814 | 0.100000 | 0.002899 | 0.500000 |
| R1_STABLE_CORE_RIDGE_HARNESS | 1 | 0.004329 | 0.900000 | 0.001571 | 0.750000 |
| R1_STABLE_CORE_RIDGE_HARNESS | 2 | 0.004897 | 0.900000 | 0.001571 | 0.750000 |
| R1_STABLE_CORE_RIDGE_HARNESS | 3 | 0.005194 | 0.900000 | 0.001571 | 0.750000 |
| R1_STABLE_CORE_RIDGE_HARNESS | 4 | 0.006597 | 0.900000 | 0.001571 | 0.750000 |
| R1_STABLE_CORE_RIDGE_HARNESS | 5 | 0.005900 | 0.900000 | 0.001571 | 0.750000 |
| C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE | 1 | 0.003741 | 0.900000 | 0.002173 | 0.750000 |
| C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE | 2 | 0.005095 | 0.900000 | 0.002173 | 0.750000 |
| C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE | 3 | 0.005296 | 0.900000 | 0.002173 | 0.750000 |
| C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE | 4 | 0.006887 | 0.900000 | 0.002173 | 0.750000 |
| C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE | 5 | 0.005914 | 0.900000 | 0.002173 | 0.750000 |

## 13. Top-K

| score | top_k | net_research_proxy_alpha | average_one_way_turnover |
|---|---|---|---|
| r0_gen2_exact_reference | 10 | -0.109857 | 0.659789 |
| r0_gen2_exact_reference | 20 | -0.076455 | 0.596893 |
| r0_gen2_exact_reference | 30 | 0.044161 | 0.541999 |
| r0_gen2_exact_reference | 50 | 0.135938 | 0.455168 |
| r1_stable_core_ridge_harness | 10 | -0.199891 | 0.671828 |
| r1_stable_core_ridge_harness | 20 | -0.037287 | 0.604950 |
| r1_stable_core_ridge_harness | 30 | 0.107981 | 0.557824 |
| r1_stable_core_ridge_harness | 50 | -0.029914 | 0.462729 |
| c1_dev_selected_signal_label_ridge | 10 | -0.148523 | 0.683840 |
| c1_dev_selected_signal_label_ridge | 20 | -0.045776 | 0.586022 |
| c1_dev_selected_signal_label_ridge | 30 | -0.025403 | 0.532657 |
| c1_dev_selected_signal_label_ridge | 50 | 0.062019 | 0.441517 |
| b3_beta_residual_rank | 10 | -0.222546 | 0.623633 |
| b3_beta_residual_rank | 20 | -0.017671 | 0.532021 |
| b3_beta_residual_rank | 30 | 0.002586 | 0.480963 |
| b3_beta_residual_rank | 50 | 0.037859 | 0.397861 |

Historical K comparisons are explicitly not production selection evidence.

## 14. Turnover

| score | policy | top_k | buffer_exit_rank | name_retention | rank_persistence | average_one_way_turnover | annualized_turnover |
|---|---|---|---|---|---|---|---|
| r0_gen2_exact_reference | r0_gen2_exact_reference_sector_top10 | 10 | nan | 0.425352 | 0.957568 | 0.659789 | 8.313345 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_sector_top20 | 20 | nan | 0.479577 | 0.957568 | 0.596893 | 7.520851 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_sector_top30 | 30 | nan | 0.515962 | 0.957568 | 0.541999 | 6.829187 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_sector_top50 | 50 | nan | 0.580000 | 0.957568 | 0.455168 | 5.735111 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_buffer20_30 | 20 | 30.000000 | 0.479577 | 0.957568 | 0.391838 | 4.937154 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_sector_top10 | 10 | nan | 0.361972 | 0.978868 | 0.671828 | 8.465038 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_sector_top20 | 20 | nan | 0.448592 | 0.978868 | 0.604950 | 7.622372 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_sector_top30 | 30 | nan | 0.499531 | 0.978868 | 0.557824 | 7.028585 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_sector_top50 | 50 | nan | 0.558873 | 0.978868 | 0.462729 | 5.830387 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_buffer20_30 | 20 | 30.000000 | 0.448592 | 0.978868 | 0.427103 | 5.381503 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_sector_top10 | 10 | nan | 0.309859 | 0.983350 | 0.683840 | 8.616383 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_sector_top20 | 20 | nan | 0.422535 | 0.983350 | 0.586022 | 7.383876 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_sector_top30 | 30 | nan | 0.506573 | 0.983350 | 0.532657 | 6.711473 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_sector_top50 | 50 | nan | 0.596338 | 0.983350 | 0.441517 | 5.563114 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_buffer20_30 | 20 | 30.000000 | 0.422535 | 0.983350 | 0.437661 | 5.514524 |
| b3_beta_residual_rank | b3_beta_residual_rank_sector_top10 | 10 | nan | 0.411268 | 0.985984 | 0.623633 | 7.857776 |
| b3_beta_residual_rank | b3_beta_residual_rank_sector_top20 | 20 | nan | 0.495070 | 0.985984 | 0.532021 | 6.703469 |
| b3_beta_residual_rank | b3_beta_residual_rank_sector_top30 | 30 | nan | 0.558216 | 0.985984 | 0.480963 | 6.060134 |
| b3_beta_residual_rank | b3_beta_residual_rank_sector_top50 | 50 | nan | 0.644789 | 0.985984 | 0.397861 | 5.013055 |
| b3_beta_residual_rank | b3_beta_residual_rank_buffer20_30 | 20 | 30.000000 | 0.495070 | 0.985984 | 0.372578 | 4.694478 |

## 15. Cost Sensitivity

| score | policy | cost_bps | gross_total_return | net_research_proxy_alpha | average_one_way_turnover |
|---|---|---|---|---|---|
| r0_gen2_exact_reference | r0_gen2_exact_reference_sector_top20 | 0 | 0.425216 | 0.148084 | 0.596893 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_sector_top20 | 10 | 0.425216 | 0.031077 | 0.596893 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_sector_top20 | 20 | 0.425216 | -0.076455 | 0.596893 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_sector_top20 | 30 | 0.425216 | -0.175269 | 0.596893 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_sector_top20 | 50 | 0.425216 | -0.349478 | 0.596893 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_buffer20_30 | 0 | 0.420556 | 0.143424 | 0.391838 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_buffer20_30 | 10 | 0.420556 | 0.065876 | 0.391838 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_buffer20_30 | 20 | 0.420556 | -0.007504 | 0.391838 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_buffer20_30 | 30 | 0.420556 | -0.076936 | 0.391838 |
| r0_gen2_exact_reference | r0_gen2_exact_reference_buffer20_30 | 50 | 0.420556 | -0.204784 | 0.391838 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_sector_top20 | 0 | 0.475007 | 0.197875 | 0.604950 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_sector_top20 | 10 | 0.475007 | 0.075267 | 0.604950 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_sector_top20 | 20 | 0.475007 | -0.037287 | 0.604950 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_sector_top20 | 30 | 0.475007 | -0.140601 | 0.604950 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_sector_top20 | 50 | 0.475007 | -0.322439 | 0.604950 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_buffer20_30 | 0 | 0.379199 | 0.102067 | 0.427103 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_buffer20_30 | 10 | 0.379199 | 0.019966 | 0.427103 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_buffer20_30 | 20 | 0.379199 | -0.057324 | 0.427103 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_buffer20_30 | 30 | 0.379199 | -0.130079 | 0.427103 |
| r1_stable_core_ridge_harness | r1_stable_core_ridge_harness_buffer20_30 | 50 | 0.379199 | -0.263019 | 0.427103 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_sector_top20 | 0 | 0.457171 | 0.180039 | 0.586022 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_sector_top20 | 10 | 0.457171 | 0.062452 | 0.586022 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_sector_top20 | 20 | 0.457171 | -0.045776 | 0.586022 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_sector_top20 | 30 | 0.457171 | -0.145380 | 0.586022 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_sector_top20 | 50 | 0.457171 | -0.321374 | 0.586022 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_buffer20_30 | 0 | 0.560701 | 0.283569 | 0.437661 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_buffer20_30 | 10 | 0.560701 | 0.188656 | 0.437661 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_buffer20_30 | 20 | 0.560701 | 0.099428 | 0.437661 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_buffer20_30 | 30 | 0.560701 | 0.015550 | 0.437661 |
| c1_dev_selected_signal_label_ridge | c1_dev_selected_signal_label_ridge_buffer20_30 | 50 | 0.560701 | -0.137402 | 0.437661 |
| b3_beta_residual_rank | b3_beta_residual_rank_sector_top20 | 0 | 0.467384 | 0.190252 | 0.532021 |
| b3_beta_residual_rank | b3_beta_residual_rank_sector_top20 | 10 | 0.467384 | 0.082382 | 0.532021 |
| b3_beta_residual_rank | b3_beta_residual_rank_sector_top20 | 20 | 0.467384 | -0.017671 | 0.532021 |
| b3_beta_residual_rank | b3_beta_residual_rank_sector_top20 | 30 | 0.467384 | -0.110464 | 0.532021 |
| b3_beta_residual_rank | b3_beta_residual_rank_sector_top20 | 50 | 0.467384 | -0.276313 | 0.532021 |
| b3_beta_residual_rank | b3_beta_residual_rank_buffer20_30 | 0 | 0.459139 | 0.182007 | 0.372578 |
| b3_beta_residual_rank | b3_beta_residual_rank_buffer20_30 | 10 | 0.459139 | 0.105982 | 0.372578 |
| b3_beta_residual_rank | b3_beta_residual_rank_buffer20_30 | 20 | 0.459139 | 0.033856 | 0.372578 |
| b3_beta_residual_rank | b3_beta_residual_rank_buffer20_30 | 30 | 0.459139 | -0.034569 | 0.372578 |
| b3_beta_residual_rank | b3_beta_residual_rank_buffer20_30 | 50 | 0.459139 | -0.161051 | 0.372578 |

## 16. Statistical Evidence

| experiment_id | mean_delta | ci_lower | ci_upper | positive_delta_ratio | fold_wins | yearly_wins |
|---|---|---|---|---|---|---|
| C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE | 0.019048 | -0.007473 | 0.048957 | 0.542538 | 4 | 4 |
| B3_BETA_RESIDUAL_RANK | 0.017268 | -0.010065 | 0.048140 | 0.548815 | 4 | 4 |
| A2_LIQUIDITY_SHOCK | 0.016810 | -0.003706 | 0.040292 | 0.532775 | 4 | 4 |
| R1_STABLE_CORE_RIDGE_HARNESS | 0.014226 | -0.007438 | 0.038998 | 0.535565 | 4 | 4 |
| C2_DEV_SELECTED_SIGNAL_LABEL_LGBM | 0.014204 | -0.005784 | 0.036050 | 0.536262 | 4 | 4 |
| A1_RESIDUAL_MOMENTUM | 0.012456 | -0.011483 | 0.035542 | 0.541841 | 4 | 4 |
| A3_PRICE_PATH_SHAPE | 0.011533 | -0.007687 | 0.035462 | 0.520223 | 4 | 4 |
| B2_SECTOR_NEUTRAL_RANK | 0.011291 | -0.009870 | 0.037546 | 0.522315 | 4 | 4 |
| B4_VOL_ADJUSTED_RANK | 0.003402 | -0.016983 | 0.023566 | 0.513250 | 3 | 3 |
| R0_GEN2_EXACT_REFERENCE | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 |
| B5_ROBUST_WINSORIZED | -0.026590 | -0.049903 | -0.006580 | 0.435844 | 1 | 1 |
| B1_RAW_FORWARD_RETURN | -0.038574 | -0.070516 | -0.011508 | 0.415621 | 2 | 2 |

All intervals use paired 20-session moving-block bootstrap with 1000 replications.

## 17. Candidate Ranking

| experiment_id | rank_ic | icir | worst_year | residual_ic | 20bps_alpha | gates_passed | status |
|---|---|---|---|---|---|---|---|
| C1_DEV_SELECTED_SIGNAL_LABEL_RIDGE | 0.068925 | 0.283996 | -0.026237 | 0.013940 | -0.045776 | 4 | REJECTED |
| B3_BETA_RESIDUAL_RANK | 0.067145 | 0.266540 | -0.027347 | 0.017147 | -0.017671 | 4 | REJECTED |
| A2_LIQUIDITY_SHOCK | 0.066687 | 0.317547 | -0.015955 | 0.011893 | nan | 4 | REJECTED |
| R1_STABLE_CORE_RIDGE_HARNESS | 0.064102 | 0.287004 | -0.016105 | 0.016239 | -0.037287 | 4 | REJECTED |
| C2_DEV_SELECTED_SIGNAL_LABEL_LGBM | 0.064081 | 0.306400 | -0.017184 | 0.012328 | nan | 4 | REJECTED |
| A1_RESIDUAL_MOMENTUM | 0.062333 | 0.272217 | -0.012746 | 0.014961 | nan | 4 | REJECTED |
| A3_PRICE_PATH_SHAPE | 0.061410 | 0.283196 | -0.010101 | 0.014066 | nan | 4 | REJECTED |
| B2_SECTOR_NEUTRAL_RANK | 0.061168 | 0.272005 | -0.019862 | 0.016006 | nan | 4 | REJECTED |
| B4_VOL_ADJUSTED_RANK | 0.053279 | 0.303589 | -0.009848 | 0.018017 | nan | 4 | REJECTED |
| R0_GEN2_EXACT_REFERENCE | 0.049877 | 0.265347 | 0.001249 | 0.022726 | -0.076455 | 3 | REJECTED |
| B5_ROBUST_WINSORIZED | 0.023286 | 0.162337 | -0.006127 | 0.015132 | nan | 2 | REJECTED |
| B1_RAW_FORWARD_RETURN | 0.011303 | 0.091689 | -0.015293 | 0.016341 | nan | 2 | REJECTED |

At most two candidates could survive; actual selected list: `[]`.

## 18. Research Integrity

PIT checks: `{'membership_pit': True, 'fundamentals_pit': True, 'industry_pit': True}`. Future leakage: false. Label contamination: false. 2026 labels read: false. 2020-2025 is reused development research, not untouched confirmation. New trailing signals use only decision-date-or-earlier observations; future market/sector returns occur only in label construction. Historical data-mining risk is now high and is the reason optimization stops.

## 19. Automatic Research Actions Performed

Initial: three signal families and five alternative label models, plus Gen2/harness references. The continuation loop read all Track A/B results, selected `liquidity_shock` and `L4_BETA_RESIDUAL_RANK` using 2020-2023 only, then automatically executed `C1` Ridge. `C2` LightGBM was executed after signal admission. No user confirmation was required because these were isolated research-only actions.

## 20. Research Suggestions

### P0

{
  "priority": "P0",
  "hypothesis": "The current plateau reflects missing information rather than insufficient model capacity.",
  "evidence": "Best new standalone residual IC was 0.0149; no family passed joint admission/candidate gates.",
  "expected_mechanism": "A genuinely new PIT source can add orthogonal earnings/event/ownership information.",
  "experiment": "Acquire and freeze a historically complete announcement/analyst/ownership source before any model test.",
  "success_gate": "Source-level PIT audit plus positive residual IC and paired CI on pre-registered future folds.",
  "risk_of_overfitting": "High if source coverage is backfilled or selected after seeing returns.",
  "expected_information_gain": "High",
  "automatic_execution": "NOT_EXECUTED_NEW_EXTERNAL_DATA_AUTHORITY_AND_COVERAGE_REQUIRED"
}

### P1

{
  "priority": "P1",
  "hypothesis": "Historical reuse is now the dominant statistical uncertainty.",
  "evidence": "2020-2025 has been used by Gen2, Gen3 and this bounded discovery cycle.",
  "expected_mechanism": "Future matured DAILY PIT outcomes provide independent evidence.",
  "experiment": "Freeze the best non-promoted research directions and collect unseen forward labels without retuning.",
  "success_gate": "Positive paired block-bootstrap lower bound and non-negative 20 bps proxy alpha on future data.",
  "risk_of_overfitting": "Low if specifications remain frozen.",
  "expected_information_gain": "High",
  "automatic_execution": "WAIT_FOR_FORWARD_EVIDENCE"
}

### P2

{
  "priority": "P2",
  "hypothesis": "Turnover fragility may require a longer-lived signal rather than more ranking complexity.",
  "evidence": "Top20 cost survival remains weak across historical candidates.",
  "expected_mechanism": "New slow-moving information could improve retention and net alpha.",
  "experiment": "Only after a new data source exists, pre-register persistence/holding-period diagnostics.",
  "success_gate": "Lower turnover with no Rank IC/residual IC deterioration.",
  "risk_of_overfitting": "Medium; policy search must remain bounded.",
  "expected_information_gain": "Medium",
  "automatic_execution": "DEFERRED_BUDGET_EXHAUSTED"
}

## 21. Automatic Continuation Decision

`ALPHA_RESEARCH_PLATEAU`. Three major signal families and 12 experiment configs have consumed the bounded historical cycle. Further historical tuning would add more selection bias than information.

## 22. Next Automatic Action

`WAIT_FOR_GENUINELY_UNSEEN_FORWARD_DATA_OR_NEW_PIT_SOURCE`. It is not executed now because the safe historical experiment budget is exhausted and genuinely unseen future labels or a newly authorized PIT source do not yet exist. This is a HARD STOP under the contract, not a request to promote a model.

## 23. Git / PR

Branch `codex/nextgen-alpha-signal-discovery`, baseline `442a88a9fc24b9c43e62ec48f38ed7858490adfd`. Research-only code/tests/artifacts; frozen, 007-012, DAILY PIT and sandbox modified count must remain zero. Final commits, PR head and CI are recorded after push; PR must remain unmerged.

## 24. Final Answer

### A. Did new alpha information materially improve prediction quality?

`NO`

### B. Did label redesign materially improve the learnability of future stock ranking?

`NO`

### C. Is there now a candidate worthy of confirmatory forward validation?

`NO`
