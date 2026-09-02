# GEN3_ALPHA_IMPROVEMENT_REPORT

## 1. Final Status

`GEN3_ALPHA_IMPROVEMENT_INCONCLUSIVE`. Assessment: `NO`. No champion promotion was performed.

## 2. Baseline

Merged SHA `e91a7b2cd97b088bb613977867417b3fa3aaa2d1`; model `GEN2-LGBM-20D-SECTOR-BALANCED-TOP20`. Reproduced Rank IC 0.049877, ICIR 0.265347, residual IC 0.022726; Top20 20 bps proxy alpha -0.076455.

## 3. Experimental Integrity

PIT checks all passed: `{'membership_pit': True, 'fundamentals_pit': True, 'industry_pit': True}`. Six annual walk-forward folds use 8-year training, one past validation year and 21-trading-day purge. No random split, future normalization, future regime or 2026 label was used. Deterministic seed is 42. This is reused development/comparative evidence; no untouched holdout exists.

## 4. Experiment Registry

| id | family | score | hypothesis |
|---|---|---|---|
| A_GEN2_BASELINE | baseline | gen2_baseline | Exact Gen2 reproduction anchors all paired comparisons. |
| B1_SHALLOW_REG | regularization | lgbm_shallow | Lower tree capacity and row/feature subsampling reduce overfit. |
| B2_STRONG_REG | regularization | lgbm_strong_reg | Stronger shrinkage improves weak-year robustness. |
| B3_EARLY_STOP_REG | regularization | lgbm_early_stop | Past-only validation can choose boosting length without OOS tuning. |
| C1_FULL61_REG | feature_selection | lgbm_full61 | Regularization may safely use the complete frozen information set. |
| C2_STABLE_CORE_REG | feature_selection | lgbm_stable_core | Stable high-activity features improve sample efficiency. |
| C3_DE_REDUNDANT_REG | feature_selection | lgbm_dedup | Removing one representative from each >0.90 cluster reduces noise. |
| C4_REMOVE_DEAD_REG | feature_selection | lgbm_dead_removed | Features inactive in all folds and weak in diagnostics add variance. |
| E_NEW_INDEPENDENT_FEATURES | new_features | lgbm_new_features | Six PIT-safe ratios/residual ranks add independent risk/liquidity/price information. |
| F_REGIME_INTERACTIONS | regime_conditioning | lgbm_regime_interactions | Simple PIT interactions address known weak slices without separate models. |
| G1_RIDGE_GEN2 | linear | ridge_gen2 | A linear model may generalize as well as trees with a smaller gap. |
| G2_RIDGE_STABLE_CORE | linear | ridge_stable_core | Stable-core linear structure may improve worst-year behavior. |
| H1_ENSEMBLE_50_50 | ensemble | ensemble_50_50 | Fixed rank averaging diversifies tree and linear errors. |
| H2_ENSEMBLE_PAST_IC | ensemble | ensemble_past_ic | Past validation IC weights adapt without reading test labels. |
| H3_ENSEMBLE_REGIME | ensemble | ensemble_regime | Shrunk past-regime weights improve weak regimes without separate models. |
| I_RESIDUALIZED_SCORE | residualization | residualized_shallow | Current-date exposure neutralization preserves more independent alpha. |

## 5. Gen2 Baseline Reproduction

The exact frozen feature-by-year and LightGBM configuration reproduced Rank IC 0.049877 versus diagnostic 0.049877, a difference of 0.00000000. This anchors paired tests.

## 6. Regularization Results

| experiment_id | train_rank_ic | rank_ic_mean | icir | train_oos_gap | worst_year_rank_ic |
|---|---|---|---|---|---|
| A_GEN2_BASELINE | 0.169366 | 0.049877 | 0.265347 | 0.119489 | 0.001249 |
| B1_SHALLOW_REG | 0.122344 | 0.048103 | 0.234760 | 0.074241 | -0.017460 |
| B2_STRONG_REG | 0.116629 | 0.047307 | 0.226977 | 0.069322 | -0.020700 |
| B3_EARLY_STOP_REG | 0.103078 | 0.045379 | 0.217130 | 0.057699 | -0.022782 |

## 7. Feature Selection Results

| experiment_id | rank_ic_mean | icir | train_oos_gap | worst_year_rank_ic |
|---|---|---|---|---|
| C1_FULL61_REG | 0.047050 | 0.233322 | 0.090179 | -0.024299 |
| C2_STABLE_CORE_REG | 0.060225 | 0.285153 | 0.049293 | -0.018128 |
| C3_DE_REDUNDANT_REG | 0.049074 | 0.242739 | 0.087786 | -0.023254 |
| C4_REMOVE_DEAD_REG | 0.047869 | 0.237052 | 0.082223 | -0.015691 |

## 8. New Feature Results

| item | family | rank_ic | evidence_basis |
|---|---|---|---|
| gen3_vol_adjusted_momentum_60 | price_behavior | -0.014424 | positive_years=3/6 |
| gen3_downside_asymmetry | risk | -0.025220 | positive_years=1/6 |
| gen3_sector_relative_momentum_60 | price_behavior | -0.015338 | positive_years=2/6 |
| gen3_extreme_reversal_5 | price_behavior | 0.003869 | positive_years=3/6 |
| gen3_liquidity_shock | liquidity | 0.019500 | positive_years=5/6 |
| gen3_trend_consistency | price_behavior | -0.016376 | positive_years=3/6 |

The group-level incremental result is recorded for `E_NEW_INDEPENDENT_FEATURES` in `feature_ablation.csv`; all definitions, lookbacks, missing behavior and provenance are in both feature registries.

## 9. Regime Results

| score | dimension | bucket | rank_ic_mean | icir | positive_ic_ratio |
|---|---|---|---|---|---|
| gen2_baseline | market_regime | risk_off | 0.018718 | 0.105839 | 0.519520 |
| gen2_baseline | volatility | low | 0.009644 | 0.055526 | 0.511158 |
| ridge_stable_core | market_regime | risk_off | 0.023793 | 0.120133 | 0.591592 |
| ridge_stable_core | volatility | low | 0.034116 | 0.205986 | 0.586471 |

Repository regimes map current-date market state to risk-on, risk-off and neutral; bull/bear/sideways are not separately relabeled because doing so after observing returns would violate the frozen PIT semantics.

## 10. Sector / Cap Results

| score | dimension | bucket | rank_ic_mean | icir | positive_ic_ratio |
|---|---|---|---|---|---|
| gen2_baseline | sector | technology | 0.007489 | 0.033262 | 0.503487 |
| gen2_baseline | market_cap | large | 0.023479 | 0.106041 | 0.524407 |
| ridge_stable_core | sector | technology | 0.037826 | 0.162685 | 0.589261 |
| ridge_stable_core | market_cap | large | 0.038641 | 0.148649 | 0.583682 |

## 11. Ridge / Linear Results

| experiment_id | rank_ic_mean | icir | train_oos_gap | worst_year_rank_ic |
|---|---|---|---|---|
| G1_RIDGE_GEN2 | 0.048281 | 0.241256 | 0.047885 | -0.005422 |
| G2_RIDGE_STABLE_CORE | 0.064102 | 0.287004 | 0.021424 | -0.016105 |

## 12. Ensemble Results

| experiment_id | rank_ic_mean | icir | worst_year_rank_ic |
|---|---|---|---|
| H1_ENSEMBLE_50_50 | 0.051027 | 0.255645 | -0.001607 |
| H2_ENSEMBLE_PAST_IC | 0.051034 | 0.255282 | -0.001590 |
| H3_ENSEMBLE_REGIME | 0.050946 | 0.254973 | -0.000763 |

Weights for adaptive ensembles came exclusively from each fold's past validation year; no full-period weight fitting occurred.

## 13. Quantile Monotonicity

| score | quantile | mean_return | monotonic_correlation | q5_minus_q1 | adjacent_consistency |
|---|---|---|---|---|---|
| gen2_baseline | 1 | 0.001915 | 0.400000 | 0.002899 | 0.500000 |
| gen2_baseline | 2 | 0.007410 | 0.400000 | 0.002899 | 0.500000 |
| gen2_baseline | 3 | 0.007678 | 0.400000 | 0.002899 | 0.500000 |
| gen2_baseline | 4 | 0.005149 | 0.400000 | 0.002899 | 0.500000 |
| gen2_baseline | 5 | 0.004814 | 0.400000 | 0.002899 | 0.500000 |
| ridge_stable_core | 1 | 0.004329 | 1.000000 | 0.001571 | 0.750000 |
| ridge_stable_core | 2 | 0.004897 | 1.000000 | 0.001571 | 0.750000 |
| ridge_stable_core | 3 | 0.005194 | 1.000000 | 0.001571 | 0.750000 |
| ridge_stable_core | 4 | 0.006597 | 1.000000 | 0.001571 | 0.750000 |
| ridge_stable_core | 5 | 0.005900 | 1.000000 | 0.001571 | 0.750000 |

## 14. Top-K

| score | top_k | net_research_proxy_alpha | average_one_way_turnover |
|---|---|---|---|
| gen2_baseline | 10 | -0.109857 | 0.659789 |
| gen2_baseline | 20 | -0.076455 | 0.596893 |
| gen2_baseline | 30 | 0.044161 | 0.541999 |
| gen2_baseline | 40 | 0.063207 | 0.498353 |
| gen2_baseline | 50 | 0.135938 | 0.455168 |
| lgbm_stable_core | 10 | -0.169959 | 0.710776 |
| lgbm_stable_core | 20 | -0.020541 | 0.596092 |
| lgbm_stable_core | 30 | 0.084763 | 0.547335 |
| lgbm_stable_core | 40 | 0.069810 | 0.497748 |
| lgbm_stable_core | 50 | -0.027935 | 0.461060 |
| lgbm_strong_reg | 10 | -0.115752 | 0.618048 |
| lgbm_strong_reg | 20 | -0.017460 | 0.549069 |
| lgbm_strong_reg | 30 | -0.041527 | 0.494267 |
| lgbm_strong_reg | 40 | 0.019530 | 0.464583 |
| lgbm_strong_reg | 50 | -0.075614 | 0.428461 |

The best historical K remains development evidence and is exposed to multiple-comparison/data-mining risk; production Top20 was not changed.

## 15. Turnover

| score | policy | top_k | buffer_exit_rank | average_one_way_turnover | annualized_turnover |
|---|---|---|---|---|---|
| gen2_baseline | gen2_baseline_sector_top20 | 20 | nan | 0.596893 | 7.520851 |
| ridge_stable_core | ridge_stable_core_sector_top20 | 20 | nan | 0.604950 | 7.622372 |
| gen2_baseline | gen2_baseline_sector_top10 | 10 | nan | 0.659789 | 8.313345 |
| gen2_baseline | gen2_baseline_sector_top30 | 30 | nan | 0.541999 | 6.829187 |
| gen2_baseline | gen2_baseline_sector_top40 | 40 | nan | 0.498353 | 6.279248 |
| gen2_baseline | gen2_baseline_sector_top50 | 50 | nan | 0.455168 | 5.735111 |
| gen2_baseline | gen2_baseline_buffer20_30 | 20 | 30.000000 | 0.391838 | 4.937154 |

## 16. Cost Sensitivity

| score | policy | cost_bps | gross_total_return | net_research_proxy_alpha | average_one_way_turnover |
|---|---|---|---|---|---|
| gen2_baseline | gen2_baseline_sector_top20 | 0 | 0.425216 | 0.148084 | 0.596893 |
| gen2_baseline | gen2_baseline_sector_top20 | 10 | 0.425216 | 0.031077 | 0.596893 |
| gen2_baseline | gen2_baseline_sector_top20 | 20 | 0.425216 | -0.076455 | 0.596893 |
| gen2_baseline | gen2_baseline_sector_top20 | 30 | 0.425216 | -0.175269 | 0.596893 |
| gen2_baseline | gen2_baseline_sector_top20 | 50 | 0.425216 | -0.349478 | 0.596893 |
| ridge_stable_core | ridge_stable_core_sector_top20 | 0 | 0.475007 | 0.197875 | 0.604950 |
| ridge_stable_core | ridge_stable_core_sector_top20 | 10 | 0.475007 | 0.075267 | 0.604950 |
| ridge_stable_core | ridge_stable_core_sector_top20 | 20 | 0.475007 | -0.037287 | 0.604950 |
| ridge_stable_core | ridge_stable_core_sector_top20 | 30 | 0.475007 | -0.140601 | 0.604950 |
| ridge_stable_core | ridge_stable_core_sector_top20 | 50 | 0.475007 | -0.322439 | 0.604950 |
| gen2_baseline | gen2_baseline_buffer20_30 | 0 | 0.420556 | 0.143424 | 0.391838 |
| gen2_baseline | gen2_baseline_buffer20_30 | 10 | 0.420556 | 0.065876 | 0.391838 |
| gen2_baseline | gen2_baseline_buffer20_30 | 20 | 0.420556 | -0.007504 | 0.391838 |
| gen2_baseline | gen2_baseline_buffer20_30 | 30 | 0.420556 | -0.076936 | 0.391838 |
| gen2_baseline | gen2_baseline_buffer20_30 | 50 | 0.420556 | -0.204784 | 0.391838 |

## 17. Residual Alpha

Scores were cross-sectionally residualized against sector, size, volatility, momentum and liquidity using only same-date exposures.

| experiment_id | rank_ic_mean | icir | positive_ic_ratio |
|---|---|---|---|
| A_GEN2_BASELINE | 0.022726 | 0.234333 | 0.611576 |
| H1_ENSEMBLE_50_50 | 0.016720 | 0.162218 | 0.592748 |
| G2_RIDGE_STABLE_CORE | 0.016239 | 0.189677 | 0.575314 |
| H2_ENSEMBLE_PAST_IC | 0.016024 | 0.155751 | 0.583682 |
| H3_ENSEMBLE_REGIME | 0.015613 | 0.151764 | 0.585774 |
| C3_DE_REDUNDANT_REG | 0.015146 | 0.152465 | 0.585077 |
| B1_SHALLOW_REG | 0.014981 | 0.147586 | 0.581590 |
| I_RESIDUALIZED_SCORE | 0.014981 | 0.147586 | 0.581590 |

## 18. Overfitting

| experiment_id | train_rank_ic | rank_ic_mean | train_oos_gap | rank_ic_std |
|---|---|---|---|---|
| G2_RIDGE_STABLE_CORE | 0.085526 | 0.064102 | 0.021424 | 0.223350 |
| G1_RIDGE_GEN2 | 0.096166 | 0.048281 | 0.047885 | 0.200122 |
| C2_STABLE_CORE_REG | 0.109518 | 0.060225 | 0.049293 | 0.211202 |
| E_NEW_INDEPENDENT_FEATURES | 0.112258 | 0.056617 | 0.055641 | 0.212870 |
| F_REGIME_INTERACTIONS | 0.117559 | 0.061241 | 0.056318 | 0.209816 |
| B3_EARLY_STOP_REG | 0.103078 | 0.045379 | 0.057699 | 0.208992 |
| B2_STRONG_REG | 0.116629 | 0.047307 | 0.069322 | 0.208424 |
| B1_SHALLOW_REG | 0.122344 | 0.048103 | 0.074241 | 0.204902 |
| C4_REMOVE_DEAD_REG | 0.130092 | 0.047869 | 0.082223 | 0.201932 |
| C3_DE_REDUNDANT_REG | 0.136860 | 0.049074 | 0.087786 | 0.202168 |
| C1_FULL61_REG | 0.137228 | 0.047050 | 0.090179 | 0.201651 |
| A_GEN2_BASELINE | 0.169366 | 0.049877 | 0.119489 | 0.187967 |

Derived ensembles have no single fitted training score and are therefore reported with a missing train/OOS gap rather than an invented value.

## 19. Statistical Comparison

| experiment_id | mean_delta | ci_lower | ci_upper | positive_daily_difference_ratio | fold_wins |
|---|---|---|---|---|---|
| A_GEN2_BASELINE | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| C2_STABLE_CORE_REG | 0.010348 | -0.006078 | 0.028155 | 0.528591 | 4 |
| F_REGIME_INTERACTIONS | 0.011365 | -0.005317 | 0.030523 | 0.529986 | 4 |
| G2_RIDGE_STABLE_CORE | 0.014226 | -0.007438 | 0.038998 | 0.535565 | 4 |

The confidence intervals use paired 20-session moving-block bootstrap with 1000 replications.

## 20. Candidate Ranking

| experiment_id | rank_ic | icir | worst_year | residual_ic | 20bps_alpha | gates_passed | status |
|---|---|---|---|---|---|---|---|
| G2_RIDGE_STABLE_CORE | 0.064102 | 0.287004 | -0.016105 | 0.016239 | -0.037287 | 4 | REJECTED |
| A_GEN2_BASELINE | 0.049877 | 0.265347 | 0.001249 | 0.022726 | -0.076455 | 4 | REJECTED |
| F_REGIME_INTERACTIONS | 0.061241 | 0.291881 | -0.026737 | 0.009539 | -0.087927 | 3 | REJECTED |
| C2_STABLE_CORE_REG | 0.060225 | 0.285153 | -0.018128 | 0.013900 | -0.020541 | 3 | REJECTED |
| E_NEW_INDEPENDENT_FEATURES | 0.056617 | 0.265970 | -0.024893 | 0.008972 | -0.245039 | 3 | REJECTED |
| H2_ENSEMBLE_PAST_IC | 0.051034 | 0.255282 | -0.001590 | 0.016024 | -0.118289 | 2 | REJECTED |
| H1_ENSEMBLE_50_50 | 0.051027 | 0.255645 | -0.001607 | 0.016720 | -0.077133 | 2 | REJECTED |
| H3_ENSEMBLE_REGIME | 0.050946 | 0.254973 | -0.000763 | 0.015613 | -0.152608 | 2 | REJECTED |
| C3_DE_REDUNDANT_REG | 0.049074 | 0.242739 | -0.023254 | 0.015146 | -0.148652 | 2 | REJECTED |
| B1_SHALLOW_REG | 0.048103 | 0.234760 | -0.017460 | 0.014981 | -0.103904 | 2 | REJECTED |
| C4_REMOVE_DEAD_REG | 0.047869 | 0.237052 | -0.015691 | 0.013178 | -0.244417 | 2 | REJECTED |
| B2_STRONG_REG | 0.047307 | 0.226977 | -0.020700 | 0.014105 | -0.017460 | 2 | REJECTED |
| C1_FULL61_REG | 0.047050 | 0.233322 | -0.024299 | 0.008955 | -0.364567 | 2 | REJECTED |
| B3_EARLY_STOP_REG | 0.045379 | 0.217130 | -0.022782 | 0.012782 | -0.031762 | 2 | REJECTED |
| G1_RIDGE_GEN2 | 0.048281 | 0.241256 | -0.005422 | 0.008383 | -0.274579 | 1 | REJECTED |
| I_RESIDUALIZED_SCORE | 0.014981 | 0.147586 | -0.008496 | 0.014981 | -0.160068 | 1 | REJECTED |

## 21. Best Research Candidate

Selected (maximum two): `[]`. The leading candidate is `G2_RIDGE_STABLE_CORE` because it passed 4 jointly pre-registered gates. Weaknesses remain its -0.007438 paired CI lower bound and lack of untouched confirmation. It remains `GEN3_RESEARCH_CANDIDATE`, not champion.

## 22. Rejected Ideas

| experiment_id | rank_ic | icir | gates_passed | status |
|---|---|---|---|---|
| G2_RIDGE_STABLE_CORE | 0.064102 | 0.287004 | 4 | REJECTED |
| A_GEN2_BASELINE | 0.049877 | 0.265347 | 4 | REJECTED |
| F_REGIME_INTERACTIONS | 0.061241 | 0.291881 | 3 | REJECTED |
| C2_STABLE_CORE_REG | 0.060225 | 0.285153 | 3 | REJECTED |
| E_NEW_INDEPENDENT_FEATURES | 0.056617 | 0.265970 | 3 | REJECTED |
| H2_ENSEMBLE_PAST_IC | 0.051034 | 0.255282 | 2 | REJECTED |
| H1_ENSEMBLE_50_50 | 0.051027 | 0.255645 | 2 | REJECTED |
| H3_ENSEMBLE_REGIME | 0.050946 | 0.254973 | 2 | REJECTED |
| C3_DE_REDUNDANT_REG | 0.049074 | 0.242739 | 2 | REJECTED |
| B1_SHALLOW_REG | 0.048103 | 0.234760 | 2 | REJECTED |
| C4_REMOVE_DEAD_REG | 0.047869 | 0.237052 | 2 | REJECTED |
| B2_STRONG_REG | 0.047307 | 0.226977 | 2 | REJECTED |
| C1_FULL61_REG | 0.047050 | 0.233322 | 2 | REJECTED |
| B3_EARLY_STOP_REG | 0.045379 | 0.217130 | 2 | REJECTED |
| G1_RIDGE_GEN2 | 0.048281 | 0.241256 | 1 | REJECTED |
| I_RESIDUALIZED_SCORE | 0.014981 | 0.147586 | 1 | REJECTED |

## 23. Gen3 Assessment

Did Gen3 materially improve predictive quality over Gen2? **NO**. This label is limited to development/comparative research evidence and is not confirmatory proof.

Answers to the required research questions:

1. Rank IC: directionally yes for Stable-Core Ridge (0.06410), but not with a positive paired CI lower bound.
2. ICIR: directionally yes (0.28700 versus 0.26535), with insufficient joint evidence.
3. Train/OOS gap: lower for Stable-Core Ridge, though its OOS weakness remains material.
4. 2025-like weakness: no; its 2025 IC is negative.
5. Risk-off: descriptively improved, not confirmed.
6. Large-cap: descriptively improved, not confirmed.
7. Technology: descriptively improved, not confirmed.
8. Low-volatility: descriptively improved, not confirmed.
9. Quantile monotonicity: improved for Stable-Core Ridge, while Q5-Q1 magnitude remains small.
10. 20 bps alpha: no Top20 challenger remained positive.
11. Turnover: no meaningful reduction for the leading ranking challenger.
12. Residual alpha: not sufficiently preserved by the leading challenger.
13. Ridge versus LGBM: Ridge remained close, and Stable-Core Ridge led raw IC.
14. Ensemble increment: small and statistically inconclusive; no ensemble passed the joint gates.
15. De-redundancy: no; the pre-registered de-redundant model underperformed baseline.
16. New features: some show standalone information, but the group did not preserve residual/cost evidence strongly enough.
17. Top20: no historical cost-survival support for a production change; production remains untouched.
18. Best K: K=30–50 appears stronger for some reused histories, with explicit data-mining risk and no authorization to select it.
19. Forward-validation candidate: none met the pre-registered `GEN3_RESEARCH_CANDIDATE` gate; Stable-Core Ridge is only an alpha-research lead.
20. Continue Gen3: insufficient evidence for confirmatory promotion; continue constrained alpha research and collect future unseen observations.

## 24. Next Phase

If a `PROMISING_RESEARCH_ONLY` candidate exists, freeze it for `GEN3_CONFIRMATORY_FORWARD_VALIDATION` on genuinely future matured observations. Otherwise continue `ALPHA_RESEARCH`; do not activate or retune against 2025. 10D and 40D remain `NOT EVALUATED` because no same-semantics frozen labels were introduced.

## 25. Git / PR

Branch `codex/gen3-alpha-improvement`. Baseline `e91a7b2cd97b088bb613977867417b3fa3aaa2d1`. Research code, tests and isolated artifacts only; frozen/007–012 modified count must remain zero. PR and CI fields are finalized in delivery metadata after push; the PR must not be auto-merged.
