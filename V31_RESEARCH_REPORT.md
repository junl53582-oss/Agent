# V31 Research Challenger / Alpha Improvement Phase 1

## 结论

V31 已按冻结协议完成一次严格历史 Purged Walk-Forward。预注册挑战者 `lightgbm_lambdarank` 未击败 V6，最终裁决为：

```text
V31_REJECTED
ranking_champion = V6
v31_status = RESEARCH_ONLY
promotion_to_candidate = false
production_prediction_ready = false
execution_authorized = false
```

V31 没有写入 prospective observation、prediction 或 mature-label ledger，也没有改变 V6、V30、V30r1-forward-r2 或 V1r4。

## Architecture

- Canonical research entrypoint：`python -m stockpilot.research_challenger.cli ...`
- 新的版本复制目录：0；新增的是非版本化 canonical package `stockpilot/research_challenger/`。
- 复用：V10 PIT 因子、V6 scorer、V4/V5 train-only 训练、A 股交易成本、turnover、V30 open-to-open 标签语义。
- 未复制 V6/V30，也未创建 `research_v31_r1/r2`。
- 初次冻结运行在任何指标写入前因非有限训练目标进入 LambdaRank 而失败。原锁 `803f31a1...` 与失败证据保留；只修复模型边界的非有限目标过滤，目标、因子、模型参数和门槛未变。有效修订锁：`7224e1a18a3dabc0cd4a5c80d78ef8b6678e3099a25dbfd070bc509056615449`。

## Dataset / PIT

- 原始缓存：947,079 行、747 只股票，2010-07-02 至 2026-08-21。
- 数据 SHA-256：`8aa5e3f2817d6bf8e5da3bd265b4f078206b58b10ee770907233595c59342b02`。
- OOS：2020–2025，共 362,520 股票日/周期；final OOS 标记为 2024–2025。
- Train：滚动八年；Validation：紧邻 OOS 前一年；训练/验证标签必须分别在 validation/OOS 开始前成熟。
- Purge gap：1D=2、5D=6、20D=21 个交易日。
- PIT membership、财务 available date、行业 effective date 全部通过；prospective rows used=0。
- 官方 benchmark open 证据尚未批准，因此 benchmark-relative 训练目标禁用。Top-K 只用 V10 已有 PIT 成分权重代理作公平研究比较器，不冒充官方指数。

## Targets

- 1D exploratory：T+1 open → T+2 open 横截面收益排名。
- 5D primary：T+1 open → T+6 open 横截面收益排名。
- 20D secondary：T+1 open → T+21 open 横截面收益排名。
- 另生成 5D/20D PIT 行业内中性收益及排名作诊断。
- Benchmark alpha target：`BENCHMARK_TARGET_DISABLED`。

## Factors

- 筛选前 61：42 fundamental、5 industry/technology、3 liquidity、7 price behavior、4 risk。
- 每折仅使用训练期的日度 RankIC、HAC、BH-FDR、年度方向一致性和相关性去冗余；每年入选 15–20 个。
- 所有年份稳定入选：利润增长变化、扣非利润变化、收入增长变化、毛利变化、毛利同比变化、流动性、行业动量、日内强弱。
- 真实 OOS 5D 较强单因子：60日低波动 RankIC 0.04420、low-volatility 0.03879、下行波动 0.03438、规模权重 0.01735。
- 从未入选的例子：价格位置、科技动量、fundamental coverage、120日动量、科技质量/估值/增长、库存周转变化、volume attention。
- 高相关冗余：`downside_volatility_60_rank` 在一折被相关性门禁剔除。

## Models / OOS RankIC

| Model | 1D RankIC | 5D RankIC | 20D RankIC | 5D ICIR | 5D Positive Ratio |
|---|---:|---:|---:|---:|---:|
| V6 | 0.00549 | -0.00469 | -0.02172 | -0.02725 | 48.93% |
| Ridge | 0.02742 | 0.03181 | 0.04218 | 0.15980 | 55.46% |
| LightGBM Regression | 0.02913 | 0.03323 | 0.04373 | 0.17781 | 57.04% |
| LightGBM LambdaRank | -0.02606 | -0.02370 | -0.02296 | -0.10928 | 46.67% |

XGBoost/CatBoost 未安装，按预注册状态记为 `DISABLED_DEPENDENCY_NOT_PRESENT`，没有为凑模型扩大依赖。

LambdaRank 5D 年度 RankIC：2020 +0.03930、2021 -0.02680、2022 -0.02941、2023 -0.09681、2024 -0.05885、2025 +0.02990。相对 V6 仅 1/6 年更好。

## Top-K / Cost / Risk（5D）

| Model | K | Gross total | Net total | Net alpha vs research proxy | Turnover annualized | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| V6 | 20 | 16.08% | -17.61% | -54.19% | 27.95x | -53.06% |
| Ridge | 20 | 27.12% | -4.63% | -41.21% | 23.46x | -33.68% |
| LGB Regression | 20 | 35.43% | 3.78% | -32.80% | 21.73x | -35.82% |
| LambdaRank | 20 | 62.03% | 32.06% | -4.52% | 16.71x | -62.43% |

LambdaRank 的头部点估计较好，但全体排序为负、五分位不单调、回撤明显恶化，不能把“擅长少量尾部”包装成全面 ranking 提升。

## Stability / Statistical Evidence

- LambdaRank 5D 在 bull/bear/sideways 的 RankIC 分别为 -0.04538/-0.01108/-0.02099；高/低波动均为负。
- 六个宽行业中只有 other 为正；technology 为 -0.01205。
- 5D RankIC 差值（LambdaRank - V6）moving-block bootstrap：均值 -0.01901，95% CI `[-0.05055, 0.00623]`。
- Top20 单期净 Alpha 差值：均值 +0.00204，95% CI `[-0.00164, 0.00545]`，跨 0。
- Promotion 十项门禁只有 Top20 点估计一项通过，其余九项失败。

## Decision

`V31_REJECTED`。Ridge 和 LightGBM Regression 的 RankIC 值值得形成“未来新预注册假设”，但它们不是本次固定 promotion challenger；看到结果后改挑赢家会构成多重检验/选择偏差，因此本轮不重跑、不晋级。

## Tests / Integrity

- V31 targeted：25 passed。
- V31 + V1r4 + V6 + forward-r2 integration：63 passed。
- Full repository：478 passed、1 xfailed、24 subtests passed。
- V18 strict xfail 保持原分类。
- V1r4、V6、V18、V20r2、V30r1-forward-r2 父锁全部完整。
- Artifact manifest SHA-256：`af6d7bb9a71faab9fdca603276136d9df321f6edeeb2c31b37ff0b297983f103`。
- 本轮网络请求：0；正式模型重训：0；prospective 写入：0。

## Remaining Risks / Next Step

- 历史 2020–2025 已被仓库过去研究多次观察，本结论是严格时间 OOS，但不是全项目层面的 pristine untouched evidence。
- PIT 成分权重代理不是已批准官方 benchmark open 序列。
- LambdaRank 头部收益与全体 RankIC 分裂，可能反映尾部识别，也可能是样本不稳；当前 Bootstrap 不支持晋级。
- 下一步不再利用本次 OOS 调 LambdaRank。应继续积累 V1r4 不可变真实 observation/prediction/settlement；若未来建立新 challenger，只能先预注册“Ridge/LGB regression 简化候选”或引入真正新增 PIT 信息，再开启新一代实验。
