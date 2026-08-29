# V30 Probabilistic Prediction Layer 完成报告

## 1. 结论

V30 概率预测层及独立缺陷修订版 V30r1 已完成真实数据验证、最新预测、不可变快照、CLI、API、网页接入和证据冻结。V30r1 修复了负斜率概率校准造成的排序反转，并加入只依赖过往 OOS 的模型冠军选择；修复生效，但剩余预测能力门禁未通过。

- `production_prediction_ready = false`
- `future_126d_confirmed = false`
- `future_confirmation_status = COLLECTING`
- `execution_authorized = false`
- 排名正式默认：V6
- 概率研究候选：V30r1

126 个未来交易日没有参与立即预测认证。本次未就绪的原因是历史严格 OOS 的校准、基线和稳定性证据不足，不是等待天数不足。

## 2. 修改与新增文件

共享文件修改：

- `stockpilot/cli.py`
- `stockpilot/api.py`
- `dashboard.py`
- `pyproject.toml`
- `README.md`
- `RESEARCH_AUTOPILOT.md`
- `RESEARCH_DECISIONS.md`
- `artifacts/active_research.json`

V30 新增模块：

- `stockpilot/prediction/labels.py`
- `stockpilot/prediction/split.py`
- `stockpilot/prediction/models.py`
- `stockpilot/prediction/calibration.py`
- `stockpilot/prediction/metrics.py`
- `stockpilot/prediction/certification.py`
- `stockpilot/prediction/drift.py`
- `stockpilot/prediction/confidence.py`
- `stockpilot/prediction/data.py`
- `stockpilot/prediction/pipeline.py`
- `stockpilot/prediction/inference.py`
- `stockpilot/prediction/storage.py`
- `stockpilot/prediction/settlement.py`
- `stockpilot/prediction/schema.py`
- `stockpilot/prediction/config.py`
- `stockpilot/prediction/freeze.py`
- `stockpilot/prediction/__init__.py`

V30r1 独立修订：`stockpilot/prediction_v30r1/`；共享结果锁工具：`stockpilot/prediction_audit.py`；测试：`tests/test_prediction_v30.py`、`tests/test_prediction_v30r1.py`。

## 3. 预测目标与标签语义

预测目标为 1D、5D、20D 上涨概率以及 5D、20D 预期收益。若 T 日收盘后生成信号：T+1 open 入场，T+2/T+6/T+21 open 分别作为 1D/5D/20D 出场价。方向标签同时保留 `raw_up` 和超过估算往返成本的 `tradable_up`，不会把 T 日收盘到未来收盘冒充可执行收益。

## 4. 数据与时间切分

- 数据：947,079 行，747 只证券
- 范围：2010-07-02 至 2026-08-21
- 特征：61 个既有 V10 PIT 特征
- 正式 OOS：2019–2025
- 支持冠军选择的先验 OOS：2018
- 每折训练上限：400,000 行
- Purge gap：1D=2、5D=6、20D=21 个交易日
- OOS 样本：1D=415,810；5D=413,252；20D=408,396

每折均验证 `train label_end_date < validation_start`，没有随机切分、shuffle 或普通 KFold。

## 5. 模型与 Baseline

方向头：LightGBM binary 与 Logistic/Ridge baseline。收益头：LightGBM regression 与 Ridge baseline。V30r1 只在先前年份 OOS 证据同时改善 Brier、LogLoss 且不降低 AUC 时保留方向 LightGBM；收益头需同时改善 MAE、RMSE 且不降低 Rank IC，否则使用 Ridge。校准使用非负单调 Platt，仅由历史 OOF/rolling validation 概率拟合。

Baseline 包括 unconditional probability、rolling market probability、momentum、Logistic/Ridge。原始概率、校准概率和横截面概率排名分别保存，不把 rank 冒充 probability。

## 6. V30r1 OOS 指标

以下为 2019–2025 逐年指标平均值：

| Horizon | Samples | ROC-AUC | PR-AUC | Brier | LogLoss | Rank IC |
|---|---:|---:|---:|---:|---:|---:|
| 1D | 415,810 | 0.52954 | 0.44795 | 0.24372 | 0.68050 | -0.00314 |
| 5D | 413,252 | 0.51055 | 0.46788 | 0.24992 | 0.69303 | 0.01835 |
| 20D | 408,396 | 0.50366 | 0.47180 | 0.25688 | 0.70745 | 0.02456 |

收益头汇总：5D selected MAE/RMSE/Rank IC 为 0.03640/0.05396/0.02885；20D 为 0.07537/0.10894/0.01870。方向模型没有稳定击败全部朴素基线，因此 `baseline_beaten=false`。

## 7. Calibration

- 5D ECE：0.03596
- 20D ECE：0.06694

| Horizon | Predicted bucket | Samples | Mean predicted | Actual up rate |
|---|---|---:|---:|---:|
| 5D | 0.00–0.45 | 217,759 | 0.4254 | 0.4712 |
| 5D | 0.45–0.50 | 187,532 | 0.4733 | 0.4498 |
| 5D | 0.50–0.55 | 7,961 | 0.5060 | 0.4452 |
| 20D | 0.00–0.45 | 197,530 | 0.4003 | 0.5091 |
| 20D | 0.45–0.50 | 172,099 | 0.4717 | 0.4541 |
| 20D | 0.50–0.55 | 38,731 | 0.5067 | 0.4343 |
| 20D | 0.55–0.60 | 36 | 0.5533 | 0.9444 |

20D 最高小桶只有 36 个样本，不能据此宣称高胜率。总体 `calibration_passed=false`。

## 8. Regime、行业与漂移

按 bull/bear/sideways/high-volatility/low-volatility 分组后没有任何 regime 通过完整门禁；5D/20D 分组平均 AUC 多数低于 0.5。七个 broad sector 中只有一个通过固定稳定性条件。详细结果位于 `regime_metrics.csv` 和 `sector_metrics.csv`。

最新特征漂移为 `SEVERE`，主要由 `fundamental_freshness_rank` PSI 2.6213 驱动。系统没有删除该特征或放宽阈值来美化结果，而是将全部最新预测置信度降为 LOW。

## 9. Certification

通过：数据完整性、PIT、标签成熟、泄漏审计、Purged Walk-Forward。

未通过：总体概率校准、全面基线比较、跨年稳定性、regime 稳定性、概率质量、成本压力测试。

`production_prediction_ready` 的计算不包含 `future_126d_confirmed`；两者与 `execution_authorized` 三状态严格分离。

## 10. 最新真实预测 Top 20

预测日期 2026-08-21；以下均为未认证研究输出，置信度 LOW：

| Rank | Symbol | Name | P(up 1D) | P(up 5D) | P(up 20D) | ER 5D | ER 20D |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 300308 | 中际旭创 | 44.57% | 45.63% | 47.55% | 0.96% | 2.53% |
| 2 | 300408 | 三环集团 | 46.18% | 45.63% | 47.82% | 1.05% | 2.98% |
| 3 | 002384 | 东山精密 | 44.49% | 45.51% | 47.63% | 0.66% | 2.24% |
| 4 | 002463 | 沪电股份 | 43.87% | 45.47% | 47.95% | 0.74% | 3.51% |
| 5 | 601899 | 紫金矿业 | 42.36% | 45.44% | 47.86% | 0.65% | 2.50% |
| 6 | 300433 | 蓝思科技 | 44.37% | 45.44% | 47.84% | 0.58% | 2.51% |
| 7 | 605117 | 德业股份 | 43.52% | 45.43% | 47.50% | 0.44% | 1.60% |
| 8 | 600803 | 新奥股份 | 42.65% | 45.43% | 47.77% | 0.53% | 1.93% |
| 9 | 002466 | 天齐锂业 | 43.42% | 45.42% | 47.24% | 0.57% | 1.92% |
| 10 | 002916 | 深南电路 | 44.61% | 45.42% | 47.71% | 0.70% | 2.90% |
| 11 | 300661 | 圣邦股份 | 45.39% | 45.42% | 47.63% | 0.95% | 3.36% |
| 12 | 000895 | 双汇发展 | 40.58% | 45.38% | 47.82% | 0.44% | 1.62% |
| 13 | 600018 | 上港集团 | 41.04% | 45.38% | 47.91% | 0.39% | 1.56% |
| 14 | 688008 | 澜起科技 | 44.19% | 45.36% | 47.49% | 0.60% | 2.15% |
| 15 | 600026 | 中远海能 | 43.28% | 45.35% | 47.53% | 0.49% | 1.95% |
| 16 | 688472 | 阿特斯 | 41.85% | 45.35% | 47.34% | 0.41% | 1.19% |
| 17 | 600176 | 中国巨石 | 45.57% | 45.34% | 47.05% | 0.28% | 0.52% |
| 18 | 002475 | 立讯精密 | 43.05% | 45.34% | 47.33% | 0.13% | 0.51% |
| 19 | 600522 | 中天科技 | 44.39% | 45.32% | 47.31% | 0.42% | 1.49% |
| 20 | 600900 | 长江电力 | 39.76% | 45.31% | 47.82% | 0.17% | 0.80% |

快照 SHA-256：`ef6111cab901450dd2a144bed2ab36d9b146cef1def539f8c2495729f523f440`。重复运行返回 `already_recorded=true`。

## 11. 测试结果

- V30/V30r1：23 passed
- Streamlit AppTest：0 exceptions
- 全仓：299 passed，24 subtests passed，1 failed

唯一失败是冻结 V18 的历史测试夹具向模型传入空回归器列表，随后调用 `np.column_stack([])`。它与 V30 无关，也不是本轮新回归；按照冻结规则保留原失败证据，不修改 V18。

## 12. Remaining Risks

- 5D/20D 概率尚未稳定击败朴素和线性基线，不能解释为可靠胜率。
- 20D 校准误差和跨 regime/sector 稳定性不足。
- 最新 PIT 特征发生严重分布漂移，所有置信度均为 LOW。
- 最新快照尚无未来 T+2/T+6/T+21 开盘，因此结算账本 747 行、成熟 0 行。
- 2019–2025 已被用于历史研究，后续不得在这些结果上反复调参数后继续称独立 OOS。
- 真正的下一步应是积累不可变 forward 结算，并引入具备来源、首次见证时间和修订链的盈利预期、资金流、公告事件或行业景气 PIT 信息。
