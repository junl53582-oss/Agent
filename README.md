# StockPilot CN

一个可本地运行的 A 股横截面预测与走步回测系统。它不猜“明天股价是多少”，而是对股票池预测未来 5 个交易日的相对收益，并输出候选、权重、样本外净值和风险指标。

> 仅供研究和软件验证，不构成投资建议。历史回测无法保证未来表现。

## 已实现

- AKShare 前复权日线下载与逐股票缓存
- 无网络的确定性演示行情
- 10 个量价特征及每日横截面排名
- 岭回归、LightGBM、动量、反转和低波动规则的同条件样本外赛马
- 次日开盘到未来开盘的 5 日板块/规模中性超额收益标签
- LightGBM 按交易日分组的 LambdaRank 排序训练
- 训练样本成熟日期约束，防止标签穿越
- 训练期、验证期和最终测试期隔离，测试前写入不可误覆盖的配置锁
- 滚动训练、定期再训练和样本外回测
- Top-N、持仓缓冲、集中度上限、等权/逆波动权重及按换手计费
- 中证指数最新成分股及权重快照
- ST、退市标识和上市时间过滤
- 主板/创业板/科创板/北交所分板块涨跌停规则
- 涨停无法买入时保留现金，跌停无法卖出时顺延退出
- 双边佣金、滑点及卖出印花税
- 累计/年化收益、Sharpe、最大回撤、胜率、Rank IC
- 今日候选、历史信号、模型解释、净值看板
- 数据健康审计：股票池快照时间、幸存者偏差、赢家集中度及异常收益
- 新浪历史流通股本、巨潮申万行业变更的 point-in-time 暴露缓存与覆盖审计
- SHA-256 冻结输入的未来126交易日影子测试协议
- FastAPI 查询接口

## 架构

```text
AKShare / CSV / 演示数据
          ↓
     数据校验与缓存
          ↓
   收盘可知的量价特征
          ↓
未来5日超额收益 多模型 Ranker
          ↓
严格按时间成熟的走步训练
          ↓
成本后回测 + 今日候选
          ↓
 Streamlit / FastAPI
```

系统默认使用稳定、透明的岭回归，并可切换 LightGBM 或单因子规则模型。所有模型必须通过同一历史成分、成交限制、交易成本与走步窗口比较，复杂模型不会被默认视为更优。

## Windows 快速开始

建议使用 Python 3.10～3.12：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[app,ml,dev]"
```

先运行完全离线的端到端演示：

```powershell
stockpilot demo
streamlit run dashboard.py
```

浏览器打开 `http://127.0.0.1:8501`。

API 可单独启动：

```powershell
uvicorn stockpilot.api:app --host 127.0.0.1 --port 8000
```

接口文档位于 `http://127.0.0.1:8000/docs`。

## 使用真实 A 股行情

示例股票池仅用于验证流程。股票数量太少时，横截面模型没有统计意义；正式研究建议使用某个指数的历史成分股，而不是今天的成分股倒推全部历史。

先获取沪深300最新成分股快照：

```powershell
stockpilot universe --index 000300
```

试跑权重前30只股票：

```powershell
stockpilot fetch-index `
  --index 000300 `
  --start 2018-01-01 `
  --end 2026-08-21 `
  --limit 30 `
  --provider auto `
  --output data/market.csv

stockpilot backtest --input data/market.csv --top-n 5
streamlit run dashboard.py
```

去掉 `--limit 30` 即下载完整的300只成分股。下载会逐股票缓存，中断后再次运行可继续复用已有文件。
`--provider auto` 会先尝试东方财富，失败后切换腾讯；如果已知东方财富连接受限，可以直接使用 `--provider tencent`。

也可以指定自选股票：

```powershell
stockpilot fetch `
  --symbols 000001,000333,600036,600519,601318 `
  --start 2018-01-01 `
  --end 2026-08-21
```

## 严格历史成分股回测

下载逐期沪深300历史权重，并压缩成成员真正发生变化的快照：

```powershell
stockpilot history-fetch `
  --index 000300 `
  --start 2022-01-01 `
  --end 2026-08-21 `
  --output data/universes/000300/history.csv
```

下载这段历史中所有成分股的行情并集：

```powershell
stockpilot fetch-history-bars `
  --membership data/universes/000300/history.csv `
  --start 2022-01-01 `
  --end 2026-08-21 `
  --provider tencent `
  --output data/market_history.csv
```

严格按照每个交易日当时可见的成分股回测：

```powershell
stockpilot backtest `
  --input data/market_history.csv `
  --membership data/universes/000300/history.csv `
  --top-n 10
```

在完全相同条件下比较全部模型（建议先用较低重训频率缩短首次运行时间）：

```powershell
stockpilot compare `
  --input data/market_history.csv `
  --membership data/universes/000300/history.csv `
  --top-n 10 `
  --retrain-every 60
```

汇总保存为 `artifacts/comparison.csv`，每个模型的完整净值、信号和年度诊断保存在 `artifacts/comparison/<model>/`。

## 验证体系 v2

先声明候选和时间边界，只用验证期选择配置；选定配置写入锁文件后，才运行一次最终测试：

```powershell
stockpilot validate-v2 `
  --input data/market_history.csv `
  --membership data/universes/000300/history.csv `
  --validation-start 2024-01-01 `
  --test-start 2025-01-01 `
  --test-end 2026-08-13 `
  --retrain-every 60
```

如果报告已存在，命令会拒绝再次打开测试期；只有明确传入 `--force` 才能重跑。即使如此，重跑后的区间也不能再被称为未触碰测试集。

基础行情CSV本身没有稳定的申万行业和自由流通市值字段。标签会优先使用额外暴露文件中的 `industry` 及 `float_market_cap`/`market_cap`；未提供或局部缺失时，明确回退到上市板块和成交额代理，报告会同时显示实际覆盖率。

补齐逐日流通市值和历史行业变更：

```powershell
stockpilot exposure-fetch `
  --membership data/universes/000300/history.csv `
  --start 2022-01-01 `
  --end 2026-08-21 `
  --output data/exposures.csv `
  --workers 1
```

接口按股票缓存，可安全断点续传。新浪历史日线的未复权收盘价乘历史流通股本得到 `float_market_cap`；行业只使用巨潮带变更日期的申万记录并向后生效，禁止未来记录回填。公开接口不保证100%可用，失败证券保存在同名 `.failures.csv`。

带真实历史暴露运行研究回测：

```powershell
stockpilot backtest `
  --input data/market_history.csv `
  --membership data/universes/000300/history.csv `
  --exposures data/exposures.csv `
  --model lightgbm `
  --top-n 20 `
  --hold-buffer 5 `
  --industry-cap 0.3
```

## 未来未触碰测试

现有历史已经被观察，不能重新命名成“未触碰测试”。冻结从下一交易日开始的影子协议：

```powershell
stockpilot future-freeze `
  --input data/market_history.csv `
  --membership data/universes/000300/history.csv `
  --exposures data/exposures.csv `
  --start 2026-08-24 `
  --minimum-days 126

stockpilot future-status --input data/market_history.csv
```

协议记录模型配置、门槛及行情/成分/暴露文件的 SHA-256，冻结后拒绝覆盖。它只允许收集影子信号，`execution_authorized` 永远为 `false`；至少积累126个新交易日后才能评估。

每个交易日收盘后追加行情；严格按冻结参数每5个观察交易日生成一次不可覆盖的影子候选：

```powershell
stockpilot shadow-update `
  --end 2026-08-24 `
  --provider tencent `
  --workers 4

stockpilot shadow-evaluate
```

首次启用完整审计时，在补齐首个信号日的全截面预测后执行一次：

```powershell
stockpilot future-complete-lock
stockpilot future-audit-verify
stockpilot future-adjudicate
```

`future-complete-lock` 不修改原始 `manifest.lock.json`，而是新增不可覆盖的协议补充锁：完整策略参数、Python及关键依赖版本、腾讯前复权数据口径、核心源码摘要和启用时已有文件摘要都会被冻结。之后任何运行环境、核心源码或已登记文件变化都会使更新失败关闭。

也可以运行 `scripts/update_shadow.ps1`，它会依次更新行情并结算已成熟信号；默认日期为本机当天，周一会先调用 `scripts/update_shadow_exposure.ps1` 刷新当期300只流通市值和行业记录。重复运行同一日期是幂等的：已有行情、信号和成熟结果只会被校验，不会被覆盖。新增行情逐日保存到 `data/shadow/bars/`，新增暴露保存到 `data/shadow/exposures/`，候选保存到 `artifacts/future_test/signals/`。

5日持有期信号需要信号日之后第1个交易日开盘进入、第6个交易日开盘退出，因此最早在第7个观察日成熟。结算只读取当时已经固化的信号，按实际可成交权重、手续费、印花税和滑点记账，不会用新数据重新挑选股票。每个信号日还会固化全部可交易股票的事前评分，用于标签成熟后计算真正的横截面 Rank IC。暴露覆盖低于95%、暴露陈旧超过7天、冻结输入摘要变化、哈希链异常或行情日期早于协议起点时，更新会失败关闭。

达到前126个观察交易日后，系统只使用该固定窗口内的信号；等待窗口末期信号全部成熟，再自动检查超额收益、Rank IC、最大回撤、正超额年度比例和暴露覆盖率，生成不可覆盖的 `decision.lock.json`。即使全部通过，状态也只是 `paper_trade_candidate`，不会授权自动实盘。

## Research V3

V3是与冻结影子协议完全分离的研究线，增加带真实公告日的PIT基本面、5/10/20日多周期标签、Ridge与LightGBM排序集成、模型一致性过滤和逐年嵌套走步选择：

```powershell
python -m research_v3.cli fundamentals-fetch --workers 4
python -m research_v3.cli run
```

当前已取得421只历史成分股、31,401条基本面记录，公告日违规为0。首次嵌套验证没有通过：策略收益16.06%、基准44.07%、超额-28.01%、平均Rank IC -0.0102、最大回撤-22.22%。因此V3只保留为研究产物，不替换现有冻结影子模型，也不能据此进入实盘。

## Research V4

V4是独立的预注册稳定因子研究线。运行前已经把输入文件摘要、质量/成长/低波/趋势四个固定因子、训练期稳定性筛选阈值、交易成本和四项通过门槛锁定在 `artifacts/research_v4/plan.lock.json`。每个测试年份的因子方向和权重只能使用该年以前已经成熟的标签学习；没有因子达标时必须持有现金。

```powershell
python -m research_v4.cli run
```

首次锁定测试没有通过：策略收益8.55%、基准44.04%、超额-35.49%、平均Rank IC 0.0262、最大回撤-30.29%，正超额年份比例1/3。平均Rank IC门槛通过，但收益、回撤和年度稳定性门槛未通过。结果保持为 `retrospective_research`，不会替换冻结模型，也不会授权交易。若修改规则或输入数据，锁校验会直接拒绝运行。

V4是保留的稳定因子基线模型。它使用基准历史行情与每日影子行情的合集生成全截面排名和Top 20候选；同一测试年的因子方向及权重只使用上一年末以前成熟的标签确定：

```powershell
python -m research_v4.cli predict
```

股票名称使用完整A股代码简称缓存；可手动刷新：

```powershell
python -m research_v4.cli names-fetch
```

每日脚本会在原冻结影子协议完成更新、结算和裁决后追加运行V4预测。V4快照保存在 `artifacts/research_v4/live/`，同一日期不允许覆盖；原冻结协议和审计链继续独立保留。V4仍是未通过全部绩效门槛的研究模型，不能据此自动下单。

## Research V5

V5是预注册的多维行业专家研究线，同时使用PIT基本面质量与成长、量价行为、风险、流动性、市场状态、全市场Ridge和大类行业专家。电子、计算机、通信、传媒会进入科技专家内部比较，而不是与银行直接使用同一组系数。规则和输入摘要锁定在 `artifacts/research_v5/plan.lock.json`。

```powershell
python -m research_v5.cli
```

首次锁定测试没有通过全部门槛：策略收益25.38%、基准44.04%、超额-18.66%、平均Rank IC 0.0215、最大回撤-20.81%，2/3测试年份取得正超额，5/6大类行业IC非负。相较V4收益和年度稳定性明显改善，但超额收益与回撤门槛仍失败；科技行业平均IC为-0.0125，说明现有可靠数据仍不足以形成有效科技专家。V5保留为 `continue_research`，不替换默认V4，也不授权交易。

当前没有伪造加入缺失数据：分析师预期修正、公告/新闻情绪、复权安全的历史估值和带历史版本时间戳的宏观数据仍列为V5数据缺口。补齐后必须另开锁定协议测试，不能回改本次V5结果。

## Research V6

V6将V5多维模型、V4训练期稳定因子和行业内排名按65%/20%/15%锁定集成，并按每日可交易股票的行业占比构建30只行业均衡组合。锁定测试结果：策略41.42%、基准44.04%、超额-2.62%、平均Rank IC 0.0253、最大回撤-18.92%，2/3年份正超额，5/6大类行业IC非负。它通过了全部相对V4替换门槛，但累计超额仍未大于0，因此只替换默认研究预测，不获得实盘授权。

```powershell
python -m research_v6.cli run
python -m research_v6.cli predict
```

每日脚本现运行V6最新预测，快照位于 `artifacts/research_v6/live/`。V4、V5和原冻结模型继续保留，不覆盖历史产物。

## Research V11

V11 在观察到 V10 失败后作为独立新版本冻结，研究了 Top30 尾部 LambdaRank、双年度收益门控、科技独立门控，以及仅使用当日 60 日动量与 20 日上涨宽度的 55%/80%/100% 防御仓位。最终冻结锁为 `C70B9C779673BDE0FB7A580D425D80729B6834949998E14C392D78CAA650A714`。

2020–2025 冻结验证结果：最终防御组合收益 14.53%、基准 31.59%、超额 -17.06%、2/6 年正超额、5 日/20 日 Rank IC 分别为 -0.0215/-0.0327、最大回撤 -29.03%。无门控模型的 Top30 精度提高到 13.84%，但入选超额仍接近零，说明尾部命中率没有转化为可交易收益。V11 未通过替换门槛，不进入影子测试；网页和每日预测继续使用 V6。完整裁决见 `artifacts/research_v11/report.json`，研究记录见 `artifacts/research_v11/progress.md`。

## Research V12

V12 将目标改为行业中性约束后的估计净边际收益，并加入 28 日禁运期、行业内 LambdaRank 和连续波动率风险预算。最终冻结锁为 `EED8368DEE95E8449537A6DBB9C96C8F05747B0B5FC3E65A3E29FC8D7CFA77E9`。

无门控组合感知模型在 2020–2025 获得 +2.39% 累计超额、14.61% Top30 精度和正的入选净边际收益，是目标对齐方向的有效进展；但仅 3/6 年正超额、全截面与科技 IC 仍为负、最大回撤 -34.50%。预注册的双年度门控没有任何测试年放行，正式风险预算线累计超额 -30.38%、最大回撤 -39.00%。因此 V12 仍判定 `keep_v6`，不得事后采用无门控消融线替换生产模型。完整报告见 `artifacts/research_v12/report.json`，研究记录见 `artifacts/research_v12/progress.md`。

## Research V13

V13 将组合感知预测拆成行业内 Top20% 概率和条件收益幅度两个模型，并使用 90% 置信下界门控与带恢复滞后的回撤状态机。冻结锁为 `AB8EA5805BE5D0693FE2D22B8F2280DBB04390CE100707F574463EC6629A5077`。

无门控两阶段模型把 Top30 精度提高到 17.63%，2020–2025 累计超额 +2.84%，但只有 3/6 年正超额，入选净边际收益接近零，全截面与科技 IC 均为负。所有测试年度的全局置信下界均为负，因此正式线没有放行主动选股；回撤状态机累计超额 -20.33%、最大回撤 -34.14%。V13 裁决仍为 `keep_v6`。完整报告见 `artifacts/research_v13/report.json`，研究记录见 `artifacts/research_v13/progress.md`。

## Research V14

V14 接入了完整分页的 PIT 公告标题历史，并在任何模型表现之前冻结数据源覆盖门槛、次交易日生效规则、16 个公告事件特征、两阶段 LightGBM、同数据 V13 增量置信门和基准相对组合约束。最终公告数据为 966,865 条、791 只股票；分析师和北向数据因覆盖不足被排除。冻结锁为 `BFA3D0B8C5D3B5345AAE2A26269588FDD0455AB1D6AFAFF036FE1EE87ECCB5C3`，83 项全项目测试通过。

无门控公告增强线在 2020–2025 年仅获得 +0.17% 累计超额、2/6 年正超额、17.40% Top30 精度，低于同数据 V13 对照的 +2.84% 超额；入选净边际收益、全截面 IC 和科技 IC 均为负。所有年度的绝对和增量置信下界均为负，正式线没有放行主动选股，累计超额 -0.52%、最大回撤 -32.42%。因此 V14 裁决为 `keep_v6`，不进入未来影子测试，也不替换网页模型。完整报告见 `artifacts/research_v14/report.json`，研究记录见 `artifacts/research_v14/progress.md`。

## Research V15

V15 是独立的公告标题文本研究：字符片段哈希表示、1/5/20 日事件目标，以及 75% 同数据基线与 25% 文本得分的固定组合。它不是公告全文大模型。原始失败验收保留，首次收益测试前通过 `artifacts/research_v15/amendment_001.json` 明确完整率分母为全部当期符合条件事件（不删除缺失目标），95% 门槛不变；修订验收为 99.2149%，全项目 100 项测试通过。

冻结锁为 `52FFCA9892F7C4FDBDD3D30F3E2ACFF2BA460A2B3A3821805C2C51623EEEE824`。2020–2025正式走步回测已完成：未门控文本增强线累计收益36.77%、共同基准31.59%、累计超额+5.18个百分点、4/6年正超额；本次同数据基线超额为+0.72个百分点。但文本线5日/20日IC为-0.03275/-0.02460，科技IC仍为负，调仓点最大回撤-33.47%，未满足替换要求。所有年度正式门控均未放行主动选股，正式线超额-0.52个百分点、回撤-32.42%；裁决为`keep_v6`，不进入V15未来影子测试。原始事件年份、标签时间边界、四线报表和V14/V15冻结完整性复核通过。完整结果见 `artifacts/research_v15/report.json`，研究记录见 `artifacts/research_v15/progress.md`。以上为历史研究结果，非逐日实盘收益。

```powershell
python -m research_v15.cli verify
```

冻结后的 `run` 不接受自定义参数或数据文件，也拒绝覆盖已有启动记录和正式报告。不要重复启动已在运行的任务，不要为重跑而删除锁或启动记录。

## Research V16

V16 在 V15 的基础上做文本表示升级：在逐字节保留 V15 的 char 2-4gram 头之外，新增一个 jieba 词级 1-2gram 头，双头 0.5/0.5 集成后按同一套 1/5/20 日目标加权；最终评分从 75%/25% 调整为 65% 同数据 V13 可比分加 35% 集成文本分。冻结前已锁定归因锚（`v15_char_replica`，必须复现 V15 未门控线）和 18 项门槛（V15 的 17 项 + 集成线累计超额不得低于复刻锚）。冻结锁为 `089AC2541AE4C801FF16F48FAE1A51BB05092320ABFC5180ACFC093C77A4F935`，运行环境为 Python 3.11.9 及锁定依赖版本。

2020–2025 走步回测完成：`v15_char_replica` 精确复现 V15（累计超额 +5.18pp、IC5/IC20 -0.0328/-0.0246、最大回撤 -33.47%），归因可靠。`v16_text_ungated` 累计超额 +5.92pp、4/6 正年、IC5/IC20 收窄至 -0.0251/-0.0168，词头净贡献 +0.74pp（4/6 年为正，2023/2024 小幅恶化）。但全截面 IC 仍为负，嵌套验证置信下界所有年度仍为负，正式门控线未放行主动选股并回落基准；18 项门槛通过 9 项，裁决 `keep_v6`。结论：词级 n-gram 表示有微弱但真实的边际价值，量级远不足以替换生产模型；下一处杠杆仍是公告正文语料或预训练语义嵌入。完整结果见 `artifacts/research_v16/report.json`，研究记录见 `artifacts/research_v16/progress.md`。

```powershell
python -m research_v16.cli verify
```

## Research V7

V7在冻结规则后测试了5/20/60日多周期行业专家、预测分歧惩罚和持仓缓冲。锁定测试结果：策略34.13%、基准44.04%、超额-9.91%、平均Rank IC 0.0152、最大回撤-17.00%，2/3年份正超额，5/6大类行业IC非负，平均单边换手39.14%。它改善了回撤和换手，但收益与IC均低于V6，没有通过相对V6替换门槛。因此默认网页和每日预测继续使用V6，V7只保留为失败但可复现的研究结果，不授权交易。

```powershell
python -m research_v7.cli
```

冻结方案位于 `artifacts/research_v7/plan.lock.json`，完整结果位于 `artifacts/research_v7/report.json`。公告事件、分析师预期、历史估值等外部数据尚未混入本次回测；其接入要求和防泄漏规则记录在 `artifacts/research_v7/data_extension.protocol.json`。

## Research V8

V8在测试前锁定为“V6基线 + PIT估值与基本面增强 + 科技专属模型 + 市场状态倾斜 + 持仓缓冲”。PIT财务记录只在 `available_date` 当日及之后可见，研发投入、公告文本和分析师预期因缺少完整历史版本快照而没有使用。

```powershell
python -m research_v8.cli
```

锁定测试结果：策略45.13%、基准44.04%、超额+1.09%、平均Rank IC 0.0248、最大回撤-18.02%，2/3年份正超额，5/6大类行业IC非负，平均单边换手41.98%。五项严格研究门槛首次全部通过，但平均IC略低于预先锁定的V6替换门槛0.0253，因此裁决仍为 `keep_v6`，网页和每日预测不切换。该结论不允许回测后修改参数，也不授权交易。

消融实验显示，仅给V6增加持仓缓冲会把换手降至44.50%，但累计收益降至35.13%；V8的改进主要来自PIT增强和状态路由，而不是单独降低换手。冻结方案、数据来源清单和完整结果分别位于 `artifacts/research_v8/plan.lock.json`、`artifacts/research_v8/data_manifest.json` 和 `artifacts/research_v8/report.json`。

成分数据来源为 `chenditc/investment_data` 公开数据库的 `ts_index_weight` 表。系统不会把第一份可用快照向过去回填；快照覆盖之前的行情会被明确排除。也可导出Qlib区间文件：

```powershell
stockpilot membership-export-qlib `
  --membership data/universes/000300/history.csv `
  --output data/universes/000300/csi300.txt
```

CSV 也可以由其他数据源提供，必需字段为：

```text
date,symbol,open,high,low,close,volume,amount
```

可选 `name` 字段会显示在候选列表中。

## 产物

运行后在 `artifacts/` 生成：

- `summary.json`：回测指标、参数和最新模型系数
- `equity.csv`：模型与股票池基准净值
- `signals.csv`：历史 Top-N 信号及事后收益
- `predictions.csv`：完整样本外预测，用于进一步诊断
- `latest_signals.csv`：最新可用交易日候选
- `yearly.csv`：逐年收益、基准收益及 Rank IC
- `comparison.csv`：多模型同条件比较（运行 `compare` 后生成）
- `validation_v2/plan.lock.json`：测试前冻结的候选、区间和选择规则
- `validation_v2/selected_config.lock.json`：仅根据验证期选出的配置
- `validation_v2/report.json`：最终测试结论和是否允许进入模拟盘
- `future_test/manifest.lock.json`：未来测试起点、门槛、锁定配置和输入摘要
- `future_test/signals/YYYY-MM-DD.csv`：按冻结再平衡频率生成、不可覆盖的影子候选
- `future_test/predictions/YYYY-MM-DD.csv`：信号日全部可交易股票的不可覆盖事前评分
- `future_test/outcomes/YYYY-MM-DD.json`：信号成熟后固化且不可覆盖的逐期成交结果
- `future_test/ledger.csv`：由固化信号和结果完整重建的净值账本
- `future_test/evaluation.json`：成熟周期、待成熟信号和累计影子收益摘要
- `future_test/protocol.addendum.lock.json`：完整配置、运行环境、源码及启用文件摘要
- `future_test/audit_chain.jsonl`：影子行情、暴露、信号、预测和结果的追加哈希链
- `future_test/adjudication_status.json`：126日收集及末期标签成熟状态
- `future_test/decision.lock.json`：所有门槛可计算后生成一次的最终裁决锁

原始和演示数据位于 `data/`，不会提交到 Git。

## 方法边界

- 演示数据只是软件测试，不用于证明策略有效。
- 免费数据接口可能中断或发生字段变化，生产使用必须保留本地快照。
- 当前流动性过滤按当日成交额横截面底部 20% 排除，名称过滤可识别 ST/退市标识，但无法还原名称的全部历史变更。
- `fetch-index` 使用最新指数成分股回溯历史，仍有幸存者偏差；可信回测应使用 `history-fetch` 和 `--membership`。
- 回测使用次日开盘成交并处理常规涨跌停，但尚未覆盖 IPO 前几日无涨跌幅限制、连续多日跌停及 100 股整数手。
- 今日候选仍需人工检查停复牌、涨跌停和最新公告。
- 当前历史最终测试未通过，系统状态为 `research_only`，不应连接自动下单。
- 新浪历史接口会限制频繁访问，暴露下载默认单线程并按股票缓存；不要盲目提高并发。

## 测试

```powershell
python -m unittest discover -s tests -v
```
