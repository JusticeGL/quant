# A-Share Alpha Lab

这是 A 股自动因子研究平台的 Phase 5 研究级数据环境。项目在 Linux Python 3.11 容器中
完成 Phase 1 的不可变数据闭环，并使用 Qlib Alpha158、确定性 LightGBM、validation
信号分析和 Top-K 组合回测生成 Phase 2 基线；Phase 3 增加严格因子契约、注册表、固定
评价程序和防泄漏门禁；Phase 4 在其上增加单点假设、候选生成、固定评价、结构化决策和
可恢复多轮日志。规格书列出的 Baostock 只在 AKShare 失败时作为显式备用源。

Phase 5 增加 2020 年起沪深 300 动态成分、退市、历史名称/ST、停牌和独立复权因子。
当前仍不包含锁定测试集评价或交易账户连接。Phase 4 的 `ACCEPT` 只表示进入人工复核，
不会自动修改已接受因子注册表。

## 研究边界

- 固定工程样本为 10 只流动性较高的沪深 A 股，时间为 `2024-01-01` 至
  `2024-06-30`。
- 该名单是当前风格的人工工程样本，存在明确的生存者偏差，`research_eligible` 为
  `false`。它只能验证数据工程，不能支持收益、因子或历史股票池结论。
- 默认使用 AKShare `stock_zh_a_hist` 的不复权日线。Phase 1 不推断复权因子、涨跌停、
  上市或退市状态；AKShare 行的停牌与 ST 状态也保持缺失。备用源实际提供的停牌和 ST
  状态会保留，并由逐列缺失率反映覆盖差异。
- 不下载全量 A 股数据，不读取券商凭据，也不连接任何交易账户。
- Phase 2 回测仅在 validation 上验证工程链路；不访问或报告锁定 test。

## 正式运行环境

- 所有正式 Python 命令均通过 `make` 或 `docker compose` 在 Linux 容器中执行。
- 容器固定 Python 3.11，依赖由容器内 `uv` 解析并锁定在 `uv.lock`。
- Apple Silicon 默认使用原生 ARM64；Compose 没有 `linux/amd64` 覆盖。
- `.env`、`data/`、Parquet、Qlib 二进制、缓存、模型和实验大文件均不会进入 Git 或
  Docker build context。

首次运行：

```bash
make lock
make build
make smoke
make lint
make test
```

`pyqlib==0.9.7` 没有可用的 Linux ARM64 PyPI 分发，因此项目锁定 Microsoft/Qlib
官方 `v0.9.7` commit `da920b7f954f48ab1bb64117c976710de198373e` 源码。
这仍是原生 ARM64 构建，不是 amd64 模拟。

## Phase 1 命令

Docker Desktop 必须处于运行状态。

```bash
# 下载固定范围；已有的完整 raw 请求区间直接命中缓存
make data-bootstrap

# 重复执行应显示 network_requests: 0
make data-bootstrap

# 可选：仅抓取当前缓存尚未覆盖的尾段
make data-update END_DATE=2024-07-31

# 对最新快照重新计算质量检查
make data-validate

# 从同一个 silver 快照生成 Qlib 文件存储
make qlib-export

# 初始化 DuckDB 目录并同步当前 manifest
make db-init

# 只读验证 schema、逻辑外键和 artifact 文件
make db-check
```

## Phase 2 基线

```bash
# 从最新 Qlib 快照运行 Alpha158 + LightGBM + validation Top-K
make baseline

# 或显式固定快照
make baseline SNAPSHOT=p1-<snapshot-id>
```

配置分别位于 `config/baseline.yaml`、`config/splits.yaml` 和
`config/costs.yaml`。切分和成本配置带稳定 SHA256 并作为锁定资产；当前短样本使用
`engineering_only` 切分。训练标签会按交易日清除跨越 train 边界的观测，validation
标签只允许使用 test 前的隔离日期，锁定 test 不被载入、打分、评价或写入报告。

每次运行的确定性身份由数据快照、Qlib 内容哈希、三份配置、固定种子和 Git commit
组成。产物写入被 Git 忽略的 `artifacts/baseline/<run-id>/`：

```text
run_manifest.json
predictions.parquet
lightgbm_model.txt
backtest_daily.parquet
trades.parquet
baseline_report.md
baseline_report.html
```

相同 run ID 再次执行时必须得到相同 `reproducibility_sha256`，否则命令失败且不会覆盖
旧产物。成功运行会把政策版本、实验、指标、artifact 和回测摘要登记到
`data/metadata.duckdb`。详细协议见 [Phase 2 基线说明](docs/phase2_baseline.md)。

## Phase 3 因子评价

```bash
# 查看注册的参考/候选因子
make factor-list

# 使用固定评价政策评价一个因子
make factor-eval ID=F0001
make factor-eval ID=F0002
make factor-eval ID=F0003
```

每个因子必须同时提供 `src/alpha_lab/factors/candidates/<ID>.py` 和 `<ID>.yaml`。
实现只能读取元数据声明字段，必须输出 `(trade_date, instrument, value)`，不得访问网络、
写文件、读取标签、使用负向 shift 或未来窗口。统一评价在因子外执行截面去极值、方向
调整和标准化。

泄漏门禁包括：

- AST 静态扫描负向 shift、居中窗口、未声明字段、网络和文件 I/O；
- 完整历史与截断历史的前缀不变性；
- 扰动未来输入后历史因子值不变；
- 输出键、重复值、无穷值和输入突变检查。

`factor_result.json` 固定输出覆盖率、IC/RankIC/ICIR、月度/年度/滚动稳定性、五分组
收益、单调性、Top-minus-Bottom、因子换手、注册因子相关性、缺失/极值、Top-K 回测和
零/基础/双倍成本敏感性。行业和市值字段尚不存在时明确输出 `unavailable`。

评价通过门槛只得到 `eligible_for_review=true`，不会自动接受。完整协议见
[Phase 3 因子评价说明](docs/phase3_factor_evaluation.md)。

## Phase 4 因子挖掘

先按 `schemas/proposal.schema.json` 创建一轮提案；Codex 可使用仓库内的
`.agents/skills/factor-mine/SKILL.md`。正式 Python 评价始终在容器中运行：

```bash
# 单轮；默认读取 experiments/<run>/proposals/round_0001.json
make mining-round RUN=phase4-example

# 或明确给出提案文件
make mining-round RUN=phase4-example PROPOSAL=/workspace/proposals/round_0001.json

# 预先准备 5 份逐轮提案后运行/恢复整个循环
make mining-loop RUN=phase4-example ROUNDS=5 PROPOSALS_DIR=/workspace/proposals

# 重建小型审计报告
make report RUN=phase4-example
```

每轮只允许一个主要变化。管线先锁定切分、成本、评价程序、泄漏测试和当前数据 manifest
的 SHA256，再做静态检查、防泄漏测试和固定 validation 评价。`decision.json` 的
ACCEPT/REJECT 由固定 promotion checks 约束，且始终标记
`human_approval_required=true`。候选文件不可变；REJECT/ERROR 记录不会删除。成功评价和
建议会登记到 DuckDB 的 `research.experiment_*` 表。完整协议见
[Phase 4 因子挖掘说明](docs/phase4_factor_mining.md)。

## Phase 5 研究级数据

Phase 5 使用独立 `data` Compose 服务，范围固定为 `000300.SH`、`2020-01-01` 至配置结束日。
凭据仅从本地 `.env` 注入：

```bash
make research-data-probe
make research-data-bootstrap
make research-data-validate SNAPSHOT=p5-<snapshot-id>
make universe-asof DATE=2021-06-01 SNAPSHOT=p5-<snapshot-id>
make research-data-update END_DATE=2026-12-31
```

原始 Tushare 响应保持不可变 Parquet 与脱敏 sidecar。研究快照将未复权日线、复权因子、
历史名称/ST、停牌事件、证券生命周期和动态指数成分分开保存。历史股票池同时检查生效区间、
公告/已知日期和上市/退市日期，不能把当前名称或当前成分回填到过去。完整说明见
[Phase 5 研究数据说明](docs/phase5_research_data.md)。

也可以显式选择快照：

```bash
make data-validate SNAPSHOT=p1-<snapshot-id>
make qlib-export SNAPSHOT=p1-<snapshot-id>
```

`data-update` 不覆盖 raw 文件。若日期范围扩大，下载器扫描已有请求区间，只请求未覆盖的
连续日期段，并把新响应追加为独立 raw artifact。AKShare 请求使用 15 秒超时、有限重试
和请求间隔；上游失败后再次运行会从已完成标的继续。若主源最终失败，当前快照的全部
标的统一改用 Baostock，避免在一个快照内混合不同来源口径；CLI 输出
`selected_provider` 与 fallback 原因，manifest 的每个 raw input 记录实际 provider 和
endpoint。Baostock 使用不需要凭据的公共数据会话，不是券商或交易账户连接。

## 数据产物

所有产物均位于被 Git 忽略的根目录 `data/`：

```text
data/
├── raw/akshare/stock_zh_a_hist/<symbol>/
│   ├── <request-sha256>.parquet
│   └── <request-sha256>.json
├── raw/baostock/query_history_k_data_plus/<symbol>/
│   ├── <request-sha256>.parquet
│   └── <request-sha256>.json
├── bronze/<snapshot-id>/daily.parquet
├── silver/<snapshot-id>/daily.parquet
├── manifests/<snapshot-id>/
│   ├── manifest.json
│   └── quality_report.json
├── qlib/<snapshot-id>/
│   ├── calendars/day.txt
│   ├── instruments/all.txt
│   ├── features/<instrument>/*.day.bin
│   └── export_manifest.json
└── state/latest_snapshot.txt
```

raw Parquet 保存数据源原始列；同目录 JSON 保存 provider、endpoint、请求参数、抓取
时间、数据源包版本、行数和文件 SHA256。缓存读取前会重新校验 SHA256，任何不完整或
被修改的 raw artifact 都会停止流水线。

快照 ID 只由标准化 schema 版本、数据源配置、样本配置和 raw SHA256 决定，不包含运行
时间或主机绝对路径。同一快照的 manifest 和 Qlib 内容哈希因此可以重复验证。

Phase 1 bronze/silver 仍是工程样本，不会被 Phase 5 覆盖。研究级时点表位于独立的
`data/research/p5-*` 快照。详细字段见 [数据字典](docs/data_dictionary.md)。

## DuckDB 数据目录

`data/metadata.duckdb` 已作为新数据库目录实现。它不重复保存大行情，而是管理版本迁移、
数据源、证券主数据、不可变 artifact、数据快照、质量结果、时点规则和研究摘要；行情、
财务、因子和回测明细继续使用 Parquet。DuckDB 通过类型化 table macro 直接读取固定
Parquet 文件。

数据库包含 `meta/ref/market/fundamental/policy/research` 六个域；Phase 5 使用第二个
不可变迁移增加提供方能力审计和证券名称历史，并扩充外部数据集契约。完整设计见
[数据库设计](docs/database_design.md)。数据库文件本身
属于可再生本地状态，继续由 Git 忽略。

## 质量门槛

`quality_report.json` 至少记录：

- 行数、标的数和日期范围；
- 每列缺失率及完全缺失的状态字段；
- 配置中存在但数据完全缺失的标的；
- `(trade_date, instrument)` 重复键；
- OHLC 关系错误、负成交量和负成交额；
- 以样本联合交易日为基准的逐标的缺失日期。

重复键、非法行情、核心数值缺失或整只标的缺失会得到 `error` 并使命令失败。仅 Phase 1
状态字段缺失时为 `warning`，不会被悄悄解释成“未停牌”“非 ST”或“没有涨跌停”。

## Qlib 导出

导出器从固定 silver Parquet 读取，不访问网络。它生成 Qlib 官方文件存储约定：逐行日历、
无表头标的区间和以全局日历索引开头的 little-endian `float32` 特征文件。当前只导出
`open/high/low/close/volume/amount` 六个未复权基础字段。

导出先写临时目录并计算逐文件 SHA256。相同快照已有导出时，只有内容哈希完全一致才视为
成功；不同内容不会静默覆盖。集成测试会用真实 `qlib.init` 和 `D.features` 读取结果。

## 已知风险

- AKShare 的 Eastmoney 上游可能限流、主动断连或改变字段；锁文件不能固定外部接口。
- Phase 5 只消除 CSI 300 范围内的当前成分回填，不代表全 A 股无生存者偏差。
- 标准表把 AKShare 的“手”成交量乘以 100 转成股；Baostock 成交量已经是股。成交额保留
  数据源原值。研究使用前仍需做抽样交叉核验。
- ARM64 依赖失败时先保留错误并诊断。项目不会自动切到 `linux/amd64`。备用端点只有在
  字段可等价标准化时才启用，且来源不会被伪装成 AKShare。
- 当前 Phase 2 使用 Phase 1 小样本 Qlib 产物跑通训练和回测，但仍只构成工程验证；
  报告中的收益、
  IC 和 Sharpe 不可解释为研究结论或未来收益预期。
