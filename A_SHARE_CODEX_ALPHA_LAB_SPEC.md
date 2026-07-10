# A 股 Codex 自动因子研究平台：项目实施规格书

版本：v1.0  
目标环境：macOS（优先兼容 Apple Silicon）  
目标用途：日频/周频 A 股截面因子自动挖掘、验证与组合回测；暂不包含实盘交易。

## 1. 最终目标

构建一个可以由 Codex 持续运行的本地研究平台：

1. 从 AKShare 获取免费 A 股行情，Tushare 作为可选增强数据源；
2. 将原始数据保存为不可变 Parquet，并维护数据版本和质量报告；
3. 转换为 Qlib 数据格式；
4. 先复现 Alpha158 + LightGBM 基线；
5. 让 Codex 按“提出假设 → 写因子 → 执行 → 评价 → 接受/拒绝”的闭环挖掘新因子；
6. 使用固定程序计算 IC、RankIC、ICIR、分层收益、换手率、覆盖率、相关性和组合回测；
7. 严格隔离训练集、验证集和最终测试集，避免 Codex 反复观察测试集导致过拟合；
8. 所有因子、代码、数据版本、指标和决策均可复现、审计和回滚。

非目标：高频交易、分钟级撮合、直接连接券商账户、自动实盘下单、依靠大模型主观判断收益是否合格。

## 2. 总体架构

```text
macOS 主机
├── Codex CLI / Codex App
├── Git 仓库
├── data/ 和 experiments/ 持久化目录
└── Docker
    └── Linux research 容器
        ├── Python 3.11
        ├── Qlib
        ├── AKShare / Tushare / Baostock
        ├── DuckDB / Parquet
        ├── LightGBM / scikit-learn
        └── 因子评估与回测程序
```

选择 Linux 容器作为标准运行环境，是因为 Qlib 官方支持重点仍是 Windows/Linux，macOS 原生安装不作为项目验收基准。Codex 在 Mac 上编辑项目，并通过 `docker compose run` 执行所有数据、测试和回测命令。

## 3. 需要使用或参考的项目

### 3.1 必需

| 项目 | 用途 | 接入方式 | 许可证/备注 |
|---|---|---|---|
| [microsoft/qlib](https://github.com/microsoft/qlib) | 数据表达式、ML 工作流、组合回测 | 优先固定 PyPI 版本；必要时从源码安装 | MIT |
| [AKShare](https://github.com/akfamily/akshare) | 免费 A 股行情和参考数据 | Python 依赖 | 研究用途；接口可能随上游变化 |
| [DuckDB](https://duckdb.org/) | 元数据、质量结果、实验索引 | Python 依赖 | 不用它保存大行情本体，行情使用 Parquet |

### 3.2 可选

| 项目 | 用途 | 使用时机 |
|---|---|---|
| [Tushare](https://tushare.pro/) | 更稳定的交易日历、股票列表、复权、指数成分和财务数据 | 用户有 Token/积分后启用 |
| [Baostock](https://www.baostock.com/) | AKShare 日线数据的免费备用源 | 数据交叉校验或 AKShare 暂时失效时 |
| [quantskills/skill-factor-mine](https://github.com/quantskills/skill-factor-mine) | 因子实验 SOP 参考 | 仅作工作流设计参考；GPL-3.0，复制代码前检查许可证影响 |

不要在第一阶段引入 RD-Agent、AlphaGen、AlphaForge、Kafka、Airflow、Kubernetes 或分布式数据库。先把单机、日频、可复现闭环跑通。

## 4. Mac 环境准备

### 4.1 主机工具

```bash
xcode-select --install

# 已安装 Homebrew 可跳过其安装步骤
brew install git git-lfs jq make cmake pkg-config
brew install --cask docker

git lfs install
```

安装 Codex CLI：

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex --version
```

启动 Docker Desktop，然后验证：

```bash
docker version
docker compose version
```

### 4.2 Python 运行环境原则

- Python 依赖全部在容器内安装；
- Mac 主机不承担正式回测计算；
- 容器使用 Python 3.11；
- 使用 `uv` 管理和锁定依赖；
- 首次成功后必须提交 `uv.lock`；
- Apple Silicon 默认构建原生 ARM64 镜像；只有依赖明确不支持时才临时使用 `linux/amd64`，并在 README 记录性能影响。

### 4.3 初始依赖集合

在 `pyproject.toml` 中先包含：

```text
pyqlib
akshare
baostock
tushare
duckdb
pyarrow
polars
pandas
numpy
scipy
scikit-learn
lightgbm
mlflow
tables
pydantic
pyyaml
typer
rich
matplotlib
seaborn
plotly
jupyterlab
pytest
pytest-cov
ruff
mypy
```

不要在规格书里假设所有最新版本天然兼容。Codex 应先解析依赖、完成最小导入测试，再生成锁文件。

## 5. 仓库组织

仓库名称建议：`a-share-alpha-lab`

```text
a-share-alpha-lab/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── compose.yaml
├── Makefile
├── .env.example
├── .gitignore
│
├── .agents/
│   └── skills/
│       └── factor-mine/
│           ├── SKILL.md
│           ├── references/
│           │   ├── factor-contract.md
│           │   ├── evaluation-policy.md
│           │   └── anti-leakage.md
│           └── scripts/
│
├── config/
│   ├── data_sources.yaml
│   ├── universe.yaml
│   ├── splits.yaml
│   ├── costs.yaml
│   ├── baseline.yaml
│   └── mining.yaml
│
├── src/alpha_lab/
│   ├── data/
│   │   ├── providers/
│   │   │   ├── akshare_provider.py
│   │   │   ├── baostock_provider.py
│   │   │   └── tushare_provider.py
│   │   ├── normalize.py
│   │   ├── calendar.py
│   │   ├── instruments.py
│   │   ├── adjustment.py
│   │   ├── quality.py
│   │   ├── snapshot.py
│   │   └── qlib_export.py
│   │
│   ├── factors/
│   │   ├── base.py
│   │   ├── operators.py
│   │   ├── registry.py
│   │   ├── metadata.py
│   │   ├── builtin/
│   │   └── candidates/
│   │
│   ├── evaluation/
│   │   ├── ic.py
│   │   ├── grouped_returns.py
│   │   ├── turnover.py
│   │   ├── correlation.py
│   │   ├── neutralization.py
│   │   ├── stability.py
│   │   ├── leakage.py
│   │   └── score.py
│   │
│   ├── backtest/
│   │   ├── qlib_runner.py
│   │   ├── portfolio.py
│   │   ├── constraints.py
│   │   └── cost_model.py
│   │
│   ├── mining/
│   │   ├── schemas.py
│   │   ├── experiment_store.py
│   │   ├── candidate_runner.py
│   │   ├── promotion.py
│   │   └── report.py
│   │
│   └── cli.py
│
├── scripts/
│   ├── bootstrap.sh
│   ├── download_data.py
│   ├── update_data.py
│   ├── validate_data.py
│   ├── export_qlib.py
│   ├── run_baseline.py
│   ├── evaluate_factor.py
│   ├── run_mining_round.py
│   └── build_report.py
│
├── schemas/
│   ├── hypothesis.schema.json
│   ├── experiment.schema.json
│   ├── factor_result.schema.json
│   └── decision.schema.json
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── leakage/
│   └── fixtures/
│
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_alpha158_baseline.ipynb
│   └── 03_factor_diagnostics.ipynb
│
├── data/                       # 全部 gitignore
│   ├── raw/                    # 原始响应，不修改
│   ├── bronze/                 # 字段初步统一
│   ├── silver/                 # 研究级时点数据
│   ├── qlib/                   # Qlib 二进制数据
│   ├── manifests/              # 每次数据快照清单和校验和
│   └── metadata.duckdb
│
├── experiments/                # 大文件 gitignore，保留摘要JSON
├── reports/
└── docs/
    ├── methodology.md
    ├── data_dictionary.md
    ├── reproducibility.md
    └── decisions/
```

## 6. 数据层设计

### 6.1 分层

1. `raw`：保存数据源原始返回结果、请求参数、抓取时间和数据源版本；只追加不修改。
2. `bronze`：统一股票代码、日期、字段名和数据类型，不做研究假设。
3. `silver`：完成交易日对齐、复权因子、停牌、ST、上市/退市状态和动态股票池处理。
4. `qlib`：由同一 silver 快照生成，不允许直接从网络生成。

### 6.2 日线最小字段

```text
trade_date
instrument
open
high
low
close
volume
amount
adj_factor
suspend
limit_up
limit_down
is_st
list_date
delist_date
source
ingested_at
```

第一版允许部分状态字段暂时缺失，但必须在质量报告里明确标记，不能默认为“没有停牌/没有ST”。

### 6.3 时点原则

- 股票池必须按历史日期构造，禁止用今天的股票列表回测过去；
- 财务数据以后加入时必须使用公告日期 `ann_date`，不是报告期结束日；
- 退市股票必须保留在历史数据中；
- 复权价格用于信号计算时，真实成交和收益核算必须保持口径一致；
- 每次数据更新生成 `manifest.json`，记录行数、日期范围、标的数、缺失率和文件 SHA256；
- AKShare 接口具有不稳定性，下载器需要重试、限速、断点续传和原始缓存；
- 同一个实验只引用一个固定 `data_snapshot_id`。

## 7. 研究切分与防泄漏

默认时间段可在数据可用后微调，但初始建议：

```text
train:      2014-01-01 ~ 2020-12-31
validation: 2021-01-01 ~ 2023-12-31
test:       2024-01-01 ~ 2025-12-31  # 锁定
forward:    2026-01-01 ~ 当前         # 只做观察
```

规则：

1. Codex 日常挖掘只能看到 train 和 validation 指标；
2. `test` 数据原始值和逐日结果不传给 Codex；
3. 测试集只在一个候选版本冻结并由人工确认后运行；
4. 测试失败不能继续针对测试结果调参后再次测试；需要开启新的研究批次并重新定义测试协议；
5. 除固定切分外，还要做 5 年训练 + 1 年验证的滚动 Walk-Forward；
6. 所有标准化、去极值、行业中性化参数只能在当时可见的数据上估计；
7. 因子值必须至少滞后到可交易时点，默认信号在收盘后计算、下一交易日开盘执行。

将 `config/splits.yaml`、评价程序和测试集访问规则视为锁定资产。因子挖掘 Agent 不得修改它们。

## 8. 因子接口契约

每个候选因子由两部分组成：

```text
src/alpha_lab/factors/candidates/<factor_id>.py
src/alpha_lab/factors/candidates/<factor_id>.yaml
```

元数据至少包含：

```yaml
factor_id: F0001
name: volume_price_divergence_20d
hypothesis: 成交量上升但价格动量减弱可能反映短期买盘衰竭
formula: "..."
inputs: [close, volume]
lookback: 20
direction: -1
family: liquidity
author: codex
parent_factor_ids: []
created_at: "..."
```

Python 实现必须：

- 只读取声明过的字段；
- 输出 `(trade_date, instrument, value)`；
- 不访问网络；
- 不写入候选目录以外位置；
- 不使用负向 shift、未来窗口或测试集专用条件；
- 明确 `min_periods`；
- 将正负无穷转换为 NaN；
- 不在因子内部读取未来收益标签；
- 在进入组合前完成统一的截面去极值、标准化和可选行业中性化，而不是让每个因子自行定义不同口径。

## 9. 固定评价程序

评价必须由普通 Python 程序完成，Codex 只能读取结构化结果，不能自行“目测图表后决定”。

至少输出：

- 样本覆盖率；
- Pearson IC 和 Spearman RankIC；
- IC 标准差、ICIR、IC 正值比例；
- 月度/年度/牛熊阶段稳定性；
- 五分组或十分组收益及单调性；
- Top-minus-Bottom 收益；
- 因子换手率；
- 与已接受因子的相关性；
- 行业、市值暴露；
- 极端值和缺失分布；
- Top-K 组合收益、回撤、Sharpe、换手和交易成本敏感性；
- 数据快照、代码提交和配置哈希。

初始晋级门槛不是“收益保证”，而是减少明显无效候选：

```text
coverage >= 70%
abs(validation RankIC mean) >= 0.015
abs(validation ICIR) >= 0.20
IC方向在多数滚动子区间一致
与任一已接受因子 abs(correlation) < 0.80
无未来函数、索引错位或标签污染
扣除成本后组合结果不发生结构性反转
```

门槛配置化，并在基线跑通后校准。禁止 Codex 为了让某个候选过关而修改门槛。

## 10. A 股回测约束

第一版回测至少处理：

- 次日开盘成交；
- T+1；
- 100 股整数手；
- 停牌不可交易；
- 涨停买不进、跌停卖不出；
- 最低佣金、佣金、过户费、印花税使用带生效日期的配置表；
- 调仓时先处理不可卖持仓，再分配可用现金；
- 新股上市初期、ST和流动性过滤规则配置化；
- 动态股票池和历史指数成分。

不要把当前费率永久写死在代码中。`config/costs.yaml` 应按生效日期保存规则。

## 11. Codex 自动挖掘闭环

### 11.1 每轮流程

```text
读取研究宪法和当前因子库摘要
→ 提出一个单点假设
→ 输出 hypothesis.json
→ 生成一个候选因子及元数据
→ 跑静态检查和防泄漏测试
→ 在 train 上计算并在 validation 上评价
→ 固定程序输出 factor_result.json
→ Codex 输出 decision.json
→ ACCEPT / REJECT / ERROR
→ 更新实验日志和因子血缘
```

每轮只允许一种主要变化：新增一个因子、修改一个算子、改变一个窗口或改变一种组合方式。禁止一次同时改变多个方向。

### 11.2 实验目录

```text
experiments/<run_id>/
├── run_manifest.json
├── research_brief.md
├── round_0001/
│   ├── hypothesis.json
│   ├── candidate/
│   ├── test_report.json
│   ├── factor_result.json
│   └── decision.json
├── round_0002/
└── final_report.md
```

### 11.3 Codex 非交互调用

新脚本使用明确沙箱，不使用已经弃用的 `--full-auto`：

```bash
codex exec \
  --sandbox workspace-write \
  --output-schema schemas/hypothesis.schema.json \
  -o experiments/<run_id>/round_0001/hypothesis.json \
  "按照 .agents/skills/factor-mine/SKILL.md 提出下一轮单点因子假设"
```

代码生成与决策分别调用，并使用不同 JSON Schema。执行指标计算的程序不由 Codex 临时生成。

### 11.4 Agent 权限

允许编辑：

```text
src/alpha_lab/factors/candidates/
experiments/<current_run>/
reports/
```

默认禁止编辑：

```text
config/splits.yaml
config/costs.yaml
src/alpha_lab/evaluation/
src/alpha_lab/backtest/
tests/leakage/
data/manifests/
```

需要修改锁定区域时，Codex 只能提出变更建议，由人工另开任务处理。

## 12. AGENTS.md 必须包含的规则

Codex 初始化项目时应创建 `AGENTS.md`，至少写入：

1. 所有正式命令通过 `make` 或 `docker compose` 执行；
2. 不直接修改 `data/raw`；
3. 不把 `.env`、Token、数据文件和实验大文件提交 Git；
4. 新增因子必须同时新增元数据和测试；
5. 禁止未来函数、标签污染和当前成分股回填历史；
6. 评价程序、切分和成本规则属于锁定区域；
7. 每轮实验只改一个主要变量；
8. 完成任务前运行 `make lint test smoke`；
9. 不以单次收益曲线作为接受依据；
10. 不删除失败实验，失败也是研究记录。

## 13. Makefile 对外命令

用户和 Codex 都只依赖以下稳定入口：

```text
make build             构建研究镜像
make shell             进入容器
make lock              解析并锁定依赖
make data-bootstrap    下载最小股票池数据
make data-update       增量更新
make data-validate     生成质量报告
make qlib-export       生成 Qlib 数据
make baseline          跑 Alpha158 + LightGBM 基线
make factor-eval ID=F0001
make mining-round RUN=<run_id>
make mining-loop RUN=<run_id> ROUNDS=5
make report RUN=<run_id>
make lint
make test
make smoke
```

## 14. 分阶段实施与验收

### Phase 0：项目骨架

交付：仓库目录、Dockerfile、compose.yaml、pyproject.toml、Makefile、AGENTS.md、基础 CI。  
验收：`make build && make smoke` 在 Mac 上通过；能导入 Qlib、AKShare、DuckDB、LightGBM。

### Phase 1：最小数据闭环

范围：先用沪深300当前小样本或 50 只股票验证工程，不宣称无生存者偏差。  
交付：AKShare 下载、原始缓存、标准化 Parquet、质量报告、快照清单、Qlib 导出。  
验收：重复运行不会重复下载完整数据；同一快照导出结果哈希一致；缺失和重复数据有明确报告。

### Phase 2：基线复现

交付：Alpha158 + LightGBM 训练、信号分析、Top-K 回测和 HTML/Markdown 报告。  
验收：固定种子下结果可复现；报告包含数据快照、配置和 Git commit。

### Phase 3：自定义因子评价

交付：因子契约、候选注册表、IC/分组/换手/相关性/稳定性评价、防泄漏测试。  
验收：至少三个手工因子（动量、反转、波动）跑通；一个故意包含未来函数的因子必须被拦截。

### Phase 4：Codex 挖掘闭环

交付：Factor Mining Skill、JSON Schema、实验目录、单轮和多轮入口。  
验收：Codex 连续运行 5 轮；每轮有完整产物；失败可恢复；不会编辑锁定目录。

### Phase 5：研究级 A 股数据

交付：历史股票列表、退市股、ST、停牌、复权和动态指数成分；Tushare 可选适配。  
验收：抽样日期能够还原当时股票池；退市标的不从历史消失；公告日逻辑有测试。

### Phase 6：稳健性与最终测试

交付：Walk-Forward、成本敏感性、行业/市值暴露、锁定测试集审批流程。  
验收：只有冻结候选版本才能运行 test；测试结果自动生成不可覆盖的报告。

## 15. 给 Codex 的第一条总任务

将下面内容连同本规格书交给 Codex：

```text
请阅读 A_SHARE_CODEX_ALPHA_LAB_SPEC.md，先只完成 Phase 0，不提前实现后续阶段。

要求：
1. 先检查当前仓库和 Mac/Docker 环境，列出假设与风险；
2. 创建规格书定义的最小项目骨架，但不要创建空壳文件填满所有未来目录；
3. 使用 Linux Docker 容器作为唯一正式 Python 运行环境，Python 3.11，使用 uv 锁依赖；
4. 创建 Dockerfile、compose.yaml、pyproject.toml、Makefile、AGENTS.md、README.md、.env.example 和 .gitignore；
5. 实现 make build、make lock、make smoke、make lint、make test；
6. smoke test 必须实际 import qlib、akshare、duckdb、pyarrow、lightgbm，并打印平台和版本；
7. Apple Silicon 默认使用原生 ARM64；如果依赖失败，先诊断，不要直接切换到 amd64；
8. 不下载全量数据，不实现回测，不连接任何交易账户；
9. 不提交密钥、缓存、数据和实验大文件；
10. 完成后给出实际执行过的命令、测试结果、未解决问题和进入 Phase 1 前的建议。

请先制定短计划，然后直接实施和验证。只有遇到会改变架构的阻塞才询问我。
```

## 16. 后续给 Codex 的任务节奏

不要一次要求 Codex 完成全部六个阶段。建议一个阶段一个任务，每阶段：

```text
阅读规格书和 AGENTS.md
→ 检查上阶段产物
→ 给出本阶段计划
→ 实施
→ 运行真实测试
→ 输出验收表
→ 提交 Git checkpoint
```

推荐首次只做 Phase 0；确认 Mac 容器兼容后再做 Phase 1 的 10～50 只股票小样本。全量 A 股数据和自动挖掘应放在基线与数据质量体系稳定之后。

## 17. 关键风险

1. AKShare 是网络数据采集工具，上游接口会变化，不能把“今天能下载”当成稳定数据合同；
2. 免费数据很难完整消除生存者偏差，第一版结果只能视为工程验证；
3. Qlib 在 macOS 不是官方主支持环境，所以正式运行放在 Linux 容器；
4. 自动搜索次数越多，验证集过拟合越严重，必须记录试验次数并锁定测试集；
5. Codex 应负责研究假设和代码实现，不应控制评价口径或决定是否隐藏失败结果；
6. 回测结果不代表未来收益，本项目只用于研究和教育。

