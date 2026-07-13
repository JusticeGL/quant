# Phase 1 数据字典

## Raw

路径：

- `data/raw/akshare/stock_zh_a_hist/<symbol>/<request-sha256>.parquet`
- `data/raw/baostock/query_history_k_data_plus/<symbol>/<request-sha256>.parquet`

raw 表保留数据源响应列和原始数据类型，不进行字段删除、单位换算、复权或状态推断。
相邻 JSON sidecar 记录 provider、endpoint、完整请求区间、数据源包版本、UTC 抓取时间、
行数和 Parquet SHA256。raw artifact 只追加，不覆盖。

## Bronze 与 silver

Phase 1 中两层使用相同标准 schema。bronze 表示字段与类型已统一；silver 是同一固定快照
的 Qlib 唯一输入，但尚未达到研究级时点数据要求。

| 字段 | Phase 1 类型/含义 | 当前限制 |
|---|---|---|
| `trade_date` | 无时区日日期 | 来自 AKShare `日期` |
| `instrument` | `SH/SZ/BJ` + 六位代码 | 按代码前缀映射交易所 |
| `open` | `float64` 开盘价 | 未复权 |
| `high` | `float64` 最高价 | 未复权 |
| `low` | `float64` 最低价 | 未复权 |
| `close` | `float64` 收盘价 | 未复权 |
| `volume` | `float64`，单位为股 | AKShare 的“手”乘以 100；Baostock 原值 |
| `amount` | `float64`，上游成交额原值 | 不做币种或缩放变换 |
| `adj_factor` | nullable `Float64` | 全部缺失，不假设为 1 |
| `suspend` | nullable boolean | AKShare 缺失；Baostock 使用 `tradestatus` |
| `limit_up` | nullable boolean | 全部缺失，不假设为 false |
| `limit_down` | nullable boolean | 全部缺失，不假设为 false |
| `is_st` | nullable boolean | AKShare 缺失；Baostock 使用 `isST` |
| `list_date` | nullable datetime | 全部缺失 |
| `delist_date` | nullable datetime | 全部缺失 |
| `source` | 字符串 provenance | 精确记录 AKShare 或 Baostock endpoint |
| `ingested_at` | UTC timestamp | 对应 raw artifact 的首次抓取时间 |

## Qlib

Qlib 文件中的 instrument 与 silver 一致。日历是 silver 的联合日期集合；每只标的的特征
从其首个可见日期开始，以全局日历位置作为 `.bin` 首个 `float32`，内部缺口写为 `NaN`。
导出的数值字段为 `open/high/low/close/volume/amount`，统一存为 little-endian
`float32`。Phase 1 不导出伪造的 `factor`、停牌或交易约束字段。

## Phase 5 研究级时点表

Phase 5 位于 `data/research/<p5-snapshot-id>/`，与 Phase 1 silver 并行：

| 数据集 | 主键 | 时点字段 | 说明 |
|---|---|---|---|
| `security_master` | `security_id` | `known_at` | 包含 L/D/P/G，退市股不删除 |
| `security_name_history` | `security_id,effective_from` | `known_at` | 名称区间与 nullable `is_st` |
| `index_membership` | `index_id,security_id,effective_from` | `known_at` | 动态成分、公告日、生效/失效日和来源方法 |
| `daily_bar` | `trade_date,security_id` | `known_at` | 未复权；成交量为股、成交额为元 |
| `adjustment_factor` | `trade_date,security_id,factor_type` | `known_at` | 与价格分离的正复权因子 |
| `daily_status` | `trade_date,security_id` | `known_at` | nullable ST 与停牌状态 |
| `universe_dates` | `as_of_date,index_id,security_id` | `as_of_date` | 已应用公告日和生命周期门禁的股票池 |

`known_at` 晚于查询日期的记录不可见。上游缺少公告日时使用生效日作为保守 fallback，并在
`known_at_source` 中明确记录。

## Phase 6 暴露快照与审批目录

Phase 6 暴露快照位于 `data/exposures/<p6x-snapshot-id>/`，清单位于
`data/manifests/<p6x-snapshot-id>/manifest.json`。清单同时锁定 Phase 5
清单、稳健性策略、质量报告、raw request 与每个 Parquet artifact 的
SHA256。

| 数据集/目录表 | 主键 | 时点字段 | 说明 |
|---|---|---|---|
| `market.exposure_market_cap` | `trade_date,security_id` | `known_at` | 总市值和流通市值，单位 CNY；按年保留在 Parquet，DuckDB 只登记 artifact |
| `ref.industry_definition` | `definition_id` | 快照身份 | SW2021 行业定义；`definition_id` 是记录内容 SHA256，`exposure_snapshot_id,industry_id` 唯一 |
| `ref.industry_membership_history` | `membership_id` | `effective_from,effective_to,known_at` | 证券行业区间；同时外键引用行业定义和 `ref.security` |
| `research.factor_freeze` | `freeze_id` | `created_at` | 锁定因子版本、Phase 5/暴露快照、策略哈希、Git commit 与最终测试区间；`freeze_id,freeze_sha256,test_start,test_end` 为唯一候选键 |
| `research.test_request` | `request_id` | `requested_at` | 携带 `freeze_id,freeze_sha256,test_start,test_end`，复合外键引用完全一致的 freeze；状态固定为 `test_requested` |
| `research.test_approval` | `approval_id` | `approved_at` | 携带 request、freeze、`confirmed_freeze_sha256` 与测试区间，复合外键保证实名审批精确绑定该 request |
| `research.final_test_run` | `test_run_id` | `started_at,finished_at` | 携带 approval/request/freeze/hash/range 并复合引用完整审批身份；错误 freeze、hash 或区间由 DuckDB constraint 拒绝 |

`effective_from <= D <= effective_to` 且 `known_at <= D` 时，行业成员记录才在
日期 `D` 可见。`known_at_source=effective_date_fallback` 表示上游无公告日，
使用生效日作为保守回退；禁止使用当前行业回填历史。

Catalog 同步在 Task 2 验证后记录 manifest SHA256，并在 DuckDB 写锁与
事务内重读 manifest、比较 SHA256，然后对 market-cap、raw、quality、
industry 与必需 Phase 5 依赖的当前文件字节逐一重算 SHA256。登记的
是实算值。Phase 5 小型引用表、质量报告与行业表使用同一次读取所得的
bytes 完成哈希与解析；写完 latest-state 后，`COMMIT` 前最后一次封口会
重算两份 canonical manifest 及全部 exposure/raw/quality/industry/Phase 5
依赖。封口发现验证后篡改、路径逃逸或中途异常时，整个同步事务回滚。
Catalog 写锁不能约束不配合的外部文件写入者，只能在提交前封口时发现其
漂移；因此 `data/raw` 必须保持 immutable、append-only，快照依赖不得覆盖。
