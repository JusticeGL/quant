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
