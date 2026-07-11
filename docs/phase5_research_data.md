# Phase 5 研究级 A 股数据

Phase 5 构建 `000300.SH` 从 2020-01-01 起的时点数据闭环。它独立于 Phase 1 的十只股票
工程样本，不覆盖旧快照，也不下载全市场日线。

## 数据源和秘密

Tushare 是证券主档、动态成分、交易日历、未复权日线、复权因子、停牌和历史名称的主源。
`TUSHARE_TOKEN` 与 `TUSHARE_HTTP_URL` 只保存在 Git 忽略的 `.env`，通过 Compose 环境变量
注入。请求缓存键、错误、sidecar、manifest 和日志均不包含 token；HTTP 地址必须使用
HTTPS。非零提供方返回码属于显式失败，不会变成空表。

先执行能力探测：

```bash
make research-data-probe
```

探测 `index_member_all`，不可用时才使用按月 `index_weight` 观测重建区间。所有降级都会写入
能力报告和快照的 `membership_method`，不会静默混合口径。

## 时点规则

某证券在日期 `D` 属于历史股票池，必须同时满足：

```text
effective_from <= D <= effective_to（空结束日视为仍有效）
known_at <= D
list_date <= D <= delist_date（空退市日视为未退市）
```

成分接口提供公告日时使用公告日作为 `known_at`；缺失时保守使用生效日并标记
`effective_date_fallback`。历史名称只在其有效区间内产生 ST 状态。停牌事件按公告日和
停复牌区间展开；完整查询确认没有事件时才标记 `is_suspended=false`。缺失名称覆盖保持
nullable，不使用当前名称回填。

日线价格保持未复权。Tushare 成交量从手转换为股，成交额从千元转换为元。复权因子单独
保存，必须为有限正数；信号计算需要调整价格时应从固定快照派生，不覆盖日线事实。

## 存储与恢复

每个 REST 请求写入：

```text
data/raw/tushare/<api_name>/<request_sha256>.parquet
data/raw/tushare/<api_name>/<request_sha256>.json
```

重复请求先校验 sidecar 与 Parquet SHA256，再直接命中缓存。网络或权限错误后重跑会复用
已完成请求。研究快照位于 `data/research/p5-*/`，大事实按年分区；manifest 与质量报告在
`data/manifests/p5-*/`。只有所有质量门禁通过后才更新
`data/state/latest_research_snapshot.txt`。

## 命令

```bash
make research-data-bootstrap
make research-data-update END_DATE=2026-12-31
make research-data-validate SNAPSHOT=p5-<snapshot-id>
make universe-asof DATE=2021-06-01 SNAPSHOT=p5-<snapshot-id>
```

`universe-asof` 是只读查询，返回在该日期已知且生命周期有效的成分。研究快照成功后会把
证券主档、生命周期、名称历史、指数成分、artifact 和质量摘要幂等同步到 DuckDB；日线、
复权和状态明细仍保留在 Parquet。

## 质量门禁与限制

以下情况阻止发布：重复主键、区间重叠、未知证券引用、成分早于上市或晚于退市、非正复权
因子、缺失 artifact 或哈希不一致。状态字段覆盖不完整产生 warning 并保持 nullable。

该数据集仅覆盖 CSI 300 历史范围，不能代表全 A 股无生存者偏差。自定义 HTTP 端点和
Tushare 积分决定实际接口可用性；能力不足时命令保留缓存并明确失败，不伪造数据。
