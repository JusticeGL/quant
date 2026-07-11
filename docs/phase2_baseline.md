# Phase 2：Alpha158 + LightGBM 工程基线

## 范围

Phase 2 从一个固定 Phase 1 silver/Qlib 快照生成官方 Qlib Alpha158 的 158 个特征，
训练单线程确定性 LightGBM 回归模型，在 validation 上计算信号指标并运行 Top-K
组合回测。它不会运行锁定 test，也不实现 Phase 3 的自定义因子契约或评价框架。

当前数据只有 10 只人工选择股票和 2024 年上半年行情，且 manifest 明确标记
`research_eligible=false`。因此所有输出均标记 `engineering_only=true`，只能证明工程
链路可运行、可审计、可重复。

## 时点和防泄漏

- 信号在交易日收盘后产生；订单在下一交易日开盘执行。
- 标签为下一交易日开盘到再下一交易日开盘的收益：
  `Ref($open,-2)/Ref($open,-1)-1`。
- train 中任何标签结果跨越 train 结束日的观测都会按交易日清除。
- validation 信号截止 2024-06-12，其标签和交易最晚使用 2024-06-14 的隔离数据。
- 锁定 test 从 2024-06-17 开始；流水线不载入、打分、评价或报告该区间。
- 特征来自 Qlib Alpha158 定义，不允许负向未来窗口作为模型输入。

## 确定性协议

- LightGBM 使用固定随机种子、单线程、`deterministic=true` 和固定列式算法。
- 数据快照、Qlib 内容、baseline/split/cost 配置都记录 SHA256。
- 报告记录 Git commit 和工作区是否 dirty。
- 排序使用稳定排序，并以 instrument 作为同分时的确定性次级键。
- predictions、信号摘要、回测摘要和约束计数形成规范化
  `reproducibility_sha256`。
- 相同 run ID 已存在时，只接受相同的 reproducibility hash，不覆盖不同结果。

## Top-K 交易约束

当前引擎处理：

- 下一交易日开盘成交；
- T+1；
- 100 股整数手；
- 停牌和零成交量不可交易；
- ST 过滤；
- 显式涨跌停标志优先；标志缺失时按开盘价相对昨收和配置阈值保守推断；
- 调仓先卖出可卖持仓，再按可用现金买入；
- 按生效日期选择佣金、最低佣金、印花税和过户费规则。

佣金和最低佣金因券商而异，在配置中明确标记为工程假设。2023-08-28 起的印花税
规则引用国家税务总局政策法规库；过户费规则记录中国结算公告链接。所有规则位于锁定的
`config/costs.yaml`，代码不永久写死当前费率。

## 输出与数据库

运行目录包含模型、预测、逐日净值、交易、机器可读 manifest，以及 Markdown/HTML
报告。DuckDB 的 `policy` 和 `research` 域登记：

- split/cost policy version 与成本规则；
- experiment run 和 validation 指标；
- 模型、信号、回测和报告 artifact；
- backtest run 与摘要。

大数据本体仍是 Parquet/文本文件，DuckDB 只保存目录与摘要。

## 已知限制

- 当前样本有生存者偏差，且规模和时间跨度不足。
- 缺少复权因子、上市/退市日期和历史动态成分。
- 涨跌停标志缺失，只能使用明示且可审计的价格推断。
- 没有行业、市值暴露、Walk-Forward 或成本敏感性；这些属于后续阶段。
- 回测结果不代表未来收益，项目仅用于研究和教育。
