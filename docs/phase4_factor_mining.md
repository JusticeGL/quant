# Phase 4 可审计因子挖掘

Phase 4 只自动化“提出单点假设 → 生成候选 → 使用 Phase 3 固定程序评价 → 记录建议”这条
工程闭环。它不访问锁定 test、不自动批准因子、不扩展数据范围，也不连接交易账户。

## 假设与风险

- 当前只有 Phase 1 的 10 只股票、约半年工程样本，存在明确生存者偏差；任何结果都不是
  收益结论。
- `ACCEPT` 仅代表固定 promotion checks 全部通过并可提交人工复核；注册表中的
  `accepted_factor_ids` 不会被自动修改。
- 候选可使用的字段受 Phase 3 因子契约限制。行业、市值、历史成分、完整停牌/ST/复权
  信息尚不可用，不能通过推断填补。
- Docker Desktop bind mount 不保证 POSIX `flock` 语义，因此实验和 DuckDB 写入使用原子
  `mkdir` 锁；异常后保留当前轮，可在同一 run ID 上重试。
- 不同挖掘 run 若选择同一个因子 ID，只允许写入完全相同的候选字节，否则立即失败。

## 产物与权限

每轮目录为 `experiments/<run_id>/round_NNNN/`，包含 hypothesis、candidate、test report、
factor result 和 decision。Git 仅允许追踪小型审计 JSON/Markdown；候选副本、大型指标和
模型产物继续忽略。正式候选代码与 YAML 发布到
`src/alpha_lab/factors/candidates/`，状态保持 `candidate`。

运行开始时记录以下锁区哈希，结束前逐字节复核：

- `config/splits.yaml`、`config/costs.yaml`、`config/factor_evaluation.yaml`；
- `src/alpha_lab/evaluation/*.py`、`tests/leakage/*.py`；
- 所选数据快照的 manifest 和 quality report。

锁区发生变化时停止当前 run，需要人工另开任务处理。

## Codex 非交互提案

仓库提供严格的 hypothesis、proposal 和 decision JSON Schema。合并提案可按以下方式生成：

```bash
codex exec \
  --sandbox workspace-write \
  --output-schema schemas/proposal.schema.json \
  -o experiments/<run_id>/proposals/round_0001.json \
  "使用 $factor-mine；读取当前 research_brief 和历史 decision，只提出一个主要变化"
```

Factor Mine skill 禁止候选访问网络、文件、子进程、标签或测试值，并要求显式
`min_periods`、仅使用历史窗口和声明字段。指标计算不由 Codex 临时生成。

## 稳定命令与恢复

```bash
make mining-round RUN=<run_id>
make mining-loop RUN=<run_id> ROUNDS=5
make report RUN=<run_id>
```

单轮默认读取 `<run>/proposals/round_NNNN.json`，也可用 `PROPOSAL=` 指定。循环可用
`PROPOSALS_DIR=` 指定逐轮提案目录。manifest 的 `current_round`、原子候选写入和幂等
decision 使进程在评价或数据库登记中断后能够从同一轮恢复，不删除失败记录。

## 人工审批边界

固定程序只输出 `ACCEPT`、`REJECT` 或 `ERROR` 建议。即使 ACCEPT，也必须由人工检查假设、
数据适用性、相关性和经济解释，再另开任务修改因子注册状态。Phase 4 不接触锁定 test；
Phase 6 才定义冻结候选和 test 审批流程。
