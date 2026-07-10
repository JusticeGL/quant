# A-Share Alpha Lab

这是 A 股自动因子研究平台的 Phase 0 骨架。当前阶段只建立可复现的 Linux Python 运行环境、依赖锁、最小导入检查、质量检查和基础 CI；不包含数据下载、因子、评价、回测或交易连接。

## 运行边界

- 唯一正式 Python 环境：Docker Desktop 中的 Linux Python 3.11。
- Apple Silicon 默认使用 Docker 原生 `linux/arm64`；项目没有设置 `linux/amd64` 覆盖。
- `uv` 在容器内解析、锁定和安装依赖；主机 Python/uv 不作为验收环境。
- `.env`、数据、缓存、本地数据库、模型和实验大文件均被 Git 与 Docker build context 排除。

## Phase 0 命令

需要 Docker Desktop 正在运行。首次使用按顺序执行：

```bash
make lock
make build
make smoke
make lint
make test
```

常用入口：

- `make lock`：在临时 Linux Python 3.11 locker 容器内更新 `uv.lock`。
- `make build`：构建安装锁定依赖的 research 镜像。
- `make smoke`：真实导入 Qlib、AKShare、DuckDB、PyArrow 和 LightGBM，并打印 Linux、CPU 架构、Python 与包版本。
- `make lint`：运行 Ruff 检查、Ruff 格式检查和 mypy。
- `make test`：运行 pytest。
- `make shell`：进入一次性 research 容器。

依赖发生变化后，先运行 `make lock`，再重新构建。`uv.lock` 是可复现环境的一部分，应提交 Git；锁文件只能由容器内的 `uv` 生成。

## 进入 Phase 1 前的就绪检查

Phase 1 仍使用小样本，但会开始产生不可提交的数据和缓存。开始前在项目根目录执行：

```bash
git lfs version
git lfs install --local
git lfs fsck
gh auth status
docker system df
df -h .
make lock build smoke lint test
```

- Git LFS 只作为仓库工具前置条件；`data/`、Parquet、DuckDB、模型与实验大文件继续完全排除在 Git 外，不应因为安装 LFS 而开始跟踪它们。
- `gh auth status` 用于读取 GitHub Actions 结果，不替代 Git 的 SSH 认证。本仓库在端口 22 受限的网络中使用 SSH-over-443 remote。
- 开始下载样本前必须查看主机和 Docker 可用空间。只清理能够确认归属的项目镜像或 cache，不运行会影响其他项目的无差别全局 prune。
- 最后一条命令必须全部通过；它同时验证锁文件、Linux 镜像、ARM64/Python 版本、真实依赖导入、静态检查和测试。

## 本机假设与已知风险

- 当前开发主机是 Apple Silicon，Docker daemon 报告 `linux/arm64`，因此不需要架构模拟。
- Qlib、AKShare 及其上游依赖会变化；锁文件能固定已解析版本，但不能保证外部数据接口长期稳定。
- PyPI 的 `pyqlib==0.9.7` 没有 Linux ARM64 wheel 或源码包，因此 Phase 0 按规格回退到 Microsoft/Qlib 官方 `v0.9.7` commit `da920b7f954f48ab1bb64117c976710de198373e` 源码安装；`uv.lock` 继续固定完整来源。
- ARM64 若缺少 wheel，容器可能从源码编译；Dockerfile 提供了 C/C++、CMake、pkg-config、OpenMP 与 HDF5 构建依赖。不要未经诊断切换 amd64。
- 完整依赖集较大，在约 8 GB 内存的 Docker Desktop 上首次构建会较慢；若失败，应先查看具体构建步骤与 OOM/磁盘信息。
- Dockerfile 先安装锁定的第三方依赖，再复制 README 和项目源码；日常文档/代码修改会复用依赖层，避免反复产生约 2.35 GB 的完整环境层。
- 当前开发机已安装 Git LFS，并在本仓库完成 local 初始化；其他开发机或全新 clone 仍须执行上述初始化。任何全量数据仍不得进入 Git。
- 本机 Docker 使用配置过的 registry mirror/proxy，并报告非默认 seccomp profile；它们属于本机运维配置，不写入项目。

## 明确不在 Phase 0 的内容

本阶段不会创建未来目录的空壳，不下载全量或样本行情，不实现 Qlib 数据导出、Alpha158、因子评价、回测、自动挖掘，也不会读取 Token 或连接交易账户。后续阶段必须由单独任务明确启动。
