CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS ref;
CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS fundamental;
CREATE SCHEMA IF NOT EXISTS policy;
CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS meta.schema_migration (
    version INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    sha256 VARCHAR NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS meta.data_source (
    source_id VARCHAR PRIMARY KEY,
    provider VARCHAR NOT NULL,
    endpoint VARCHAR,
    package_name VARCHAR,
    package_version VARCHAR,
    license_note VARCHAR,
    priority INTEGER NOT NULL DEFAULT 100,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS meta.ingestion_run (
    run_id VARCHAR PRIMARY KEY,
    source_id VARCHAR NOT NULL REFERENCES meta.data_source(source_id),
    dataset_name VARCHAR NOT NULL,
    requested_start DATE,
    requested_end DATE,
    request_params JSON,
    status VARCHAR NOT NULL CHECK (status IN ('running', 'success', 'partial', 'failed')),
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    rows_received BIGINT,
    rows_accepted BIGINT,
    error_message VARCHAR,
    code_commit VARCHAR,
    config_sha256 VARCHAR
);

CREATE TABLE IF NOT EXISTS meta.artifact (
    artifact_id VARCHAR PRIMARY KEY,
    run_id VARCHAR REFERENCES meta.ingestion_run(run_id),
    source_id VARCHAR REFERENCES meta.data_source(source_id),
    layer VARCHAR NOT NULL CHECK (layer IN ('raw', 'bronze', 'silver', 'qlib', 'report', 'research')),
    dataset_name VARCHAR NOT NULL,
    relative_path VARCHAR NOT NULL,
    format VARCHAR NOT NULL,
    sha256 VARCHAR NOT NULL,
    file_size_bytes BIGINT,
    row_count BIGINT,
    min_event_date DATE,
    max_event_date DATE,
    schema_version INTEGER NOT NULL,
    immutable BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(relative_path, sha256)
);

CREATE TABLE IF NOT EXISTS meta.dataset_snapshot (
    snapshot_id VARCHAR PRIMARY KEY,
    snapshot_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL CHECK (status IN ('building', 'valid', 'invalid', 'frozen')),
    identity_sha256 VARCHAR NOT NULL,
    schema_version INTEGER NOT NULL,
    code_commit VARCHAR,
    config_sha256 VARCHAR,
    source_config JSON,
    universe_config JSON,
    row_count BIGINT,
    security_count BIGINT,
    start_date DATE,
    end_date DATE,
    quality_status VARCHAR,
    parent_snapshot_id VARCHAR,
    notes VARCHAR,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS meta.snapshot_artifact (
    snapshot_id VARCHAR NOT NULL REFERENCES meta.dataset_snapshot(snapshot_id),
    artifact_id VARCHAR NOT NULL REFERENCES meta.artifact(artifact_id),
    dataset_name VARCHAR NOT NULL,
    PRIMARY KEY(snapshot_id, artifact_id)
);

CREATE TABLE IF NOT EXISTS meta.quality_result (
    snapshot_id VARCHAR NOT NULL REFERENCES meta.dataset_snapshot(snapshot_id),
    dataset_name VARCHAR NOT NULL,
    check_name VARCHAR NOT NULL,
    severity VARCHAR NOT NULL CHECK (severity IN ('info', 'warning', 'error')),
    status VARCHAR NOT NULL CHECK (status IN ('pass', 'fail')),
    observed_value DOUBLE,
    threshold_value DOUBLE,
    affected_rows BIGINT,
    details JSON,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY(snapshot_id, dataset_name, check_name)
);

CREATE TABLE IF NOT EXISTS meta.dataset_contract (
    dataset_name VARCHAR PRIMARY KEY,
    storage_layer VARCHAR NOT NULL,
    storage_format VARCHAR NOT NULL,
    primary_key_columns JSON NOT NULL,
    partition_columns JSON NOT NULL,
    required_columns JSON NOT NULL,
    point_in_time_column VARCHAR,
    description VARCHAR NOT NULL,
    schema_version INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS meta.repository_state (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS ref.exchange (
    exchange_code VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    country_code VARCHAR NOT NULL DEFAULT 'CN',
    timezone VARCHAR NOT NULL DEFAULT 'Asia/Shanghai',
    currency VARCHAR NOT NULL DEFAULT 'CNY'
);

CREATE TABLE IF NOT EXISTS ref.security (
    security_id VARCHAR PRIMARY KEY,
    asset_type VARCHAR NOT NULL CHECK (asset_type IN ('stock', 'index', 'fund', 'bond')),
    exchange VARCHAR NOT NULL REFERENCES ref.exchange(exchange_code),
    board VARCHAR,
    currency VARCHAR NOT NULL,
    lot_size INTEGER NOT NULL CHECK (lot_size > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS ref.security_identifier_history (
    identifier_id VARCHAR PRIMARY KEY,
    security_id VARCHAR NOT NULL REFERENCES ref.security(security_id),
    identifier_type VARCHAR NOT NULL,
    identifier_value VARCHAR NOT NULL,
    valid_from DATE,
    valid_to DATE,
    known_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    source_artifact_id VARCHAR,
    CHECK (valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from),
    UNIQUE(identifier_type, identifier_value, valid_from)
);

CREATE TABLE IF NOT EXISTS ref.security_lifecycle (
    security_id VARCHAR PRIMARY KEY REFERENCES ref.security(security_id),
    list_date DATE,
    delist_date DATE,
    first_trade_date DATE,
    last_trade_date DATE,
    listing_status VARCHAR NOT NULL CHECK (listing_status IN ('unknown', 'prelisted', 'listed', 'suspended_listing', 'delisted')),
    delist_reason VARCHAR,
    known_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    source_artifact_id VARCHAR,
    CHECK (delist_date IS NULL OR list_date IS NULL OR delist_date >= list_date)
);

CREATE TABLE IF NOT EXISTS ref.trading_calendar (
    exchange VARCHAR NOT NULL REFERENCES ref.exchange(exchange_code),
    calendar_date DATE NOT NULL,
    is_open BOOLEAN NOT NULL,
    previous_trade_date DATE,
    next_trade_date DATE,
    session_open TIME,
    session_close TIME,
    source_artifact_id VARCHAR,
    PRIMARY KEY(exchange, calendar_date)
);

CREATE TABLE IF NOT EXISTS ref.industry_classification (
    classification_id VARCHAR PRIMARY KEY,
    provider VARCHAR NOT NULL,
    standard_name VARCHAR NOT NULL,
    version VARCHAR NOT NULL,
    level_count INTEGER NOT NULL CHECK (level_count > 0)
);

CREATE TABLE IF NOT EXISTS ref.industry_node (
    classification_id VARCHAR NOT NULL REFERENCES ref.industry_classification(classification_id),
    industry_code VARCHAR NOT NULL,
    industry_name VARCHAR NOT NULL,
    level INTEGER NOT NULL,
    parent_code VARCHAR,
    PRIMARY KEY(classification_id, industry_code)
);

CREATE TABLE IF NOT EXISTS ref.security_industry_history (
    membership_id VARCHAR PRIMARY KEY,
    security_id VARCHAR NOT NULL REFERENCES ref.security(security_id),
    classification_id VARCHAR NOT NULL REFERENCES ref.industry_classification(classification_id),
    industry_code VARCHAR NOT NULL,
    valid_from DATE NOT NULL,
    valid_to DATE,
    announced_at TIMESTAMPTZ,
    source_artifact_id VARCHAR,
    CHECK (valid_to IS NULL OR valid_to >= valid_from)
);

CREATE TABLE IF NOT EXISTS ref.index_definition (
    index_id VARCHAR PRIMARY KEY,
    index_code VARCHAR NOT NULL UNIQUE,
    index_name VARCHAR NOT NULL,
    exchange VARCHAR REFERENCES ref.exchange(exchange_code),
    provider VARCHAR NOT NULL,
    base_date DATE
);

CREATE TABLE IF NOT EXISTS ref.index_membership_history (
    membership_id VARCHAR PRIMARY KEY,
    index_id VARCHAR NOT NULL REFERENCES ref.index_definition(index_id),
    security_id VARCHAR NOT NULL REFERENCES ref.security(security_id),
    effective_from DATE NOT NULL,
    effective_to DATE,
    announced_at TIMESTAMPTZ,
    weight DOUBLE,
    source_artifact_id VARCHAR,
    CHECK (effective_to IS NULL OR effective_to >= effective_from)
);

CREATE TABLE IF NOT EXISTS market.corporate_action (
    action_id VARCHAR PRIMARY KEY,
    security_id VARCHAR NOT NULL,
    action_type VARCHAR NOT NULL,
    announcement_date DATE NOT NULL,
    record_date DATE,
    ex_date DATE,
    payment_date DATE,
    cash_dividend DOUBLE,
    share_ratio DOUBLE,
    rights_price DOUBLE,
    rights_ratio DOUBLE,
    revision_no INTEGER NOT NULL DEFAULT 1,
    known_at TIMESTAMPTZ NOT NULL,
    source_artifact_id VARCHAR
);

CREATE TABLE IF NOT EXISTS fundamental.metric_definition (
    metric_code VARCHAR PRIMARY KEY,
    metric_name VARCHAR NOT NULL,
    statement_type VARCHAR NOT NULL,
    default_unit VARCHAR,
    description VARCHAR
);

CREATE TABLE IF NOT EXISTS fundamental.filing_catalog (
    filing_id VARCHAR PRIMARY KEY,
    security_id VARCHAR NOT NULL,
    report_period DATE NOT NULL,
    report_type VARCHAR NOT NULL,
    announcement_date DATE NOT NULL,
    actual_release_time TIMESTAMPTZ,
    revision_no INTEGER NOT NULL DEFAULT 1,
    is_latest BOOLEAN NOT NULL DEFAULT true,
    source_artifact_id VARCHAR,
    known_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS fundamental.fact_artifact (
    filing_id VARCHAR NOT NULL REFERENCES fundamental.filing_catalog(filing_id),
    artifact_id VARCHAR NOT NULL,
    metric_count BIGINT,
    PRIMARY KEY(filing_id, artifact_id)
);

CREATE TABLE IF NOT EXISTS policy.policy_version (
    policy_id VARCHAR PRIMARY KEY,
    policy_type VARCHAR NOT NULL,
    version VARCHAR NOT NULL,
    config_sha256 VARCHAR NOT NULL,
    effective_from DATE,
    effective_to DATE,
    locked BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS policy.price_limit_rule (
    rule_id VARCHAR PRIMARY KEY,
    policy_id VARCHAR NOT NULL REFERENCES policy.policy_version(policy_id),
    exchange VARCHAR NOT NULL,
    board VARCHAR,
    security_status VARCHAR,
    effective_from DATE NOT NULL,
    effective_to DATE,
    limit_ratio DOUBLE,
    ipo_no_limit_days INTEGER,
    CHECK (effective_to IS NULL OR effective_to >= effective_from)
);

CREATE TABLE IF NOT EXISTS policy.cost_rule (
    rule_id VARCHAR PRIMARY KEY,
    policy_id VARCHAR NOT NULL REFERENCES policy.policy_version(policy_id),
    market VARCHAR NOT NULL,
    side VARCHAR NOT NULL CHECK (side IN ('buy', 'sell', 'both')),
    effective_from DATE NOT NULL,
    effective_to DATE,
    commission_rate DOUBLE NOT NULL,
    minimum_commission DOUBLE NOT NULL,
    stamp_duty_rate DOUBLE NOT NULL,
    transfer_fee_rate DOUBLE NOT NULL,
    CHECK (effective_to IS NULL OR effective_to >= effective_from)
);

CREATE TABLE IF NOT EXISTS research.universe_definition (
    universe_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    description VARCHAR,
    construction_rule JSON NOT NULL,
    survivorship_free BOOLEAN NOT NULL,
    research_eligible BOOLEAN NOT NULL,
    config_sha256 VARCHAR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS research.universe_membership (
    membership_id VARCHAR PRIMARY KEY,
    universe_id VARCHAR NOT NULL REFERENCES research.universe_definition(universe_id),
    security_id VARCHAR NOT NULL,
    effective_from DATE NOT NULL,
    effective_to DATE,
    announced_at TIMESTAMPTZ,
    exclusion_reason VARCHAR,
    source_artifact_id VARCHAR,
    CHECK (effective_to IS NULL OR effective_to >= effective_from)
);

CREATE TABLE IF NOT EXISTS research.factor_definition (
    factor_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE,
    family VARCHAR NOT NULL,
    description VARCHAR,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS research.factor_version (
    factor_version_id VARCHAR PRIMARY KEY,
    factor_id VARCHAR NOT NULL REFERENCES research.factor_definition(factor_id),
    formula VARCHAR NOT NULL,
    implementation_path VARCHAR NOT NULL,
    code_sha256 VARCHAR NOT NULL,
    metadata_sha256 VARCHAR NOT NULL,
    lookback INTEGER NOT NULL,
    direction INTEGER NOT NULL CHECK (direction IN (-1, 1)),
    parent_factor_version_id VARCHAR,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS research.experiment_run (
    experiment_id VARCHAR PRIMARY KEY,
    factor_version_id VARCHAR REFERENCES research.factor_version(factor_version_id),
    data_snapshot_id VARCHAR NOT NULL,
    universe_id VARCHAR NOT NULL REFERENCES research.universe_definition(universe_id),
    split_policy_sha256 VARCHAR NOT NULL,
    cost_policy_sha256 VARCHAR,
    code_commit VARCHAR NOT NULL,
    random_seed BIGINT,
    status VARCHAR NOT NULL CHECK (status IN ('queued', 'running', 'success', 'failed')),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS research.experiment_metric (
    experiment_id VARCHAR NOT NULL REFERENCES research.experiment_run(experiment_id),
    split_name VARCHAR NOT NULL,
    metric_name VARCHAR NOT NULL,
    metric_value DOUBLE,
    period VARCHAR NOT NULL DEFAULT 'all',
    PRIMARY KEY(experiment_id, split_name, metric_name, period)
);

CREATE TABLE IF NOT EXISTS research.experiment_decision (
    experiment_id VARCHAR PRIMARY KEY REFERENCES research.experiment_run(experiment_id),
    decision VARCHAR NOT NULL CHECK (decision IN ('accept', 'reject', 'error')),
    reason JSON NOT NULL,
    policy_version VARCHAR NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS research.backtest_run (
    backtest_id VARCHAR PRIMARY KEY,
    experiment_id VARCHAR REFERENCES research.experiment_run(experiment_id),
    data_snapshot_id VARCHAR NOT NULL,
    signal_artifact_id VARCHAR,
    position_artifact_id VARCHAR,
    trade_artifact_id VARCHAR,
    report_artifact_id VARCHAR,
    status VARCHAR NOT NULL CHECK (status IN ('queued', 'running', 'success', 'failed')),
    summary JSON
);

CREATE INDEX IF NOT EXISTS idx_artifact_dataset ON meta.artifact(dataset_name, created_at);
CREATE INDEX IF NOT EXISTS idx_snapshot_status ON meta.dataset_snapshot(status, created_at);
CREATE INDEX IF NOT EXISTS idx_security_exchange ON ref.security(exchange, board);
CREATE INDEX IF NOT EXISTS idx_identifier_value ON ref.security_identifier_history(identifier_type, identifier_value);
CREATE INDEX IF NOT EXISTS idx_calendar_date ON ref.trading_calendar(calendar_date);
CREATE INDEX IF NOT EXISTS idx_universe_membership_dates ON research.universe_membership(universe_id, effective_from, effective_to);
CREATE INDEX IF NOT EXISTS idx_index_membership_dates ON ref.index_membership_history(index_id, effective_from, effective_to);

CREATE OR REPLACE MACRO market.read_daily_bar(file_pattern) AS TABLE
SELECT
    CAST(trade_date AS DATE) AS trade_date,
    CAST(instrument AS VARCHAR) AS instrument,
    CAST(open AS DOUBLE) AS open,
    CAST(high AS DOUBLE) AS high,
    CAST(low AS DOUBLE) AS low,
    CAST(close AS DOUBLE) AS close,
    CAST(volume AS DOUBLE) AS volume_shares,
    CAST(amount AS DOUBLE) AS amount_cny,
    CAST(adj_factor AS DOUBLE) AS adj_factor,
    CAST(suspend AS BOOLEAN) AS is_suspended,
    CAST(limit_up AS BOOLEAN) AS hit_limit_up,
    CAST(limit_down AS BOOLEAN) AS hit_limit_down,
    CAST(is_st AS BOOLEAN) AS is_st,
    CAST(list_date AS DATE) AS list_date,
    CAST(delist_date AS DATE) AS delist_date,
    CAST(source AS VARCHAR) AS source,
    CAST(ingested_at AS TIMESTAMPTZ) AS ingested_at
FROM read_parquet(file_pattern, union_by_name = true);

CREATE OR REPLACE MACRO market.read_adjustment_factor(file_pattern) AS TABLE
SELECT * FROM read_parquet(file_pattern, union_by_name = true);

CREATE OR REPLACE MACRO market.read_daily_basic(file_pattern) AS TABLE
SELECT * FROM read_parquet(file_pattern, union_by_name = true);

CREATE OR REPLACE MACRO market.read_daily_status(file_pattern) AS TABLE
SELECT * FROM read_parquet(file_pattern, union_by_name = true);

CREATE OR REPLACE MACRO fundamental.read_financial_fact(file_pattern) AS TABLE
SELECT * FROM read_parquet(file_pattern, union_by_name = true);

CREATE OR REPLACE MACRO research.read_factor_value(file_pattern) AS TABLE
SELECT * FROM read_parquet(file_pattern, union_by_name = true);

CREATE OR REPLACE MACRO research.read_backtest_daily(file_pattern) AS TABLE
SELECT * FROM read_parquet(file_pattern, union_by_name = true);
