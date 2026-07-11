CREATE TABLE IF NOT EXISTS meta.provider_capability (
    capability_id VARCHAR PRIMARY KEY,
    source_id VARCHAR NOT NULL REFERENCES meta.data_source(source_id),
    api_name VARCHAR NOT NULL,
    available BOOLEAN NOT NULL,
    request_params_sha256 VARCHAR NOT NULL,
    requested_fields JSON NOT NULL,
    returned_fields JSON,
    error_type VARCHAR,
    details JSON,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS ref.security_name_history (
    name_history_id VARCHAR PRIMARY KEY,
    security_id VARCHAR NOT NULL REFERENCES ref.security(security_id),
    security_name VARCHAR NOT NULL,
    is_st BOOLEAN,
    effective_from DATE NOT NULL,
    effective_to DATE,
    announced_at TIMESTAMPTZ,
    known_at TIMESTAMPTZ NOT NULL,
    source_artifact_id VARCHAR,
    CHECK (effective_to IS NULL OR effective_to >= effective_from),
    UNIQUE(security_id, effective_from, security_name)
);

ALTER TABLE ref.index_membership_history
ADD COLUMN IF NOT EXISTS known_at TIMESTAMPTZ;

ALTER TABLE ref.index_membership_history
ADD COLUMN IF NOT EXISTS membership_method VARCHAR;

CREATE INDEX IF NOT EXISTS idx_security_name_effective
ON ref.security_name_history(security_id, effective_from, effective_to);

CREATE INDEX IF NOT EXISTS idx_provider_capability_api
ON meta.provider_capability(source_id, api_name, checked_at);

CREATE OR REPLACE MACRO ref.read_security_master(file_pattern) AS TABLE
SELECT * FROM read_parquet(file_pattern, union_by_name = true);

CREATE OR REPLACE MACRO ref.read_security_name_history(file_pattern) AS TABLE
SELECT * FROM read_parquet(file_pattern, union_by_name = true);

CREATE OR REPLACE MACRO ref.read_index_membership(file_pattern) AS TABLE
SELECT * FROM read_parquet(file_pattern, union_by_name = true);

CREATE OR REPLACE MACRO research.read_universe_dates(file_pattern) AS TABLE
SELECT * FROM read_parquet(file_pattern, union_by_name = true);
