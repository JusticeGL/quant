CREATE TABLE IF NOT EXISTS ref.industry_definition (
    definition_id VARCHAR PRIMARY KEY
        CHECK (regexp_full_match(definition_id, '[0-9a-f]{64}')),
    exposure_snapshot_id VARCHAR NOT NULL,
    industry_id VARCHAR NOT NULL,
    source_index_code VARCHAR NOT NULL,
    industry_code VARCHAR,
    industry_name VARCHAR NOT NULL,
    level VARCHAR NOT NULL CHECK (level IN ('L1', 'L2', 'L3')),
    classification_standard VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    source_artifact_id VARCHAR NOT NULL
        CHECK (regexp_full_match(source_artifact_id, '[0-9a-f]{64}')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(exposure_snapshot_id, industry_id)
);

CREATE TABLE IF NOT EXISTS ref.industry_membership_history (
    membership_id VARCHAR PRIMARY KEY
        CHECK (regexp_full_match(membership_id, '[0-9a-f]{64}')),
    exposure_snapshot_id VARCHAR NOT NULL,
    definition_id VARCHAR NOT NULL REFERENCES ref.industry_definition(definition_id),
    security_id VARCHAR NOT NULL REFERENCES ref.security(security_id),
    effective_from DATE NOT NULL,
    effective_to DATE,
    announced_at TIMESTAMPTZ,
    known_at TIMESTAMPTZ NOT NULL,
    known_at_source VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    source_artifact_id VARCHAR NOT NULL
        CHECK (regexp_full_match(source_artifact_id, '[0-9a-f]{64}')),
    CHECK (effective_to IS NULL OR effective_to >= effective_from),
    UNIQUE(exposure_snapshot_id, security_id, definition_id, effective_from)
);

CREATE TABLE IF NOT EXISTS research.factor_freeze (
    freeze_id VARCHAR PRIMARY KEY,
    freeze_sha256 VARCHAR NOT NULL UNIQUE
        CHECK (regexp_full_match(freeze_sha256, '[0-9a-f]{64}')),
    factor_version_id VARCHAR NOT NULL
        REFERENCES research.factor_version(factor_version_id),
    phase5_snapshot_id VARCHAR NOT NULL,
    exposure_snapshot_id VARCHAR NOT NULL,
    robustness_policy_sha256 VARCHAR NOT NULL
        CHECK (regexp_full_match(robustness_policy_sha256, '[0-9a-f]{64}')),
    cost_policy_sha256 VARCHAR NOT NULL
        CHECK (regexp_full_match(cost_policy_sha256, '[0-9a-f]{64}')),
    code_commit VARCHAR NOT NULL,
    test_start DATE NOT NULL,
    test_end DATE NOT NULL,
    manifest_artifact_id VARCHAR NOT NULL
        CHECK (regexp_full_match(manifest_artifact_id, '[0-9a-f]{64}')),
    status VARCHAR NOT NULL DEFAULT 'frozen' CHECK (status = 'frozen'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    CHECK (test_end >= test_start),
    UNIQUE(freeze_id, freeze_sha256, test_start, test_end)
);

CREATE TABLE IF NOT EXISTS research.test_request (
    request_id VARCHAR PRIMARY KEY,
    request_sha256 VARCHAR NOT NULL UNIQUE
        CHECK (regexp_full_match(request_sha256, '[0-9a-f]{64}')),
    freeze_id VARCHAR NOT NULL,
    freeze_sha256 VARCHAR NOT NULL
        CHECK (regexp_full_match(freeze_sha256, '[0-9a-f]{64}')),
    robustness_report_sha256 VARCHAR NOT NULL
        CHECK (regexp_full_match(robustness_report_sha256, '[0-9a-f]{64}')),
    test_start DATE NOT NULL,
    test_end DATE NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'test_requested'
        CHECK (status = 'test_requested'),
    requested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    CHECK (test_end >= test_start),
    FOREIGN KEY (freeze_id, freeze_sha256, test_start, test_end)
        REFERENCES research.factor_freeze(
            freeze_id, freeze_sha256, test_start, test_end
        ),
    UNIQUE(request_id, freeze_id, freeze_sha256, test_start, test_end)
);

CREATE TABLE IF NOT EXISTS research.test_approval (
    approval_id VARCHAR PRIMARY KEY,
    approval_sha256 VARCHAR NOT NULL UNIQUE
        CHECK (regexp_full_match(approval_sha256, '[0-9a-f]{64}')),
    request_id VARCHAR NOT NULL,
    freeze_id VARCHAR NOT NULL,
    confirmed_freeze_sha256 VARCHAR NOT NULL
        CHECK (regexp_full_match(confirmed_freeze_sha256, '[0-9a-f]{64}')),
    test_start DATE NOT NULL,
    test_end DATE NOT NULL,
    approver VARCHAR NOT NULL CHECK (length(trim(approver)) > 0),
    status VARCHAR NOT NULL DEFAULT 'approved' CHECK (status = 'approved'),
    approved_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    CHECK (test_end >= test_start),
    FOREIGN KEY (
        request_id, freeze_id, confirmed_freeze_sha256, test_start, test_end
    ) REFERENCES research.test_request(
        request_id, freeze_id, freeze_sha256, test_start, test_end
    ),
    UNIQUE(
        approval_id, request_id, freeze_id, confirmed_freeze_sha256,
        test_start, test_end
    )
);

CREATE TABLE IF NOT EXISTS research.final_test_run (
    test_run_id VARCHAR PRIMARY KEY,
    run_sha256 VARCHAR NOT NULL UNIQUE
        CHECK (regexp_full_match(run_sha256, '[0-9a-f]{64}')),
    approval_id VARCHAR NOT NULL,
    request_id VARCHAR NOT NULL,
    freeze_id VARCHAR NOT NULL,
    freeze_sha256 VARCHAR NOT NULL
        CHECK (regexp_full_match(freeze_sha256, '[0-9a-f]{64}')),
    result_artifact_id VARCHAR NOT NULL
        CHECK (regexp_full_match(result_artifact_id, '[0-9a-f]{64}')),
    report_artifact_id VARCHAR NOT NULL
        CHECK (regexp_full_match(report_artifact_id, '[0-9a-f]{64}')),
    status VARCHAR NOT NULL CHECK (status IN ('success', 'failed')),
    test_start DATE NOT NULL,
    test_end DATE NOT NULL,
    summary JSON,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL,
    CHECK (test_end >= test_start),
    CHECK (finished_at >= started_at),
    FOREIGN KEY (
        approval_id, request_id, freeze_id, freeze_sha256, test_start, test_end
    ) REFERENCES research.test_approval(
        approval_id, request_id, freeze_id, confirmed_freeze_sha256,
        test_start, test_end
    )
);

CREATE INDEX IF NOT EXISTS idx_industry_definition_snapshot
ON ref.industry_definition(exposure_snapshot_id, industry_id);

CREATE INDEX IF NOT EXISTS idx_industry_membership_dates
ON ref.industry_membership_history(
    security_id, effective_from, effective_to, known_at
);

CREATE INDEX IF NOT EXISTS idx_factor_freeze_factor
ON research.factor_freeze(factor_version_id, created_at);

CREATE INDEX IF NOT EXISTS idx_test_request_freeze
ON research.test_request(freeze_id, requested_at);

CREATE INDEX IF NOT EXISTS idx_test_approval_request
ON research.test_approval(request_id, approved_at);

CREATE INDEX IF NOT EXISTS idx_final_test_approval
ON research.final_test_run(approval_id, started_at);
