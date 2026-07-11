COMPOSE := docker compose
RESEARCH_SERVICE := research
DATA_SERVICE := data
LOCK_SERVICE := locker
CONFIG_DIR ?= config
DATA_DIR ?= data
EXPERIMENTS_DIR ?= experiments
ARTIFACTS_DIR ?= artifacts
ROUNDS ?= 5
DATA_END_DATE_ARG = $(if $(END_DATE),--end-date $(END_DATE),)
PROPOSAL_ARG = $(if $(PROPOSAL),--proposal $(PROPOSAL),)
PROPOSALS_DIR_ARG = $(if $(PROPOSALS_DIR),--proposals-dir $(PROPOSALS_DIR),)

.PHONY: build shell lock smoke lint test data-bootstrap data-update data-validate qlib-export research-data-probe research-data-bootstrap research-data-update research-data-validate universe-asof db-init db-check baseline factor-list factor-eval mining-round mining-loop report

build:
	$(COMPOSE) build $(RESEARCH_SERVICE)

shell:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) bash

lock:
	$(COMPOSE) run --rm --build $(LOCK_SERVICE) uv lock

data-bootstrap:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.cli data-bootstrap --config-dir $(CONFIG_DIR) --data-dir $(DATA_DIR)

data-update:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.cli data-update --config-dir $(CONFIG_DIR) --data-dir $(DATA_DIR) $(DATA_END_DATE_ARG)

data-validate:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.cli data-validate --data-dir $(DATA_DIR) $(if $(SNAPSHOT),--snapshot $(SNAPSHOT),)

qlib-export:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.cli qlib-export --data-dir $(DATA_DIR) $(if $(SNAPSHOT),--snapshot $(SNAPSHOT),)

research-data-probe:
	$(COMPOSE) run --rm --build $(DATA_SERVICE) python -m alpha_lab.cli research-data-probe --config-dir $(CONFIG_DIR) --data-dir $(DATA_DIR)

research-data-bootstrap:
	$(COMPOSE) run --rm --build $(DATA_SERVICE) python -m alpha_lab.cli research-data-bootstrap --config-dir $(CONFIG_DIR) --data-dir $(DATA_DIR)

research-data-update:
	$(COMPOSE) run --rm --build $(DATA_SERVICE) python -m alpha_lab.cli research-data-update --end-date $(END_DATE) --config-dir $(CONFIG_DIR) --data-dir $(DATA_DIR)

research-data-validate:
	$(COMPOSE) run --rm --build $(DATA_SERVICE) python -m alpha_lab.cli research-data-validate --data-dir $(DATA_DIR) $(if $(SNAPSHOT),--snapshot $(SNAPSHOT),)

universe-asof:
	$(COMPOSE) run --rm --build $(DATA_SERVICE) python -m alpha_lab.cli universe-asof --date $(DATE) --data-dir $(DATA_DIR) $(if $(SNAPSHOT),--snapshot $(SNAPSHOT),)

db-init:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.cli db-init --config-dir $(CONFIG_DIR) --data-dir $(DATA_DIR)

db-check:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.cli db-check --data-dir $(DATA_DIR)

baseline:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.cli baseline --config-dir $(CONFIG_DIR) --data-dir $(DATA_DIR) $(if $(SNAPSHOT),--snapshot $(SNAPSHOT),)

factor-list:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.cli factor-list --config-dir $(CONFIG_DIR)

factor-eval:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.cli factor-eval --id $(ID) --config-dir $(CONFIG_DIR) --data-dir $(DATA_DIR) $(if $(SNAPSHOT),--snapshot $(SNAPSHOT),)

mining-round:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.cli mining-round --run $(RUN) --config-dir $(CONFIG_DIR) --data-dir $(DATA_DIR) --experiments-dir $(EXPERIMENTS_DIR) --artifacts-dir $(ARTIFACTS_DIR) $(PROPOSAL_ARG)

mining-loop:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.cli mining-loop --run $(RUN) --rounds $(ROUNDS) --config-dir $(CONFIG_DIR) --data-dir $(DATA_DIR) --experiments-dir $(EXPERIMENTS_DIR) --artifacts-dir $(ARTIFACTS_DIR) $(PROPOSALS_DIR_ARG)

report:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.cli mining-report --run $(RUN) --experiments-dir $(EXPERIMENTS_DIR)

smoke:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.smoke

lint:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) sh -c 'ruff check . && ruff format --check . && mypy src'

test:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) pytest -q
