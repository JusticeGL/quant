COMPOSE := docker compose
RESEARCH_SERVICE := research
LOCK_SERVICE := locker
CONFIG_DIR ?= config
DATA_DIR ?= data
DATA_END_DATE_ARG = $(if $(END_DATE),--end-date $(END_DATE),)

.PHONY: build shell lock smoke lint test data-bootstrap data-update data-validate qlib-export db-init db-check baseline factor-list factor-eval

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

smoke:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.smoke

lint:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) sh -c 'ruff check . && ruff format --check . && mypy src'

test:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) pytest -q
