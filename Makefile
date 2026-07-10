COMPOSE := docker compose
RESEARCH_SERVICE := research
LOCK_SERVICE := locker

.PHONY: build shell lock smoke lint test

build:
	$(COMPOSE) build $(RESEARCH_SERVICE)

shell:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) bash

lock:
	$(COMPOSE) run --rm --build $(LOCK_SERVICE) uv lock

smoke:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) python -m alpha_lab.smoke

lint:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) sh -c 'ruff check . && ruff format --check . && mypy src'

test:
	$(COMPOSE) run --rm --build $(RESEARCH_SERVICE) pytest -q
