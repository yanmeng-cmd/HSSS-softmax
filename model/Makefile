SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

BOOTSTRAP_PYTHON ?= python3
MODEL_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
VENV_PYTHON := $(MODEL_ROOT).venv/bin/python
PYTHON ?= $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),$(BOOTSTRAP_PYTHON))

.PHONY: help bert longformer download package clean env

help:
	@$(PYTHON) "$(MODEL_ROOT)/run.py" help

bert longformer download package clean env:
	@if [ "$@" = "$(firstword $(MAKECMDGOALS))" ]; then \
		$(PYTHON) "$(MODEL_ROOT)/run.py" $@ $(filter-out $@,$(MAKECMDGOALS)); \
	fi

%:
	@:
