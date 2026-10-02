JULIA ?= julia
TEST_ENV_RUN = JULIA="$(JULIA)" ./test/Quality/run-test-env.sh
TARGET ?= .
FILE ?=

.PHONY: quality quality-target quality-aqua quality-jet quality-staticlint quality-formatter fmt fmt-check fmt-file

quality: quality-aqua quality-jet quality-staticlint quality-formatter

quality-target: quality

quality-aqua:
	$(TEST_ENV_RUN) test/Quality/aqua.jl "$(TARGET)"

quality-jet:
	$(TEST_ENV_RUN) test/Quality/jet.jl "$(TARGET)"

quality-staticlint:
	$(TEST_ENV_RUN) test/Quality/staticlint.jl "$(TARGET)"

quality-formatter:
	$(TEST_ENV_RUN) test/Quality/formatter.jl check "$(TARGET)"

fmt:
	$(TEST_ENV_RUN) test/Quality/formatter.jl write "$(TARGET)"

fmt-check:
	$(TEST_ENV_RUN) test/Quality/formatter.jl check "$(TARGET)"

fmt-file:
	@if [ -z "$(FILE)" ]; then echo "Usage: make fmt-file FILE=path/to/file.jl"; exit 2; fi
	$(TEST_ENV_RUN) test/Quality/formatter.jl write "$(FILE)"
