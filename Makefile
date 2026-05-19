# Local developer shortcuts. Keep targets self-explanatory and idempotent.
.PHONY: help refresh-seed test test-fast install

help:
	@echo "apple-notes-brain Makefile targets:"
	@echo "  make install       — uv sync --extra dev --extra semantic"
	@echo "  make test          — full pytest suite (1670+ cases)"
	@echo "  make test-fast     — drop slow + property tests"
	@echo "  make refresh-seed  — regenerate src/apple_notes_brain/data/seed-models.json"
	@echo "                       from the MTEB Python registry (idempotent)"

install:
	uv sync --extra dev --extra semantic

test:
	uv run pytest tests/ -m "not live" -q

test-fast:
	uv run pytest tests/ -m "not live and not slow and not property" -q

# Refresh the bundled MTEB-derived seed metadata. Lives in its own venv
# (~/.venv-anb-mteb) so its heavyweight torch/scipy/mteb deps don't
# pollute the project's .venv. Creates the venv on first run, reuses
# thereafter. CI mirrors this pattern via the `warm-mteb-venv` job in
# .github/workflows/ci.yml — both paths produce bit-identical output
# given the same MTEB version pin.
refresh-seed:
	@if [ ! -d "$$HOME/.venv-anb-mteb" ]; then \
	  echo "creating MTEB venv at ~/.venv-anb-mteb (one-time ~60s install)..."; \
	  python3 -m venv "$$HOME/.venv-anb-mteb"; \
	  "$$HOME/.venv-anb-mteb/bin/pip" install --upgrade pip > /dev/null; \
	  "$$HOME/.venv-anb-mteb/bin/pip" install -r scripts/requirements-build-seed.txt; \
	fi
	"$$HOME/.venv-anb-mteb/bin/python" scripts/build-seed.py
	@echo "if 'git diff' shows changes, commit the regenerated seed."
