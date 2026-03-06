.PHONY: handoff handoff-next handoff-validate

ARGS ?= next

# Usage:
#   make handoff ARGS="next"
#   make handoff ARGS="claim --agent Codex"
#   make handoff ARGS="log"
#   make handoff ARGS="close --owner Human"

handoff:
	python3 scripts/handoff.py $(ARGS)

handoff-next:
	python3 scripts/handoff.py next

handoff-validate:
	python3 scripts/handoff.py validate
