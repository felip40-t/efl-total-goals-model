PY := .venv/bin/python

# make <name>  ->  runs src/<name>.py
%: src/%.py
	$(PY) $<
