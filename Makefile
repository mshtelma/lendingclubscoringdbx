env:
	rm -rf .venv && python -m venv .venv && source .venv/bin/activate && pip install -r unit-requirements.txt && pip install -e .

test:
	pytest tests/unit

job:
	dbx launch --job=lendingclub_scoring_dbx-sample-integration-test --trace

clean:
	rm -rf *.egg-info && rm -rf .pytest_cache