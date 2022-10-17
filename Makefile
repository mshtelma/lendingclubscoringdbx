env:
	rm -rf .venv && python -m venv .venv && source .venv/bin/activate && pip install --upgrade pip &&  pip install -e .\[test]

test:
	pytest tests/unit

job:
	dbx launch --job=lendingclub_scoring_dbx-sample-integration-test --trace

clean:
	rm -rf *.egg-info && rm -rf .pytest_cache

format:
	black .

lint:
	prospector   --profile prospector.yaml && black --check lendingclub_scoring
