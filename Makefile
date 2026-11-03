install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt pytest

test:
	PYTHONPATH=src python -m pytest tests

worker:
	cd src && celery -A worker worker --loglevel=info -E --concurrency=1 --queues=evilflowers_equation
