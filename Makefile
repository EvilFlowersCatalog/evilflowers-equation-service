PYTHON := python

install:
	pip install -r requirements.txt

run:
	$(PYTHON) src/main.py