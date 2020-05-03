COVERAGE_OUTPUT = coverage
COVERAGE_OPTIONS = --cov-config coverage/.coveragerc --cov-report term --cov-report html

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements.dev.txt

test: install-dev
	pytest tests/

coverage: install-dev
	py.test $(COVERAGE_OPTIONS) --cov=eml tests/ | tee coverage/coverage.txt
	mv -f .coverage coverage/.coverage
	mkdir -p $(COVERAGE_OUTPUT)/notebooks
	cp -r coverage/html/* $(COVERAGE_OUTPUT)/notebooks
