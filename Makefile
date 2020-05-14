COVERAGE_OUTPUT = coverage
COVERAGE_OPTIONS = --cov-config coverage/.coveragerc --cov-report term --cov-report html
AUTODOC_OPTIONS = -d 1 --no-toc --separate --force --private

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements.dev.txt

unittest:
	pytest tests/units/${TESTPATH}

test: install-dev
	pytest tests/

coverage: install-dev
	py.test $(COVERAGE_OPTIONS) --cov=eml tests/ | tee coverage/coverage.txt
	mv -f .coverage coverage/.coverage
	mkdir -p $(COVERAGE_OUTPUT)/notebooks
	cp -r coverage/html/* $(COVERAGE_OUTPUT)/notebooks

init-doc: install-dev
	sphinx-quickstart doc/

doc: install-dev
	rm -rf doc/source/generated/
	sphinx-apidoc $(AUTODOC_OPTIONS) -o doc/source/generated/ eml eml/__version__.py eml/dev/
	cd doc; make html

