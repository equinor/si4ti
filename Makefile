# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = simpli
SOURCEDIR     = .
BUILDDIR      = _build

PYTHON        = python

python:
	$(PYTHON) setup.py build

test:
	$(PYTHON) setup.py test

clean:
	rm -rf build "$(BUILDDIR)"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile python test clean doc

%:
	python

# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
doc: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
