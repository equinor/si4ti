# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = si4ti
SOURCEDIR     = .
BUILDDIR      = _build

PYTHON        = python

default: run view

python:
	$(PYTHON) -m pip install . --user --no-deps
	$(PYTHON) ~/.local/bin/ts \
		vint0.sgy \
		vint1.sgy \
		vint2.sgy \

test:
	$(PYTHON) setup.py test

clean:
	rm -rf build "$(BUILDDIR)"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile python test clean doc run view build

gdb: build
	gdb -ex=run --args build/timeshift/timeshift \
		-i 5 -x 21 \
		vint0.sgy \
		vint1.sgy \
		vint2.sgy \

valgrind: build
	valgrind build/timeshift/timeshift \
		-i 5 -x 21 \
		vint0.sgy \
		vint1.sgy \
		vint2.sgy \

build:
	cd build && make

ITER = 100

run: build
	build/timeshift/timeshift -i 5 -x 21 \
		--max-iter $(ITER) \
		vint0.sgy \
		vint1.sgy \
		vint2.sgy \

VIEW = timeshift-0.sgy

view:
	$(shell source /project/res/komodo/stable/enable \
	&& /project/res/komodo/stable/root/bin/python \
	/project/res/komodo/stable/root/bin/segyviewer -i 5 -x 21 \
	$(VIEW))

%:
	run view

# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
doc: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
