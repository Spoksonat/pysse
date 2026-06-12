# Minimal Sphinx Makefile for local documentation builds.

SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean

help:
	@echo "Usage:"
	@echo "  make html   Build HTML documentation in _build/html"
	@echo "  make clean  Remove the build directory"

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
