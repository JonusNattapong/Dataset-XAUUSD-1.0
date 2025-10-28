# Makefile for XAUUSD Research Paper

PAPER = XAUUSD_Research_Paper
TEX_FILE = $(PAPER).tex
PDF_FILE = $(PAPER).pdf
BIB_FILE = references.bib

# Default target
all: $(PDF_FILE)

# Compile PDF from LaTeX
$(PDF_FILE): $(TEX_FILE) $(BIB_FILE)
	pdflatex $(TEX_FILE)
	bibtex $(PAPER)
	pdflatex $(TEX_FILE)
	pdflatex $(TEX_FILE)

# Quick compile without bibliography
quick: $(TEX_FILE)
	pdflatex $(TEX_FILE)

# Clean auxiliary files
clean:
	del /Q *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fdb_latexmk *.fls

# Clean all generated files including PDF
cleanall: clean
	del /Q $(PDF_FILE)

# View PDF (requires PDF viewer)
view: $(PDF_FILE)
	start $(PDF_FILE)

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Compile full PDF with bibliography"
	@echo "  quick    - Quick compile without bibliography"
	@echo "  clean    - Remove auxiliary files"
	@echo "  cleanall - Remove all generated files"
	@echo "  view     - Open PDF in default viewer"
	@echo "  help     - Show this help message"

.PHONY: all quick clean cleanall view help