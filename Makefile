.PHONY: paper site clean

# Build paper (PDF via LaTeX)
paper:
	$(MAKE) -C paper paper

# Build paper figures (requires validated findings)
figures:
	$(MAKE) -C paper figures

# Build site (Astro static site)
site:
	cd site && npm run build

# Dev mode: paper + site watch
dev:
	@echo "Paper (LaTeX): make -C paper paper"
	@echo "Site (Astro): cd site && npm run dev"

# Clean all build artifacts
clean:
	$(MAKE) -C paper clean
	cd site && rm -rf dist/ .astro/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
