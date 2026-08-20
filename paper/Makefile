.PHONY: paper figures clean

paper:
	latexmk -pdf kinetic_ai.tex

figures:
	python3 figures/make_paper_figures.py

clean:
	latexmk -C
	rm -f *.auxlock
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
