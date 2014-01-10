all:
	pdflatex compress
	pdflatex compress
	bibtex compress
	bibtex compress
	pdflatex compress
	pdflatex compress
	open compress.pdf

clean:
	rm *.aux *.brf *.pdf *.log *.bbl *.blg *.dvi *.ps *.out

pdfcrop:
	pdfcrop img/conv.pdf
