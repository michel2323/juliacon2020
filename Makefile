
PAPER_SRC=paper.md 
PAPER_DST=paper.pdf 

paper: 
	pandoc --filter pandoc-citeproc -s $(PAPER_SRC) -o $(PAPER_DST)

