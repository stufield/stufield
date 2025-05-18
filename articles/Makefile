# ----------------------------------- #
# GNU Make Automatic Variables:
# ----------------------------------- #
# $*: stem of target
# $@: filename of target 
# $%: target member name
# $<: the first prerequisites for target
# $^: prerequisites for target
# $?: prerequisites newer than target
#######################
# Local Variables:
#######################
MV = mv -f
RM = rm -rf
CP = cp -f
TOUCH = touch
PDFLATEX = pdflatex
RSCRIPT = Rscript --vanilla
#########################
.PRECIOUS: %.tex Makefile
.PHONY: clean
#########################

all: quarto clean

quarto:
	@ echo "Rendering Rmarkdown *.md files"
	@ for i in *.[Qq]md; do quarto render $$i --to default; done
	@ for i in *[.]md; do sed -i '1,2G' $$i; done

%.pdf: %.tex
	@ $(PDFLATEX) $*
	@ $(PDFLATEX) $*

clean:
	@ $(RM) *.log *.aux *.out *.html

