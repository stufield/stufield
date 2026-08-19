# ----------------------------------- #
# GNU Make Automatic Variables:
# ----------------------------------- #
# $*: stem of target
# $@: filename of target
# $<: the first prerequisites for target
# $^: prerequisites for target
# $?: prerequisites newer than target

MV = mv -f
RM = rm -rf
CP = cp -f
SED = sed

.PRECIOUS: %.tex Makefile
.PHONY: all clean

QMD := $(wildcard *.[Qq]md)
MD  := $(patsubst %.qmd,%.md,$(patsubst %.Qmd,%.md,$(QMD)))

all: $(MD)

define render
	@ echo "rendering $<"
	@ quarto render $< --to gfm
	#@ $(SED) -i.bak '1,2G' $@ && $(RM) $@.bak
endef

%.md: %.Qmd
	$(render)

%.md: %.qmd
	$(render)

clean:
	@ $(RM) *.log *.aux *.out *.html *.bak
