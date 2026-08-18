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

# Quarto's gfm writer emits title, author and date on consecutive
# lines, which GitHub collapses into a single run-together paragraph.
# Appending a blank line after each of the first two separates them.
#
# `-i.bak` is the only in-place form portable across BSD sed (macOS)
# and GNU sed (CI runners); a bare `-i` fails on one or the other.

define render
	@ echo "rendering $<"
	@ quarto render $< --to gfm
	@ $(SED) -i.bak '1,2G' $@ && $(RM) $@.bak
endef

%.md: %.Qmd
	$(render)

%.md: %.qmd
	$(render)

clean:
	@ $(RM) *.log *.aux *.out *.html *.bak
