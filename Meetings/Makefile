# Generate PDFs from the Markdown source files
#
# In order to use this makefile, you need some tools:
# - GNU make
# - Pandoc
# - LuaLaTeX
# - DejaVu Sans fonts

# Convert the list of source files (Markdown files in directory src/)
# into a list of output files (PDFs in directory print/).
objects := $(patsubst %.md,%.pdf,$(wildcard *.md))

all: $(objects)

# Recipe for converting a Markdown file into PDF using Pandoc
%.pdf : %.md
	pandoc \
		--variable mainfont="DejaVu Sans" \
		--variable fontsize=12pt \
		--variable geometry:"top=1.5cm, bottom=2.5cm, left=1.5cm, right=1.5cm" \
		--variable geometry:a4paper \
		-f markdown_github  $< \
		--latex-engine=xelatex \
		-o $@

.PHONY : clean

clean:
	rm -f $(output)/*.pdf
