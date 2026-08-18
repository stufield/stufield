#!/usr/bin/env Rscript
# Rebuild the article grid in README.md from articles/index.csv.
#
# The manifest is deliberately hand-maintained rather than scraped.
# Article H1 headings do not match the display names used here (e.g.
# "Mind Your P's & Q's" is listed as "False Discovery"), and the
# thumbnails are hand-picked rather than each file's first figure.
# Scraping would silently discard both choices.
#
# This script only handles layout, so changing the column count is a
# one-argument edit instead of rewriting every row.
#
# Usage:
#   Rscript scripts/build-articles.R          # 4 cols, 190px
#   Rscript scripts/build-articles.R 5        # 5 cols
#   Rscript scripts/build-articles.R 5 150    # 5 cols, 150px thumbs
#
# With 11 articles, 4 and 5 columns both produce 3 rows, so 4 columns
# buys larger thumbnails at no cost in vertical space.

readme <- "README.md"
manifest <- "articles/index.csv"
start_tag <- "<!-- ARTICLES:START -->"
end_tag <- "<!-- ARTICLES:END -->"

# GitHub renders README content in a column roughly this wide. Past it
# the browser shrinks cells or the table overflows, so warn rather
# than silently produce a grid that looks wrong only once pushed.
max_render_px <- 880L


# Build ------

read_manifest <- function(path) {
  items <- read.csv(path, colClasses = "character")
  missing <- !file.exists(c(items$path, items$figure))
  if (any(missing)) {
    stop(
      "Manifest references files that do not exist:\n  ",
      paste(c(items$path, items$figure)[missing], collapse = "\n  "),
      call. = FALSE
    )
  }
  items
}

# One <td> per article: thumbnail above the caption. Putting the title
# under the image lets it use the full cell width, so long names like
# "The Birthday Paradox" do not wrap mid-grid.
build_cell <- function(title, path, figure, width) {
  sprintf(
    paste0(
      '<td align="center" width="%d">',
      '<a href="%s"><img src="%s" width="%d"></a>',
      "<br><sub><b>%s</b></sub></td>"
    ),
    width + 20L, path, figure, width, title
  )
}

build_grid <- function(items, ncol = 4L, width = 190L) {
  cells <- build_cell(items$title, items$path, items$figure, width)
  starts <- seq(1L, length(cells), by = ncol)
  rows <- lapply(starts, function(.i) {
    chunk <- cells[.i:min(.i + ncol - 1L, length(cells))]
    c("<tr>", chunk, "</tr>")
  })
  c("<table>", unlist(rows), "</table>")
}


# Splice ------

# Replaces only the marked region, so the rest of README.md is never
# touched. Errors loudly if the markers are absent or out of order,
# rather than appending a duplicate grid.
splice_block <- function(lines, body, start_tag, end_tag) {
  i <- match(start_tag, trimws(lines))
  j <- match(end_tag, trimws(lines))
  if (is.na(i) || is.na(j)) {
    stop("Markers not found in ", readme, call. = FALSE)
  }
  if (j <= i) {
    stop("End marker precedes start marker.", call. = FALSE)
  }
  c(lines[1:i], body, lines[j:length(lines)])
}


# Main ------

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  ncol <- if (length(args) > 0L) {
    suppressWarnings(as.integer(args[1L]))
  } else {
    4L
  }
  width <- if (length(args) > 1L) {
    suppressWarnings(as.integer(args[2L]))
  } else {
    190L
  }
  if (is.na(ncol) || ncol < 1L) {
    stop("Column count must be a positive integer.", call. = FALSE)
  }
  if (is.na(width) || width < 1L) {
    stop("Thumbnail width must be a positive integer.", call. = FALSE)
  }
  items <- read_manifest(manifest)
  lines <- readLines(readme, warn = FALSE)
  grid <- build_grid(items, ncol = ncol, width = width)
  writeLines(splice_block(lines, grid, start_tag, end_tag), readme)
  total <- ncol * (width + 20L)
  if (total > max_render_px) {
    warning(
      sprintf(
        "Grid is %dpx wide; GitHub renders ~%dpx. Reduce columns or width.",
        total, max_render_px
      ),
      call. = FALSE
    )
  }
  message(sprintf(
    "%s: %d articles, %d cols x %dpx = %dpx wide (%d rows)",
    readme, nrow(items), ncol, width, total,
    ceiling(nrow(items) / ncol)
  ))
}

# Runs under Rscript but not when sourced, keeping the functions
# testable in isolation.
if (sys.nframe() == 0L) {
  main()
}
