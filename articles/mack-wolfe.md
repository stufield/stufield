# The Mack-Wolfe Test
Stu Field

# Mack-Wolfe Non-Parametric Peak Tests

Given an expected group ordering
(i.e. `Group A < Group B < Group C < Group D`), the Mack-Wolfe test
counts for *each* value in `Group A`, the number of values in
`Group B` that are greater in magnitude (+0.5 for ties), and repeats
this calculation for all *each* $n \choose 2$, $n \ge 2$, pairwise
combinations ($A-B$, $A-C$, $A-D$, $B-C$, $B-D$, $C-D$). In the
extreme, if there are 9 samples in `Group A` and 15 samples in
`Group B`, and *all* the samples in `Group B` are greater than the
highest value in `Group A`, this results in $9 \times 15 = 135$ for
that pairwise combination. The process is repeated and summed for the
other 5 combinations. See below for 4 different possible JT-test
scenarios (Mack-Wolfe with peak at end).

``` r
mack_fake_data <- function() {
  withr::local_seed(100)
  data.frame(
    equal_degree        = factor(rep(LETTERS[1:4L], each = 10L)),
    unequal_degree      = factor(rep(LETTERS[1:4L], c(5, 9, 16, 10))),
    equal_response      = c(rnorm(10, 10, 2), rnorm(10,10,2), 
                            rnorm(10, 10, 2), rnorm(10, 10, 2)),
    increasing_response = c(rnorm(10, 5, 2), rnorm(10, 10, 2),
                            rnorm(10,15,2), rnorm(10, 20 ,2)),
    dogleg_response     = c(rnorm(10, 5, 2), rnorm(10, 5, 2),
                            rnorm(10, 15, 2), rnorm(10, 20, 2)),
    dogleg_response2    = c(rnorm(5, 5, 2), rnorm(9, 2, 2),
                            rnorm(16, 15, 2), rnorm(10,20,2))
  )
}

test_mack_data <- mack_fake_data()
response_vec   <- c("equal_response", "increasing_response", "dogleg_response") |>
  helpr::set_Names()

mack_tests <- lapply(test_mack_data[, response_vec], function(.x) {
  libml::mack_wolfe(.x, test_mack_data$equal_degree, peak = "jt")
})

mack_tests$dogleg_response2 <- libml::mack_wolfe(
  test_mack_data$dogleg_response2, test_mack_data$unequal_degree, peak = "jt"
)

p <- lapply(response_vec, function(.p) {
   SomaPlotr::boxplotBeeswarm(
     split(test_mack_data[[.p]], test_mack_data$equal_degree),
           notch = FALSE, main = .p)
})

p[[4L]] <- SomaPlotr::boxplotBeeswarm(
  split(test_mack_data$dogleg_response2, test_mack_data$unequal_degree),
  notch = FALSE, main = "dogleg_response unequal groups")

p[[1L]] + p[[2L]] + p[[3L]] + p[[4L]]
```

![cap_4_plot](figures/mack-wolfe-four-scenario-beeswarm-1.png)

----------------------------------------------------------------------

For scenarios where there is an expected peak, the test statistic
($A_p$) is calculated as essentially the sum of two JT-tests, summing
over the *upward* side and *downward* side JT-tests.

For example, if **peak = Group C**,
i.e. `Group A < Group B < Group C > Group D`, two tests are calculated
and summed:

$$
\begin{equation}
  (Group\ A < Group\ B < Group\ C) + (Group\ C > Group\ D)
\end{equation}
$$

The significance is inferred typically using a large-sample
approximation to the Gaussian distribution ($A^*$), i.e. conversion to
a $Z_{score}$, from which an associated p-value can be calculated.

----------------------------------------------------------------------

# Session Info

<details class="code-fold">
<summary>Code</summary>

``` r
get_session_info()
#> $packages
#>  package      * version    date (UTC) lib source
#>  cellranger     1.1.0      2016-07-27 []  RSPM
#>  cli            3.6.6      2026-04-09 []  RSPM
#>  digest         0.6.39     2025-11-19 []  RSPM
#>  dplyr        * 1.2.1      2026-04-03 []  RSPM
#>  evaluate       1.0.5      2025-08-27 []  RSPM
#>  farver         2.1.2      2024-05-13 []  RSPM
#>  fastmap        1.2.0      2024-05-15 []  RSPM
#>  gbm            2.3.1      2026-07-09 []  RSPM
#>  generics       0.1.4      2025-05-09 []  RSPM
#>  ggplot2        4.0.3      2026-04-22 []  RSPM
#>  glue           1.8.1      2026-04-17 []  RSPM
#>  gtable         0.3.6      2024-10-25 []  RSPM
#>  helpr        * 0.0.2.9000 2026-08-19 []  Github (stufield/helpr@db72926)
#>  htmltools      0.5.9      2025-12-04 []  RSPM
#>  igraph         2.3.3      2026-06-26 []  RSPM
#>  jsonlite       2.0.0      2025-03-27 []  RSPM
#>  kknn           1.4.1      2025-05-19 []  any (@1.4.1)
#>  knitr          1.51       2025-12-20 []  any (@1.51)
#>  labeling       0.4.3      2023-08-29 []  RSPM
#>  lattice        0.22-9     2026-02-09 []  CRAN (R 4.6.1)
#>  libml        * 0.0.1.9000 2026-08-31 []  Github (stufield/libml@fddfa9e)
#>  lifecycle      1.0.5      2026-01-08 []  RSPM
#>  magrittr       2.0.5      2026-04-04 []  RSPM
#>  Matrix         1.7-5      2026-03-21 []  CRAN (R 4.6.1)
#>  otel           0.2.0      2025-08-29 []  RSPM
#>  patchwork    * 1.3.2      2025-08-25 []  any (@1.3.2)
#>  pillar         1.11.1     2025-09-17 []  RSPM
#>  pkgconfig      2.0.3      2019-09-22 []  RSPM
#>  purrr          1.2.2      2026-04-10 []  RSPM
#>  R6             2.6.1      2025-02-15 []  RSPM
#>  RColorBrewer   1.1-3      2022-04-03 []  RSPM
#>  Rcpp           1.1.2      2026-07-05 []  RSPM
#>  readxl         1.5.0      2026-05-16 []  RSPM
#>  rlang          1.3.0      2026-07-05 []  RSPM
#>  rmarkdown      2.31       2026-03-26 []  RSPM
#>  S7             0.2.2      2026-04-22 []  RSPM
#>  scales         1.4.0      2025-04-24 []  RSPM
#>  sessioninfo    1.2.4      2026-06-04 []  any (@1.2.4)
#>  SomaDataIO     6.6.1      2026-05-15 []  RSPM
#>  SomaPlotr    * 0.0.1      2026-08-19 []  Github (stufield/SomaPlotr@246a626)
#>  survival       3.8-6      2026-01-16 []  CRAN (R 4.6.1)
#>  tibble         3.3.1      2026-01-11 []  any (@3.3.1)
#>  tidyr          1.3.2      2025-12-19 []  any (@1.3.2)
#>  tidyselect     1.2.1      2024-03-11 []  RSPM
#>  vctrs          0.7.3      2026-04-11 []  RSPM
#>  withr          3.0.3      2026-06-19 []  RSPM
#>  wranglr      * 0.0.2.9000 2026-08-19 []  Github (stufield/wranglr@cd4c5f4)
#>  xfun           0.60       2026-07-09 []  RSPM
#>  yaml           2.3.12     2025-12-10 []  RSPM
#> 
#>  * ── Packages attached to the search path.
#> 
#> $platform
#>  setting  value
#>  version  R version 4.6.1 (2026-06-24)
#>  os       Ubuntu 24.04.4 LTS
#>  system   x86_64, linux-gnu
#>  ui       X11
#>  language (EN)
#>  collate  C.UTF-8
#>  ctype    C.UTF-8
#>  tz       UTC
#>  pandoc   3.1.3 @ /usr/bin/ (via rmarkdown)
#>  quarto   1.10.18 @ /usr/local/bin/quarto
```

</details>
