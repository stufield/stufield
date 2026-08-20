# The Birthday Paradox
Stu Field

# The Birthday Paradox

In probability theory, the birthday problem asks for the probability
that, in a set of n randomly chosen people, at least two will share a
birthday. The birthday paradox refers to the counter-intuitive fact
that only 23 people are needed for that probability to exceed 50%.

This is the [Birthday
Paradox](https://en.wikipedia.org/wiki/Birthday_problem).

## Summary

- you take 23 random people
- what is the probability that any two birthdays occur on the same
  day?
- To answer, first calculate the probability that all 23 birthdays
  occur on *different* days ($P(x)$), then the two are mutually
  exclusive, $P(x') = 1 - P(x)$ is the probability that all birthdays
  are *not* on different days

which becomes,

$$
P(x) = \frac{365}{365} \times \frac{364}{365} \times \frac{363}{365} \times\ ...\ \times \frac{343}{365}
$$

which can be simplified to,

$$
P(x) = \bigg(\frac{1}{365}\bigg)^{23} \times \big(365 \times 364 \times\ ...\ \times 343\big)
$$

This can be represented in `R` like so:

``` r
px <- (1 / 365)^23 * prod(365:343)
px
#> [1] 0.4927028

# probability of NOT all different
(1 - px)
#> [1] 0.5072972
```

We can next write a generalized function for this “duplicated”
probability representing *any* number of people for any number of days
in a year (leap year?) and call it `prob_fun()`.

``` r
# x   = number of people (trials), can be vectorized
# dpy = days/year

prob_fun <- function(x, dpy = 365L) {
  x <- as.integer(x)
  vapply(x, function(.x) {
    v <- seq(dpy, dpy - .x + 1L)
    1 - (1 / dpy)^.x * prod(v)
  }, NA_real_, USE.NAMES = FALSE)
}

ppl <- 23

prob_fun(ppl)   # confirms the above
#> [1] 0.5072972
```

## Visualize `n` People

This can be generalized across an arbitrary number of people to get a
overall view of how probability changes as a function of the
opportunity for duplicate birthdays. Using the `prob_fun()` function
above, the red-dashed line represents the $n =$ 23 days described in
classic example. The cyan-dashed lines are reference lines for 25% and
75% respectively.

``` r
  "The birthday paradox visualized as a cumulative
  distribution function (CDF)."
#> [1] "The birthday paradox visualized as a cumulative\n  distribution function (CDF)."
base <- ggplot() + xlim(1, 50)

base +
  geom_function(fun = prob_fun, colour = "navy", linewidth = 1) +
  labs(title = "The Birthday Paradox", x = "No. People",
       y = "P(duplicated birthday)") +
  annotate("segment",
           x = c(ppl, 1), xend = rep(ppl, 2),
           y = c(0, prob_fun(ppl)), yend = rep(prob_fun(ppl), 2),
           linetype = "dashed", colour = "red"
  ) +
  annotate("segment",
           x = c(32, 1), xend = rep(32, 2),
           y = c(0, 0.75), yend = c(0.75, 0.75),
           linetype = "dashed", colour = "cyan"
  ) +
  annotate("segment",
           x = c(15, 1), xend = rep(15, 2),
           y = c(0, 0.25), yend = c(0.25, 0.25),
           linetype = "dashed", colour = "cyan"
  ) +
 theme(text = element_text(size = 15))
```

![](figures/birthday-paradox-ggplot-prob-fun-1.png)

## Simulation

Next, let’s simulate this. I draw 23 “people” from the uniform
distribution:

$$
X \sim U(1, 365),
$$

and count the number of times the same number is drawn *exactly* twice
on *any* day of the year. Repeat this 10000 times and determine an
empirical probability of the event occurring.

``` r
reps <- 10000
rsim <- withr::with_seed(101,              # set seed for reproducibility
  replicate(
    reps,                                  # nsims
    round(runif(ppl, min = 1, max = 365))) # n = 23; X ~ U(1, 365)
)
counts <- apply(rsim, 2, function(.x) table(.x))        # tabulate counts of duplicate wells
dupes  <- vapply(counts, function(.x) any(.x == 2), NA) # any (>1) duplicates present!
prob   <- mean(dupes)
prob
#> [1] 0.507
```

### Caveats

1.  the *same* day could be selected *more* than twice; this
    simulation reflects the probability that *exactly* 2 people are
    chosen with the same birthday (not 3 or more).

2.  the *same* birthday could be selected twice multiple times; this
    simulation reflects the probability that two people are chosen
    with the same birthday *at least* once (but could be more).

Considering the 2 caveats above, the simulation agrees fairly well
with the closed form solution described above and from `prob_fun()`.

----------------------------------------------------------------------

# Session Info

<details class="code-fold">
<summary>Code</summary>

``` r
get_session_info()
#> $packages
#>  package      * version date (UTC) lib source
#>  cli            3.6.6   2026-04-09 []  RSPM
#>  digest         0.6.39  2025-11-19 []  RSPM
#>  dplyr          1.2.1   2026-04-03 []  RSPM
#>  evaluate       1.0.5   2025-08-27 []  RSPM
#>  farver         2.1.2   2024-05-13 []  RSPM
#>  fastmap        1.2.0   2024-05-15 []  RSPM
#>  generics       0.1.4   2025-05-09 []  RSPM
#>  ggplot2      * 4.0.3   2026-04-22 []  RSPM
#>  glue           1.8.1   2026-04-17 []  RSPM
#>  gtable         0.3.6   2024-10-25 []  RSPM
#>  htmltools      0.5.9   2025-12-04 []  RSPM
#>  jsonlite       2.0.0   2025-03-27 []  RSPM
#>  knitr          1.51    2025-12-20 []  any (@1.51)
#>  labeling       0.4.3   2023-08-29 []  RSPM
#>  lifecycle      1.0.5   2026-01-08 []  RSPM
#>  magrittr       2.0.5   2026-04-04 []  RSPM
#>  otel           0.2.0   2025-08-29 []  RSPM
#>  pillar         1.11.1  2025-09-17 []  RSPM
#>  pkgconfig      2.0.3   2019-09-22 []  RSPM
#>  R6             2.6.1   2025-02-15 []  RSPM
#>  RColorBrewer   1.1-3   2022-04-03 []  RSPM
#>  rlang          1.3.0   2026-07-05 []  RSPM
#>  rmarkdown      2.31    2026-03-26 []  RSPM
#>  S7             0.2.2   2026-04-22 []  RSPM
#>  scales         1.4.0   2025-04-24 []  RSPM
#>  sessioninfo    1.2.4   2026-06-04 []  any (@1.2.4)
#>  tibble         3.3.1   2026-01-11 []  any (@3.3.1)
#>  tidyselect     1.2.1   2024-03-11 []  RSPM
#>  vctrs          0.7.3   2026-04-11 []  RSPM
#>  withr          3.0.3   2026-06-19 []  RSPM
#>  xfun           0.60    2026-07-09 []  RSPM
#>  yaml           2.3.12  2025-12-10 []  RSPM
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
#>  date     2026-08-20
#>  pandoc   3.1.3 @ /usr/bin/ (via rmarkdown)
#>  quarto   1.10.18 @ /usr/local/bin/quarto
```

</details>
