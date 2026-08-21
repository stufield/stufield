# The Monty Hall Paradox
Stu Field

Suppose you’re on a game show, and you’re given the choice of three
doors: behind one door is a car; behind the others are goats. You pick
a door, say `A`, and the host, who knows where the care is located,
opens another door, say `C`, which has a goat. He then asks you, “Do
you want to pick door `B`?” Is it to your advantage to switch your
choice?

### Key Assumptions

For the paradox to work, there are some key assumptions (rules) about
the host’s behavior:

1.  The host must always open a door that was not picked by the
    contestant
2.  The host must always open a door to reveal a goat and never the
    car.
3.  The host must always offer the chance to switch between the
    originally chosen door and the remaining closed door.

### Common Mistake

Most people come to the conclusion that switching does not matter
because there are two unopened doors and one car and that it is a
50/50 choice. This would be true if the host opens a door randomly,
but that is not the case; the door opened depends on the player’s
initial choice, so the assumption of independence does not hold. As we
see below, breaking the independence assumption drastically alters the
probabilities of the remaining unrevealed doors.

----------------------------------------------------------------------

### Solution

The key insight is that the host does *not* reveal the remaining
(non-chosen) doors randomly. He *always* reveals a goat, and thus has
knowledge of where the car actually is. Incorporating this information
into the probability calculation adjusts the probability, shifting it
away from the newly revealed door to the remaining unrevealed and
un-chosen door. In a way this represents a Bayesian update to the
probability of door `B` with the knowledge that the car is *not*
behind door `C`. The posterior probability of door `B` is updated from
0.33 -\> 0.66 following the reveal that door `C` is not an option.

The problem can be reduced to a binary problem (switch or stay):

1.  The player chooses correctly and loses by switching (1/3)
2.  The player chooses incorrectly and wins by switching (2/3)

| Door A | Door B | Door C | Stay Strategy | Switch Strategy |
|:------:|:------:|:------:|:-------------:|:---------------:|
|  Goat  |  Goat  |  Car   |   Wins goat   |    Wins car     |
|  Goat  |  Car   |  Goat  |   Wins goat   |    Wins car     |
|  Car   |  Goat  |  Goat  |   Wins car    |    Wins goat    |
|        |        |        | P(car) = 1/3  |  P(car) = 2/3   |

#### Visual: probability tree

The bifurcated tree below assumes the player has chosen Door 1:

![](figures/monty-hall-tree.png)

----------------------------------------------------------------------

## Simple Simulation

Perhaps the easiest way to visualize the solution is through
simulation.

### Code

``` r
# Single Monty-Hall trial; win by switching?
mh_switch_win <- function() {
  true   <- sample(1:3, 1L)    # true correct door; 3 possible doors
  choose <- sample(1:3, 1L)    # player's door choice: 1/3
  # if player chooses incorrect door,
  # player wins by switching (TRUE/FALSE)
  true != choose
}
```

> Once you convince yourself that the probability of winning by
> switching is the same as the probability of choosing incorrectly,
> i.e. 1 - 1/3, the function can be simplified further.

``` r
mh_switch_win <- function() {
  runif(1) > 1 / 3
}
```

Run the simulation with the `runif()` function directly rather than
`mh_switch_win()`:

``` r
n_trials <- 1000
sim_tbl  <- tibble::tibble(
  n_sim           = seq_len(n_trials),
  switch_win      = withr::with_seed(833, runif(n_trials) > 1 / 3),
  stay_win        = !switch_win,
  sum_switch_wins = cumsum(switch_win),
  sum_stay_wins   = cumsum(stay_win),
  prob_switch_win = sum_switch_wins / (sum_switch_wins + sum_stay_wins),
  prob_stay_win   = 1 - prob_switch_win
)

# simulation results
sim_tbl
#> # A tibble: 1,000 × 7
#>    n_sim switch_win stay_win sum_switch_wins sum_stay_wins prob_switch_win prob_stay_win
#>    <int> <lgl>      <lgl>              <int>         <int>           <dbl>         <dbl>
#>  1     1 TRUE       FALSE                  1             0           1             0    
#>  2     2 TRUE       FALSE                  2             0           1             0    
#>  3     3 TRUE       FALSE                  3             0           1             0    
#>  4     4 TRUE       FALSE                  4             0           1             0    
#>  5     5 TRUE       FALSE                  5             0           1             0    
#>  6     6 FALSE      TRUE                   5             1           0.833         0.167
#>  7     7 TRUE       FALSE                  6             1           0.857         0.143
#>  8     8 FALSE      TRUE                   6             2           0.75          0.25 
#>  9     9 FALSE      TRUE                   6             3           0.667         0.333
#> 10    10 TRUE       FALSE                  7             3           0.7           0.3  
#> # ℹ 990 more rows
```

### Plot Simulations

``` r
# Cumulative wins
plot_df <- sim_tbl |>
  tidyr::pivot_longer(
  cols     = c(sum_switch_wins, sum_stay_wins),
  names_to = "strategy", values_to = "Wins"
)

p1 <- plot_df |>
  ggplot(aes(x = n_sim, y = Wins, color = strategy)) +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c("#24135F", "#00A499")) +
  labs(y = "Cumulative Wins", x = "Trial",
       title = "Cumulative Wins by Strategy")

# Prob winning
plot_df <- sim_tbl |>
  tidyr::pivot_longer(
  cols     = c(prob_switch_win, prob_stay_win),
  names_to = "strategy", values_to = "prob"
)

p2 <- plot_df |>
  ggplot(aes(x = n_sim, y = prob, color = strategy)) +
  geom_line(linewidth = 1) +
  ylim(c(0, 1)) +
  geom_hline(yintercept = sim_tbl$prob_switch_win[n_trials],
             linetype = "dashed") +
  scale_color_manual(values = c("#24135F", "#00A499")) +
  labs(y = "P(win)", x = "Trial",
       subtitle = sprintf("P(switch win) = %0.2f",
                          sim_tbl$prob_switch_win[n_trials]),
       title = "Probability of Winning by Strategy")

p1 + p2
```

![](figures/monty-hall-plot-sim-1.png)

----------------------------------------------------------------------

### Links

<https://en.wikipedia.org/wiki/Monty_Hall_problem>

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
#>  patchwork    * 1.3.2   2025-08-25 []  any (@1.3.2)
#>  pillar         1.11.1  2025-09-17 []  RSPM
#>  pkgconfig      2.0.3   2019-09-22 []  RSPM
#>  purrr          1.2.2   2026-04-10 []  RSPM
#>  R6             2.6.1   2025-02-15 []  RSPM
#>  RColorBrewer   1.1-3   2022-04-03 []  RSPM
#>  rlang          1.3.0   2026-07-05 []  RSPM
#>  rmarkdown      2.31    2026-03-26 []  RSPM
#>  S7             0.2.2   2026-04-22 []  RSPM
#>  scales         1.4.0   2025-04-24 []  RSPM
#>  sessioninfo    1.2.4   2026-06-04 []  any (@1.2.4)
#>  tibble       * 3.3.1   2026-01-11 []  any (@3.3.1)
#>  tidyr        * 1.3.2   2025-12-19 []  any (@1.3.2)
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
#>  date     2026-08-21
#>  pandoc   3.1.3 @ /usr/bin/ (via rmarkdown)
#>  quarto   1.10.18 @ /usr/local/bin/quarto
```

</details>
