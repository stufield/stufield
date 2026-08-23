# Effect of Class Imbalance on Prediction Accuracy
Stu Field

*A cautionary tale of balance*

# Overview of Setup

- logistic regression binary classification model
- initial global data set of 800 samples \>\>\> 50/50 classes
- random split data into 400 training set, 400 test set
- test set is used for Accuracy calculations
- at each iteration the training set is bootstrap sampled from class 1 *only*
- at the same time, class 2 is down-sampled by the same amount, resulting in a
  larger and larger class-imbalance (1 \> 2)

## Simulate Data

Simulate some bivariate data to predict classes

``` r
# total global sample set size
n <- 800

# create tibble of simulated data
sim_data <- withr::with_seed(1, {
  tibble(F1 = rnorm(n), F2 = rnorm(n)) |> # 2 features; F1 & F2
    dplyr::mutate(
      y  = rep(1:2L, each = n / 2),
      # For F1 only class 1; bump noise [1,2]
      F1 = ifelse(y == 1, F1 + runif(n, 1, 2), F1),
      # For F2 only class 1; bump unif noise [1,3]
      F2 = ifelse(y == 1, F2 + runif(n, 1, 3), F2),
      y  = factor(y),           # convert -> factor for model building
      id = dplyr::row_number()  # add ID to track samples
    )
})
```

## Look at the Data

Plot the simulated data based on the 2 predictors (by class)

``` r
sim_data |>
  ggplot(aes(x = F2, y = F1, colour = y)) +
  geom_point(alpha = 0.5, size = 3) +
  scale_colour_manual(values =  c("#00A499", "#24135F"))
```

![](figures/class-imbalance-plot-scatter-1.png)

## Split Data

Random split of simulated data into 50/50 training and test sets

``` r
# random select half of 200 samples for training
train <- withr::with_seed(101, dplyr::sample_frac(sim_data, 0.5))
# merge on `id`s NOT present in the training set
test <- dplyr::anti_join(sim_data, train, by = "id") |>
  dplyr::select(-id)               # rm tracking field `id` from test
train <- dplyr::select(train, -id) # rm tracking field `id` from train
```

## Run Simulation

- increasing the class imbalance as it proceeds
- one-in-one out algorithm. Bootstrap add 1 sample from class 1
- randomly remove 1 sample from class 2
- this generates a class imbalance but maintains training size

``` r
set.seed(1234)
n1 <- table(train$y)[["1"]]
n2 <- table(train$y)[["2"]]
simres <- lapply(seq(190), function(.x) { # more iterations -> no class 2 left; trouble fitting
    # create c1 training samples for this round
    c1_boot <- dplyr::filter(train, y == 1) |> # filter only the class 1 samples
      dplyr::sample_n(size    = n1 + .x,       # bootstrap class 1 samples
                      replace = TRUE)          # with replacement

    # create c2 training samples for this round
    c2_down <- dplyr::filter(train, y == 2) |> # filter only the class 2 samples
      dplyr::sample_n(size = n2 - .x)    # randomly down-sample class 2; no replacement

    train_boot <- rbind(c1_boot, c2_down)  # combine boot c1 w down-sampled c2
    class_prop <- mean(train_boot$y == 1) # calc. proportion c1

    stopifnot(nrow(c1_boot) + nrow(c2_down) == nrow(train)) # sanity check; stable size
    
    # fit logistic-regression model
    logr <- stats::glm(y ~ F1 + F2,          # y ~ F1 + F2
                       data = train_boot,    # use the new imbalanced training set
                       family = "binomial")  # `binomial` = logistic regression

    acc <- data.frame(
      true_class = test$y,                    # true class names from test set
      pred       = predict(logr,
                           newdata = dplyr::select(test, -y), # predicted `probabilities`
                           type = "response") # this ensures prob. space; not log-odds
    )
    acc$pred_class <- factor(ifelse(acc$pred < 0.5, 1L, 2L)) # probs -> classes (cutoff = 0.5)
    conf <- table(acc$true_class, acc$pred_class, dnn = list("Actual", "Predicted"))
    tibble(
      n1            = nrow(c1_boot),    # collect output in `tibble`
      n2            = nrow(c2_down),    # number of class 2
      class_balance = class_prop,       # class 1 proportion
      accuracy      = sum(diag(conf)) / sum(conf) # accuracy
    )              
  }) |>
  dplyr::bind_rows(.id = "iter")

simres  # view the `tibble` of simulation results
#> # A tibble: 190 × 5
#>    iter     n1    n2 class_balance accuracy
#>    <chr> <int> <int>         <dbl>    <dbl>
#>  1 1       206   194         0.515    0.838
#>  2 2       207   193         0.518    0.848
#>  3 3       208   192         0.52     0.85 
#>  4 4       209   191         0.522    0.845
#>  5 5       210   190         0.525    0.838
#>  6 6       211   189         0.528    0.848
#>  7 7       212   188         0.53     0.85 
#>  8 8       213   187         0.532    0.845
#>  9 9       214   186         0.535    0.848
#> 10 10      215   185         0.538    0.842
#> # ℹ 180 more rows
```

## Class Imbalance vs. Prediction Accuracy

- Simulation starts with class 1 at 51.5% and Accuracy = 0.838.
- Simulation ends with class 1 at 98.75% and Accuracy = 0.573.

``` r
simres |>
  ggplot(aes(x = class_balance, y = accuracy)) +
  geom_point(alpha = 0.5, size = 3) +
  geom_smooth(formula = y ~ x, method = "loess") +
  labs(x = "Class 1 Proportion (Imbalance)", y = "Prediction Accuracy",
       title = "Logistic Regression | Accuracy vs. Class Imbalance")
```

![](figures/class-imbalance-plot-imbalance-1.png)

--------------------------------------------------------------------------------

# Session Info

<details class="code-fold">
<summary>Code</summary>

``` r
get_session_info()
#> $packages
#>  package      * version date (UTC) lib source
#>  cli            3.6.6   2026-04-09 []  RSPM
#>  digest         0.6.39  2025-11-19 []  RSPM
#>  dplyr        * 1.2.1   2026-04-03 []  RSPM
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
#>  lattice        0.22-9  2026-02-09 []  CRAN (R 4.6.1)
#>  lifecycle      1.0.5   2026-01-08 []  RSPM
#>  magrittr       2.0.5   2026-04-04 []  RSPM
#>  Matrix         1.7-5   2026-03-21 []  CRAN (R 4.6.1)
#>  mgcv           1.9-4   2025-11-07 []  CRAN (R 4.6.1)
#>  nlme           3.1-169 2026-03-27 []  CRAN (R 4.6.1)
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
#>  tibble       * 3.3.1   2026-01-11 []  any (@3.3.1)
#>  tidyselect     1.2.1   2024-03-11 []  RSPM
#>  utf8           1.2.6   2025-06-08 []  RSPM
#>  vctrs          0.7.3   2026-04-11 []  RSPM
#>  withr        * 3.0.3   2026-06-19 []  RSPM
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
#>  date     2026-08-23
#>  pandoc   3.1.3 @ /usr/bin/ (via rmarkdown)
#>  quarto   1.10.18 @ /usr/local/bin/quarto
```

</details>
