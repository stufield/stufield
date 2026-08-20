# Technical note: Logistic Regression
Stu Field

## Multivariate Logistic Regression

As in univariate logistic regression, let $\pi(x)$ represent the
probability of an event that depends on $p$ covariates or independent
variables. Then, using an *inverse logit* formulation, which is simply
the inverse of log-odds, for modeling the probability, we have:

\$\$

\$\$

The form is identical to univariate logistic regression, but now with
more than one covariate.

To obtain the corresponding log-odds (*logit*) function we get:

$$
\begin{eqnarray}
  logit(\pi(X)) &=& log\bigg(\frac{\pi(X)}{1-\pi(X)}\bigg) \\
                &=& log\Bigg[\frac{\frac{e^{\beta_0+\beta_1X_1+\beta_2X_2+\dots+\beta_pX_p}}{1+e^{\beta_0+\beta_1X_1+\beta_2X_2+\dots+\beta_pX_p}}}{1-\frac{e^{\beta_0+\beta_1X_1+\beta_2X_2+\dots+\beta_pX_p}}{1+e^{\beta_0+\beta_1X_1+\beta_2X_2+\dots+\beta_pX_p}}}\Bigg] \\
                &=& log\Bigg[\frac{\frac{e^{\beta_0+\beta_1X_1+\beta_2X_2+\dots+\beta_pX_p}}{1+e^{\beta_0+\beta_1X_1+\beta_2X_2+\dots+\beta_pX_p}}}{\frac{1}{1+e^{\beta_0+\beta_1X_1+\beta_2X_2+\dots+\beta_pX_p}}}\Bigg] \\
                &=& log\big(e^{\beta_0+\beta_1X_1+\beta_2X_2+\dots+\beta_pX_p}\big) \\
                &=& \beta_0+\beta_1X_1+\beta_2X_2+\dots+\beta_pX_p,
\end{eqnarray}
$$

which gives the log-odds defined by a standard multivariate linear
regression model. Notice that this transform changes the range of
$\pi(X)$ from $(0,1)$ to $(-\infty, +\infty)$, as is usual for linear
regression. Notice also that it is trivial to convert from log-odds to
probability via:

$$
\begin{eqnarray}
  odds &=& \frac{\pi}{1-\pi} \\
  && \\
  \pi  &=& \frac{odds}{1+odds}
\end{eqnarray}
$$

Plots of the logit (‘link’) function and its inverse, the logistic
function. The logistic function maps any value on the y-axis of the
logit function to a value on $(0, 1)$.

``` r
par(mfrow=c(1, 2L))
curve(log(x / (1 - x)), from = 0, to = 1, col = "navy", lwd = 2,
      main = "The Logit Function",
      ylab = bquote("log-odds = logit(x) = x / (1 - x)"))
abline(h = 0, lty = 2)
curve(exp(x) / (1 + exp(x)), from = -6, to = 6, col = "navy", lwd = 2,
      main = "The Logistic (inverse logit) Function",
      ylab = expression(pi))
abline(v = 0, lty = 2)
```

![](figures/logistic-logit-1.png)

Similar to linear regression, and analogously to univariate logistic
regression, the above equations represent the mean or expected
probability, $\pi(X)$, given $X$.

As this is an estimate, each data point will have an error
distribution, but rather than a normal distribution (linear
regression), we use a binomial distribution, to match the dichotomous
outcomes. The mean of the binomial distribution is $\pi(X)$, and the
variance is $\pi(X)(1-\pi(X))$. Of course, now $X$ is a vector,
whereas it is a scalar value in the univariate case.

Let $\cal{L} = L$$(Data; \theta)$ be the likelihood of the data given
the model, where $\theta = {\beta_0, \beta_1,\dots,\beta_p}$ are the
parameters of the model. The parameters are estimated by the principle
of maximum likelihood. **Technical point**: there is no error term for
the overall logistic regression model, unlike in linear regressions.

----------------------------------------------------------------------

# Session Info

<details class="code-fold">
<summary>Code</summary>

``` r
get_session_info()
#> $packages
#>  package     * version date (UTC) lib source
#>  cli           3.6.6   2026-04-09 []  RSPM
#>  digest        0.6.39  2025-11-19 []  RSPM
#>  dplyr         1.2.1   2026-04-03 []  RSPM
#>  evaluate      1.0.5   2025-08-27 []  RSPM
#>  fastmap       1.2.0   2024-05-15 []  RSPM
#>  generics      0.1.4   2025-05-09 []  RSPM
#>  glue          1.8.1   2026-04-17 []  RSPM
#>  htmltools     0.5.9   2025-12-04 []  RSPM
#>  jsonlite      2.0.0   2025-03-27 []  RSPM
#>  knitr         1.51    2025-12-20 []  any (@1.51)
#>  lifecycle     1.0.5   2026-01-08 []  RSPM
#>  magrittr      2.0.5   2026-04-04 []  RSPM
#>  otel          0.2.0   2025-08-29 []  RSPM
#>  pillar        1.11.1  2025-09-17 []  RSPM
#>  pkgconfig     2.0.3   2019-09-22 []  RSPM
#>  R6            2.6.1   2025-02-15 []  RSPM
#>  rlang         1.3.0   2026-07-05 []  RSPM
#>  rmarkdown     2.31    2026-03-26 []  RSPM
#>  sessioninfo   1.2.4   2026-06-04 []  any (@1.2.4)
#>  tibble        3.3.1   2026-01-11 []  any (@3.3.1)
#>  tidyselect    1.2.1   2024-03-11 []  RSPM
#>  vctrs         0.7.3   2026-04-11 []  RSPM
#>  xfun          0.60    2026-07-09 []  RSPM
#>  yaml          2.3.12  2025-12-10 []  RSPM
#> 
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
