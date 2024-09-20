# Longitudinal Data Analysis via Linear Mixed-Effects Models
Stu Field
20 September 2024

------------------------------------------------------------------------

# The Linear Model

For a simple linear-*fixed* effect model, the model equation fits the
equation:

<span id="eq-lme">$$
\begin{eqnarray} 
  y_i      & \sim & \beta_0 + \beta_1 x_i + \epsilon_i \\
  && \\
  \epsilon & \sim & N(0, \sigma^2_n),
\end{eqnarray}
 \qquad(1)$$</span>

for the $i^{th}$ sample/observation. Thus, we can estimate the model
coefficients via,

<span id="eq-coefs">$$
\begin{equation} 
   \hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i 
\end{equation}
 \qquad(2)$$</span>

where,

<span id="eq-betas">$$
\begin{eqnarray}
   \hat{\beta_1} &=& \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}
                          {\sum_{i=1}^n (x_i - \bar{x})^2} \\
   \hat{\beta_0} &=& \bar{y} - \hat{\beta_1} \bar{x} \\
   \epsilon_i &=& r_i  \\
              &=& y_i - \hat{y_i} \\
    RSS &=& \sum_{i=1}^n \epsilon_i^2 = \sum_{i=1}^n (y_i - \hat{y_i})^2 \\
        &=& (y_1 - \hat{\beta_0} - \hat{\beta_1}x_1)^2 +
            (y_2 - \hat{\beta_0} - \hat{\beta_1}x_2)^2 + \dots +
            (y_n - \hat{\beta_0} - \hat{\beta_1}x_n)^2 \\
    R^2 &=& (TSS - RSS) / TSS \\
    TSS &=& \sum (y_i - \bar{y})^2
\end{eqnarray}
 \qquad(3)$$</span>

The Total Sum of Squares (TSS) measures the total variance in the
response (*Y*), thus $R^2$ is the proportion of the total variance in
*Y* that can be explained by the model. The remaining variance is packed
into $\epsilon$. To calculate the accuracy of the coefficient estimates,
we need a measure of variance in *Y*:

$$
\begin{eqnarray}
   \text{SE}(\hat\beta_0)^2 &=& \sigma^2 \bigg[ \frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n(x_i-\bar{x})^2} \bigg] \\
       && \\
   \text{SE}(\hat\beta_1)^2 &=& \frac{\sigma^2}{\sum_{i=1}^n(x_i-\bar{x})^2} \\
   \text{where,} && \\
   \sigma^2 &=& Var(\epsilon), \\
\end{eqnarray}
$$

and $\sigma^2$ can be estimated from the data via the Residual Standard
Error (RSE):

<span id="eq-RSE">$$
\begin{equation*}
    RSE = \sqrt{RSS/(n-2)}
\end{equation*}
 \qquad(4)$$</span>

# Multiple-Linear Regression

The linear model can easily be expanded to include multiple predictors
(regressors/variables/proteins). The extension of
(<a href="#eq-lme" class="quarto-xref">Equation 1</a>) can be modeled
as:

<span id="eq-lme-multivariate">$$
\begin{eqnarray}
   y_i &=& \beta_0 x_{0i} + \beta_1 x_{1i} + \dots + \beta_p x_{pi} + \epsilon_i \\
   && \\
   \epsilon_i &\sim& \text{NID}(0,\sigma^2)
\end{eqnarray}
 \qquad(5)$$</span>

for the $i^{th}$ sample and $p^{th}$ covariate. We typically set
$x_{0i}=1$ so that $\beta_0$ is a constant intercept. In this form, the
only *random-effect* is the error term, $\epsilon_i$. This equation can
be rewritten in matrix form as:

<span id="eq-lme-matrix">$$
\begin{eqnarray}
   \begin{pmatrix}
       y_1 \\ y_2 \\ \vdots \\ y_n
    \end{pmatrix}
       &=&
    \begin{pmatrix}
        x_{11} & x_{12} & \dots & x_{1p} \\
        x_{21} & x_{22} & \dots & x_{2p} \\
        \vdots & \vdots & \ddots & \vdots \\
        x_{n1} & x_{n2} & \dots & x_{np} \\
    \end{pmatrix}
    *
    \begin{pmatrix}
        \beta_1 \\ \beta_2 \\ \vdots \\ \beta_p
    \end{pmatrix}
    +
    \begin{pmatrix}
        \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n
    \end{pmatrix}
    \\ \nonumber && \\ \nonumber
    \vec y &=& \mathbf{X} \vec \beta + \vec \epsilon
\end{eqnarray}
 \qquad(6)$$</span>

where $\vec y$ is a $n \times 1$ vector, $\mathbf{X}$ is the
$n \times p$ data matrix, $\vec \beta$ is a $p\times1$ vector of fixed
effects coefficients for the $p$ covariates, and $\vec \epsilon$ is the
$n \times 1$ vector of observation specific errors.

# The Mixed-Effect Model

Mixed-effects models fit both fixed effects *and* random effects in the
same model. We assume a grouping or dependence structure (typically
temporal or spatial) that is sampled from a larger population (e.g.
plots within a farm, schools within districts, or observations within
subjects). We also assume *compound symmetry*, which is a fancy way of
saying that the related samples vary in the same fashion for each
`group` (subject), i.e. the off-diagonal of the variance-covariance
matrix are all equal. For longitudinal data, samples from the same
individual reflect serial dependence and the model must account for this
non-independence in sample structure. Typically we are interested in
fitting subject-specific (random) intercepts (i.e. parallel linear fits
can hit the y-axis independently) and define the following:

$$
\begin{eqnarray}
  y_{ij} &=& b_{0i} + \beta_1 x_{ij} + \epsilon_{ij} \\
  && \\
    b_i &\sim& N(0,\sigma^2_b) \\
  && \\
    \epsilon &\sim& N(0,\sigma^2_n)
\end{eqnarray}
$$

for the $j^{th}$ observation of the $i^{th}$ subject, where $b$ is a
parameter treated as a random variable (i.e. $i$ subjects sampled from a
large, infinite population of potential subjects). These random effects
are thus assumed to vary by group (in this case subjects) the goal is to
capture this variation in the coefficients and prevent it from being
packaged into $\epsilon$!

------------------------------------------------------------------------

# Syntax in `R`

Use the `nlme` or `lme4` package. `R` uses the “Wilkinson” notation in
specifying model formulas.

### Random Intercept

``` r
lme(y ~ time * group, random = ~ 1 | pid, data = adat)
```

### Random Slope

``` r
lme(y ~ time * group, random = ~ time | pid, data = adat)
```

### Random Intercept & Random Slope

``` r
nlme::lme(y ~ time * group, random = ~ 1 + time | pid, data = adat)
```

### Syntax Definitions

| Symbol         |               |                  Corresponds to                  |
|----------------|---------------|:------------------------------------------------:|
| $y$            | $\rightarrow$ |                     $y_{ij}$                     |
| $∼ 1$          | $\rightarrow$ |            random effects ($b_{0i}$)             |
| `pid`          | $\rightarrow$ | field containing the dependent groups (subjects) |
| `time * group` | $\rightarrow$ |      fixed effects (with interaction term)       |
| `time * group` | $\rightarrow$ |          `time + group + time * group`           |

------------------------------------------------------------------------

# Example 1: Response to Drug

First, simulate longitudinal data for 20 subjects, having $3-10$ serial
samples each, intercept ($\beta_0 = 1000$), slope ($\beta_1 = 400$), and
a serial autocorrelation parameter of 0.1. The default simulation values
can be seen below:

``` r
args(simulateLongData)
> function (nsubj = 20, beta0 = 1000, beta1 = 10, max.obs = 10, 
>     group = NULL, auto.cor = 0.5, sd.pars = list(sigma = beta0/4, 
>         tau0 = 2, tau1 = 2, tau01 = 0.5), r.seed = 1234) 
> NULL
```

``` r
# simulate longitudinal data
# createLongData() is a wrapper for simulateLongData()
p <- list(nsubj = 20, beta0 = 1000, beta1 = 400, max.obs = 10, r.seed = 101,
          sd.pars = list(sigma = 250), auto.cor = 0.1)
ResponseData <- createLongData(subject = p)
> Warning: The `sd.pars` argument list is missing these parameter(s): 'tau0', 'tau1', 'tau01'.
> Using default parameters: 2, 2, 0.5.
ResponseData
> # A tibble: 127 × 5
>    pid          time    eij   yij Group  
>  * <chr>       <dbl>  <dbl> <dbl> <chr>  
>  1 subject_001     1   81.1 1480  subject
>  2 subject_001     3  300.  2498. subject
>  3 subject_001     5  184.  3180. subject
>  4 subject_002     1 -185.  1217  subject
>  5 subject_002     4  337.  2944. subject
>  6 subject_002     6 -331.  3079  subject
>  7 subject_003     1 -204.  1194. subject
>  8 subject_003     2 -530.  1266. subject
>  9 subject_003     3  -93.8 2101  subject
> 10 subject_003     4  167.  2760. subject
> # ℹ 117 more rows
table(ResponseData$pid)
> 
> subject_001 subject_002 subject_003 subject_004 subject_005 subject_006 
>           3           3           8           9           9           3 
> subject_007 subject_008 subject_009 subject_010 subject_011 subject_012 
>           4           9           7           6           5           8 
> subject_013 subject_014 subject_015 subject_016 subject_017 subject_018 
>           5          10           5          10           3          10 
> subject_019 subject_020 
>           5           5
```

``` r
# plot longitudinal traces by subject ("pid")
ResponseData |>
  ggplot(aes(x = time, y = yij, group = pid, colour = pid)) +
  geom_line() +
  geom_point( size = 4, shape = 21, fill = "white") +
  ylab(y_lab) +
  guides(colour = guide_legend(ncol = 1)) +
  facet_wrap(vars(pid)) +
  ggtitle(bquote(y[ij]~"~"~time))
```

<img src="figures/mixed-plot_long_pid-1.png" data-fig-align="center" />

``` r
# plot longitudinal traces together
ResponseData |>
  ggplot(aes(x = time, y = yij, group = pid, colour = pid)) +
  geom_line() +
  geom_point(size = 4, shape = 21, fill = "white") +
  ylab(y_lab) +
  guides(colour = guide_legend(ncol = 1)) +
  facet_wrap(vars(Group)) +
  ggtitle(bquote(y[ij]~"~"~time))
```

<img src="figures/mixed-plot_long_combined-1.png"
data-fig-align="center" />

## Analysis in `R`

The model we wish to fit:

$$
\begin{equation}
   y_{ij} = b_{0i} + \beta_1 time_{ij} + \epsilon_{ij},
\end{equation}
$$

again for the $j^{th}$ observation of the $i^{th}$ subject, where
$b_{0i}$ is the subject-specific random *intercept*, and $\beta_1$ is
the fixed-effect for *time*.

``` r
fit1 <- fit_lme_safely(yij ~ time, random = ~ 1 | pid, data = ResponseData)
summary(fit1)
> Linear mixed-effects model fit by REML
>   Data: ResponseData 
>        AIC      BIC    logLik
>   1749.402 1760.715 -870.7009
> 
> Random effects:
>  Formula: ~1 | pid
>         (Intercept) Residual
> StdDev:  0.03568658 244.4632
> 
> Fixed effects:  yij ~ time 
>                Value Std.Error  DF  t-value p-value
> (Intercept) 1023.567  43.04121 106 23.78110       0
> time         391.360   7.17510 106 54.54415       0
>  Correlation: 
>      (Intr)
> time -0.864
> 
> Standardized Within-Group Residuals:
>        Min         Q1        Med         Q3        Max 
> -2.2109147 -0.7422681  0.0232049  0.7453189  2.0609776 
> 
> Number of Observations: 127
> Number of Groups: 20
```

From the fixed-effects column in `Value`, the estimates are reasonably
close to the original parameters:

- $\hat{\beta_0}=$ 1023.57
- $\hat{\beta_1}=$ 391.36

Both the slope and intercepts differ significantly from zero.

------------------------------------------------------------------------

# Example 2: Group Dependent Response (Interaction)

In clinical studies, it is often of interest to understand how the
treatment of one group of subjects differs from another group of
subjects (e.g. a control group). In a mixed-model setting, this is
accomplished by adding an interaction term to the model, which sets up a
conditional response variable, given that a subject belongs to a
particular group/class.

Next, simulate longitudinal data for 20 subjects, having $3-10$ serial
samples each, intercepts of $\beta_0=1000$, slopes of $\beta_1=0$ (crtl)
and $\beta_1=500$ (treat), and a serial autocorrelation parameter of
0.1. The parameters that differ from the default simulation values can
be seen below:

``` r
# simulate longitudinal data with group-specific response
control <- list(nsubj = 20, beta1 = 0, max.obs = 10, r.seed = 10,
                sd.pars = list(sigma = 150), auto.cor = 0.1)
treatment <- list(nsubj = 20, beta1 = 500, max.obs = 10, r.seed = 11,
                  sd.pars = list(sigma = 350), auto.cor = 0.1)
GroupResponseData <- createLongData(control, treatment)
```

Now plot the longitudinal time traces:

``` r
GroupResponseData |>
  ggplot(aes(x = time, y = yij, group = pid, colour = pid)) +
  geom_line() +
  geom_point(size = 4, shape = 21, fill = "white") +
  ylab(y_lab) +
  guides(colour = guide_legend(ncol = 2)) +
  facet_wrap(vars(Group)) +
  ggtitle(bquote(y[ij]~"~"~time))
```

<img src="figures/mixed-plot_long_combined2-1.png"
data-fig-align="center" />

## Analysis in `R`

The model specification to fit:

$$
\begin{equation}
   y_{ij} = b_{0i} + \beta_1 time_{ij} + \beta_2 group_i + 
     \beta_3 (group_i \times time_{ij}) + \epsilon_{ij},
\end{equation}
$$

again for the $j^{th}$ observation of the $i^{th}$ subject, $b_{0i}$ is
still the subject-specific random intercept and $\beta_1$ is still the
fixed-effect for *time*. However, now there is a fixed-effect for
*group* and an interaction term ($\beta_3$ term) that is multiplied by a
dummy variable such that:

<span id="eq-dummy">$$
\begin{equation}
  G = \begin{cases}
    0, & control \\ 
    1, & treatment
    \end{cases}
\end{equation}
 \qquad(7)$$</span>

giving,

<span id="eq-lme-dummy">$$
\begin{equation}
   y_{ij} = b_{0i} + \beta_1 time_{ij} + \beta_2 group_i + 
     \beta_3 (group_i \times time_{ij}) * G + \epsilon_{ij},
\end{equation}
 \qquad(8)$$</span>

``` r
fit2 <- fit_lme_safely(yij ~ time * Group, random = ~ 1 | pid,
                       data = GroupResponseData)
```

## Check Subject-specific Linear Models with `lmList`

It can be a good idea to check the variation on the coefficients of
subject-specific linear fits of the data. The `mixr::lmeDiagnostic()`
function generates diagnostic plots of linear models for each of the *i*
subjects.

``` r
lmeDiagnostic(fit2)
```

<img src="figures/mixed-diag-1.png" data-fig-align="center" />

Notice that the subject-subject offsets ($\beta_0$) are approximately
centered about 1000 (RFU) for both treatment and control (as they should
be; we set them to the same intercept), and a vertical line up from 1000
would cross the confidence interval for most subjects. The more variable
the estimates (and intervals) of $\beta_0$ seen in the left panel, the
greater the improvement in model fit can be found in fitting
subject-specific offsets in a mixed-effects model framework. Secondly,
as can be expected, the slope coefficient ($\beta_1$) is vastly
different between treated and control. Depending on the statistical
question, subject-specific slopes may be desired (but typically not for
longitudinal time-series data in a clinical setting).

## Summary

``` r
summary(fit2)
> Linear mixed-effects model fit by REML
>   Data: GroupResponseData 
>        AIC      BIC    logLik
>   3816.172 3837.829 -1902.086
> 
> Random effects:
>  Formula: ~1 | pid
>         (Intercept) Residual
> StdDev:    34.88397 243.5147
> 
> Fixed effects:  yij ~ time * Group 
>                         Value Std.Error  DF  t-value p-value
> (Intercept)         1002.8524  42.32105 235 23.69630  0.0000
> time                  -5.2117   6.81643 235 -0.76457  0.4453
> Grouptreatment       -57.0508  60.96284  38 -0.93583  0.3553
> time:Grouptreatment  517.7512   9.86489 235 52.48422  0.0000
>  Correlation: 
>                     (Intr) time   Grptrt
> time                -0.856              
> Grouptreatment      -0.694  0.594       
> time:Grouptreatment  0.591 -0.691 -0.856
> 
> Standardized Within-Group Residuals:
>         Min          Q1         Med          Q3         Max 
> -3.21074964 -0.64030693  0.02129678  0.59148825  3.32489176 
> 
> Number of Observations: 277
> Number of Groups: 40
```

From the fixed-effects column in `Value`, the estimates are reasonably
close to the original parameters:

- $\hat{\beta_0}=$ 1002.85
- $\hat{\beta_1}=$ -5.21 (essentially zero).

This indicates that the effect of time has no effect on $y_{ij}$ (at
least for the controls!). The significant **interaction** tells a
different story and indicates that the **entire** fixed-effect for
*time* occurs in the *treatment* group, $\hat{\beta_3}=$ 517.75; in this
case the *control* group does not vary with time.

------------------------------------------------------------------------

# Example 3

To illustrate how the model output is constructed, we simulate
identically to the above, however give the control group a slope of
$\beta_1 = 250$. This results in a significant fixed-effect for time
*and* an significant interaction; the additional effect of being in the
*treatment* group. Relate the values back to the original
(<a href="#eq-lme-dummy" class="quarto-xref">Equation 8</a>) above in
<a href="#sec-group-dependent" class="quarto-xref">Section 6</a> to see
how the terms are partitioned.

``` r
# simulate longitudinal data with group-specific response
p1 <- list(nsubj = 20, beta1 = 250, max.obs = 10, r.seed = 10,
           sd.pars = list(sigma = 150), auto.cor = 0.1)
p2 <- list(nsubj = 20, beta1 = 500, max.obs = 10, r.seed = 101,
           sd.pars = list(sigma = 350), auto.cor = 0.1)
GroupResponseData2 <- createLongData(control = p1, treatment = p2)
GroupResponseData2 |>
  ggplot(aes(x = time, y = yij, group = pid, colour = pid)) +
  geom_line() +
  geom_point(size = 4, shape = 21, fill = "white") +
  ylab(y_lab) +
  guides(colour = guide_legend(ncol = 2)) +
  facet_wrap(vars(Group)) +
  ggtitle(bquote(y[ij]~"~"~time))
```

<img src="figures/mixed-plot_long_combined3-1.png"
data-fig-align="center" />

``` r
fit3 <- fit_lme_safely(yij ~ time * Group, random = ~ 1 | pid,
                       data = GroupResponseData2)
summary(fit3)
> Linear mixed-effects model fit by REML
>   Data: GroupResponseData2 
>        AIC      BIC    logLik
>   3759.704 3781.228 -1873.852
> 
> Random effects:
>  Formula: ~1 | pid
>         (Intercept) Residual
> StdDev:    26.48875 257.0739
> 
> Fixed effects:  yij ~ time * Group 
>                         Value Std.Error  DF  t-value p-value
> (Intercept)         1002.6579  44.25897 229 22.65435  0.0000
> time                 244.8556   7.18949 229 34.05744  0.0000
> Grouptreatment        30.3784  63.62028  38  0.47750  0.6357
> time:Grouptreatment  243.5068  10.43136 229 23.34372  0.0000
>  Correlation: 
>                     (Intr) time   Grptrt
> time                -0.864              
> Grouptreatment      -0.696  0.601       
> time:Grouptreatment  0.595 -0.689 -0.859
> 
> Standardized Within-Group Residuals:
>         Min          Q1         Med          Q3         Max 
> -2.87824541 -0.63823023  0.04982133  0.54893419  2.75872522 
> 
> Number of Observations: 271
> Number of Groups: 40
```

Notice that the $\hat{\beta_0}=$ 1002.66 has not changed, as we haven’t
changed any baseline values, but now half of the effect has moved into
the *time* effect ($\hat{\beta_1}=$ 244.86), and away from the
interaction coefficient ($\hat{\beta_3}=$ 243.51) which was carrying the
entire slope effect for the treatment group. Stated alternatively, the
significance of the interaction has dropped, though still a significant
$p-$value, shifting from 52.48 to 23.34.

------------------------------------------------------------------------

# Notes

## Mixed-effects Analysis Guidelines

Below are some initial guidelines for the implementation of linear
mixed-effects modeling in `R`.

1.  Make sure you are fitting the correct model for your desired
    question.
2.  Be sure you can write the full equation for the model(s) you are
    fitting.
3.  A wrapper for `nlme::lme()` to avoid convergence failures to stop
    your code from running to completion can be found via
    `mixr::fit_lme_safely()`.
4.  Prior to model fitting, plot the subject-specific coefficients with
    the diagnostic wrapper `mixr::lmeDiagnostic()`.
5.  The `lme()` function uses Expectation Maximization combined with
    Newton-Raphson iterations to fit the various model coefficients.
6.  See also: `mixr::fitMixedEffectsModels()` and
    `mixr::createMixedEffectsTable()`.

## Pitfalls

- **Mike Hinterberg**: effects are fit sequentially according to the
  magnitude of the effect.
