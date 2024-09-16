# Naïve Bayes Classifiers
Stu Field
16 September 2024

------------------------------------------------------------------------

# Overview

The general goal is to use a probabilistic modeling framework to predict
the class of an unknown sample. Such mathematical models, or
classifiers, are based on training data and are built (i.e. model
fitting) in order to make class predictions about the unknown sample.

------------------------------------------------------------------------

# Bayes’ Theorem

The Bayes Theorem states:

$$
\begin{equation}
  Posterior\ Probability = \frac{Likelihood(data) \times Prior}{Evidence}
\end{equation}
$$

written in terms of probabilities,

<span id="eq-bayes-full">$$
\begin{equation}
  P(outcome\ |\ data) = \frac{P(data\ |\ outcome) \times P(outcome)}{P(data)}
\end{equation}
 \qquad(1)$$</span>

where the term $P(data)$ is a normalizing constant that is independent
of the *outcome*, and is often ignored if *relative* posteriors are
desired over *absolute* posteriors.
(<a href="#eq-bayes-full" class="quarto-xref">Equation 1</a>) then
simplifies to,

$$
\begin{equation}
  P(outcome\ |\ data) \propto P(data\ |\ outcome) \times P(outcome)
\end{equation}
$$

# Naïve Bayes in Practice

Consider an example where *proteomic* measurements (i.e. features =
proteins), assumed to have Gaussian (i.e. Normal) distributions, are
taken from an individual and the posterior probability of interest is
whether that individual belongs to one of $k$ possible outcomes/classes.
For example, if there are $k=2$ possible classes, disease or control
(i.e. a *binary* classifier), this can be re-written as:

$$
\begin{eqnarray}
   P(control\ |\ protein\ conc) &=& P(protein\ conc\ |\ control) \times P(control), \\
   P(disease\ |\ protein\ conc) &=& P(protein\ conc\ |\ disease) \times P(disease).
\end{eqnarray}
$$

If we have $p$ features and naïvely assume all are independent, we can
multiply their individual probabilities to produce a cumulative
probability. For the disease posterior this gives:

$$
\begin{eqnarray}
   P(control\ |\ protein\ conc) &=& P(control) \times P(prot_1\ |\ control) \times P(prot_2\ |\ control) \times ... \times P(prot_p\ |\ control), \\
   P(disease\ |\ protein\ conc) &=& P(disease) \times P(prot_1\ |\ disease) \times P(prot_2\ |\ disease) \times ... \times P(prot_p\ |\ disease).
\end{eqnarray}
$$

Naïve Bayes models contain $2pk + k$ parameters, where $k$ is the number
of classes, $p$ is the number of features; a **mean** ($\mu$) and
**standard deviation** ($\sigma$) for each feature $\times$ class
combination, plus the class-specific **prior**, which is often
determined by the training class proportions (i.e. an *uninformative*
prior). Naïve Bayes models assume Gaussian densities and are calculated
via the probability density function (PDF) given class-specific
parameters $\mu$ and $\sigma$:

<span id="eq-pdf">$$
\begin{eqnarray}
   f(x\ |\ \mu_k,\sigma_k) &= \frac{1}{\sqrt{2\pi\sigma_k^2}} \; exp\Bigg( \frac{-(x-\mu_k)^2}{2\sigma_k^2} \Bigg),
\end{eqnarray}
 \qquad(2)$$</span>

To classify an unknown sample with $p$ feature measurements
($\vec x = x_1,...,x_p$) and $k$ classes, calculate the following:

<span id="eq-bayes">$$
\begin{eqnarray}
   P(k\ |\ \vec x) &= 
      \Bigg[ \prod_{i=1}^{p} \frac{1}{\sqrt{2\pi\sigma_{ik}^2}} \; exp\Bigg( \frac{-(x_i-\mu_{ik})^2}{2\sigma_{ik}^2} \Bigg) \Bigg] \times P(k),
\end{eqnarray}
 \qquad(3)$$</span>

The result of (<a href="#eq-bayes" class="quarto-xref">Equation 3</a>)
gives a probability *density* for each class, which is not constrained
on the interval $[0,\ 1]$. Normalized posterior probabilities ($Pr$) are
obtained by calculating the class-specific proportion of the total
density,

<span id="eq-bayes-prob">$$
\begin{eqnarray}
  Pr(k=j\ |\ \vec x) &=& \frac{P(j\ |\ \vec x)}{\sum_{i=1}^k P(i\ |\ \vec x)}
\end{eqnarray}
 \qquad(4)$$</span>

# Example Calculation

## Training data

``` r
tibble::as_tibble(train)
#> # A tibble: 297 × 3
#>    feat1 feat2 Response
#>    <dbl> <dbl> <fct>   
#>  1  3.87  2.69 control 
#>  2  3.39  2.80 control 
#>  3  3.79  2.60 control 
#>  4  3.46  2.70 control 
#>  5  3.77  2.72 control 
#>  6  3.75  2.68 control 
#>  7  3.57  2.73 control 
#>  8  3.63  2.64 control 
#>  9  3.84  2.86 control 
#> 10  3.82  2.67 control 
#> # ℹ 287 more rows
```

## The Model

Consider a $k = 2$ class (disease vs. control) and $p = 2$ feature
example, with $2pk + k = 10$ parameters. The naïve Bayes model looks
like this:

``` r
bayes_model
#> 
#> Robust Naive Bayes Classifier for Discrete Predictors
#> 
#> Call:
#> robustNaiveBayes.default(x = X, y = Response)
#> 
#> A-priori probabilities:
#> Response
#>   control   disease 
#> 0.7542088 0.2457912 
#> 
#> Conditional densities:
#>                   feat1      feat2
#> control_mu    3.7338882 2.71638422
#> disease_mu    4.0007477 2.83842689
#> control_sigma 0.1757146 0.10198388
#> disease_sigma 0.2523813 0.09695953
```

With specific parameters:

| **Model Parameters**                          | **Control** | **Disease** |
|:----------------------------------------------|:------------|:------------|
| Training samples                              | 224         | 73          |
| Prevalence (prior)                            | 0.7542088   | 0.2457912   |
| Feature 1 ($\hat{\mu}_c,\ \hat{\mu}_d$)       | 3.7338882   | 4.0007477   |
| Feature 1 ($\hat{\sigma}_c,\ \hat{\sigma}_d$) | 0.1757146   | 0.2523813   |
| Feature 2 ($\hat{\mu}_c,\ \hat{\mu}_d$)       | 2.7163842   | 2.8384269   |
| Feature 2 ($\hat{\sigma}_c,\ \hat{\sigma}_d$) | 0.1019839   | 0.0969595   |

### Sample “new” data

5 unknown samples with 2 measurements each:

``` r
sample_data <- train[c(3L, 4L, 5L, 257L, 267L), 1:2L]
s1 <- as.numeric(sample_data[1L, ])
s2 <- as.numeric(sample_data[2L, ])
s3 <- as.numeric(sample_data[3L, ])
s4 <- as.numeric(sample_data[4L, ])
s5 <- as.numeric(sample_data[5L, ])
tibble::as_tibble(sample_data)
#> # A tibble: 5 × 2
#>   feat1 feat2
#>   <dbl> <dbl>
#> 1  3.79  2.60
#> 2  3.46  2.70
#> 3  3.77  2.72
#> 4  4.10  2.85
#> 5  4.08  2.84
```

### Raw calculation sample 1:

For unknown **sample 1**, the naïve Bayes posterior conditional
probability densities are calculated using
(<a href="#eq-bayes-prob" class="quarto-xref">Equation 4</a>) as
follows:

**Control posterior density:**

$$
\begin{eqnarray*}
   P(control\ |\ x_1=3.7930077,x_2=2.5993371) &=& P(x_1=3.7930077\ |\ control) \times \\
               && P(x_2=2.5993371\ |\ control) \times \\
               && P(control) \\
               &=& P(x_1=3.7930077\ |\ \mu_c=3.7338882, \sigma_c=0.1757146) \times \\
               && P(x_2=2.5993371\ |\ \mu_c=2.7163842, \sigma_c=0.1019839) \times \\
               && (224\ /\ (224 + 73)) \\
               &=& 1.1264967 \times 0.1967663 \times 0.7542088 \\
               &=& 0.1671753 \\
\end{eqnarray*}
$$

**Disease posterior density:**

$$
\begin{eqnarray*}
   P(disease\ |\ x_1=3.7930077,x_2=2.5993371) &=& P(x_1=3.7930077\ |\ disease) \times \\ 
              && P(x_2=2.5993371\ |\ disease) \times \\
              && P(disease) \\
              &=& P(x_1=3.7930077\ |\ \mu_d=4.0007477, \sigma_d=0.2523813) \times \\
              && P(x_2=2.5993371\ |\ \mu_d=2.8384269, \sigma_d=0.0969595) \times \\
              && (73\ /\ (224 + 73)) \\
              &=& 2.1454638 \times 2.0246405 \times 0.2457912 \\
              &=& 1.0676663 \\
\end{eqnarray*}
$$

**Normalized Posterior Probabilities:**

From (<a href="#eq-bayes-prob" class="quarto-xref">Equation 4</a>), the
relative proportion of each density is:

$$
\begin{eqnarray*}
   Pr(control\ |\ \vec x) &=&
      \frac{ 3.2761267 }{ 0.0544812 + 3.2761267 } = 0.9836423 \\
   && \\
   Pr(disease\ |\ \vec x) &=&
      \frac{ 0.0544812 }{ 0.0544812 + 3.2761267 } = 0.0163577 \\
\end{eqnarray*}
$$

Normalized posterior probabilities for 5 unknown samples are shown
below. The disease class prediction is based on a decision cutoff of
$Pr(disease) \ge 0.5$

``` r
cutoff <- 0.5
preds  <- tibble::as_tibble(predict(bayes_model, sample_data, type = "raw"))
preds$class  <- ifelse(preds$disease >= cutoff, "disease", "control")
names(preds) <- c("Pr(control)", "Pr(disease)", "Class Prediction")
preds
#> # A tibble: 5 × 3
#>   `Pr(control)` `Pr(disease)` `Class Prediction`
#>           <dbl>         <dbl> <chr>             
#> 1         0.984        0.0164 control           
#> 2         0.970        0.0299 control           
#> 3         0.929        0.0711 control           
#> 4         0.180        0.820  disease           
#> 5         0.231        0.769  disease
```

------------------------------------------------------------------------

# Naïve Bayes Visualization

## Probability density functions (PDFs)

Probability density functions (scaled by sample size) of the training
data used to fit the naïve Bayes model. Curves are colored by class and
the feature measurements for Sample 1 ($\vec x =$ 3.7930077, 2.5993371)
are represented by the dashed vertical line. This graphically indicates
that Sample 1 is more likely to have come from the control distribution.

``` r
library(patchwork)
p1 <- train |>
  dplyr::select(feat1, Response) |>
  ggplot(aes(x = feat1, fill = Response)) +
  geom_density(alpha = 0.25, linewidth = 0.1) +
  labs(y = "Probability Density",  title = "Feature 1", x = "value") +
  geom_vline(xintercept = s1f1, linetype = "dashed")
p2 <- train |>
  dplyr::select(feat2, Response) |>
  ggplot(aes(x = feat2, fill = Response)) +
  geom_density(alpha = 0.25, linewidth = 0.1) +
  labs(y = "Probability Density",  title = "Feature 2", x = "value") +
  geom_vline(xintercept = s1f2, linetype = "dashed")
p1 + p2
```

![](figures/bayes-PDFs-1.png)

## Bivariate Plots and Decision Boundary

``` r
p1 <- ggplot(train, aes(x = feat1, y = feat2)) + 
  geom_point(aes(fill = Response), alpha = 0.5, size = 3,
             stroke = 1, shape = 21) +
  geom_vline(xintercept = c(feat1_control_mu, feat1_disease_mu),
             color = c("#F8766D", "#00BFC4"), linetype = "dashed") +
  geom_hline(yintercept = c(feat2_control_mu, feat2_disease_mu),
             color = c("#F8766D", "#00BFC4"), linetype = "dashed") +
  geom_rug(colour = "navy", linewidth = 0.25, length = unit(0.01, "npc")) +
  labs(title = "Training Data")
```

``` r
plotBayesBoundary <- function(data, pos.class, res = 50, main = NULL) {
  stopifnot(ncol(data) == 3, is.numeric(res))
  train <- data |>
    dplyr::rename_if(is.factor, function(.x) "class") |> # rename response
    dplyr::rename_at(1:2, function(.x) c("F1", "F2"))    # rename features 1,2
  train$class <- factor(train$class,    # pos.class 2nd
                        levels = c(setdiff(train$class, pos.class), pos.class))
  model <- robustNaiveBayes(class ~ F1 + F2, data = train)
  df <- expand_grid(
    list(
      F1 = seq(min(train$F1), max(train$F1), length = res),
      F2 = seq(min(train$F2), max(train$F2), length = res)
    )
  )
  df$Pr <- predict(model, newdata = df, type = "raw")[, 2L]

  p <- ggplot(df, aes(x = F1, y = F2))
  p + geom_raster(aes(fill = Pr), alpha = 0.5) +
    scale_fill_gradient(high = "#00BFC4", low = "#F8766D") +
    geom_contour(aes(x = F1, y = F2, z = Pr), binwidth = 0.5,
                 color = "darkorchid4", linetype = "dashed") +
    geom_point(data = train, mapping = aes(x = F1, y = F2, color = class),
               size = 2.5, alpha = 0.5) +
    geom_point(data = train, aes(x = F1, y = F2),
               size = 2.5, shape = 21, color = "black") +
    labs(x = "Feature 1", y = "Feature 2", title = main)
}
p2 <- plotBayesBoundary(
  train, pos.class = "disease", main = "Bayes Decision Boundary"
  ) +
  geom_point(data = sample_data, aes(x = feat1, y = feat2),
             shape = "cross", color = "green", size = 3, stroke = 2)
```

Bivariate plots of training data used to fit the two feature naïve Bayes
model. Dotted lines are the class specific means of the model
parameters, points are colored by class.

The non-linear Bayes decision boundary reflecting the $p = 0.5$ cutoff
is represented by the “purple” dashed line. The green `X`’s represent
the bivariate coordinates of samples 1–5.

![](figures/bayes-bivariate-decision-boundary-1.png)
