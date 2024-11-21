# Baseball Analytics: Strike Classifier
Stu Field
26 November 2024

# Overview

Add general overview of the aim of the analysis etc …

Pitch data were obtained from [FanGraphs](https://www.fangraphs.com/)

------------------------------------------------------------------------

## Modeling Approach

1.  **Explore features**:
    - there are 54705 pitches (strikes) for analysis
    - visually and heuristically to identify likely candidates that may
      be predictive with the response variable `is_strike`
2.  **Feature reduction**: using a combination of
    - step-wise forward/backward feature selection
    - [Stability
      Selection](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00740.x)
    - [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)
3.  **Fit binary classification models**:
    - Logistic Regression
    - [Random
      Forest](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/)
    - Naive Bayes
4.  **Evaluate and Refine**:
    - make predictions
    - evaluate metrics

## Model Features

The `pitch_data` object contains 21 possible features that could be used
in a putative classification model to predict `is_strike`:

``` r
pitch_data
#> # A tibble: 54,705 × 22
#>    is_strike inning is_bottom balls strikes outs_before is_lhp is_lhb pitch_type
#>        <int>  <int>     <int> <int>   <int>       <int>  <int>  <int> <chr>     
#>  1         1      1         0     0       0           0      0      0 FF        
#>  2         0      1         0     0       1           0      0      0 FF        
#>  3         0      1         0     0       0           0      0      0 SL        
#>  4         0      1         0     1       0           0      0      0 FF        
#>  5         1      1         0     2       0           0      0      0 SL        
#>  6         0      1         0     0       0           1      0      1 CH        
#>  7         0      1         0     1       0           1      0      1 FF        
#>  8         0      1         0     2       1           1      0      1 FF        
#>  9         1      1         0     3       1           1      0      1 FF        
#> 10         1      1         1     0       0           0      0      1 FT        
#> # ℹ 54,695 more rows
#> # ℹ 13 more variables: bat_score_before <int>, field_score <int>,
#> #   basecode_before <int>, batterid <chr>, pitcherid <chr>, cid <chr>,
#> #   hp_umpid <chr>, plate_location_x <dbl>, plate_location_z <dbl>,
#> #   rel_speed <dbl>, spin_rate <dbl>, induced_vert_break <dbl>,
#> #   horizontal_break <dbl>

table(pitch_data$is_strike)  # slight class imbalance
#> 
#>     0     1 
#> 37530 17175

# rm first row: is_strike is the response!
tibble::enframe(sapply(pitch_data, class)) |> tail(-1L)
#> # A tibble: 21 × 2
#>    name             value    
#>    <chr>            <chr>    
#>  1 inning           integer  
#>  2 is_bottom        integer  
#>  3 balls            integer  
#>  4 strikes          integer  
#>  5 outs_before      integer  
#>  6 is_lhp           integer  
#>  7 is_lhb           integer  
#>  8 pitch_type       character
#>  9 bat_score_before integer  
#> 10 field_score      integer  
#> # ℹ 11 more rows
```

Notice there is a class imbalance in the strike response (~ 2:1) which
could be problematic generalizing to new data outside of these training
data. See my other tutorial on the dangers of class imbalance [here]().
Since there samples (pitches) are not in short supply (unlike in, for
examples, biological data sets), I will simply down sample the major
class for training.

``` r
pitch_data2 <- rebalance(pitch_data, is_strike)

table(pitch_data2$is_strike)  # class imbalance removed
#> 
#>     0     1 
#> 17175 17175
```

I used a combination of univariate testing, forward- and
backward-feature selection, stability selection, and simple heuristics
(i.e common sense to exclude certain variables) to arrive at a final
feature set.

The following 4 features were chosen:

``` r
feats <- c("plate_location_x",
           "plate_location_z",
           "strikes",
           "balls")
```

Not surprisingly, `plate_location_*` coordinates were by far the most
significant predictors in most model building exercises, followed by
`balls` and `strikes`. As one would expect, the pitch count was highly
influential on upcoming pitch location. Incidentally, PCA revealed
`spin_rate` dominated the variance, as it was the first (principal)
component containing over 99% of the total variance, however this
variation was *not* associated with `is_strike`.

## Fit Model

I evaluated numerous model types and eventually decided on a Random
Forest model. In my experience CART methods can perform especially well
with discrete variables/predictors (i.e. `strikes` and `balls`).

``` r
rf_model <- withr::with_seed(123, {  # set seed for reproducibility
  randomForest::randomForest(        # use randomForest package
    as.matrix(pitch_data2[, feats]),  # feature data matrix
    as.factor(pitch_data2$is_strike), # convert response to factor
    ntree = 250
  )
})

# Gini Importance by feature
get_gini(rf_model)
#> # A tibble: 4 × 2
#>   Feature          Gini_Importance
#>   <chr>                      <dbl>
#> 1 plate_location_z           7624.
#> 2 plate_location_x           7306.
#> 3 strikes                     842.
#> 4 balls                       233.
```

and predict strike probability:

``` r
rf_probs <- predict(rf_model,
                    newdata = pitch_data2[, feats], # predict on *training* data
                    type = "prob")[, 2L]            # class 2 = strike

cmat <- calc_confusion(pitch_data2$is_strike, rf_probs, pos_class = 1L) # confusion matrix

summary(cmat) # evaluate performance
#> ══ Confusion Matrix Summary ════════════════════════════════════════════════════
#> ── Confusion ───────────────────────────────────────────────────────────────────
#> 
#> Positive Class: 1
#> 
#>      Predicted
#> Truth     0     1
#>     0 16523   652
#>     1    91 17084
#> 
#> ── Performance Metrics (CI95%) ─────────────────────────────────────────────────
#> 
#> # A tibble: 10 × 5
#>    metric              n estimate CI95_lower CI95_upper
#>    <chr>           <int>    <dbl>      <dbl>      <dbl>
#>  1 Sensitivity     17175   0.995      0.993      0.996 
#>  2 Specificity     17175   0.962      0.959      0.965 
#>  3 PPV (Precision) 17736   0.963      0.960      0.966 
#>  4 NPV             16614   0.995      0.993      0.996 
#>  5 Accuracy        34350   0.978      0.977      0.980 
#>  6 Bal Accuracy    34350   0.978      0.977      0.980 
#>  7 Prevalence      34350   0.5        0.494      0.506 
#>  8 AUC             34350   0.999      0.999      0.999 
#>  9 Brier Score     34350   0.0178     0.0162     0.0194
#> 10 MCC                NA   0.957     NA         NA     
#> 
#> ── Additional Statistics ───────────────────────────────────────────────────────
#> 
#> F_measure    G_mean    Wt_Acc 
#>     0.979     0.978     0.987
```

Model performance was surprisingly accurate. Stark contrast to my
experience in Life Sciences (proteomics) where performance is typically
*much* lower and `AUC > 0.95` are uncommon.

Also keep in mind that performance here was evaluated on the *training*
data, and a typical machine learning training and test setup would
certainly generate reduced *test* performance.

That said, it should be noted that random forest models do perform a
sort of *quasi*-internal cross-validation, out-of-bag (OOB) samples,
that should guard (somewhat) against over-fitting.

## Append predictions to original data

It is often safer to immediately append the predicted probabilities to
the original data set so they do not become out-of-sync:

``` r
pitch_data2$strike_prob <- rf_probs

dplyr::select(pitch_data2, all_of(feats), is_strike, strike_prob)
#> # A tibble: 34,350 × 6
#>    plate_location_x plate_location_z strikes balls is_strike strike_prob
#>               <dbl>            <dbl>   <int> <int>     <int>       <dbl>
#>  1            1.16             2.30        1     0         0       0.02 
#>  2            0.038            0.972       1     1         0       0    
#>  3           -0.016            1.62        1     1         0       0.456
#>  4            0.936            2.97        1     0         0       0.324
#>  5            1.52             1.84        0     0         0       0.016
#>  6           -0.28             3.62        0     0         0       0.104
#>  7            0.13             3.5         0     1         0       0.324
#>  8           -0.958            2.32        0     0         0       0.556
#>  9            1.18             3.47        0     0         0       0.008
#> 10           -1.56             2.50        2     2         0       0.008
#> # ℹ 34,340 more rows
```

## ROC

At this point generating a ROC curve of the predictive performance of
the model is superfluous, but I’ll do it anyway:

``` r
plot_emp_roc(pitch_data2$is_strike, pitch_data2$strike_prob, pos_class = 1L,
             plot_fit = TRUE, lwd = 1, cutoff_shape = 21,
             cutoff_size = 2.5, outline = FALSE, col = "#002D72") +
  ggtitle("Strike Classifier ROC Curve")
```

![](figures/strike-roc-1.png)

Perhaps another visual that can be useful is a log-odds plot, where the
predictions are plotted against the decision boundary to see how close
they are. Sort of a visual representation of the Brier Score.

Because there are 34350 samples, this plot can become cluttered so I
will randomly sample 250 pitches to represent patterns in the
predictions.

``` r
odds_data <- withr::with_seed(123, dplyr::sample_n(pitch_data2, size = 250L))
plot_log_odds(odds_data$is_strike, odds_data$strike_prob, pos_class = 1L) +
  ggplot2::ggtitle("Log-Odds RF Strike Classifier")
```

![](figures/strike-log-odds-1.png)

A curious pattern emerges:

1.  incorrectly classed pitches are directly next to the boundary line
    (which is good!).
2.  there are a subset (majority?) of pitches with extreme
    probabilities, that have been thresholded by the plotting routine.
    The classifier is *absolutely* sure about these predictions,
    however, what is responsible for the gap between these clusters?
    This is odd. **TODO:** look into this further … something with a
    Random Forest?

------------------------------------------------------------------------

Created in `RStudio` (`v2024.09.1+394`), by
[Quarto](https://quarto.org/) (`v1.4.555`), and R version 4.4.1
(2024-06-14).
