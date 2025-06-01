# Decision boundaries: KNN vs Naïve Bayes
Stu Field
1 June 2025

# Overview

It can be useful to visualize the decision boundaries of various
models. Here we compare 2 commonly used models:

- Naïve Bayes
- *k*-nearest neighbors (KNN)

----------------------------------------------------------------------

## KNN vs Naïve Bayes

Below are decision boundaries for 2 simulated data sets using
*k*-nearest neighbors and Naïve Bayes models. In the first data set
(upper 2 panels) the true class boundary is simulated such that the
disease (purple) Feature_1 \> 44 and Feature_2 \> 74, these data are
simulated with an *unrealistic* threshold to form the classes. The
lower panels are simulated from bivariate normal distributions,
somewhat more realistic, and show the difference in the boundary
between the two methods.

![](figures/knn-bayes-knn-vs-bayes-1.png)

![](figures/knn-bayes-knn-vs-bayes-2.png)

## Choosing *k* in KNN

``` r
withr::with_par(
  list(mgp = c(2.00, 0.75, 0.00), mar = c(3, 4, 3, 1), mfrow = c(3L, 3L)), {
  for ( i in 2:10L ) {
    plot_decision_boundary(sim_data(), k = i)
  }
})
```

![](figures/knn-bayes-knn-k-1.png)

----------------------------------------------------------------------

### Code Reference

``` r
# simulated data set 1
sim_data
#> function(n = 200) {
#>   withr::local_seed(1001)
#>   # deterministic model
#>   # a = 3; b = 2.14; sd = 250
#>   # y_hat <- a * x^b
#>   # y <- rnorm(length(y.det), mean = y.det, sd = sd) # stochastic 'y'
#>   df   <- data.frame(x1 = rnorm(n, 45, 2), x2 = rnorm(n, 75, 2))
#>   df$y <- ifelse(df$x1 < 44 | df$x2 <= 74, "control", "disease") |> factor()
#>   df
#> }
#> <bytecode: 0x6144d4e6cbf0>

# simulated data set 1
sim_data2
#> function(n = 100) {
#>   withr::local_seed(999)
#>   df1 <- data.frame(x1 = rnorm(n, 46, 1.5), x2 = rnorm(n, 75, 1.5)) # control
#>   df2 <- data.frame(x1 = rnorm(n, 44, 1), x2 = rnorm(n, 78, 1))     # disease
#>   df  <- rbind(df1, df2)
#>   df$y <- factor(rep(c("control", "disease"), each = n))
#>   df
#> }
#> <bytecode: 0x6144cf6ccde8>

# predicting nearest neighbors from scratch
predict_bivariate_knn
#> function(X, y, k, newdata, method = "minkowski") {
#>   if ( k < 2L ) {
#>     stop("Neighborhood (k) must be > 1: ", k, call. = FALSE)
#>   }
#>   if ( missing(newdata) ) {
#>     newdata <- X
#>   }
#>   X   <- data.matrix(X)
#>   ntr <- nrow(X)
#>   if ( length(y) != ntr ) {
#>     stop(
#>        sprintf("Length of class vector [y=%i] unequal to n training samples (n=%i)",
#>                length(y), ntr),
#>        call. = FALSE)
#>   }
#>   if ( ntr < k ) {
#>     warning(
#>       sprintf("Neighborhood (k=%i) exceeds training data (n=%i) ... resetting k=%i",
#>               k, ntr, ntr),
#>       call. = FALSE)
#>       k <- ntr
#>   }
#>   nte <- nrow(newdata)
#>   class_names <- names(table(y))
#>   neighbor_list <- lapply(seq_len(nte), function(.i) {
#>                           new_vals <- newdata[.i, ]
#>                           if ( length(new_vals) != 2L ) {
#>                             stop("Problem with new values ... length =",
#>                                  length(new_vals), call. = FALSE)
#>                           }
#>                           # kknn::kknn() uses Minkowski; class::knn() uses Euclid
#>                           dist_vec <- head(dist(rbind(new_vals, X),
#>                                                 method = method), ntr) |>
#>                             setNames(seq_len(ntr)) |> sort()
#>                           head(dist_vec, k) |>  # get neighborhood (closest dist)
#>                             names() |>
#>                             as.integer()
#>   })
#>   neighbor_prop_disease <- vapply(neighbor_list, function(.x) {
#>     prop.table(table(y[.x]))[[class_names[2L]]] # prop disease in neighborhood
#>   }, double(1))
#>   classes <- ifelse(
#>     neighbor_prop_disease == 0.5,
#>     sample(class_names, 1, prob = prop.table(table(y))), # random tie-break
#>     ifelse(neighbor_prop_disease >= 0.5, class_names[2L], class_names[1L])
#>   )
#>   data.frame(class = classes, prob = neighbor_prop_disease)
#> }
#> <bytecode: 0x6144d642e5c0>

# plotting routine for decision boundary
plot_decision_boundary
#> function(data, res = 50L,
#>                                    model_type = c("knn", "bayes"),
#>                                    k = 15L, line_col = "#00A499",
#>                                    lwd = 2, lty = 1, contours = 0.5) {
#>   y  <- data$y
#>   X  <- data[, c("x1", "x2")]
#>   x1 <- data$x1
#>   x2 <- data$x2
#>   x1grid <- seq(min(x1), max(x1), length = res)
#>   x2grid <- seq(min(x2), max(x2), length = res)
#>   grid <- expand.grid(x1 = x1grid, x2 = x2grid, KEEP.OUT.ATTRS = FALSE)
#>   model_type <- match.arg(model_type)
#>   if ( model_type == "bayes" ) {
#>     rm(k)
#>     model <- libml::fit_nb(y ~ ., data = data)
#>     prob_vec <- predict(model, grid, type = "raw")[, "disease"]
#>     title <- "Naive Bayes | disease (Pr>=0.5) space: "
#>   } else if ( model_type == "knn" ) {
#>     model <- predict_bivariate_knn(X, y, k, grid)
#>     prob_vec <- model$prob
#>     title <- sprintf("KNN (k=%i) | disease (Pr>=0.5) space: ", k)
#>   }
#>   prob_grid <- matrix(prob_vec, nrow = res)
#>   pos_space <- round(sum(prob_grid >= 0.5) / res^2, 3L)
#>   contour(x = x1grid, y = x2grid, z = prob_grid, levels = contours,
#>           lwd = lwd, lty = lty, labcex = 1, vfont = c("sans serif", "bold"),
#>           col = line_col, xlab = "Feature1", ylab = "Feature2",
#>           main = paste0(title, pos_space), axes = TRUE)
#>   col_d <- ggplot2::alpha("#840B55", 0.5)
#>   col_c <- ggplot2::alpha("steelblue", 0.5)
#>   points(grid, pch = "\u2022", cex = 0.75,
#>          col = ifelse(prob_vec >= 0.5, col_d, col_c))
#>   points(X, cex = 1.25, pch = 21, col = 1,
#>          bg = ifelse(y == "disease", col_d, col_c))
#>   invisible(data)
#> }
#> <bytecode: 0x6144d53742f8>
```
