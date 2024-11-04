# R script to create plots of EI (expected improvement) for a 'forgetful' Bayesian Optimization framework
# where the function is a real-valued function on R^2

# Load necessary packages
library(tidyverse)
library(here)
library(latex2exp)
library(ggpubr)

# Fix constants (prior + hyperparameters) 
lenscale = c(0.25, 0.25)
mu_f = 0
sigma_f = 1
sigma_e = 0

# Define an "unknown" function on [0, 1] for Bayesian optimization
f_true <- function(x1, x2){
  return(sin(pi * x1) - 4 * (x2 - 1/2)^2)
}

# Matérn(1/2) kernel 
matern_kernel <- function(X1, X2, ls, os){
  # Distance between X1 and X2
  # ls is lengthscale parameter
  # os is outputscale, i.e. k(xi, xi)

  # scale values by lengthscale
  X1_scaled <- t(t(X1)/ls)
  X2_scaled <- t(t(X2)/ls)

  # compute euclidean distances
  euc_dists <- outer(
    1:nrow(X1_scaled),
    1:nrow(X2_scaled),
    FUN = Vectorize(function(i, j) sqrt(sum((X1_scaled[i,] - X2_scaled[j,])^2)))
  )

  return(os * exp(-euc_dists))
}

# Prior mean function (fixed zero function)
prior_mean <- function(x1, x2){
  return (0 * x1 + 0 * x2 + mu_f)
}

# Returns mean and covariance of posterior predictive distribution
posterior_pred <- function(x_new, x_obs, y_obs, noise_sd, ls, os){
  # Returns posterior predictive mean and covariance matrix evaluated at x_new
  # based on observations (x_obs, y_obs) and hyperparameters
  x_new <- as.matrix(x_new)
  x_obs <- as.matrix(x_obs)
  y_obs <- as.matrix(y_obs)
  
  # Compute kernels
  new_to_new <- matern_kernel(x_new, x_new, ls, os) + noise_sd^2 * diag(nrow(x_new))
  new_to_obs <- matern_kernel(x_new, x_obs, ls, os)
  obs_to_obs <- matern_kernel(x_obs, x_obs, ls, os) + noise_sd^2 * diag(nrow(x_obs))
  
  # Posterior predictive mean and variance
  post_mean <- mu_f + new_to_obs %*% solve(obs_to_obs, y_obs - mu_f)
  post_covmat <- new_to_new - new_to_obs %*% solve(obs_to_obs, t(new_to_obs))
  return(list(mean = post_mean, cov = post_covmat))
}

# Dataset of "test" points for the function
X_test <- expand_grid(x1 = seq(0.01, 0.99, 0.02), x2 = seq(0.01, 0.99, 0.02))
X_test <- X_test %>% mutate(y = f_true(x1, x2))

# Function to compute + plot GP posterior and EI
GP_posterior <- function(x_test, obs_df, ls, os){
  # Compute posterior 
  test_post <- posterior_pred(x_test, obs_df[,1:2], obs_df[,3], noise_sd = sigma_e, ls, os)
  
  posterior_df <- tibble(x1 = x_test$x1, x2 = x_test$x2, mean = test_post$mean[,1], 
                         sd = sqrt(diag(test_post$cov)))

  # Plot expected improvement
  y_best <- max(obs_df$y)
  posterior_df <- posterior_df %>%
    mutate(Z = (mean - y_best)/sd) %>%
    mutate(EI = sd * (Z * pnorm(Z) + dnorm(Z)))
  
  gp_EI_plot <- ggplot() + 
    geom_tile(aes(x = x1, y = x2, fill = EI), colour = "#00000080", data = posterior_df) + 
    geom_point(aes(x = x1, y = x2), colour = "red", data = obs_df) + 
    geom_text(aes(x=x1, y = x2, label = paste0("y = ", formatC(y, digits = 3, format = "f"))),
              vjust = -1, hjust = 0.25, data = obs_df) +
    labs(title = TeX(r"(Expected Improvement from querying $f(x_{1}, x_{2})$)"),
         subtitle = TeX(str_glue(r"($\mu_{{f}} = {mu_f}, \sigma_{{f}} = {sigma_f}, \sigma_{{\epsilon}} = {sigma_e}, ls = ({ls[1]}, {ls[2]}), os = {os}$)")),
         x = TeX("$x_{1}$"), y = TeX("$x_{2}$")) +
    coord_fixed() +
    theme_minimal() +
    scale_fill_gradient(low = "#ffffff", high = "#0000ff") + 
    theme(plot.title = element_text(hjust = 0.5), 
          plot.subtitle = element_text(hjust = 0.5))
  
  return(gp_EI_plot)
}

# Do plots
for(i in 2:10){
  obs_df <- tibble(x1 = runif(3), x2 = runif(3), y = f_true(x1,x2))
  for(j in c(5, 10, 25, 50, 100)){
    gp_plot <- GP_posterior(X_test %>% select(x1, x2), obs_df, ls = c(j/100, j/100), os = 1)
    ggsave(paste0("gp_posteriors_", i, "_", formatC(j, width = 3, flag = "0"), ".jpg"),
           plot = gp_plot, device = "jpg", path = here("Vanilla-HighDim/Code/Plots/BayesOpt-2D-3Obs"), 
           width = 8, height = 8, units = "in")
  }
}
