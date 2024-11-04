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

# RBF kernel
rbf_kernel <- function(X1, X2, ls, os){
  # Squared exponential kernel distance between X1 and X2
  # ls is lengthscale parameter
  # os is outputscale, i.e. k(x, x)
  
  # scale values by lengthscale
  X1_scaled <- cbind(X1[,1]/ls[1], X1[,2]/ls[2])
  X2_scaled <- cbind(X2[,1]/ls[1], X2[,2]/ls[2])
  
  # compute squared euclidean distances
  sq_euc_dists <- outer(
    1:nrow(X1_scaled), 
    1:nrow(X2_scaled),
    FUN = Vectorize(function(i, j) sum((X1_scaled[i,] - X2_scaled[j,])^2))
  )
  
  return(os * exp(-0.5 * sq_euc_dists))
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
  new_to_new <- rbf_kernel(x_new, x_new, ls, os) + noise_sd^2 * diag(nrow(x_new))
  new_to_obs <- rbf_kernel(x_new, x_obs, ls, os)
  obs_to_obs <- rbf_kernel(x_obs, x_obs, ls, os) + noise_sd^2 * diag(nrow(x_obs))
  
  # Posterior predictive mean and variance
  post_mean <- mu_f + new_to_obs %*% solve(obs_to_obs, y_obs - mu_f)
  post_covmat <- new_to_new - new_to_obs %*% solve(obs_to_obs, t(new_to_obs))
  return(list(mean = post_mean, cov = post_covmat))
}

# Dataset of "test" points for the function
X_test <- expand_grid(x1 = seq(0.025, 0.975, 0.05), x2 = seq(0.025, 0.975, 0.05))
X_test <- X_test %>% mutate(y = f_true(x1, x2))

# Plot the true function 
f_true_plot <- ggplot(X_test) +
  geom_tile(aes(x = x1, y = x2, fill = y), colour = "#00000080", linewidth = 0.15) +
  labs(title = TeX(r"(Plot of the "true" function $f^{*}(x_{1}, x_{2}) = \sin(\pi{x_{1}}) - (2x_{2} - 1)^2$)"),
       x = TeX("$x_{1}$"), y = TeX("$x_{2}$")) +
  coord_fixed() +
  theme_minimal() +
  scale_fill_gradient2(low = "#ff0088", mid = "#ffffff", high = "#00ff88") +
  guides(fill = guide_colourbar(title = TeX(r"($f^{*}(x_{1}, x_{2})$)"),
                                barwidth = 1, barheight = 15, nbin = 12)) +
  theme(plot.title = element_text(hjust = 0.5), legend.title = element_text(hjust = 0.5))

ggsave("f_true_plot-2D.jpg", plot = f_true_plot, device = "jpg", 
       path = here("Vanilla-HighDim/Code/Plots"), width = 8, height = 6, units = "in")


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
    geom_tile(aes(x = x1, y = x2, alpha = EI), fill = "#0000b0", data = posterior_df) + 
    geom_point(aes(x = x1, y = x2), colour = "red", data = obs_df) + 
    geom_text(aes(x=x1, y = x2, label = paste0("y = ", formatC(y, digits = 3, format = "f"))),
              vjust = -1, hjust = 0.25, data = obs_df) +
    labs(title = TeX(r"(Expected Improvement from querying $f(x_{1}, x_{2})$)"),
         subtitle = TeX(str_glue(r"($\mu_{{f}} = {mu_f}, \sigma_{{f}} = {sigma_f}, \sigma_{{\epsilon}} = {sigma_e}, ls = ({ls[1]}, {ls[2]}), os = {os}$)")),
         x = TeX("$x_{1}$"), y = TeX("$x_{2}$")) +
    coord_fixed() +
    theme_minimal() +
    scale_alpha_continuous(range = c(0, 1)) + 
    theme(plot.title = element_text(hjust = 0.5), 
          plot.subtitle = element_text(hjust = 0.5))
  
  return(gp_EI_plot)
}

# Do plots
for(i in 1:5){
  obs_df <- tibble(x1 = runif(2), x2 = runif(2), y = f_true(x1,x2))
  for(j in c(5, 10, 25, 50, 100)){
    gp_plot <- GP_posterior(X_test %>% select(x1, x2), obs_df, ls = c(j/100, j/100), os = 1)
    ggsave(paste0("gp_posteriors_", i, "_", formatC(j, width = 3, flag = "0"), ".jpg"),
           plot = gp_plot, device = "jpg", path = here("Vanilla-HighDim/Code/Plots/GP-Fixed-Obs-2D"), 
           width = 8, height = 8, units = "in")
  }
}
