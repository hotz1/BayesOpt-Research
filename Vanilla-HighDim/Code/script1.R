# R script to create plots of EI (expected improvement) for a 'forgetful' Bayesian Optimization framework

# Load necessary packages
library(tidyverse)
library(here)
library(latex2exp)
library(ggpubr)

# Fix constants (prior + hyperparameters) 
lenscale = 0.25
mu_f = 0
sigma_f = 1
sigma_e = 0

# Define an "unknown" function on [0, 1] for Bayesian optimization
f_true <- function(x){
  return(2 * sin(pi * x) - 1)
}

# Plot the true function 
f_true_plot <- ggplot() + 
  stat_function(fun = f_true, n = 501, color = "#ff8000",
                linewidth = 0.75, xlim = c(0,1)) + 
  labs(x = "x", y = TeX(r"($f^{*}(x)$)"), 
       title = TeX(r"(Plot of the "true" function $f^{*}(x) = 2\sin(\pi{x}) - 1$)")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Add the global maximum to the plot
f_true_plot <- f_true_plot +
  geom_point(aes(x = 0.5, y = f_true(0.5)), 
             fill = "#0000b0", size = rel(3), shape = 23)

ggsave("f_true_plot.jpg", plot = f_true_plot, device = "jpg", 
       path = here("Vanilla-HighDim/Code/Plots"), width = 8, height = 6, units = "in")

# RBF kernel
rbf_kernel <- function(X1, X2, ls = 1, os = 1){
  # Squared exponential kernel distance between X1 and X2
  # ls is lengthscale parameter
  # os is outputscale, i.e. k(x, x)
  
  sq_dists <- outer(X1, X2, \(a,b) (b - a)^2)
  return(os * exp(-0.5 * sq_dists/ls^2))
  
}

# Prior mean function (fixed zero function)
prior_mean <- function(x){
  return (0 * x + mu_f)
}

# Returns mean and covariance of posterior predictive distribution
posterior_pred <- function(x_new, x_obs, y_obs, noise_sd = 0, ls = 1, os = 1){
  # Returns posterior predictive mean and covariance matrix evaluated at x_new
  # based on observations (x_obs, y_obs) and hyperparameters
  
  
  # Compute kernels
  new_to_new <- rbf_kernel(x_new, x_new, ls, os) + noise_sd^2 * diag(length(x_new))
  new_to_obs <- rbf_kernel(x_new, x_obs, ls, os)
  obs_to_obs <- rbf_kernel(x_obs, x_obs, ls, os) + noise_sd^2 * diag(length(x_obs))
  
  # Posterior predictive mean and variance
  post_mean <- prior_mean(x_new) + new_to_obs %*% solve(obs_to_obs, y_obs - prior_mean(x_obs))
  post_covmat <- new_to_new - new_to_obs %*% solve(obs_to_obs, t(new_to_obs))
  return(list(mean = post_mean, cov = post_covmat))
}

# Dataset of "test" points for the function
x_test <- seq(0, 1, length.out = 501)
X_test_df <- tibble(x = x_test, mean = prior_mean(x_test), sd = 1)

# Plot GP prior
gp_prior_plot <- X_test_df %>%
  ggplot() +
  geom_ribbon(aes(x = x, ymin = qnorm(0.025, mean, sd), ymax = qnorm(0.975, mean, sd)),
              fill = "#dedede80", colour = "black", linetype = "dashed") +
  geom_line(aes(x = x, y = mean), colour = "black", linewidth = 1) +
  labs(x = "x", y = TeX(r"($f^{*}(x)$)"),
       title = TeX(r"(GP prior for $f^{*}(x)$)"),
       subtitle = TeX(str_glue(r"($\mu_{{f}} = {mu_f}, \sigma_{{f}} = {sigma_f}, \sigma_{{\epsilon}} = {sigma_e}, ls = {lenscale}$)"))) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("gp_prior.jpg", plot = gp_prior_plot, device = "jpg", 
       path = here("Vanilla-HighDim/Code/Plots"), width = 6, height = 6, units = "in")


# Function to compute + plot GP posterior and EI
GP_posterior <- function(x_test, obs_df){
  # Compute posterior 
  test_post <- posterior_pred(x_test, obs_df$x, obs_df$y, noise_sd = sigma_e, ls = lenscale, os = sigma_f)
  
  posterior_df <- tibble(x = x_test, mean = test_post$mean, 
                         sd = sqrt(diag(test_post$cov)))
  
  # Plot GP posterior
  gp_posterior_plot <- ggplot() +
    geom_ribbon(aes(x = x, ymin = qnorm(0.025, mean, sd), ymax = qnorm(0.975, mean, sd)),
                data = posterior_df, fill = "#dedede80", colour = "black", linetype = "dashed") +
    geom_line(aes(x = x, y = mean), data = posterior_df, colour = "black", linewidth = 1) +
    geom_point(aes(x = x, y = y), data = obs_df, colour = "red", size = rel(2)) +
    labs(x = "x", y = TeX(r"($f^{*}(x)$)"),
         title = TeX(r"(GP posterior for $f^{*}(x)$)"),
         subtitle = TeX(str_glue(r"($\mu_{{f}} = {mu_f}, \sigma_{{f}} = {sigma_f}, \sigma_{{\epsilon}} = {sigma_e}, ls = {lenscale}$)"))) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
  
  # Plot expected improvement
  y_best <- max(obs_df$y)
  posterior_df <- posterior_df %>%
    mutate(Z = (mean - y_best)/sd) %>%
    mutate(EI = sd * (Z * pnorm(Z) + dnorm(Z)))
  
  # Plot GP posterior
  gp_EI_plot <- ggplot() +
    geom_vline(aes(xintercept = x), data = obs_df, colour = "red", linetype = "dashed") +
    geom_area(aes(x = x, y = EI), data = posterior_df, fill = "#80ff4080") +
    geom_line(aes(x = x, y = EI), data = posterior_df, colour = "black", linewidth = 1) +
    labs(x = "x", y = "Expected Improvement",
         title = TeX(r"(Expected Improvement from querying $f^{*}(x)$)"),
         subtitle = TeX(str_glue(r"($\mu_{{f}} = {mu_f}, \sigma_{{f}} = {sigma_f}, \sigma_{{\epsilon}} = {sigma_e}, ls = {lenscale}$)"))) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
  
  ### TODO: NUMERICALLY FIND ARGMAX EI AND ADD TO PLOT
  
  return(ggarrange(gp_posterior_plot, gp_EI_plot, ncol = 1))
}

# Do plots
for(i in 1:20){
  obs_df <- tibble(x = runif(2), y = f_true(x))
  gp_post_plots <- GP_posterior(x_test, obs_df)
  
  ggsave(paste0("gp_posteriors_", i, ".jpg"), plot = gp_post_plots, device = "jpg",
         path = here("Vanilla-HighDim/Code/Plots/GP-Posteriors"), 
         width = 8, height = 8, units = "in")
}
