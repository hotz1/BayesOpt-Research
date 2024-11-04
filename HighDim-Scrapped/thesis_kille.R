# Thesis Killer (maybe?) 
# "FAIL FAST!"
# Last Updated: October 23, 2024

# Goal: See if there is really something worth exploring after all in this project.


library(tidyverse)

# Function to calculate EI given ||y||, ymax, ||k*|| and cos(theta)
EI <- function(y_Kinv, y_max, k_Kinv, cosine){
  mu_post <- k_Kinv * cosine * y_Kinv
  sigma_post <- sqrt(1 - k_Kinv^2)
  zscore <- (mu_post - y_max)/sigma_post
  return((mu_post - y_max)*pnorm(zscore,0,1) + sigma_post * dnorm(zscore,0,1))
}

# Get a whole bunch of test cases
my_tbl <- expand_grid(
  y = c(0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50,
        0.65, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00),
  ymax = c(0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50,
           0.65, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00),
  kstar = seq(0.0125, 0.9875, length.out = 40),
  cosine = seq(0.02, 1, by = 0.02)
)

# Add in test cases when ||k*|| = cos(theta) = 0
my_tbl_zeros <- expand_grid(
  y = c(0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50,
        0.65, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00),
  ymax = c(0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50,
           0.65, 0.80, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00),
  kstar = 0,
  cosine = 0
)

my_tbl <- rbind(my_tbl, my_tbl_zeros)

# Filter out cases where ymax > y 
my_tbl <- my_tbl %>% filter(ymax <= y)

# Compute EI for each casr
my_tbl <- my_tbl %>% mutate(EI = EI(y, ymax, kstar, cosine))

# Find theoretical max EI per combo of y, ymax
EI_maxes <- my_tbl %>% 
  group_by(y, ymax) %>% 
  top_n(1, EI)

# Find theoretical min EI per combo of y, ymax
EI_mins <- my_tbl %>% 
  group_by(y, ymax) %>% 
  top_n(1, -EI)


ggplot() +
  geom_tile(aes(x = kstar, y = cosine, fill = EI), colour = NA, data = my_tbl) +
  geom_point(aes(x = kstar, y = cosine), fill = 'green', shape = 24, data = EI_maxes) +
  geom_point(aes(x = kstar, y = cosine), fill = 'red', shape = 25, data = EI_mins) +
  theme_bw() +
  scale_fill_gradient(low = "#ffffff", high = "#0000ff") + 
  facet_grid(y ~ ymax, labeller = label_both)

ggsave("Thesis_Killer_ALT.png", device = png, scale = 5)
