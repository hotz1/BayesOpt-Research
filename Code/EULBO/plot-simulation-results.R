library(here)
library(tidyverse)

# Read data
sims <- read_csv(here("Code/EULBO/Sim-Results/Combined", "500Epochs_All.csv"))

# Make Simulation no. categorical
sims <- sims %>%
  mutate(Simulation = as.factor(Simulation))

sim_summary <- sims %>%
  group_by(Epoch, ActsName) %>%
  summarize(y_mean = mean(obsBest), y_sd = sd(trueBest))

# # Create plot
# sim_summary %>% 
#   filter(ActsName == "15 Actions") %>%
#   ggplot() +
#   geom_ribbon(aes(x = Epoch, ymin = y_mean + qnorm(0.025) * y_sd, ymax = y_mean + qnorm(0.975) * y_sd), 
#               color = "#CEEDEE", linewidth = 1.5, fill = "#CEEDEE88") +
#   geom_line(aes(x = Epoch, y = y_mean), colour = "#7E87F2", linewidth = 1) +
#   labs(x = "Epoch", y = "trueBest", 
#        title = "Average true best value over 6 simulations of Separate Optimization",
#        subtitle = "Projection of T = 15 Actions") +
#   theme_bw() +
#   theme(plot.title = element_text(hjust = 0.5), 
#         plot.subtitle = element_text(hjust = 0.5))
# 
# # Save plot
# ggsave(here("Code/EULBO/Sim-Results/Plots/T15_Plot.png"), device = "png",
#        width = 10, height = 10, units = "in")
# 
# # Create plot
# sim_summary %>% 
#   filter(ActsName == "sqrt(N) Actions") %>%
#   ggplot() +
#   geom_ribbon(aes(x = Epoch, ymin = y_mean + qnorm(0.025) * y_sd, ymax = y_mean + qnorm(0.975) * y_sd), 
#               color = "#CEEDEE", linewidth = 1.5, fill = "#CEEDEE88") +
#   geom_line(aes(x = Epoch, y = y_mean), colour = "#7E87F2", linewidth = 1) +
#   labs(x = "Epoch", y = "trueBest", 
#        title = "Average true best value over 6 simulations of Separate Optimization",
#        subtitle = "Projection of T = 15 Actions") +
#   theme_bw() +
#   theme(plot.title = element_text(hjust = 0.5), 
#         plot.subtitle = element_text(hjust = 0.5))
# 
# # Save plot
# ggsave(here("Code/EULBO/Sim-Results/Plots/sqrtN_Plot.png"), device = "png",
#        width = 10, height = 10, units = "in")


# Create plot
sim_summary %>%
  ggplot() +
  geom_ribbon(aes(x = Epoch, ymin = y_mean + qnorm(0.025) * y_sd, ymax = y_mean + qnorm(0.975) * y_sd), 
              color = "#CEEDEE", linewidth = 1.5, fill = "#CEEDEE88") +
  geom_line(aes(x = Epoch, y = y_mean), colour = "#7E87F2", linewidth = 1) +
  labs(x = "Epoch", y = "trueBest", 
       title = "Average true best value over 11 total simulations of Separate Optimization") +
  facet_wrap(~ActsName) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), 
        plot.subtitle = element_text(hjust = 0.5))

ggsave(here("Code/EULBO/Sim-Results/Plots/Comparison_Plots1.png"), device = "png",
       width = 12, height = 8, units = "in")

# Create plot
sims %>%
  ggplot() +
  geom_line(aes(x = Epoch, y = trueBest, color = Simulation), linewidth = 1) +
  labs(x = "Epoch", y = "trueBest", 
       title = "Average true best value over 11 total simulations of Separate Optimization") +
  facet_wrap(~ActsName) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), 
        plot.subtitle = element_text(hjust = 0.5))

ggsave(here("Code/EULBO/Sim-Results/Plots/Comparison_Plots2.png"), device = "png",
       width = 12, height = 8, units = "in")

