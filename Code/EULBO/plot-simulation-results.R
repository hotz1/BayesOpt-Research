library(here)
library(tidyverse)

# Read data
# sims <- read_csv(here("Code/EULBO/Sim-Results/Combined/___.csv"))

# Make Simulation no. categorical
sims <- sims %>%
  mutate(Simulation = as.factor(Simulation))

# Create plot
sims %>%
  ggplot(aes(x = Epoch, y = trueBest, group = Simulation, colour = Simulation)) +
  geom_line() +
  facet_wrap(~Actions, labeller = as_labeller(function(t) paste(t, "Actions"))) +
  theme_bw() +
  labs(title = "10 Simulations of Separate Optimization for (S, x)",
       subtitle = "N = 200") +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 3.32237, color = "red", linetype = "dashed")

# Save plot
ggsave(here("Code/EULBO/Sim-Results/Plots/N200_Plots.png"), device = "png",
       width = 10, height = 10, units = "in")
