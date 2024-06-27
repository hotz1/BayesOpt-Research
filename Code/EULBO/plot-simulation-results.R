library(here)
library(tidyverse)

sims <- read_csv("./Code/EULBO/Sim-Results/Separate_Simulation_Results_Full.csv")
sims <- sims %>%
  mutate(Simulation = as.factor(Simulation))

sims %>%
  ggplot(aes(x = Epoch, y = trueBest, group = Simulation, colour = Simulation)) +
  geom_line() +
  facet_wrap(~Actions, labeller = as_labeller(function(t) paste(t, "Actions"))) +
  theme_bw() +
  labs(title = "20 Simulations of Separate Optimization for (S, x)") +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 3.32237, color = "red", linetype = "dashed")

ggsave("./Code/EULBO/Sim-Results/Separate_Simulation_Plots.png", device = "png",
       width = 10, height = 10, units = "in")
