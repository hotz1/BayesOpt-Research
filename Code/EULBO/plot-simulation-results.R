library(here)
library(tidyverse)

# Read in data
all_csvs <- list.files(here("Code/EULBO/Sim-Results/RawData"), 
                       pattern = ".csv", full.names = T)
csv_list <- list()
for(i in 1:length(all_csvs)){
  csv_list[[i]] <- read_csv(all_csvs[i])
}

# Combine data
all_sims <- bind_rows(csv_list)

# Make Simulation no. categorical
all_sims <- all_sims %>%
  mutate(Simulation = as.factor(Simulation))

# Check number of epochs for each thing
all_sims %>% 
  group_by(Simulation, ActsName) %>% 
  summarize(Total = n(), 
            Epochs = max(Epoch))
  

# Summarize by type and by epoch
epoch_summary <- all_sims %>%
  group_by(Epoch, ActsName) %>%
  summarize(y_mean = mean(trueBest), 
            y_sd = sd(trueBest), 
            Sims = n())

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

