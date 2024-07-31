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

# Check number of epochs per simulation and filter
all_sims <- all_sims %>% 
  group_by(Simulation, ActsName) %>% 
  mutate(Total = n()) %>%
  filter(Total == 501)
  

# Summarize by type and by epoch
epoch_summary <- all_sims %>%
  group_by(Epoch, ActsName) %>%
  summarize(y_mean = mean(trueBest), 
            y_sd = sd(trueBest), 
            time_mean = mean(cpuTime),
            time_sd = sd(cpuTime),
            Sims = n()) %>%
  mutate(ActsName = paste0(ActsName, " (", Sims, " Simulations)"))

# Create plot of trueBest
epoch_summary %>%
  ggplot() +
  geom_ribbon(aes(x = Epoch, ymin = y_mean + qnorm(0.025) * y_sd, ymax = y_mean + qnorm(0.975) * y_sd), 
              color = "#CEEDEE", linewidth = 1.5, fill = "#CEEDEE88") +
  geom_line(aes(x = Epoch, y = y_mean), colour = "#7E87F2", linewidth = 1) +
  labs(x = "Epoch", y = "trueBest", 
       title = "Average true best value from Separate Optimization") +
  facet_wrap(~ActsName) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), 
        plot.subtitle = element_text(hjust = 0.5))

ggsave(here("Code/EULBO/Sim-Results/Plots/TrueBest_Comparison.png"), device = "png",
       width = 12, height = 8, units = "in")

# Create plot of trueBest
epoch_summary %>%
  filter(Epoch > 0) %>%
  ggplot() +
  geom_ribbon(aes(x = Epoch, ymin = time_mean + qnorm(0.025) * time_sd, 
                  ymax = time_mean + qnorm(0.975) * time_sd), 
              color = "#CEEDEE", linewidth = 1.5, fill = "#CEEDEE88") +
  geom_line(aes(x = Epoch, y = time_mean), colour = "#7E87F2", linewidth = 1) +
  labs(x = "Epoch", y = "CPU Time (s)", 
       title = "Average time taken per epoch of Separate Optimization") +
  facet_wrap(~ActsName) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), 
        plot.subtitle = element_text(hjust = 0.5))

ggsave(here("Code/EULBO/Sim-Results/Plots/TimeTaken_Comparison.png"), device = "png",
       width = 12, height = 8, units = "in")
