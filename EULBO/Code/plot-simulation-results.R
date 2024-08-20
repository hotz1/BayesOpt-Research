library(here)
library(tidyverse)

# Select number of epochs per simulation (for finding correct filenames)
cat("Select the number of epochs per simulation.\n")
n_epochs = as.integer(readLines(con = "stdin", n = 1))
# n_epochs <- 500

# Get filenames corresponding to desired number of epochs
all_csvs <- list.files(here("EULBO/Code/Sim-Results/RawData"), 
                       pattern = paste0("_", n_epochs, "E.csv"), full.names = T)

# Read in the data
csv_list <- list()
for(i in 1:length(all_csvs)){
  csv_list[[i]] <- read_csv(all_csvs[i])
}

# Combine data
all_sims <- bind_rows(csv_list)

# Make Simulation number a categorical variable
all_sims <- all_sims %>%
  mutate(Simulation = as.factor(Simulation))

# Check number of epochs per simulation and filter
all_sims <- all_sims %>% 
  group_by(Simulation, ActsName) %>% 
  mutate(Total = n()) %>%
  filter(Total == n_epochs + 1) %>%
  select(-Total)

# Summarize by simulation type and by epoch
epoch_summary <- all_sims %>%
  group_by(Epoch, ActsName) %>%
  summarize(y_mean = mean(trueBest), 
            y_sd = sd(trueBest), 
            time_mean = mean(cpuTime),
            time_sd = sd(cpuTime),
            Sims = n()) %>%
  mutate(ActsName = paste0(ActsName, " (", Sims, " Simulations)"))

# Create plot of trueBest per epoch
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

# Save plot locally
ggsave(filename = paste0("TrueBest_Comparison_", n_epochs, "E.png"),
       path = here("EULBO/Code/Sim-Results/Plots"), 
       device = "png", width = 12, height = 8, units = "in")

# Create plot of time taken
epoch_summary %>%
  filter(Epoch > 0) %>%
  ggplot() +
  geom_ribbon(aes(x = Epoch, ymin = time_mean + qnorm(0.025) * time_sd, 
                  ymax = time_mean + qnorm(0.975) * time_sd), 
              color = "#CEEDEE", linewidth = 1.5, fill = "#CEEDEE88") +
  geom_line(aes(x = Epoch, y = time_mean), colour = "#7E87F2", linewidth = 1) +
  labs(x = "Epoch", y = "CPU Time (sec)", 
       title = "Average time taken per epoch of Separate Optimization") +
  facet_wrap(~ActsName) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), 
        plot.subtitle = element_text(hjust = 0.5))

# Save plot locally
ggsave(filename = paste0("TimeTaken_Comparison_", n_epochs, "E.png"),
       path = here("EULBO/Code/Sim-Results/Plots"), 
       device = "png", width = 12, height = 8, units = "in")
