# LIBRARIES AND PREP------------------------------------------------------------
library(missForest)
library(dplyr)
library(imputeR) # Parkinson's dataset
library(kernlab) # Musk dataset

# Check versions for missForest, randomForest, and R
cat("missForest version:", as.character(packageVersion("missForest")), "\n")
cat("randomForest version:", as.character(packageVersion("randomForest")), "\n")
cat("R version:", R.version.string, "\n")

# Function to run the experiments
# tag is only used to name the .rds file that saves the data after every missing rate
run_exp <- function(true, n_sim, rates = c(0.10, 0.20, 0.30), seed = 2208, backend = "randomForest", tag = "exp") {
  set.seed(seed)
  result <- vector("list", length(rates) * n_sim)
  k <- 0
  for (rate in rates) {
    for (i in 1:n_sim) {
      # prodNA is a built-in function in missForest that creates the artificially missing values with a given missing rate
      data_na <- prodNA(true, noNA = rate)
      # missForest with all the default hyperparameters (ntree=100, mtry=sqrt(p)), timed to compare with the runtimes in the paper
      # Modern missForest defaults to ranger as the backend for speed, but the original paper uses randomForest, so we default to it
      tm <- system.time(
        mf <- missForest(data_na,
                         xtrue = true,
                         ntree = 100,
                         mtry = floor(sqrt(ncol(true))),
                         backend = backend,
                         verbose = FALSE)
      )[["elapsed"]]
      
      # Pull the OOB error that's automatically calculated by missForest
      oob <- mf$OOBerror
      names(oob) <- paste0("OOB_", names(oob))
      
      k <- k + 1
      result[[k]] <- data.frame(rate = rate, sim = i, t(mf$error), t(oob), seconds = tm)
    }
    # Save data into a dedicated .rds file after each completed missingness rate
    saveRDS(bind_rows(result[seq_len(k)]), paste0(tag, ".rds"))
  }
  bind_rows(result)
}

# Summary of the simulations
exp_summary <- function(r) {
  r %>%
    group_by(rate) %>%
    # Calculate the mean and standard error of every metric: true error (NRMSE/PFC), OOB estimate, and the runtime (s)
    summarise(across(-sim, list(mean = mean, se = ~ sd(.x) / sqrt(length(.x))))) %>%
    as.data.frame()
}

# CONTINUOUS DATA --------------------------------------------------------------

# PARKINSON'S DATASET ----------------------------------------------------------

# Pull the dataset from the imputeR library
data(parkinson)
park_matrix <- as.matrix(parkinson)

# Check if the dimensions of the dataset match the ones in the paper (195 rows, 22 cols)
stopifnot(identical(dim(park_matrix), c(195L, 22L)))

# Parkinson's is a relatively computationally cheap dataset, so we run 50 simulations per missingness rate
exp_park <- run_exp(true = park_matrix, n_sim = 50, tag = "park")

cat("\n Parkinson's Average NRMSE \n")
print(exp_summary(exp_park))


# CATEGORICAL DATA -------------------------------------------------------------

# SPECT DATASET ----------------------------------------------------------------

# SPECT data had to be pulled from the UCI archive. It's already pre-split, so we need to re-join it.
spect_train <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train", header = FALSE)
spect_test <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test", header = FALSE)
spect_raw <- rbind(spect_train, spect_test)

# Remove column 1 (the diagnosis label) to keep only the predictor columns
spect_data <- spect_raw %>% select(2:23)

# Factorize categorical variables to make sure missForest uses classification trees
spect_df <- as.data.frame(lapply(spect_data, as.factor))

# Check that the dimensions match (267 rows, 22 cols)
stopifnot(identical(dim(spect_df), c(267L, 22L)))

# SPECT is also a cheap dataset to run, so we do 50 simulations per missingness rate like with Parkinson's
exp_spect <- run_exp(true = spect_df, n_sim = 50, tag = "spect")

cat("\n SPECT Average PFC \n")
print(exp_summary(exp_spect))


# MIXED DATA -------------------------------------------------------------------

# MUSK DATASET -----------------------------------------------------------------

data(musk)
# Unlike the continuous case, mixed data has to stay as a data frame: 166 continuous columns and categorical class label in col 167
musk_df <- as.data.frame(musk)

# Check that the dimensions match (476 rows, 167 cols)
stopifnot(identical(dim(musk_df), c(476L, 167L)))
# Check the categorical part of the dataset
stopifnot(is.factor(musk_df[[167]]))
# Check the continuous part of the dataset
stopifnot(all(sapply(musk_df[, 1:166], is.numeric)))

# Musk is a rather computationally expensive dataset, so we reduce the number of simulations per missingness rate to 10 to save time
exp_musk <- run_exp(true = musk_df, n_sim = 10, tag = "musk")

cat("\n Musk (Mixed) Average NRMSE and PFC \n")
print(exp_summary(exp_musk))