# Load required libraries for data manipulation, visualization, and modeling
library(tidyverse)
library(ggplot2)

# Read the dataset from CSV file
data <- read_csv("data-UG-SCH-sitelevel.csv")

# Filter data to keep only rows where Species is "mansoni"
# Clean and convert relevant columns to appropriate types
data_mansoni <- data %>%
  filter(Species == "mansoni") %>%
  mutate(
    # Replace "null" strings with NA in latitude and longitude columns
    latitude = na_if(latitude, "null"),
    longitude = na_if(longitude, "null"),
    # Convert latitude and longitude to numeric values
    latitude = as.numeric(latitude),
    longitude = as.numeric(longitude),
    # Convert LocationName to character type
    LocationName = as.character(LocationName),
    # Create a logical flag for primary school locations based on "P/S" in LocationName
    is_primary_school = str_detect(LocationName, "P/S"),
    # Convert SurveyYear to integer type
    SurveyYear = as.integer(SurveyYear),
    # Replace "null" with NA and convert Age_start and Age_end to numeric
    Age_start = na_if(Age_start, "null"),
    Age_start = as.numeric(Age_start),
    Age_end = na_if(Age_end, "null"),
    Age_end = as.numeric(Age_end)
  )

# Summary statistics on the filtered mansoni data
total_surveys <- n_distinct(data_mansoni$SiteID)  # Number of unique survey sites
total_tested <- sum(data_mansoni$Examined, na.rm = TRUE)  # Total number of people tested
total_positive <- sum(data_mansoni$Positive, na.rm = TRUE)  # Total positive cases for S. mansoni
completeness <- mean(!is.na(data_mansoni)) * 100  # Percentage of non-missing data in dataset

# Print summary information to console
cat("Total surveys:", total_surveys, "\n")
cat("Total people tested:", total_tested, "\n")
cat("Total positive cases (S. mansoni):", total_positive, "\n")
cat("Data completeness:", round(completeness, 2), "%\n")

# Calculate yearly prevalence statistics: mean, SD, sample size, and 95% confidence intervals
prev_yearly <- data_mansoni %>%
  group_by(SurveyYear) %>%
  summarise(
    mean_prev = mean(Prevalence, na.rm = TRUE),
    sd_prev = sd(Prevalence, na.rm = TRUE),
    n = n()
  ) %>%
  mutate(
    se = sd_prev / sqrt(n),          # Standard error of the mean prevalence
    lower = mean_prev - 1.96 * se,  # Lower bound of 95% CI
    upper = mean_prev + 1.96 * se   # Upper bound of 95% CI
  )

# Conduct t-test comparing prevalence before and after year 2000
before_2000 <- data_mansoni %>% filter(SurveyYear < 2000)
after_2000 <- data_mansoni %>% filter(SurveyYear >= 2000)
t_test_result <- t.test(before_2000$Prevalence, after_2000$Prevalence)
print(t_test_result)
tidy(t_test_result)  # Tidy output for further use if needed

# Ensure is_primary_school column is logical type (TRUE/FALSE)
data_mansoni_school <- data_mansoni %>%
  mutate(is_primary_school = as.logical(is_primary_school))

# Summarize mean prevalence and confidence intervals for all sites by year
all_sites <- data_mansoni_school %>%
  filter(!is.na(SurveyYear), !is.na(Prevalence)) %>%
  group_by(SurveyYear) %>%
  summarise(
    mean_prev = mean(Prevalence, na.rm = TRUE),
    ci = 1.96 * sd(Prevalence, na.rm = TRUE) / sqrt(n()),  # 95% CI
    group = "All Sites"
  )

# Summarize mean prevalence and confidence intervals for primary school sites by year
school_sites <- data_mansoni_school %>%
  filter(is_primary_school == TRUE, !is.na(SurveyYear), !is.na(Prevalence)) %>%
  group_by(SurveyYear) %>%
  summarise(
    mean_prev = mean(Prevalence, na.rm = TRUE),
    ci = 1.96 * sd(Prevalence, na.rm = TRUE) / sqrt(n()),
    group = "Primary Schools"
  )

# Combine the summaries for plotting
combined <- bind_rows(all_sites, school_sites)

# Plot mean prevalence over time with 95% confidence intervals, grouped by site type
ggplot(combined, aes(x = SurveyYear, y = mean_prev, color = group)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +  
  geom_ribbon(aes(ymin = mean_prev - ci, ymax = mean_prev + ci, fill = group), alpha = 0.2, color = NA) +
  scale_x_continuous(
    breaks = seq(1950, 2020, by = 5),
    limits = c(1951, 2015)
  ) +
  labs(
    title = "Mean Prevalence of S. mansoni in Uganda (with 95% CI) with yearly population size of school-aged children",
    x = "Survey Year",
    y = "Mean Prevalence (%)",
    color = "Group",
    fill = "Group"
  ) +
  theme_minimal()

# Calculate prevalence statistics by region (ADMIN1_NAME)
region_prev <- data_mansoni %>%
  group_by(ADMIN1_NAME) %>%
  summarise(
    mean_prev = mean(Prevalence, na.rm = TRUE),
    sd_prev = sd(Prevalence, na.rm = TRUE),
    n = n(),
    se = sd_prev / sqrt(n),
    lower = mean_prev - 1.96 * se,
    upper = mean_prev + 1.96 * se
  ) %>%
  arrange(desc(mean_prev))  # Sort regions by descending mean prevalence
print(region_prev)

# --------------------------------
# Model building to predict prevalence
# --------------------------------

# Load additional libraries for modeling
library(xgboost)
library(glmnet)
library(Matrix)
library(randomForest)
library(caret)

# Select relevant variables and remove rows with missing prevalence
data_clean <- data_mansoni %>%
  select(Prevalence, SurveyYear, Age_start, Age_end, latitude, longitude, is_primary_school) %>%
  filter(!is.na(Prevalence))

# Convert variables to correct data types
data_clean <- data_clean %>%
  mutate(
    is_primary_school = as.factor(is_primary_school),
    SurveyYear = as.numeric(SurveyYear),
    Age_start = as.numeric(Age_start),
    Age_end = as.numeric(Age_end),
    latitude = as.numeric(latitude),
    longitude = as.numeric(longitude)
  )

# Split dataset into training (80%) and testing (20%) sets with reproducibility
set.seed(42)
train_index <- createDataPartition(data_clean$Prevalence, p = 0.8, list = FALSE)
train_data <- data_clean[train_index, ] %>%
  na.omit()  # Remove rows with any NA values in training set
test_data <- data_clean[-train_index, ] %>%
  na.omit()  # Remove rows with any NA values in test set

# ---------------------------
# MODEL 1: Random Forest
# ---------------------------
# Train Random Forest model on training data with 500 trees
rf_model <- randomForest(Prevalence ~ ., data = train_data, importance = TRUE, ntree = 500)

# Predict prevalence on test data
pred_rf <- predict(rf_model, test_data)

# Calculate RMSE (Root Mean Squared Error) as model performance metric
rf_rmse <- sqrt(mean((pred_rf - test_data$Prevalence)^2))

# Plot variable importance from Random Forest model
varImpPlot(rf_model)

# ---------------------------
# MODEL 2: Linear Regression
# ---------------------------
# Fit a linear regression model on training data
lm_model <- lm(Prevalence ~ ., data = train_data)
summary(lm_model)  # Show model summary

# Predict on test data and calculate RMSE
pred_lm <- predict(lm_model, test_data)
lm_rmse <- sqrt(mean((pred_lm - test_data$Prevalence)^2))

# ---------------------------
# MODEL 3: XGBoost
# ---------------------------
# Prepare numeric matrices for xgboost from training and test data (excluding intercept)
train_matrix <- model.matrix(Prevalence ~ ., data = train_data)[, -1]
test_matrix <- model.matrix(Prevalence ~ ., data = test_data)[, -1]

# Create DMatrix objects for xgboost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_data$Prevalence)
dtest <- xgb.DMatrix(data = test_matrix, label = test_data$Prevalence)

# Train XGBoost regression model with 100 rounds, suppress verbose output
xgb_model <- xgboost(data = dtrain, nrounds = 100, objective = "reg:squarederror", verbose = 0)

# Predict on test set and calculate RMSE
pred_xgb <- predict(xgb_model, dtest)
xgb_rmse <- sqrt(mean((pred_xgb - test_data$Prevalence)^2))

# ---------------------------
# MODEL 4: Lasso Regression (glmnet)
# ---------------------------
# Prepare matrices for glmnet (lasso regression)
x_train <- model.matrix(Prevalence ~ ., train_data)[, -1]
y_train <- train_data$Prevalence

x_test <- model.matrix(Prevalence ~ ., test_data)[, -1]
y_test <- test_data$Prevalence

# Fit cross-validated Lasso regression model (alpha=1 for Lasso)
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 5)

# Extract best lambda value from cross-validation
best_lambda <- lasso_model$lambda.min

# Predict on test set using best lambda and calculate RMSE
pred_lasso <- predict(lasso_model, s = best_lambda, newx = x_test)
lasso_rmse <- sqrt(mean((pred_lasso - y_test)^2))

# ---------------------------
# Print RMSE results for all models
# ---------------------------
cat("Random Forest RMSE:", round(rf_rmse, 2), "\n")
cat("Linear Regression RMSE:", round(lm_rmse, 2), "\n")
cat("XGBoost RMSE:", round(xgb_rmse, 2), "\n")
cat("Lasso Regression RMSE:", round(lasso_rmse, 2), "\n")

