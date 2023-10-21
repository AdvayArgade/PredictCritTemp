df = read.csv("train.csv")

#Preprocessing----
library(caTools)
library(mlbench)
library(xgboost)
library(caret)
library(MLmetrics)
library(FSelectorRcpp)
library(car)
library(MASS)
library(brms)
library(FSelector)
#Min max scaling----
normalize<-function(x){
  if (is.vector(x)) {
    min_val <- min(x)
    max_val <- max(x)
  } else {
    min_val <- max_val <- x
  }
  
  if (min_val == max_val) {
    return(rep(0, length(x)))  # Handle case where min = max
  }
  
  return((x - min_val) / (max_val - min_val))
}
df = cbind(as.data.frame(lapply(df, normalize)))


# Feature selection----
# information gain----
# critical_temp = df$critical_temp
# gains = information_gain(x = subset(df, select = -critical_temp), y= df$critical_temp, type = 'infogain', equal = TRUE)
# top_attrs = cut_attrs(attrs = gains, k = 0.4)
# df = subset(df, select = top_attrs)
# df = cbind(df, critical_temp)

#fischer score----
# library(MASS)
# data <- read.csv('train.csv')
# X <- data[, 1:80]  
# y <- data$critical_temp
# # Perform Fisher Score calculation using LDA
# lda_result <- lda(X, y)
# fisher_scores <- numeric(ncol(X))
# for (i in seq_along(fisher_scores)) {
#   between_class_var <- sum((lda_result$means[, i] - mean(lda_result$means))^2)
#   within_class_var <- sum(lda_result$scaling^2 * table(y)[i])
#   fisher_scores[i] <- between_class_var / within_class_var
# }
# feature_scores <- data.frame(Feature = colnames(X), Fisher_Score = fisher_scores)
# # Sort the features by Fisher Score in descending order
# sorted_features <- feature_scores[order(-feature_scores$Fisher_Score), ]
# # Select the top-k features (e.g., top 10)
# k <- 10
# selected_features <- sorted_features$Feature[1:k]
# # Print the selected features and their Fisher Scores
# print(sorted_features)
# cat("\nSelected Features:\n", selected_features, sep = "\n")



#lASSO----
# library(readr)
# df<-read.csv('train.csv')
# #str(df)
# library(caret)
# set.seed(123)
# index<-createDataPartition(df$critical_temp,p=.8,list=FALSE,times=1)
# df<-as.data.frame(df)
# train_df<-df[index,]
# test_df<-df[-index,]
# #cross validation
# ctrlspecs<-trainControl(method ="cv",number=10,savePredictions = "all")
# lambda_vector<- 10^seq(5,-5,length=500)
# model1<-train(critical_temp~.,
#               data = train_df,
#               preProcess=c("center","scale"),
#               method="glmnet",
#               tuneGrid=expand.grid(alpha=1, lambda=lambda_vector),
#               na.action = na.omit)
# model1$bestTune
# model1$bestTune$lambda
# #lasso reg ceof
# coefficient<-round(coef(model1$finalModel, model1$bestTune$lambda),3)
# #print(coefficient)
# abs_coef<-abs(coefficient)
# ordered_coef<-abs_coef[order(-abs_coef[,]),]
# print(ordered_coef)
# #var imp---gives top 20 features----
# varImp(model1)


#CFS----
#failed
# library(caret)
# library(mlbench)
# data <- read.csv('train.csv')
# y<-data$critical_temp
# x<-subset(data,select = -c(y))
# cfs_feature<-findCorrelation(cor(x),cutoff = 0.5)
# print(cfs_feature)


# Get the optimal feature subset
top5attrs = intersect(results$optVariables[1:30], top_attrs[1:30])

# Train test split----
# All features
set.seed(42)
split = sample.split(df, SplitRatio = 0.8)
train = subset(df, split==TRUE)
test = subset(df, split==FALSE)
x_train = as.matrix(subset(train, select = -critical_temp))
x_test = as.matrix(subset(test, select = -critical_temp))
y_train = as.matrix(subset(train, select = critical_temp))
y_test = subset(test, select = critical_temp)
y_test <- test$critical_temp

# Top 5 features

x_train_top = x_train[, top5attrs]
x_test_top = x_test[, top5attrs]

# RFE
# y_train = train$critical_temp
# 
# set.seed(42)
# control = rfeControl(functions= rfFuncs, method="cv", number = 3)
# results <- rfe(x_train, y_train,
#                c(25:55), "Rsquared", maximize=TRUE, rfeControl = control)
# # summarize the results
# print(results)




# print(head(x_test))
# print(head(y_test))

# Regression----
# XGBoost----
# folds = createFolds(y_train, k = 5)
# cat("All features:")
# xgb_train = xgb.DMatrix(data = x_train, label = y_train)
# xgb_test = xgb.DMatrix(data = x_test, label = y_test)
# watchlist = list(train=xgb_train, test=xgb_test)
# start_time = Sys.time()
# regressor = xgb.train(data = xgb_train, nrounds = 200, max.depth = 5, watchlist = watchlist
#                       , eta = 0.285, lambda = 1.01)
# 
# end_time = Sys.time()
# print(end_time-start_time)
# y_pred = predict(regressor, xgb_test)
# standard_error = RMSE(y_pred = y_pred, y_true = y_test)
# lower_interval <- y_pred - 1.96 * standard_error
# upper_interval <- y_pred + 1.96 * standard_error
# predictions = data.frame(cbind(y_pred, lower_interval, upper_interval))
# print(head(predictions))
# 
# print(MAE(y_pred = y_pred, y_true = y_test))
# print(R2(pred = y_pred, obs = y_test))
# print(RMSE(y_pred = y_pred, y_true = y_test))
# 
# cat("\nTop 5 features:")
# xgb_train = xgb.DMatrix(data = x_train_top, label = y_train)
# xgb_test = xgb.DMatrix(data = x_test_top, label = y_test)
# watchlist = list(train=xgb_train, test=xgb_test)
# start_time = Sys.time()
# regressor = xgb.train(data = xgb_train, nrounds = 200, max.depth = 5, watchlist = watchlist
#                       , eta = 0.285, lambda = 1.01)
# 
# end_time = Sys.time()
# print(end_time-start_time)
# y_pred_top = predict(regressor, xgb_test)
# standard_error = RMSE(y_pred = y_pred, y_true = y_test)
# lower_interval <- y_pred_top - 1.96 * standard_error
# upper_interval <- y_pred_top + 1.96 * standard_error
# predictions = data.frame(cbind(y_pred_top, lower_interval, upper_interval))
# print(head(predictions))
# 
# print(MAE(y_pred = y_pred_top, y_true = y_test))
# print(R2(pred = y_pred_top, obs = y_test))
# print(RMSE(y_pred = y_pred_top, y_true = y_test))
#Anova(regressor)
# while(max(Anova(regressor)$'Pr(>F)') > 0.05) {
#   remove <- names(which.max(Anova(regressor)$'Pr(>F)'))
#   regressor <- update(regressor, . ~ . - remove)
# }




# Bayesian NN----
# Define a Bayesian regression model with uncertainty
# model <- brm(critical_temp ~ ., data = train, family = gaussian())
# Generate predictions with uncertainty
# predictions <- predict(model, newdata = test, nsamples = 1000)
# Extract mean and credible intervals
# mean_predictions <- apply(predictions, 2, mean)
# credible_intervals <- apply(predictions, 2, quantile, c(0.025, 0.975))





# ANN----
critical_temp = train$critical_temp
train_top = cbind(x_train_top, critical_temp)
critical_temp = test$critical_temp
test_top = cbind(x_test_top, critical_temp)
library(h2o)
h2o.init(nthreads = -1)
# All features----
# model = h2o.deeplearning(standardize = FALSE,
#                         y = 'critical_temp',
#                          training_frame = as.h2o(train),
#                         validation_frame = as.h2o(test),
#                          activation = 'RectifierWithDropout',
#                          hidden = c(100,100, 75),
#                          epochs = 100,
#                         loss = "Absolute",
#                         verbose = TRUE,
#                          train_samples_per_iteration = -2)
# 
# y_pred = h2o.predict(model, as.h2o(x_test))
# y_pred = as.vector(as.numeric(y_pred))
# print(MAE(y_pred = y_pred, y_true = y_test))
# print(R2(pred = y_pred, obs = y_test))
# print(RMSE(y_pred = y_pred, y_true = y_test))


# Top 5 features----
start_time = Sys.time()
model = h2o.deeplearning(standardize = FALSE,
                         y = 'critical_temp',
                         training_frame = as.h2o(train_top),
                         validation_frame = as.h2o(test_top),
                         activation = 'RectifierWithDropout',
                         hidden = c(100,100, 75),
                         epochs = 100,
                         loss = "Absolute",
                         verbose = TRUE,
                         train_samples_per_iteration = -2)

end_time = Sys.time()
print(end_time - start_time)
y_pred = h2o.predict(model, as.h2o(x_test_top))
y_pred = as.vector(as.numeric(y_pred))
print(MAE(y_pred = y_pred, y_true = y_test))
print(R2(pred = y_pred, obs = y_test))
print(RMSE(y_pred = y_pred, y_true = y_test))





# K-Fold cross validation----
# rmse = c()
# r2 = c()
# mae = c()
# cv = lapply(folds, function(x) {
#   x_kf_train = as.matrix(subset(df, select = -critical_temp)[-x, ])
#   x_kf_test = as.matrix(subset(df, select = -critical_temp)[x, ])
#   y_kf_train = as.matrix(df$critical_temp[-x])
#   y_kf_test = as.matrix(df$critical_temp[x])
#   xgb_train = xgb.DMatrix(data = x_kf_train, label = y_kf_train)
#   xgb_test = xgb.DMatrix(data = x_kf_test, label = y_kf_test)
#   watchlist = list(train=xgb_train, test=xgb_test)
#   regressor = xgb.train(data = xgb_train, nrounds = 200, max.depth = 3,
#                         watchlist = watchlist)
#   y_pred = predict(regressor, xgb_test)
#   #mape = MAE(y_pred = y_pred, y_true = y_kf_test, epsilon)
#   non_zero_indices <- y_kf_test != 0
#   y_kf_test_filtered <- y_kf_test[non_zero_indices]
#   y_pred_filtered <- y_pred[non_zero_indices]
#   rmse = append(rmse, RMSE(y_pred_filtered, y_kf_test_filtered))
#   r2 = append(r2, R2(y_pred_filtered, y_kf_test_filtered))
#   mae = append(mae, MAE(y_pred_filtered, y_kf_test_filtered))
#   print(rmse[length(rmse)])
#   print(r2[length(r2)])
#   print(mae[length(mae)])
#   Sys.sleep(3)
# })
# 
# print(mean(rmse))
# print(mean(r2))
# print(mean(mae))


# Gradient Boost----
# library(gbm)
# library(caTools)
# library(caret)
# start_time <- Sys.time()
# df <- read.csv("train.csv")

# # Preprocessing
# normalize <- function(x) {
#   if (is.vector(x)) {
#     min_val <- min(x)
#     max_val <- max(x)
#   } else {
#     min_val <- max_val <- x
#   } 
#   if (min_val == max_val) {
#     return(rep(0, length(x)))  # Handle case where min = max
#   } 
#   return((x - min_val) / (max_val - min_val))
# }
# df_scaled <- as.data.frame(lapply(df, normalize))

# # Train-test split
# set.seed(123)
# split <- sample.split(df_scaled$critical_temp, SplitRatio = 0.7)
# train <- subset(df_scaled, split == TRUE)
# test <- subset(df_scaled, split == FALSE)

# gbm_model <- gbm(critical_temp ~ ., data = train, distribution = "gaussian", n.trees = 100, interaction.depth = 2, shrinkage = 0.1)

# predictions <- predict(gbm_model, newdata = test, n.trees = 100, type = "response")
# current_time <- Sys.time()

# print((current_time) - (start_time))
# mse <- mean((predictions - test$critical_temp)^2)  # Corrected
# rmse <- sqrt(mse)
# mae <- mean(abs(predictions - test$critical_temp))
# #print(paste("Mean Squared Error:", mse))
# actual_values <- test$critical_temp
# ss_total <- sum((actual_values - mean(actual_values))^2)
# ss_residual <- sum((actual_values - predictions)^2)
# rsquared <- 1 - (ss_residual / ss_total)
# print(paste("R2:", rsquared))
# print(paste("RMSE:", rmse))
# print(paste("MAE:", mae))

#PCA----
library(stats)
data <- read.csv("train.csv")
column_names <- colnames(data)
scaled_data <- scale(data)
pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)
explained_variance <- pca_result$sdev^2 / sum(pca_result$sdev^2)
cat("Explained Variance for Principal Components:\n")
for (i in 1:length(explained_variance)) {
  cat("PC", i, " (", column_names[i], "): ", explained_variance[i], "\n")
}
cumulative_variance <- cumsum(explained_variance)
plot(cumulative_variance, type = "b", xlab = "Number of Principal Components", ylab = "Cumulative Explained Variance", main = "Cumulative Explained Variance Plot")
cumulative_variance <- cumsum(explained_variance)
desired_variance <- 0.9
selected_pc <- which(cumulative_variance >= desired_variance)[1
cat("The first principal component where cumulative variance exceeds 0.9 is PC", selected_pc, "\n")


#Random Forest Whole dataset----

library(randomForest)
library(Metrics)
library(caret)

# Load your data
A <- read.csv("train.csv")
A$critical_temp <- as.factor(A$critical_temp)

# Set the seed for reproducibility
set.seed(123)

# Create an index for sampling
index <- sample(2, nrow(A), replace = TRUE, prob = c(0.7, 0.3))

# Create training and test sets
train <- A[index == 1, ]
test <- A[index == 2, ]

# Convert "critical_temp" to numeric
train$critical_temp <- as.numeric(train$critical_temp)
test$critical_temp <- as.numeric(test$critical_temp)

# Function to normalize numeric variables
normalize_data <- function(data) {
  numeric_cols <- sapply(data, is.numeric)
  data[, numeric_cols] <- lapply(data[, numeric_cols], function(x) (x - min(x)) / (max(x) - min(x)))
  return(data)
}

# Normalize the training and test datasets
train <- normalize_data(train)
test <- normalize_data(test)

# Build the random forest model
T1 <- system.time({RFM <- randomForest(critical_temp ~ ., data = train)})
cat("\n\nTraining Time:", T1)

# Print the model summary
print(RFM)

# Model Accuracy
T2 <- system.time({Temp_pred <- predict(RFM, test)})
result <- data.frame(test$critical_temp, Temp_pred)
cat("\n\nPrediction Time:", T2)
print(result)
plot(result)

# R-square
r_squared <- R2(Temp_pred, test$critical_temp)
cat("\nR2:", r_squared)

# RMSE
rmse <- rmse(Temp_pred, test$critical_temp)
cat("\nRMSE:", rmse)

# MAE
mae <- mae(Temp_pred, test$critical_temp)
cat("\nMAE:", mae)



#Random Forest on Features----
library(randomForest)
library(Metrics)
library(caret)


A <- read.csv("train.csv")
A$critical_temp <- as.factor(A$critical_temp)

# Load your feature selection results
results <- readRDS('rfeResult.rds')
top_attrs <- readRDS('InfoGainResult.rds')

# Extract common features
common_features <- intersect(results$optVariables[1:50], top_attrs[1:50])

# Filter the dataset to include only common features
data_filtered <- A[, c("critical_temp", common_features)]

# Set the seed for reproducibility
set.seed(123)

# Create an index for sampling
index <- sample(2, nrow(data_filtered), replace = TRUE, prob = c(0.7, 0.3))

# Create training and test sets
train <- data_filtered[index == 1, ]
test <- data_filtered[index == 2, ]

# Convert "critical_temp" to numeric
train$critical_temp <- as.numeric(train$critical_temp)
test$critical_temp <- as.numeric(test$critical_temp)

# Function to normalize numeric variables
normalize_data <- function(data) {
  numeric_cols <- sapply(data, is.numeric)
  data[, numeric_cols] <- lapply(data[, numeric_cols], function(x) (x - min(x)) / (max(x) - min(x)))
  return(data)
}

# Normalize the training and test datasets
train <- normalize_data(train)
test <- normalize_data(test)

# Build the random forest model
T1 <- system.time({RFM <- randomForest(critical_temp ~ ., data = train)})
cat("\n\nTraining Time:", T1)

# Print the model summary
print(RFM)

# Model Accuracy
T2 <- system.time({Temp_pred <- predict(RFM, test)})
result <- data.frame(test$critical_temp, Temp_pred)
cat("\n\nPrediction Time:", T2)
print(result)
plot(result)

# R-square
r_squared <- R2(Temp_pred, test$critical_temp)
cat("\nR2:", r_squared)

# RMSE
rmse <- rmse(Temp_pred, test$critical_temp)
cat("\nRMSE:", rmse)

# MAE
mae <- mae(Temp_pred, test$critical_temp)
cat("\nMAE:", mae)

#MARS whole dataset----
# Load the earth package
library(earth)
library(caret)
library(Metrics)

# Load your dataset
data <- read.csv("train.csv")

# Function to normalize numeric variables
n2 <- function(b) {
  (b - min(b)) / (max(b) - min(b))
}

# Apply the normalization function to numeric columns
numeric_cols <- sapply(data, is.numeric)
data[, numeric_cols] <- lapply(data[, numeric_cols], n2)

# Split the dataset into a training set (70%) and a testing set (30%)
set.seed(42)
sample_index <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[sample_index, ]
test_data <- data[-sample_index, ]

# Build a MARS model using the training data
T1 <- system.time({
  mars_model <- earth(critical_temp ~ ., data = train_data)
})
cat("Training Time:", T1)

# Summary of the MARS model
print(mars_model)

# Make predictions on the testing set
T2 <- system.time({
  predictions <- predict(mars_model, newdata = test_data)
})
print(predictions)
cat("Prediction Time:", T2)

# Calculate performance metrics (e.g., RMSE) for the predictions
RMSE <- rmse(predictions, test_data$critical_temp)
cat("\nRoot Mean Squared Error (RMSE):", RMSE)

MAE <- mae(predictions, test_data$critical_temp)
cat("\nMAE:", MAE)

r_square <- R2(predictions, test_data$critical_temp)
cat("\nR2:", r_square)



#MARS on Features selected----

                                 
# Load the earth package
library(earth)
library(caret)
library(Metrics)

# Load your dataset
data <- read.csv("train.csv")

# Load your feature selection results
results <- readRDS('rfeResult.rds')
top_attrs <- readRDS('InfoGainResult.rds')

# Extract common features
common_features <- intersect(results$optVariables[1:50], top_attrs[1:50])

# Filter the dataset to include only common features
data_filtered <- data[, c("critical_temp", common_features)]

# Function to normalize numeric variables
n2 <- function(b) {
  (b - min(b)) / (max(b) - min(b))
}

# Apply the normalization function to numeric columns
numeric_cols <- sapply(data_filtered, is.numeric)
data_filtered[, numeric_cols] <- lapply(data_filtered[, numeric_cols], n2)

# Split the filtered dataset into a training set (70%) and a testing set (30%)
set.seed(42)
sample_index <- sample(1:nrow(data_filtered), 0.7 * nrow(data_filtered))
train_data <- data_filtered[sample_index, ]
test_data <- data_filtered[-sample_index, ]

# Build a MARS model using the training data
T1 <- system.time({
  mars_model <- earth(critical_temp ~ ., data = train_data)
})
cat("Training Time:", T1)

# Summary of the MARS model
print(mars_model)

# Make predictions on the testing set
T2 <- system.time({
  predictions <- predict(mars_model, newdata = test_data)
})
cat("Prediction Time:", T2)

# Calculate performance metrics (e.g., RMSE) for the predictions
RMSE <- rmse(predictions, test_data$critical_temp)
cat("\nRoot Mean Squared Error (RMSE):", RMSE)

MAE <- mae(predictions, test_data$critical_temp)
cat("\nMAE:", MAE)

r_square <- R2(predictions, test_data$critical_temp)
cat("\nR2:", r_square)
                                 



