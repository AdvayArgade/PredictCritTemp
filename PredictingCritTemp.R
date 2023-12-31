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
library(randomForest)
library(earth)
library(e1071)
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

# RFE
# y_train = train$critical_temp
# 
# set.seed(42)
# control = rfeControl(functions= rfFuncs, method="cv", number = 3)
# results <- rfe(x_train, y_train,
#                c(1:55), "Rsquared", maximize=TRUE, rfeControl = control)
# # summarize the results
# saveRDS(top_attrs, file = "rfeResult.rds")
# print(results)

# Information gain
# critical_temp = df$critical_temp
# gains = information_gain(x = subset(df, select = -critical_temp), y= df$critical_temp, type = 'infogain', equal = TRUE)
# 
# top_attrs = cut_attrs(attrs = gains, k = 0.99)
# saveRDS(top_attrs, file = "InfoGainResult.rds")

# Get the optimal feature subset
results = readRDS('rfeResult.rds')
top_attrs = readRDS('InfoGainResult.rds')
common_features = intersect(results$optVariables[1:50], top_attrs[1:50])

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



# Correlation Matrix Implementation----

# Compute the correlation matrix
correlation_matrix <- cor(df)

# Find pairs of columns with a correlation greater than 0.6
highly_correlated_pairs <- which(correlation_matrix > 0.6 & correlation_matrix < 1, arr.ind = TRUE)

# Identify and exclude one column from each highly correlated pair
columns_to_exclude <- character(0)

for (i in 1:nrow(highly_correlated_pairs)) {
  row <- highly_correlated_pairs[i, 1]
  col <- highly_correlated_pairs[i, 2]

  col1 <- names(df)[col]

  # Check if the column is not already excluded
  if (!(col1 %in% columns_to_exclude)) {
    # Calculate the mean correlation of the current column with all other columns
    mean_corr <- mean(correlation_matrix[, col])

    # Calculate the mean correlation of the other column in the pair
    mean_corr_other <- mean(correlation_matrix[, row])

    # Exclude the column with the higher mean correlation
    if (mean_corr > mean_corr_other) {
      columns_to_exclude <- c(columns_to_exclude, col1)
    } else {
      columns_to_exclude <- c(columns_to_exclude, names(df)[row])
    }
  }
}

# Create a new data frame with the excluded columns
df_filtered <- df[, !names(df) %in% columns_to_exclude]
print(df_filtered)
print(ncol(df_filtered))
print(nrow(df_filtered))
print(colnames(df_filtered))
data = df_filtered
data = normalize(data)
# Split the data into features (X) and target variable (y)
X <- data[, -ncol(data)]  # All columns except the last one
y <- data[, ncol(data)]   # Last column

# Regression----
# All features
# XGBoost----

cat("All features:")
xgb_train = xgb.DMatrix(data = x_train, label = y_train)
xgb_test = xgb.DMatrix(data = x_test, label = y_test)
watchlist = list(train=xgb_train, test=xgb_test)
start_time = Sys.time()
set.seed(42)
regressor = xgb.train(data = xgb_train, nrounds = 200, max.depth = 5, watchlist = watchlist
                      , eta = 0.285, lambda = 1.01)

end_time = Sys.time()
print(end_time-start_time)
y_pred = predict(regressor, xgb_test)

print(MAE(y_pred = y_pred, y_true = y_test))
print(R2(pred = y_pred, obs = y_test))
print(RMSE(y_pred = y_pred, y_true = y_test))

#Plotting XGBoost performance----
# Get RMSE for each boosting round
evals <- regressor$evaluation_log

# Extract RMSE values
train_rmse <- evals$train_rmse
test_rmse <- evals$test_rmse

# Create a data frame for plotting
eval_df <- data.frame(
  Round = 1:length(train_rmse),
  Train_RMSE = train_rmse,
  Test_RMSE = test_rmse
)

# Plot RMSE over training rounds
plot <- ggplot(data = eval_df, aes(x = Round)) +
  geom_line(aes(y = Train_RMSE, color = "Train")) +
  geom_line(aes(y = Test_RMSE, color = "Test")) +
  scale_color_manual(values = c("Train" = "blue", "Test" = "red")) +
  labs(
    x = "Training Round",
    y = "RMSE",
    title = "XGBoost Training Progress with RMSE Metric"
  ) +
  theme_minimal()

num_features <- c(1:20)
# Plot R2
p = plot(num_features, r2, type = "b", pch = 19, col = "blue", xlab = "Number of Features", ylab = "R2")
# Add connecting lines between points
lines(num_features, rmse, col = "blue")
# Add a grid for better visualization (optional)
grid()
print(p)

# Plot RMSE
p = plot(num_features, rmse, type = "b", pch = 19, col = "orange", xlab = "Number of Features", ylab = "RMSE")
# Add connecting lines between points
lines(num_features, rmse, col = "orange")
# Add a grid for better visualization (optional)
grid()
print(p)

# Plotting MAE vs number of features
p = plot(num_features, mae, type = "b", pch = 19, col = "forestgreen", xlab = "Number of Features", ylab = "MAE")
# Add connecting lines between points
lines(num_features, mae, col = "forestgreen")
# Add a grid for better visualization (optional)
grid()
print(p)

# Plot training time
p = plot(num_features, training_time, type = "b", pch = 19, col = "green", xlab = "Number of Features", ylab = "Training Time")
# Add connecting lines between points
lines(num_features, training_time, col = "green")
# Add a grid for better visualization (optional)
grid()
print(p)

# Plot prediction time
p = plot(num_features, pred_time, type = "b", pch = 19, col = "cyan", xlab = "Number of Features", ylab = "Prediction Time")
# Add connecting lines between points
lines(num_features, pred_time, col = "cyan")
# Add a grid for better visualization (optional)
grid()
print(p)

# ANN----
critical_temp = train$critical_temp
train_top = cbind(x_train_top, critical_temp)
critical_temp = test$critical_temp
test_top = cbind(x_test_top, critical_temp)
library(h2o)
h2o.init(nthreads = -1)
# All features----
model = h2o.deeplearning(standardize = FALSE,
                        y = 'critical_temp',
                         training_frame = as.h2o(train),
                        validation_frame = as.h2o(test),
                         activation = 'RectifierWithDropout',
                         hidden = c(100,100, 75),
                         epochs = 100,
                        loss = "Absolute",
                        verbose = TRUE,
                         train_samples_per_iteration = -2)

y_pred = h2o.predict(model, as.h2o(x_test))
y_pred = as.vector(as.numeric(y_pred))
print(MAE(y_pred = y_pred, y_true = y_test))
print(R2(pred = y_pred, obs = y_test))
print(RMSE(y_pred = y_pred, y_true = y_test))


# Gradient Boost----
gbm_model <- gbm(critical_temp ~ ., data = train, distribution = "gaussian", n.trees = 100, interaction.depth = 2, shrinkage = 0.1)

predictions <- predict(gbm_model, newdata = test, n.trees = 100, type = "response")
current_time <- Sys.time()

print((current_time) - (start_time))
mse <- mean((predictions - test$critical_temp)^2)  # Corrected
rmse <- sqrt(mse)
mae <- mean(abs(predictions - test$critical_temp))
#print(paste("Mean Squared Error:", mse))
actual_values <- test$critical_temp
ss_total <- sum((actual_values - mean(actual_values))^2)
ss_residual <- sum((actual_values - predictions)^2)
rsquared <- 1 - (ss_residual / ss_total)
print(paste("R2:", rsquared))
print(paste("RMSE:", rmse))
print(paste("MAE:", mae))

#Random Forest Whole dataset----

# Set the seed for reproducibility
set.seed(42)
                           
# Build the random forest model
T1 <- system.time({RFM <- randomForest(critical_temp ~ ., data = train,ntree=1000,mtry=15)})
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


# SVR----
# Training SVR model
start_time <- Sys.time()

svr_model <- svm(y_train ~ ., data = data.frame(cbind(y_train, X_train)), kernel = "radial", cost = 1)

end_time <- Sys.time()
# Predict on the test set
y_pred <- predict(svr_model, newdata = data.frame(X_test))


# Print the results
cat("R-squared (R2): ", r2, "\n")
cat("Root Mean Squared Error (RMSE): ", rmse, "\n")
cat("Mean Absolute Error (MAE): ", mae, "\n")
elapsed_time <- end_time - start_time
cat("\nTime taken: ", elapsed_time, " seconds\n")











# Top 10 features----
# XGBoost
cat("\nTop 10 features:")
set.seed(42)
top_i_features = common_features[1:10]
print(top_i_features)
x_train_top = as.matrix(x_train[, top_i_features])
x_test_top = as.matrix(x_test[, top_i_features])
xgb_train = xgb.DMatrix(data = x_train_top, label = y_train)
xgb_test = xgb.DMatrix(data = x_test_top, label = y_test)
watchlist = list(train=xgb_train, test=xgb_test)
start_time = Sys.time()
regressor = xgb.train(data = xgb_train, nrounds = 200, max.depth = 5, watchlist = watchlist
                      , eta = 0.285, lambda = 1.01)

end_time = Sys.time()
print(end_time-start_time)
y_pred_top = predict(regressor, xgb_test)
print(head(predictions))

print(MAE(y_pred = y_pred_top, y_true = y_test))
print(R2(pred = y_pred_top, obs = y_test))
print(RMSE(y_pred = y_pred_top, y_true = y_test))

# ANN
x_train_top = as.data.frame(x_train[, top_i_features])
colnames(x_train_top) = top_i_features
x_test_top = as.data.frame(x_test[, top_i_features])
colnames(x_test_top) = top_i_features
critical_temp = train$critical_temp
train_top = cbind(x_train_top, critical_temp)
critical_temp = test$critical_temp
test_top = cbind(x_test_top, critical_temp)
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

#Random Forest on Features----

# Load your feature selection results
results <- readRDS('rfeResult.rds')
top_attrs <- readRDS('InfoGainResult.rds')

# Extract common features
common_features <- intersect(results$optVariables[1:50], top_attrs[1:50])

# Filter the dataset to include only common features
data_filtered <- A[, c("critical_temp", common_features)]

# Set the seed for reproducibility
set.seed(42)

# Create an index for sampling
index <- sample(2, nrow(data_filtered), replace = TRUE, prob = c(0.7, 0.3))

# Create training and test sets
train <- data_filtered[index == 1, ]
test <- data_filtered[index == 2, ]

# Build the random forest model
T1 <- system.time({RFM <- randomForest(critical_temp ~ ., data = train,ntree=500,mtry=10)})
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


#MARS on Features selected----

set.seed(42)
sample_index <- sample(1:nrow(data_filtered), 0.8 * nrow(data_filtered))
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



# Support Vector Regression Top 10 RFE and InfoGain----

common_features <- intersect(results$optVariables[1:50], top_attrs[1:50])
common_features = common_features[1:20]

data_filtered <- data[, c("critical_temp", common_features)]

X <- data_filtered[, -ncol(data_filtered)]  # All columns except the last one
y <- data_filtered[, ncol(data_filtered)]   # Last column

set.seed(42)  # For same random values
train_idx <- sample(nrow(data_filtered), nrow(data_filtered) * 0.7)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[-train_idx, ]
y_test <- y[-train_idx]

# Start measuring time
start_time <- Sys.time()

# Training SVR model
svr_model <- svm(y_train ~ ., data = data.frame(cbind(y_train, X_train)), kernel = "radial", cost = 1)

# Stop measuring time
end_time <- Sys.time()

# Predict on the test set
y_pred <- predict(svr_model, newdata = data.frame(X_test))

# Calculate R-squared (R2) using Metrics package
r_square <- R2(y_pred, y_test)
cat("\nR2:", r_square)

# Calculate Root Mean Squared Error (RMSE) using Metrics package
RMSE <- RMSE(y_pred, y_test)
cat("\nRoot Mean Squared Error (RMSE):", RMSE)

# Calculate Mean Absolute Error (MAE) using Metrics package
MAE <- MAE(y_pred, y_test)
cat("\nMAE:", MAE)

# Calculate and print the time taken
elapsed_time <- end_time - start_time
cat("\nTime taken: ", elapsed_time, " seconds\n")




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

