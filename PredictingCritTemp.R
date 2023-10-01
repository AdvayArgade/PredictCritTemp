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
# att<-df[,1:80]
# class_labels<-df$critical_temp
# data_df<-data.frame(att,Class=class_labels)
# fs_score<-information.gain(Class~.,data=data_df)
# print(fs_score)
# top_attrs_fs = cut_attrs(fs_score, k=0.2)
#sorted_fs_score <- fs_score[order(fs_score)]
#print(sorted_fs_score)


#lasso----
# tryCatch({
#   f <- read.csv('train.csv')
# }, error = function(e) {
#   stop("Error reading data file: ", e$message)
# })
# att <- as.matrix(f[, 1:80])
# class_labels <- f$critical_temp
# #find the best lambda
# cv_model <- cv.glmnet(att, class_labels, alpha = 1)
# best_lambda <- cv_model$lambda.min
# best_model <- glmnet(att, class_labels, alpha = 1, lambda = best_lambda)
# # Extract coefficients 
# coefficients <- coef(best_model)
# abs_coefficients <- abs(coefficients)
# important_features <- which(abs_coefficients > 0)
# sorted_important_features <- important_features[order(-abs_coefficients[important_features])]
# cat("Important Features (Decreasing Order of Absolute Coefficient Magnitude):\n")
# cat(rownames(coefficients)[sorted_important_features], sep = "\n")

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




