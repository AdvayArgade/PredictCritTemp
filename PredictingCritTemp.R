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
# information gain 
# critical_temp = df$critical_temp
# gains = information_gain(x = subset(df, select = -critical_temp), y= df$critical_temp, type = 'infogain', equal = TRUE)
# top_attrs = cut_attrs(attrs = gains, k = 0.2)
# df = subset(df, select = top_attrs)
# df = cbind(df, critical_temp)

# Train test split----
set.seed(42)
split = sample.split(df, SplitRatio = 0.8)
train = subset(df, split==TRUE)
test = subset(df, split==FALSE)
x_train = as.matrix(subset(train, select = -critical_temp))
x_test = as.matrix(subset(test, select = -critical_temp))
y_train = as.matrix(subset(train, select = critical_temp))
y_test = subset(test, select = critical_temp)
y_test <- test$critical_temp


# RFE
# set.seed(42)
# # control = rfeControl(functions= rfFuncs, method="cv", number=3)
# results <- rfe(x = x_train, y = y_train,
#                sizes=c(20:50), metric = "Rsquared", maximize=TRUE)
# # summarize the results
# print(results)

# print(head(x_test))
# print(head(y_test))

# Regression----
folds = createFolds(y_train, k = 5)
xgb_train = xgb.DMatrix(data = x_train, label = y_train)
xgb_test = xgb.DMatrix(data = x_test, label = y_test)
watchlist = list(train=xgb_train, test=xgb_test)
start_time = Sys.time()
regressor = xgb.train(data = xgb_train, nrounds = 200, max.depth = 5, watchlist = watchlist
                      , eta = 0.285, lambda = 1.01)

end_time = Sys.time()
#Anova(regressor)
# while(max(Anova(regressor)$'Pr(>F)') > 0.05) {
#   remove <- names(which.max(Anova(regressor)$'Pr(>F)'))
#   regressor <- update(regressor, . ~ . - remove)
# }

#print(end_time-start_time)
# y_pred = predict(regressor, xgb_test)
# print(MAE(y_pred = y_pred, y_true = y_test))
# print(R2(pred = y_pred, obs = y_test))
# print(RMSE(y_pred = y_pred, y_true = y_test))


# Bayesian NN----
# Define a Bayesian regression model with uncertainty
# model <- brm(critical_temp ~ ., data = train, family = gaussian())
# Generate predictions with uncertainty
# predictions <- predict(model, newdata = test, nsamples = 1000)
# Extract mean and credible intervals
# mean_predictions <- apply(predictions, 2, mean)
# credible_intervals <- apply(predictions, 2, quantile, c(0.025, 0.975))





# ANN----
# library(h2o)
# h2o.init(nthreads = -1)
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




