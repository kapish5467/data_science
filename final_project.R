#Reading from the file
df <- read.csv("house-votes.txt", header = FALSE, na.strings = c("?","NA"))


install.packages("caret")
library(caret)
install.packages("irr")
library(irr)
install.packages("randomForest")
library(randomForest)
install.packages("e1071")
library(e1071)
install.packages("C50")
library(C50)



#Imputing the missing values in the dataset
install.packages("plyr")
library(plyr)
install.packages("gmodels")
library(gmodels)
count(df, "V2")

replace_Val <- ifelse(count(df, "V2")[1,2] > count(df, "V2")[2,2], "n", "y")
replace_Val
df$V2
df$V2 <- as.factor(ifelse(is.na(df$V2), replace_Val, as.character(df$V2)))
df$V2

replace_Val <- ifelse(count(df, "V3")[1,2] > count(df, "V3")[2,2], "n", "y")
replace_Val
df$V3
df$V3 <- as.factor(ifelse(is.na(df$V3), replace_Val, as.character(df$V3)))
df$V3

replace_Val <- ifelse(count(df, "V4")[1,2] > count(df, "V4")[2,2], "n", "y")
replace_Val
df$V4
df$V4 <- as.factor(ifelse(is.na(df$V4), replace_Val, as.character(df$V4)))
df$V4

replace_Val <- ifelse(count(df, "V5")[1,2] > count(df, "V5")[2,2], "n", "y")
replace_Val
df$V5
df$V5 <- as.factor(ifelse(is.na(df$V5), replace_Val, as.character(df$V5)))
df$V5

replace_Val <- ifelse(count(df, "V6")[1,2] > count(df, "V6")[2,2], "n", "y")
replace_Val
df$V6
df$V6 <- as.factor(ifelse(is.na(df$V6), replace_Val, as.character(df$V6)))
df$V6

replace_Val <- ifelse(count(df, "V7")[1,2] > count(df, "V7")[2,2], "n", "y")
replace_Val
df$V7
df$V7 <- as.factor(ifelse(is.na(df$V7), replace_Val, as.character(df$V7)))
df$V7

replace_Val <- ifelse(count(df, "V8")[1,2] > count(df, "V8")[2,2], "n", "y")
replace_Val
df$V8
df$V8 <- as.factor(ifelse(is.na(df$V8), replace_Val, as.character(df$V8)))
df$V8

replace_Val <- ifelse(count(df, "V9")[1,2] > count(df, "V9")[2,2], "n", "y")
replace_Val
df$V9
df$V9 <- as.factor(ifelse(is.na(df$V9), replace_Val, as.character(df$V9)))
df$V9

replace_Val <- ifelse(count(df, "V10")[1,2] > count(df, "V10")[2,2], "n", "y")
replace_Val
df$V10
df$V10 <- as.factor(ifelse(is.na(df$V10), replace_Val, as.character(df$V10)))
df$V10

replace_Val <- ifelse(count(df, "V11")[1,2] > count(df, "V11")[2,2], "n", "y")
replace_Val
df$V11
df$V11 <- as.factor(ifelse(is.na(df$V11), replace_Val, as.character(df$V11)))
df$V11

replace_Val <- ifelse(count(df, "V12")[1,2] > count(df, "V12")[2,2], "n", "y")
replace_Val
df$V12
df$V12 <- as.factor(ifelse(is.na(df$V12), replace_Val, as.character(df$V12)))
df$V12

replace_Val <- ifelse(count(df, "V13")[1,2] > count(df, "V13")[2,2], "n", "y")
replace_Val
df$V13
df$V13 <- as.factor(ifelse(is.na(df$V13), replace_Val, as.character(df$V13)))
df$V13

replace_Val <- ifelse(count(df, "V14")[1,2] > count(df, "V14")[2,2], "n", "y")
replace_Val
df$V14
df$V14 <- as.factor(ifelse(is.na(df$V14), replace_Val, as.character(df$V14)))
df$V14

replace_Val <- ifelse(count(df, "V15")[1,2] > count(df, "V15")[2,2], "n", "y")
replace_Val
df$V15
df$V15 <- as.factor(ifelse(is.na(df$V15), replace_Val, as.character(df$V15)))
df$V15

replace_Val <- ifelse(count(df, "V16")[1,2] > count(df, "V16")[2,2], "n", "y")
replace_Val
df$V16
df$V16 <- as.factor(ifelse(is.na(df$V16), replace_Val, as.character(df$V16)))
df$V16

replace_Val <- ifelse(count(df, "V17")[1,2] > count(df, "V17")[2,2], "n", "y")
replace_Val
df$V17
df$V17 <- as.factor(ifelse(is.na(df$V17), replace_Val, as.character(df$V17)))
df$V17

df_train <- df[1:334, ]
df_train_c5.0 <- df[1:334, ]

df_test <- df[335:435, ]
df_test_c5.0 <- df[335:435, ]

df_train_label <- df[1:334, 1]
df_train_label_c5.0 <- df[1:334, 1]

df_test_label <- df[335:435, 1]
df_test_label_c5.0 <- df[335:435, 1]

df_classifier <- naiveBayes(df_train, df_train_label)
df_test_pred <- predict(df_classifier, df_test)
CrossTable(df_test_pred, df_test_label,prop.chisq = FALSE, prop.t = FALSE,dnn = c('predicted', 'actual'))

df_classifier_c5.0 <- C5.0(df_train_c5.0[-1], df_train_label_c5.0)
df_test_pred_c5.0 <- predict(df_classifier_c5.0, df_test_c5.0)
CrossTable(df_test_pred_c5.0, df_test_label_c5.0,prop.chisq = FALSE, prop.t = FALSE,dnn = c('predicted', 'actual'))


#Creating 10 folds
num_folds <- createFolds(df$V1, k=10)
df_fold_test <- df[num_folds$Fold01, ]
df_fold_train <- df[-num_folds$Fold01, ]


df_cv_results <- lapply(num_folds, function(x){
  df_cv_train <- df[-x, ]
  df_cv_test <- df[x, ]
  df_cv_model <- naiveBayes(V1 ~., data = df_cv_train)
  df_cv_pred <- predict(df_cv_model, df_cv_test)
  df_actual <- df_cv_test$V1
  kappa <- kappa2(data.frame(df_actual, df_cv_pred))$value
  return(kappa)
})

df_cv_results_c5.0 <- lapply(num_folds, function(x){
  df_cv_train <- df[-x, ]
  df_cv_test <- df[x, ]
  df_cv_model <- C5.0(V1 ~., data = df_cv_train)
  df_cv_pred <- predict(df_cv_model, df_cv_test)
  df_actual <- df_cv_test$V1
  kappa <- kappa2(data.frame(df_actual, df_cv_pred))$value
  return(kappa)
})

#Mean of kappa values for 10 folds
str(df_cv_results)
mean(unlist(df_cv_results))

str(df_cv_results_c5.0)
mean(unlist(df_cv_results_c5.0))


#Automated parameter tunning
modelLookup("nb")
set.seed(2)
param_tunning <- train(V1 ~., data = df, method = "nb")
param_tunning

modelLookup("c5.0")
set.seed(2)
param_tunning_c5.0 <- train(V1 ~., data = df, method = "C5.0")
param_tunning_c5.0


#Ensemble learning for naiveBayes
rf <- randomForest(V1 ~., data = df)
rf
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
set.seed(2)
grid_rf <- expand.grid(.fL = c(0,1),.usekernel=c(TRUE,FALSE),.adjust=c(FALSE))
model_rf_nb <- train(V1 ~., data=df, method = "nb", metric = "Kappa", trControl = ctrl, tuneGrid = grid_rf)
model_rf_nb
plot(model_rf_nb)

set.seed(2)
grid_rf_c5.0 <- expand.grid(.trials = c(1, 10, 20, 30, 40, 50),.model="tree",.winnow=c(TRUE, FALSE))
model_rf_c5.0 <- train(V1 ~., data=df, method = "C5.0", metric = "Kappa", trControl = ctrl, tuneGrid = grid_rf_c5.0)
model_rf_c5.0
plot(model_rf_c5.0)
