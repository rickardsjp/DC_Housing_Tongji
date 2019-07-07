# libraries
library(dplyr)
library(xgboost)
library(Metrics)
library(caTools)
library(caret)
library(stringr)
library(car)

# working directory
setwd("C:/Users/D070697/Documents/Git/DC_Housing_Tongji_qiqi/DC_Housing_Tongji/data/")

# load data
data <- read.csv("data_cleaned.csv")
split = sample.split(data$PRICE, SplitRatio = 0.7)

#one hot
ohe_feats = c("ROOF", "HEAT", "AC", "STYLE", "STRUCT", "GRADE", "CNDTN", "EXTWALL", "INTWALL", "ASSESSMENT_SUBNBHD")

dummies <- dummyVars(~ ROOF + HEAT + AC + STYLE + STRUCT + GRADE + CNDTN + EXTWALL + INTWALL + ASSESSMENT_SUBNBHD, data = data)
all_ohe <- as.data.frame(predict(dummies, newdata = data))
data.ohe <- cbind(data[, -c(which(colnames(data) %in% ohe_feats))], all_ohe)

data.train <- data.ohe %>% filter(split==TRUE)
#data.train <- data.frame(lapply(data.train, function(x) ((x-min(x))/(max(x)-min(x))))
data.test <- data.ohe %>% filter(split==FALSE)

x.train <- subset(data.train, select = -c(PRICE))
y.train <- data.train["PRICE"]

x.test <- subset(data.test, select = -c(PRICE))
y.test <- data.test["PRICE"]


x.train <- as.matrix(x.train)
x.train <- xgb.DMatrix(data = x.train, label = y.train$PRICE)

x.test <- as.matrix(x.test)
x.test <- xgb.DMatrix(data = x.test, label = y.test$PRICE)

# train model 
model <- xgboost(data = x.train, 
                 label = y.train,
                 nthread = 8,
                 max_depth=12,
                 nrounds = 30)

# predict test data
y.pred <- predict(model, x.test)

#scores
R2(y.pred, y.test$PRICE)
mae(y.test$PRICE, y.pred)
mape(y.test$PRICE, y.pred)
mse(y.test$PRICE, y.pred)
