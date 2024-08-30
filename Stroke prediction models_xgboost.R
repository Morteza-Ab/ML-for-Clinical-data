library(randomForest)
library(readxl)
library(writexl)
library(dplyr)
library(UpSetR)
library(randomForest)
library(RWeka)
library(mice)
library(Hmisc)
library(VIM)
library(reshape2)
library(readxl)
library(ggplot2)
library(openxlsx)
library(dplyr)
library(UpSetR)
library(writexl)
library(pheatmap)
#install.packages("caTools")
#install.packages("caret")
#install.packages("recipes")
library(recipes)
library(caret)
library(caTools)
#install.packages("ggcorrplot")
library(ggcorrplot)
#install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
library(class)
library(pROC)
#install.packages("xgboost")
library(xgboost)
install.packages("ROSE")
library(ROSE)


setwd("/Users/mortezaabyadeh/Desktop")
healthcare.dataset.stroke.data <- read.csv("healthcare-dataset-stroke-data.csv")
healthcare.dataset.stroke.data$stroke <- as.factor(healthcare.dataset.stroke.data$stroke)
summary(healthcare.dataset.stroke.data$stroke)
### first check my data

head(healthcare.dataset.stroke.data)
summary(healthcare.dataset.stroke.data)

str(healthcare.dataset.stroke.data)
is.na(healthcare.dataset.stroke.data)
md.pattern(healthcare.dataset.stroke.data)
data <- healthcare.dataset.stroke.data
class(data)
describe(data)
levels(data$gender)

mice_plot <- aggr(healthcare.dataset.stroke.data, col=c("green","red"),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(healthcare.dataset.stroke.data), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))
dim(healthcare.dataset.stroke.data)
colSums(is.na(healthcare.dataset.stroke.data))


hist(healthcare.dataset.stroke.data$stroke, main="Histogram of Variable", xlab="Variable", col="blue")
ggplot(healthcare.dataset.stroke.data, aes(x=age, y=avg_glucose_level)) + geom_point() + theme_minimal()

data <- healthcare.dataset.stroke.data

table(data$gender)

cross_tab <- table(data$ever_married, data$stroke)
margin.table(cross_tab,1)
margin.table(cross_tab, 2)
prop.table(cross_tab)

chisq.test(cross_tab)
### if less than 0.05 means there is a relation between them, otherwise means there is no relation between two parameters



##### there is other as gender type; its only one and can not be analyzed as separate group; either remove or include in the highest freauent gender

data$gender <- ifelse(data$gender == "Other", 'Female', data$gender)
table(data$gender)

### Data preprocessing 

data$gender <- as.factor(data$gender)
data$hypertension <- as.factor(data$hypertension)
data$heart_disease <- as.factor(data$heart_disease)
data$ever_married <- as.factor(data$ever_married)
data$work_type <- as.factor(data$work_type)
data$Residence_type <- as.factor(data$Residence_type)
data$smoking_status <- as.factor(data$smoking_status)
data$stroke <- as.factor(data$stroke)



head(data)


data <- data[,-1]
data$bmi <- as.numeric(data$bmi)
summary(data)

set.seed(123)

trainIndex <- createDataPartition(data$stroke, p = 0.7, list = FALSE)

# Split the data into training and test sets
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Check the proportion of 'stroke' in both datasets
prop.table(table(trainData$stroke))
prop.table(table(testData$stroke))


m1 <- glm(stroke~., data = trainData, family = binomial)

summary(m1)

###################################### data preprocessing
head(data)
summary(data)
data$bmi <- as.numeric(data$bmi) ### this was factor, made many problems :/
data <- data[!is.na(data$bmi), ]

dummy_columns <- c("gender", "hypertension", "heart_disease", "ever_married", "Residence_type", "work_type", "smoking_status")

dummy_formula <- as.formula(paste("~", paste(dummy_columns, collapse = " + ")))

dummy_model <- dummyVars(dummy_formula, data = data, fullRank = TRUE)

data_encoded <- predict(dummy_model, newdata = data)

data_final <- cbind(data[, !names(data) %in% dummy_columns], as.data.frame(data_encoded))



#m1 <- glm(stroke~., data = data, family = "binomial")

#summary(m1)

#modelchi1 <- m1$null.deviance - m1$deviance
#chidf1 <- m1$df.null - m1$df.residual
#chisq_prob1 <- 1 - pchisq(modelchi1,chidf1)

#chisq_prob1
#CIs <- exp(confint(m1))
#Odd <- exp(1.4313) # data$ever_marriedYes = 1.4313

################# Data splitting 
trainIndex <- createDataPartition(data_final$stroke, p = 0.7, list = FALSE)

# Split the data into training and test sets
trainData <- data_final[trainIndex, ]
testData <- data_final[-trainIndex, ]

# Check the proportion of 'stroke' in both datasets
prop.table(table(trainData$stroke))
prop.table(table(testData$stroke))

logistic_model <- glm(stroke ~ ., data = trainData, family = binomial)
summary(logistic_model)


logistic_pred <- predict(logistic_model, testData, type = "response")
logistic_pred_class <- ifelse(logistic_pred >= 0.5, 1, 0)

logistic_accuracy <- mean(logistic_pred_class == testData$stroke)
print(paste("Logistic Regression Accuracy:", logistic_accuracy))

summary(logistic_pred)

#logistic_model <- glm(stroke ~ ., data = data_final, family = binomial)
#summary(logistic_model)


set.seed(123)

split <- sample.split(data_final$stroke, SplitRatio = 0.7)
train_data <- subset(data_final, split == TRUE)

test_data <- subset(data_final, split == FALSE)

prop.table(table(train_data$stroke))
prop.table(table(test_data$stroke))

logistic_model <- glm(stroke ~ ., data = train_data, family = binomial)
summary(logistic_model)


logistic_pred <- predict(logistic_model, test_data, type = "response")
logistic_pred_class <- ifelse(logistic_pred >= 0.3, 1, 0)

logistic_accuracy <- mean(logistic_pred_class == test_data$stroke)
print(paste("Logistic Regression Accuracy:", logistic_accuracy))


summary(logistic_pred_class)

##########
# Select only numerical columns
numeric_data <- data[, sapply(data, is.numeric)]

correlation_matrix <- cor(numeric_data, use = "complete.obs")

ggcorrplot(correlation_matrix, 
           method = "circle",   
           type = "lower",      
           lab = TRUE) 

ggcorrplot(correlation_matrix, 
           method = "circle", 
           type = "lower", 
           lab = TRUE,
           title = "Correlation Matrix",
           colors = c("red", "white", "blue"))



### Decision tree

tree_model <- rpart(stroke ~ ., data = train_data, method = "class")
tree_pred <- predict(tree_model, test_data, type = "class")
tree_accuracy <- mean(tree_pred == test_data$stroke)
print(paste("Decision Tree Accuracy:", tree_accuracy))

#rpart.plot(tree_model)


### SVM
svm_model <- svm(stroke ~ ., data = train_data, probability = TRUE)
svm_pred <- predict(svm_model, test_data)
svm_accuracy <- mean(svm_pred == test_data$stroke)
print(paste("SVM Accuracy:", svm_accuracy))

### KNN
train_data_norm <- scale(train_data[ , -which(names(train_data) == "stroke")])
test_data_norm <- scale(test_data[ , -which(names(test_data) == "stroke")])
knn_pred <- knn(train_data_norm, test_data_norm, train_data$stroke, k = 5)
knn_accuracy <- mean(knn_pred == test_data$stroke)
print(paste("k-NN Accuracy:", knn_accuracy))

print(paste("Logistic Regression Accuracy:", logistic_accuracy))
print(paste("Decision Tree Accuracy:", tree_accuracy))
print(paste("SVM Accuracy:", svm_accuracy))
print(paste("k-NN Accuracy:", knn_accuracy))




confusionMatrix(as.factor(logistic_pred_class), as.factor(test_data$stroke))

confusionMatrix(as.factor(tree_pred), as.factor(test_data$stroke))

confusionMatrix(as.factor(rf_pred), as.factor(test_data$stroke))

confusionMatrix(as.factor(svm_pred), as.factor(test_data$stroke))

confusionMatrix(as.factor(knn_pred), as.factor(test_data$stroke))


describe(logistic_pred_class)
describe(test_data$stroke)

describe(data)


####### age and stroke

ggplot(data, aes(x = age, y = stroke, color = stroke)) +
  geom_point(alpha = 0.6) +
  labs(title = "Scatter Plot of Age vs Stroke",
       x = "Age",
       y = "Stroke",
       color = "Stroke") +
  theme_minimal()

ggplot(data, aes(x = stroke, y = age, fill = stroke)) +
  geom_violin() +
  labs(title = "Violin Plot of Age by Stroke Status",
       x = "Stroke",
       y = "Age",
       fill = "Stroke") +
  theme_minimal()
summary(data$age)

ggplot(data, aes(x = age, fill = stroke)) +
  geom_bar(position = "dodge") +
  labs(title = "Number of Strokes by Age",
       x = "Age",
       y = "Count",
       fill = "Stroke Status") +
  theme_minimal()

####### to see at what age stroke started
data$stroke <- as.numeric(as.character(data$stroke))
age_intervals <- cut(data$age, breaks = seq(0, 100, by = 5), right = FALSE)
data_interval <- data.frame(age_interval = age_intervals, stroke = data$stroke)
stroke_counts <- data_interval %>%
  group_by(age_interval) %>%
  summarise(count_stroke = sum(stroke), .groups = 'drop')


ggplot(stroke_counts, aes(x = age_interval, y = count_stroke)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Number of Strokes by Age Interval",
       x = "Age Interval",
       y = "Count of Strokes") +
  theme_minimal()



data_long <- melt(data, id.vars = "stroke", variable.name = "parameter", value.name = "value")

continuous_params <- sapply(data, is.numeric)
continuous_params <- names(continuous_params[continuous_params])

# Determine which parameters are categorical
categorical_params <- setdiff(names(data), continuous_params)

ggplot(data_long[data_long$parameter %in% categorical_params, ], aes(x = value, fill = factor(stroke))) +
  geom_bar(position = "dodge") +
  facet_wrap(~ parameter, scales = "free") +
  labs(title = "Association of Categorical Parameters with Stroke",
       x = "Parameter",
       y = "Count",
       fill = "Stroke") +
  theme_minimal()


summary(data$work_type)
data <- data[data$work_type != "children", ]


ggplot(data, aes(x = age, fill = work_type)) +
  geom_bar(position = "dodge") +
  labs(title = "Number of Strokes by Age",
       x = "Age",
       y = "Count",
       fill = "Stroke Status") +
  theme_minimal()

ggplot(data, aes(x = age, fill = work_type == "Children")) +
  geom_bar(position = "dodge") +
  labs(title = "Number of Strokes by Age",
       x = "Age",
       y = "Count",
       fill = "Stroke Status") +
  theme_minimal()

data$age <- as.integer(data$age)
head(data$age)
summary(data$age)



######################### all above models are not good due to imbalance between positive and negative cases of stroke

# Convert categorical columns to dummy variables
summary(data)
summary(data$stroke)
categorical_cols <- c("gender", "ever_married", "work_type", "Residence_type", "smoking_status")

dummy_model <- dummyVars(paste("~", paste(categorical_cols, collapse = "+")), data = data, fullRank = TRUE)
data_dummies <- predict(dummy_model, newdata = data)

# Combine the dummy variables with the numerical columns
data_encoded <- cbind(data_dummies, data[, !(names(data) %in% categorical_cols)])

# Convert 'stroke' to a binary numeric variable (0 or 1)
data_encoded$stroke <- as.numeric(data$stroke) - 1

set.seed(123)
split <- sample.split(data_encoded$stroke, SplitRatio = 0.7)
train_data <- subset(data_encoded, split == TRUE)
test_data <- subset(data_encoded, split == FALSE)

# Convert data to xgboost DMatrix

train_data_numeric <- data.frame(lapply(train_data, function(x) as.numeric(as.character(x))))
test_data_numeric <- data.frame(lapply(test_data, function(x) as.numeric(as.character(x))))

train_data_numeric <- train_data_numeric[, colSums(is.na(train_data_numeric)) == 0]
test_data_numeric <- test_data_numeric[, colSums(is.na(test_data_numeric)) == 0]


balanced_data <- ovun.sample(stroke ~ ., data = train_data_numeric, method = "both", N = nrow(train_data_numeric))$data

dtrain <- xgb.DMatrix(data = as.matrix(balanced_data[,-ncol(balanced_data)]), 
                      label = balanced_data$stroke)
dtest <- xgb.DMatrix(data = as.matrix(test_data_numeric[,-ncol(test_data_numeric)]), 
                     label = test_data_numeric$stroke)



summary(train_data_numeric)
table(train_data_numeric$stroke)
table(balanced_data$stroke)
table(test_data_numeric$stroke)
# Define XGBoost parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  max_depth = 3,
  eta = 0.1
)

### give more weight to 1 stroke to predict
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",  
  max_depth = 3,
  eta = 0.1,
  scale_pos_weight = sum(balanced_data$stroke == 0) / sum(balanced_data$stroke == 1)  
)


set.seed(123)  
cv <- xgb.cv(
  params = params,
  data = dtrain,
  nfold = 5,
  nrounds = 100,
  stratified = TRUE,
  metrics = "auc",
  verbose = 0
)

print(cv)
str(cv)

best_nrounds <- which.max(cv$evaluation_log$test_auc_mean)
print(best_nrounds)


xgb_model <- xgboost(params = params, data = dtrain, nrounds = best_nrounds)

xgb_pred <- predict(xgb_model, dtest)

xgb_pred_class <- ifelse(xgb_pred > 0.5, 1, 0)

xgb_accuracy <- mean(xgb_pred_class == test_data$stroke)
print(paste("XGBoost Accuracy:", xgb_accuracy))

roc_obj <- roc(test_data$stroke, xgb_pred)
print(paste("AUC-ROC:", auc(roc_obj)))
plot(roc_obj, main = "ROC Curve for XGBoost Model")


conf_matrix <- confusionMatrix(as.factor(xgb_pred_class), as.factor(test_data$stroke))
print(conf_matrix)
