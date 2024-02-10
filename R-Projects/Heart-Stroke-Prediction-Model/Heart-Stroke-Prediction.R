heartstk <- read.csv(file.choose(), header= T)
str(heartstk)
names(heartstk)
##checking for Missing data
colSums(is.na(heartstk))
##no missing data 
heartstk$bmi = as.integer(heartstk$bmi)
heartstk <- na.omit(heartstk)

##converting all categorical columns into factor
library(tidyverse)
heartstk<- heartstk %>% mutate_if(is.character,as.factor)
heartstk<- heartstk %>% mutate_if(is.numeric,as.integer)
heartstk$hypertension = as.factor(heartstk$hypertension)
heartstk$heart_disease = as.factor(heartstk$heart_disease)
heartstk$stroke = as.factor(heartstk$stroke)
##checking summary of the data 
summary(heartstk)
attach(heartstk)
# Exploring MULTIPLE CONTINUOUS features
ColsForHist=c("age","bmi","avg_glucose_level")

#Splitting the plot window into four parts
par(mfrow=c(2,2))

hist(heartstk[,c(ColumnName)], main=paste('Histogram of:', ColumnName), 
       col=brewer.pal(8,"Paired"))

# Exploring MULTIPLE CATEGORICAL features
ColsForBar=c("gender","hypertension","heart_disease","ever_married")
ColsForBar1 = c("work_type","Residence_type","smoking_status","stroke")

#Splitting the plot window into four parts
par(mfrow=c(2,2))


  barplot(table(heartstk[,c(ColumnName)]), main=paste('Barplot of:', ColumnName), 
          col=brewer.pal(8,"Paired"), cex.names = 0.8)


# Categorical Vs Continuous Visual analysis: Boxplot
library(RColorBrewer)
############################################################
############################################################
 Barplot
#, 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'
## Gender

barplot(CrossTabResult, beside=T, col=c('Red','Green'), legend=T, main = 'Grouped Bar plot of Gender vs Stroke')

## hypertension
barplot(CrossTabResult, beside=T, col=c('Light Blue','Light Green'), legend=T, main = 'Grouped Bar plot of Hypertension vs Stroke')

## heart_disease
barplot(CrossTabResult, beside=T, col=c('Blue','Light Green'), legend=T, main = 'Grouped Bar plot of Heart Disease vs Stroke')

## ever_married
barplot(CrossTabResult, beside=T, col=c('Red','Light Green'), legend=T, main = 'Grouped Bar plot of ever_married vs Stroke')

##Smoking status
barplot(CrossTabResult, beside=T, col=c('Red','Light Green', 'blue', 'green'), legend=T, main = 'Grouped Bar plot of Smoking_status vs Stroke')


############################################################

InputData=heartstk

# Specifying the Target Variable
TargetVariableName="stroke"

# Making sure the class of Target variable is FACTOR
InputData[, c(TargetVariableName)]=as.factor(InputData[, c(TargetVariableName)])
class(InputData[, c(TargetVariableName)])

# Summarizing the Target Variable
summary(InputData[, c(TargetVariableName)])


#############################################################################################

# Extracting Target and predictor variables from data 
TargetVariable=InputData[, c(TargetVariableName)]
str(TargetVariable)


PredictorVariables=InputData[, !names(InputData) %in% TargetVariableName]
str(PredictorVariables)

DataForML=data.frame(TargetVariable,PredictorVariables)
str(DataForML)


#############################################################################################
# Sampling | Splitting data into 70% for training 30% for testing
TrainingSampleIndex=sample(1:nrow(DataForML), size=0.7 * nrow(DataForML) )
DataForMLTrain=DataForML[TrainingSampleIndex, ]
DataForMLTest=DataForML[-TrainingSampleIndex, ]
DataForMLTrain = na.omit(DataForMLTrain)
DataForMLTest = na.omit(DataForMLTest)
dim(DataForMLTrain)
dim(DataForMLTest)

#############################################################################################
#############################################################################################
# Performing stepwise regression model for feature selection and comparing it with full model
###### Logistic Regression #######

LR_Model=glm(TargetVariable ~ ., data=DataForMLTrain, family='binomial')
summary(LR_Model)

##Splitwise selection
library(MASS)
step.model <- LR_Model %>% stepAIC(trace = FALSE)
coef(step.model)
summary(step.model)
## as per stepwise regression model best predictor variables are age,hypertension1,ever_marriedYes, avg_glucose_level

# Make predictions for full model
probabilities <- LR_Model %>% predict(DataForMLTest, type = "response")
predicted.classes <- ifelse(probabilities > 0.3, 1,0)

# Prediction accuracy
observed.classes <- DataForMLTest$TargetVariable
mean(predicted.classes == observed.classes)


# Make predictions for stepwise model
probabilities <- step.model %>% predict(DataForMLTest, type = "response")
predicted.classes1 <- ifelse(probabilities > 0.3, 1,0)

# Prediction accuracy
observed.classes1 <- DataForMLTest$TargetVariable
mean(predicted.classes1 == observed.classes1)

## after testing the accuracy on both the models - step.model has higher accuracy

# Creating Predictive models on training data to check the accuracy on test data
###### Logistic Regression #######
LR_ModelFinal=glm(TargetVariable ~ age + avg_glucose_level + hypertension + heart_disease , data=DataForMLTrain, family='binomial')
LR_ModelFinal
summary(LR_ModelFinal)


# Checking Accuracy of model on Training data
PredictionProb=predict(LR_ModelFinal, DataForMLTrain, type = "response")
DataForMLTrain$Prediction=ifelse(PredictionProb>=0.04, 1, 0)
DataForMLTrain$Prediction=as.factor(DataForMLTrain$Prediction)
head(DataForMLTrain)
str(DataForMLTrain)
median(PredictionProb)
mean(PredictionProb)
max(PredictionProb)
min(PredictionProb)


# Checking Accuracy of model on Testing data
PredictionProb=predict(LR_ModelFinal, DataForMLTest, type = "response")
DataForMLTest$Prediction=ifelse(PredictionProb>=0.04, 1, 0)
DataForMLTest$Prediction=as.factor(DataForMLTest$Prediction)
head(DataForMLTest)
str(DataForMLTest)
max(PredictionProb)
min(PredictionProb)
mean(PredictionProb)
class(DataForMLTest$TargetVariable)
class(DataForMLTest$Prediction)
levels(DataForMLTest$TargetVariable)
levels(DataForMLTest$Prediction)

# Creating the Confusion Matrix to calculate overall accuracy, precision and recall on TESTING data
library(caret)
Accuracy1 =confusionMatrix(DataForMLTest$Prediction, DataForMLTest$TargetVariable, mode = "prec_recall")
Accuracy1

## checking accuracy of training dataset

Accuracy2 =confusionMatrix(DataForMLTrain$Prediction, DataForMLTrain$TargetVariable, mode = "prec_recall")
Accuracy2


cm <- confusionMatrix(DataForMLTest$TargetVariable,DataForMLTest$Prediction)
cm                      
# Since AccuracyResults is a list of multiple items, fetching useful components only
Accuracy1[['table']]
Accuracy1[['byClass']]

##Plotting the Receiver Operator Characterstics Curve
install.packages("pROC")
library(pROC)
lr_predict <- predict(LR_ModelFinal, DataForMLTest, probability =TRUE)
auc_gbm = roc(DataForMLTest$TargetVariable, PredictionProb,plot = TRUE, col = "blue")

## Area under the curve
auc(auc_gbm)






