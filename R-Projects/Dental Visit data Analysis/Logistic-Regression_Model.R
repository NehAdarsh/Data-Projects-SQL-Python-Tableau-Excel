#Installing required libraries
install.packages("dplyr")
install.packages("ggplot2")
install.packages("descr")
install.packages("psych")
library(dplyr)
library(ggplot2)
library(descr)
library(psych)
install.packages("DataExplorer")
library(DataExplorer)
install.packages('inspectdf')
library(inspectdf)
install.packages("tidyverse")
library(tidyverse)

#Data import
dental_data = read.csv(file.choose(), header = T)
View(dental_data)  

#Few basic checks on data
nrow(dental_data)                        #rows in dataset
ncol(dental_data)                        #columns in dataset
names(dental_data)
str(dental_data)                         #structure of the dataset
summary(dental_data)                     #summary of the dataset
describe(dental_data)                    #descriptive statistics of the dataset
data.frame(sapply(dental_data, class))   #columns data types
plot_str(dental_data)                    #Dataset structure

#Checking NA or NULL values
any(is.na(dental_data))
any(is.null(dental_data))
colSums(is.na(dental_data))             #NA count in each of the columns

#Imputing mode of each of the variable in place of NA values
getmode <- function(x) {
  uniqv <- unique(x)
  uniqv[which.max(tabulate(match(x, uniqv)))]
}                                                     #mode function
m1 = getmode(dental_data$healthgroup)
dental_data$healthgroup[is.na(dental_data$healthgroup)] = m1      #healthgroup mode imputation

m2 = getmode(dental_data$agegrp)
dental_data$agegrp[is.na(dental_data$agegrp)] = m2                #Agegroup mode imputation

m3 = getmode(dental_data$race)
dental_data$race[is.na(dental_data$race)] = m3                    #race mode imputation

m4 = getmode(dental_data$employ.ins)
dental_data$employ.ins[is.na(dental_data$employ.ins)] = m4        #employ.ins mode imputation

m5 = getmode(dental_data$insured )
dental_data$insured [is.na(dental_data$insured )] = m5            #insured  mode imputation

m6 = getmode(dental_data$employ)
dental_data$employ [is.na(dental_data$employ )] = m6              #employ  mode imputation

m7 = getmode(dental_data$marital.stat)
dental_data$marital.stat[is.na(dental_data$marital.stat)] = m7    #marital.stat  mode imputation

m8 = getmode(dental_data$postponed.care)
dental_data$postponed.care[is.na(dental_data$postponed.care)] = m8    #postponed.care  mode imputation

m9 = getmode(dental_data$emergency )
dental_data$emergency [is.na(dental_data$emergency )] = m9    #emergency   mode imputation

m10 = getmode(dental_data$specialist)
dental_data$specialist[is.na(dental_data$specialist)] = m10     #specialist    mode imputation

m11 = getmode(dental_data$meds)
dental_data$meds[is.na(dental_data$meds)] = m11            #meds mode imputation

m12 = getmode(dental_data$health)
dental_data$health[is.na(dental_data$health)] = m12            #health mode imputation

m13 = getmode(dental_data$educ)
dental_data$educ[is.na(dental_data$educ)] = m13            #educ mode imputation

#Checking NA one more time after imputations
colSums(is.na(dental_data))             

#Exploring variables
ggplot(data = dental_data) +
  geom_bar(mapping = aes(x = dental.visit))

ggplot(data = dental_data) +
  geom_bar(mapping = aes(x = dental.visit, fill = sex)) +
  labs(title = "Dental", subtitle = "Based on Gender")

ggplot(data = dental_data) +
  geom_bar(mapping = aes(x = dental.visit, fill = insured)) +
  labs(title = "Dental", subtitle = "Based on Insurance")

ggplot(data = dental_data) +
  geom_bar(mapping = aes(x = dental.visit, fill = health)) +
  labs(title = "Dental", subtitle = "Based on Health")

ggplot(data = dental_data) +
  geom_bar(mapping = aes(x = dental.visit, fill = healthgroup)) +
  labs(title = "Dental", subtitle = "Based on Healthgroup")

ggplot(data = dental_data) +
  geom_bar(mapping = aes(x = dental.visit, fill = race)) +
  labs(title = "Dental", subtitle = "Based on Race")

ggplot(data = dental_data) +
  geom_bar(mapping = aes(x = dental.visit, fill = agegrp)) +
  labs(title = "Dental", subtitle = "Based on Age Group")

ggplot(data = dental_data) +
  geom_bar(mapping = aes(x = dental.visit, fill = emergency)) +
  labs(title = "Dental", subtitle = "Based on Emergency")

ggplot(data = dental_data) +
  geom_bar(mapping = aes(x = dental.visit, fill = educ)) +
  labs(title = "Dental", subtitle = "Based on Education")

ggplot(data = dental_data) +
  geom_bar(mapping = aes(x = dental.visit, fill = meds)) +
  labs(title = "Dental", subtitle = "Based on Medications")

ggplot(data = dental_data) +
  geom_bar(mapping = aes(x = dental.visit, fill = marital.stat)) +
  labs(title = "Dental", subtitle = "Based on Marital Status")

ggplot(data = dental_data) +
  geom_bar(mapping = aes(x = dental.visit, fill = children)) +
  labs(title = "Dental", subtitle = "Based on Children")

#Converting the response variable to a factor
dental_data$dental.visit = as.factor(dental_data$dental.visit )

#Logistic regression

#Split into train and test
install.packages('caTools')
library(caTools)
install.packages('randomForest')
library(randomForest)

set.seed(823)
split <- sample.split(dental_data, SplitRatio = 0.75)
train <- subset(dental_data, split == "TRUE")
test <- subset(dental_data, split == "FALSE")
str(train)
str(test)

#Feature selection
set.seed(823)
VariableImportancePlot <- randomForest(as.factor(dental.visit) ~. , data = train, importance=TRUE)
varImpPlot(VariableImportancePlot)

#Logistic regression Model
model = glm(dental.visit~., family = "binomial", data = train)             #With all the variables as independent
summary(model)

model_2 = glm(dental.visit~employ+insured+marital.stat+postponed.care+emergency+meds+specialist+health+confident+educ, 
              data = train, family = "binomial")             #With all the variables as independent
summary(model_2)

#Chi square
anova(model, model_2, test="Chisq")

#Predict on the test set
thresh <- 0.5        #Threshold set to 0.5
predictedNumLog <- predict(model_2,newdata=test,type='response')
predictedLog <- ifelse(predictedNumLog > thresh,1,0) 
test$predicted <- predictedLog


