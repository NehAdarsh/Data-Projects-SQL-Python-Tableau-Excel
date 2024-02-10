install.packages('gmodels')
install.packages('ggpubr')
install.packages("dplyr")
install.packages("ggplot2")
library(dplyr)
library(ggplot2)
library(gmodels) # Cross Tables [CrossTable()]
library(corrplot) # Correlation plot [corrplot()]
library(ggpubr) # Arranging ggplots together [ggarrange()]
library(caret) # ML [train(), confusionMatrix(), createDataPartition(), varImp(), trainControl()]
library(cowplot) # Arranging ggplots together [plot_grid()]
library(ROCR) # Model performance [performance(), prediction()]

#Data import
data_bank = read.csv(file.choose(), header = T)
View(data_bank)
names(data_bank)
dim(data_bank)
CrossTable(data_bank$y)

#Making a cpoy of the original dataset
data2 = data_bank

#Converting to factors
data2 = data2 %>% 
  mutate(y = factor(if_else(y == "yes", "1", "0"), 
                    levels = c("0", "1")))

#NA Check
any(is.na(data2))
colSums(is.na(data2)) 

#########################################################################################
#########################################################################################
#########################################################################################

######### Predictive Modeling ##########

# Data Preparartion

# re-ordering levels from factor variable
fun_reorder_levels = function(data2, variable, first){   
  remaining = unique(data2[, variable])[which(unique(data2[, variable]) != first)]
  x = factor(data2[, variable], levels = c(first, remaining))
  return(x)
}

#labeling “0” in the pdays_dummy variable for customers who have not been contacted in previous campaign.
data2 = data2 %>% 
  mutate(pdays_dummy = if_else(pdays == 999, "0", "1")) %>% 
  select(-pdays)

#Converting variables to factors
data2$age = fun_reorder_levels(data2, "age", "low")
data2$job = fun_reorder_levels(data2, "job", "unemployed")
data2$marital = fun_reorder_levels(data2, "marital", "single")
data2$education = fun_reorder_levels(data2, "education", "basic.4y")
data2$contact = fun_reorder_levels(data2, "contact", "telephone")
data2$month = fun_reorder_levels(data2, "month", "(03)mar")
data2$day_of_week = fun_reorder_levels(data2, "day_of_week", "(01)mon")
data2$campaign = fun_reorder_levels(data2, "campaign", "1")
data2$previous = fun_reorder_levels(data2, "previous", "0")
data2$poutcome = fun_reorder_levels(data2, "poutcome", "nonexistent")
data2$pdays_dummy = fun_reorder_levels(data2, "pdays_dummy", "0")



#Splitting data into train and test set
set.seed(1234)

split_data = createDataPartition(data2$y,
                          times = 1,
                          p = 0.8,
                          list = F)
train = data2[split_data, ]
test = data2[-split_data, ]
dim(train)
dim(test)
#########################################################################################

#Logistic Regression
logistic_reg = glm(y ~ .,
               data = train,
               family = "binomial")

summary(logistic_reg)
summary(logistic_reg)$coef


## Prediction ##  
pred = predict(logistic_reg, train, type = "response")  
probs = ifelse(pred>.547, 1,0)  

#Predicted scores for test and train data
logistic_train_score = predict(logistic_reg,
                               newdata = train,
                               type = "response")

logistic_test_score = predict(logistic_reg,
                              newdata = test,
                              type = "response")

#Confusion Matrix from the training data
logistic_train_cut = 0.2
logistic_train_class = fun_cut_predict(logistic_train_score, logistic_train_cut)
# matrix
logistic_train_confm = confusionMatrix(logistic_train_class, train$y, 
                                       positive = "1",
                                       mode = "everything")
#Confusion Matric (Training data)
logistic_train_confm

#Confusion matrix from the testing data
logistic_test_class = fun_cut_predict(logistic_test_score, logistic_train_cut)
                                       # matrix
                                       logistic_test_confm = confusionMatrix(logistic_test_class, test$y, 
                                                                             positive = "1",
                                                                             mode = "everything")
#Confusion Matrix (Test Data)                                       
logistic_test_confm


## ROC Curve
res<-prediction(pred, train$y) 
eval<-performance(res, "acc")  
plot(eval)  
abline(h=.95, v=.547)  
max<-which.max(slot(eval, "y.values")[[1]])  
slot(eval, "y.values")[[1]][max]  
slot(eval, "x.values")[[1]][max]  
roc<-performance(res, "tpr", "fpr")  
plot(roc)  
abline(a=0,b=1)


## AUC
plot(roc)  
abline(a=0,b=1)  
auc<-performance(res, "auc")  
auc<-unlist(slot(auc, "y.values"))  
auc<-round(auc,3)  
legend(.8,.4,auc, title = "AUC")


#########################################################################################
#########################################################################################

model2 = glm(y ~ pdays_dummy+euribor3m+emp.var.rate+cons.price.idx+poutcome+duration+day_of_week+month+
               contact+default,
                   data = train,
                   family = "binomial")
summary(model2)



#########################################################################################
#########################################################################################

