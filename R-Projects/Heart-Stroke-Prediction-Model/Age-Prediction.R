

install.packages("dplyr")
install.packages("ggplot2")
install.packages("descr")
install.packages("pastecs")
install.packages("psych")
install.packages("plotly")
library(tidyverse)
library(dplyr)
library(ggplot2)
library(descr)
library(pastecs)
library(psych)
library(plotly)

####################################################################################################################################
####################################################################################################################################

#importing and basic checks on data
data = read.csv(file.choose(), header = T)
data = data.frame(data)

head(data)      #first few records
tail(data)      #last few records

describe(data)   #descriptive statistics
str(data)    #structure of data

nrow(data)                    #total rows
ncol(data)                  #total columns
names(data)                 #variable names
data.frame(sapply(data, class))   #columns data types



####################################################################################################################################
####################################################################################################################################


##########  Data Cleaning / categorical columns / NA-missing values ############

# View all distinct categorical variable
lapply(subset(data, select = c(gender, ever_married, work_type, Residence_type, smoking_status, bmi)), unique)

#Gender 
table(data$gender)      # unique variables in the gender column
data$gender = ifelse(data$gender == "Other", "Female", data$gender)  # As 'Female' has the majority counts, replace 'Other' to 'Female'
table(data$gender) 

#BMI
data$bmi = as.numeric(data$bmi) # Convert BMI to numeric
data$bmi[is.na(data$bmi)] = mean(data$bmi,na.rm=TRUE)  # Replace N/A's in BMI column with mean
summary(data$bmi)   #New summary of the variable BMI
plot(data$age, data$bmi)

#Smoking Status
table(data$smoking_status)
#data$smoking_status = as.numeric(data$smoking_status) # Convert smoking status to numeric

# Calculating the probability of formerly smoker, current smokers and non-smokers given that there's only this three categories in the smoking_status 
prob.F = 885 / (885 + 1892 + 789)
prob.N = 1892 / (885 + 1892 + 789)
prob.S = 789 / (885 + 1892 + 789)

data1 = data       #creating a copy of the data
view(data1)

# Replacing 'Unknown' in smoking_status by the other 3 variables according to their proportions we calculated
data1$rand = runif(nrow(data1))
data1 = data1%>%mutate(Probability = ifelse(rand <= prob.F, "formerly smoked", 
                                             ifelse(rand <= (prob.F+prob.N), "never smoked", ifelse(rand <= 1, "smokes", "Check"))))
data1 = data1%>%mutate(smoking.status = ifelse(smoking_status == "Unknown", Probability, smoking_status))

table(data1$smoking.status)   #new smoking status

# Remove columns that are not needed
data1 = subset(data1, select = -c(rand,Probability,smoking_status, id))
view(data1)

#NA/Null values
any(is.na(data1))     #no NA values available in the data
any(is.null(data1))   #no null/missing values


####################################################################################################################################
####################################################################################################################################

#######   EDA   #############

#Gender 
par(mfrow = c(2, 2))
ggplot(data1,aes(factor(gender), fill = factor(gender)))+
  geom_bar() + theme_classic()

#Hypertension
hypercounts = as.data.frame(table(data1$hypertension))    #hypertension counts table
hypercounts$Var1 = ifelse(hypercounts$Var1 == 0, "No", 'Yes')    # Replace num to char

ggplot(hypercounts, aes(x = Var1, y = Freq, fill = Var1)) +          # Bar Chart of Hypertension : No vs. Yes   
  geom_bar(stat = "identity") + theme(legend.position="none") +
  geom_text(aes(label = Freq), vjust = 0) +
  labs(title="Pateient's Hypertension Status",x ="Hypertension", y = "Frequency") +
  theme(plot.title = element_text(hjust = 0.5))

#Heart Disease
heartcounts = as.data.frame(table(data1$heart_disease))    #heart disease counts table
ggplot(heartcounts, aes(x = Var1, y = Freq, fill = Var1)) +     # Bar Chart of Heart Disease : No vs. Yes     
  geom_bar(stat = "identity") + theme(legend.position="none") +
  geom_text(aes(label = Freq), vjust = 0) +
  labs(title="Pateient's Heart Disease Status",x ="Heart Disease", y = "Frequency") +
  theme(plot.title = element_text(hjust = 0.5))

#Stroke
strokecounts = as.data.frame(table(data1$stroke))  # Create stroke counts table
strokecounts$Var1 = ifelse(strokecounts$Var1 == 0, "No", 'Yes')  # As 'Female' has the majority counts, replace num to char 
ggplot(strokecounts, aes(x = Var1, y = Freq, fill = Var1)) +   #Bar graph
  geom_bar(stat = "identity") + theme(legend.position="none") +
  geom_text(aes(label = Freq), vjust = 0) +
  labs(title="Stroke Status",x ="Stroke", y = "Frequency") +
  theme(plot.title = element_text(hjust = 0.5))

#Work type
workcounts = as.data.frame(table(data1$work_type))   # Create work type counts table
ggplot(workcounts, aes(x = Var1, y = Freq, fill = Var1)) +       # Bar Chart of Patient Work Type  
  geom_bar(stat = "identity") + geom_text(aes(label = Freq), vjust = 0) +
  labs(title="Patient Work Type",x ="Work Type", y = "Frequency") 

#Ever married
marriedcounts = as.data.frame(table(data1$ever_married))   # Create ever married counts table
ggplot(marriedcounts, aes(x = Var1, y = Freq, fill = Var1)) +     # Bar Chart of Patients Who Have Been Married 
  geom_bar(stat = "identity") + geom_text(aes(label = Freq), vjust = 0) +
  labs(title="Bar Chart of Patients Who Have Been Married",x ="Ever Married", y = "Frequency") 

#Residence type
rescounts = as.data.frame(table(data1$Residence_type))    # Create residence type counts table
ggplot(rescounts, aes(x = Var1, y = Freq, fill = Var1)) +     # Bar Chart of Patients' residence
  geom_bar(stat = "identity") + theme(legend.position="none") +
  geom_text(aes(label = Freq), vjust = 0) +
  labs(title="Residence Type of the Patients",x ="Residence Type", y = "Frequency") +
  theme(plot.title = element_text(hjust = 0.5))

#Smoking status
smokecounts = as.data.frame(table(data1$smoking.status))
ggplot(smokecounts, aes(x = Var1, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity") + theme(legend.position="none") +
  geom_text(aes(label = Freq), vjust = 0) +
  labs(title="Smoking Status",x ="Smoking Status", y = "Frequency") +
  theme(plot.title = element_text(hjust = 0.5))


# Histogram of Age with normal distribution overlay
par(mfrow = c(2, 2))
mean(data1$age)
histage = hist(data1$age,xlim=c(0,100),
                main="Histogram of Age with Normal Distribution Overlay",
                xlab="Age",las=1)
xfit = seq(min(data1$age),max(data1$age))
yfit = dnorm(xfit,mean=mean(data1$age),sd=sd(data1$age))
yfit = yfit*diff(histage$mids[1:2])*length(data1$age)
lines(xfit,yfit,col="red",lwd=2)

# Average Glucose Level 
histglucose = hist(data1$avg_glucose_level,xlim=c(0,300),
                    main="Avg. Glucose distribution",
                    xlab="Avg. Glucose",las=1)
mean(data1$avg_glucose_level)


# BMI
mean(data1$bmi)
histbmi = hist(data1$bmi,xlim=c(0,80),
                   main="BMI", xlab="BMI")

#Stroke
histstroke = plot(data1$stroke)


#Boxplots
par(mfrow = c(1, 3))
boxplot(data1$age,main="Age",ylab="Age",las=1)
boxplot(data1$avg_glucose_level,main="Average Glucose Level",
        ylab="Avg. Glucose",las=1)
boxplot(data1$bmi,main="BMI",
        ylab="BMI",las=1)

names(data1)
par(mfrow = c(1, 3))
boxplot(age ~ hypertension, data = data1, main="Hypertension by age",
        las=1,names=c("No","Yes"))

boxplot(age ~ stroke, data = data1, main="Stroke by age",
        las=1,names=c("No Stroke","Stroke"))

boxplot(age ~heart_disease , data = data1, main="Heart disease by age",
        las=1,names=c("No","Yes"))

boxplot(age ~work_type , data = data1, main="WorkType and Age", 
        names=c("Children","govt", 'Never', 'Private', 'Self'), xlab = 'work type')

boxplot(age ~ever_married , data = data1, main="EverMarried and Age")


boxplot(age ~smoking.status , data = data1, main="SmokingStatus and Age")

#Violinplots
par(mfrow = c(2, 2))
temp = data1     # Create a temp table for violin plots
temp$stroke = ifelse(temp$stroke == 0, "No", 'Yes')    # Replace num to char
ggplot(temp, aes(x=stroke, y=age, fill = stroke)) +    # Violin Plot of Age in Patients With and Without Strokes
  geom_violin(trim=FALSE, fill='#A4A4A4', color="darkred")+
  geom_boxplot(width=0.1) + theme_minimal()


temp$heart_disease = ifelse(temp$heart_disease == 0, "No", 'Yes')   # Replace num to char
ggplot(temp, aes(x=heart_disease, y=age, fill = heart_disease)) +       # Violin Plot of Age in Patients With and Without Heart Disease
  geom_violin(trim=FALSE, fill='#A4A4A4', color="darkred")+
  geom_boxplot(width=0.1) + theme_minimal()


temp$hypertension = ifelse(temp$hypertension == 0, "No", 'Yes')    # Replace num to char
ggplot(temp, aes(x=hypertension, y=age, fill = hypertension)) +     # Violin Plot of Age in Patients With and Without Hypertension
  geom_violin(trim=FALSE, fill='#A4A4A4', color="darkred")+
  geom_boxplot(width=0.1) + theme_minimal()


####################################################################################################################################
####################################################################################################################################

##########  Correlation  ##############

install.packages("corrplot")
library(corrplot)
library(corrgram)
library(caret)
library(corrr)
library(rpart)
library(rpart.plot)

#corrgram of all numerical variables
corrgram(data1, order=NULL, panel=panel.shade, text.panel=panel.txt,    
         diag.panel=panel.minmax, main="Correlogram")

#correlation table of numerical variables
round(cor(subset(data1, select=c(age,hypertension, heart_disease,avg_glucose_level, bmi, stroke))),2)   

#numeric data 
num_var = data1 %>%
  select(age, hypertension, heart_disease, avg_glucose_level, bmi , stroke)  #extracting all the numeric variables


#corrplot
corrplot(cor(num_var))    #corrplot

#scatter plot matrix
library(psych)
pairs.panels(num_data, 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE) # show correlation ellipses


#Correlogram of the Numeric Variables and Categorical Variables
# Convert the Categorical Variables to Numerical Variables
data2 = data1   #creating a copy
dummy = dummyVars(" ~ .", data = data2)
temp2 = data.frame(predict(dummy, newdata = data2))
names(temp2)       #headers of the new dataset
pairs(temp2)
cor_temp2 = cor(temp2, use = "complete.obs")   #complete.obs removes all the NA values if any
cor_temp3 = corrplot(cor(cor_temp2))    #corrplot of all variables


##############################################################################################################################
#######################################################################################################################

##############################################################################################################################
#######################################################################################################################

############ Exploring more ###########
library(hrbrthemes)

# Breaking the age into multiple age groups
data2$age = cut(data2$age,
                 breaks = c(-Inf
                            ,5 ,10 ,15,20,25,30,35,40,45,50,55,60 ,65,70,75,80
                            , Inf),
                 
                 labels = c("0-4"
                            ,"5-9","10-14","15-19","20-24"
                            ,"25-29","30-34","35-39","40-44"
                            ,"45-49","50-54","55-59","60-64"
                            ,"65-69","70-74","75-79","80-84"
                 ),
                 right = FALSE)
stroke = data2$stroke
agetablestroke = as.data.frame(table(data2$age, stroke))
agetablestroke

# Bar plot showing patient age groups separated by whether or not the patient experienced a stroke
ggplot(agetablestroke, aes(x=Var1, y=Freq, fill=stroke)) + geom_bar(stat="identity") +
  theme_ipsum() + scale_x_discrete(name = "Age Group") +
  ggtitle("Age groups of patients with stroke experience") + ylab("Number of Patients\n") +
  scale_fill_brewer(palette="Paired", labels=c("No","Yes")) +
  theme(axis.title.x = element_text(face="bold", size=14, hjust = 0.5),
        axis.title.y = element_text(face="bold", size=20, hjust=0.5))

#Hypertension vs age
hypertension = data2$hypertension
agetablehypertension = as.data.frame(table(data2$age, hypertension))
agetablehypertension

# Bar plot showing patient age groups separated by whether or not the patient experienced hypertension  
ggplot(agetablehypertension, aes(x=Var1, y=Freq, fill=hypertension)) + geom_bar(stat="identity") +
  theme_ipsum() + scale_x_discrete(name = "Age Group") + 
  ggtitle("Age groups of patients with Hypertension experience") + ylab("Number of Patients\n") +
  scale_fill_brewer(palette="Paired", labels=c("No","Yes")) +
  theme(axis.title.x = element_text(face="bold", size=14, hjust = 0.5),
        axis.title.y = element_text(face="bold", size=20, hjust=0.5))

#Hypertension vs age
heart_disease = data2$heart_disease
agetableheart_disease = as.data.frame(table(data2$age, heart_disease))
agetableheart_disease

# Bar plot showing patient age groups separated by whether or not the patient experienced hypertension  
ggplot(agetableheart_disease, aes(x=Var1, y=Freq, fill=heart_disease)) + geom_bar(stat="identity") +
  theme_ipsum() + scale_x_discrete(name = "Age Group") + 
  ggtitle("Age groups of patients with heart_diseases") + ylab("Number of Patients\n") +
  scale_fill_brewer(palette="Paired", labels=c("No","Yes")) +
  theme(axis.title.x = element_text(face="bold", size=14, hjust = 0.5),
        axis.title.y = element_text(face="bold", size=20, hjust=0.5))

##############################################################################################################################
#######################################################################################################################
                       ########### Feature selection ######
# Forward selection method 

step(lm(age ~ 1, data = data1), direction = 'forward', scope = ~ gender + hypertension + heart_disease + ever_married + 
       work_type + Residence_type + avg_glucose_level + bmi + smoking.status + 
       stroke) 
model_forward = lm(formula = age ~ work_type + ever_married + heart_disease + 
                     stroke + hypertension + smoking.status + avg_glucose_level + 
                     bmi, data = data1) 
summary(model_forward) 

data1
# Backward selection method 
step(lm(age ~ ., data = data1), direction = 'backward') 
model_back = step(lm(age ~ ., data = data1), direction = 'backward') 
summary(model_back) 

# Stepwise selection method 
step(lm(age ~ ., data = data1), direction = 'both') 
model_step <- step(lm(mpg ~ ., data = mtcars), direction = 'both') 
summary(model_step) 

AIC(model_step) 
BIC(model_step) 

# Using the leaps package 
library(leaps) 

# Review the data set 
summary(data1) 

data1 = data1 %>% 
  na.omit() 

# Best subsets with regsubsets 
best_subset = regsubsets(age ~ ., data = data1, nbest = 4) 
reg.summary = summary(best_subset) 
reg.summary 
plot(best_subset, scale = 'adjr2')
names(reg.summary)


##############################################################################################################################
#######################################################################################################################

###### Splitting the data into test and train sets ########

# Create Train and Test set - maintain % of event rate (70/30 split) 
library(caret) 
trainIndex = sort(sample(x = nrow(data1), size = nrow(data1) * 0.7)) 
sample_train = data1[trainIndex,] 
sample_test = data1[-trainIndex,] 
dim(sample_train)
dim(sample_test)


##############################################################################################################################
#######################################################################################################################

###### Model Building ########

############ model 1 ##############
model1 = lm(age ~ hypertension + heart_disease + ever_married + 
              work_type + avg_glucose_level + bmi + smoking.status + stroke, data = sample_train)
summary(model1)
summary(model1)$coefficient
AIC(model1)
BIC(model1)


########### model 2  ###############
model2 = update(model1, ~.-bmi)   # remove the less significant features
summary(model2)  
model2$coefficients



##########  Diagnostic plots  #######
'Residuals vs Fitted : Linearity
Normal QQ : Normality
Scale - Location : Homoscedasticity (constant variance)
Residuals vs Leverage : Unusual obervations'

par(mfrow=c(2,2))
plot(model2)


AIC(model2)
BIC(model2)

prediction = predict(model2, newdata = sample_test)
print(prediction)

sigma(model2)/mean(sample_train$age)

cooksd = cooks.distance(model2)

# references
# https://www.analyticsvidhya.com/blog/2021/05/how-to-create-a-stroke-prediction-model/
#https://www.stat.cmu.edu/capstoneresearch/spring2021/315files/team16.html
#https://www.kaggle.com/adrynh/exploratory-data-analysis-on-stroke-dataset
#https://www.analyticsvidhya.com/blog/2021/06/25-questions-to-test-your-skills-on-linear-regression-algorithm/
#https://www.statology.org/diagnostic-plots-in-r/
