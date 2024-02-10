#Importing the Libraries
import warnings

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

#importing churn data
churn_data = pd.read_csv('Customer_Churn.csv')

#a few basic checks
churn_data.head() #Checking top 5 records of the data
churn_data.tail() #Checking last 5 records of the data
churn_data.columns #Checking column names
churn_data.info()  #Checking information of the variable in the data
churn_data.shape #Checking the rows and columns
churn_data.dtypes #Checking the data types of all the variables
churn_data.describe() #Checking the descriptive statistics of the numeric variables

#churn_data.describe(): 
# Although Senior citizen is given as an integer variable but the distribution 25%-50%-75% is not properly
# done, So we can say it is actually a categorical variable
# 75% customers have the tenure less than 55 months and 50% of the customers has the tensure less than 29 months
#Average monthly charges are 64.76$ whereas 50% customers pay more than 70.35$ and 75% customers pay 89.85$

#-------------------------------------------------------------DATA CLEANING-------------------------------------------------------------
#Checking the ratio of male and female customers
churn_data['gender'].value_counts()/len(churn_data['gender'])*100     # M: 50.4%  F: 49.5% - The ratio is quite same for both the genders

#Plotting the ratio of gender variable
churn_data['gender'].value_counts().plot(kind='barh', figsize=(8,6))
plt.xlabel('Count')
plt.ylabel('Gender')
plt.title('Distribution of gender variable')
plt.show()

#Checking the ratio of male and female customers
churn_data['Churn'].value_counts()/len(churn_data['Churn'])*100       # Y: 26.5%  N: 73% - Highly unbalanced data

#Plotting the Distribution of target variable (Churn) 
churn_data['Churn'].value_counts().plot(kind='barh', figsize=(8,6))
plt.xlabel('Count')
plt.ylabel('Churn')
plt.title('Distribution of target variable (Churn)')
plt.show()

#Plotting the missing data 
missing = pd.DataFrame((churn_data.isnull().sum())*100/churn_data.shape[0]).reset_index()
plt.figure(figsize=(8,10))
ax = sns.pointplot('index',0, data =missing)
plt.xticks(rotation=90,fontsize=7)
plt.title('Missing values count')
plt.ylabel('Percentage')
plt.show()

#Copying the original data to a new variable to keep the original one as it is and to perform the modifications in a new one
new_data = churn_data.copy()
new_data.head()
new_data.info()

#Converting total charges to a numerical variable as it should not be an object type bcz similar variable monthly charges ia a float type data
#After converting to numerical, we can see it has null values which were not there in object type. We cannot say for sure that an object data type had null values or not when its actually a numeric type
new_data.TotalCharges = pd.to_numeric(new_data.TotalCharges, errors='coerce')
new_data.isnull().sum()

#Location of the misisng values
new_data.loc[new_data['TotalCharges'].isnull()==True]

#Imputing the mean values inplace of the nulls as its only 11 records that has the nulls so we can drop or impute them. I am going to impute the mean value
new_data['TotalCharges'] = new_data['TotalCharges'].fillna(new_data['TotalCharges'].mean())

#Checking nulls after imputation
new_data.isnull().sum()  #Now we don't have any null values

#Analyzing the Tenure variable; As it has ranges, I am going to divide it in groups for better understanding
new_data['tenure'].max() #What is the max tenure
labels = ["{0} - {1}".format(i, i+11)for i in range(1, 72, 12)]
new_data['tenure_group'] = pd.cut(new_data.tenure, range(1,80,12), right = False, labels=labels)

new_data['tenure_group'].value_counts()

#Dropping columns which are not making that much sense
new_data.drop(columns=['customerID', 'tenure'], axis = 1, inplace =True)
new_data.head()

#-------------------------------------------------------------EXPLORATORY DATA ANALYSIS-------------------------------------------------------------
#-------------------------------------------------------------UNIVARIATE ANALYSIS-------------------------------------------------------------

new_data2 = new_data.copy()     #created a new dataset to not disturb the old one
new_data2.drop(columns=['TotalCharges', 'MonthlyCharges'], axis = 1, inplace =True)   #dropped the numeric columns as we are going to plot only categorical data
new_data2.columns #checked the columns of new data after dropping a few columns

#Instead of writing individual code for each of the variables, I have written one for loop which will keep track of the index and the value on that index and used seaborn library to plot all the categorical variables
for i, predictor in enumerate(new_data2):
    plt.figure(i)
    sns.countplot(data=new_data2, x=predictor, hue='Churn')
    plt.show()

 # Another way: we can write code for each of the variables like this individually:   
sns.countplot(data=new_data2, x='partner', hue='Churn')
plt.show()

sns.countplot(data=new_data2, x='StreamingTV', hue='Churn')
plt.show()

# CONVERTING TARGET VARIABLE 'CHURN' INTO A NUMERIC VARIABLE
new_data['Churn']=np.where(new_data.Churn == 'Yes', 1, 0)
new_data.head

#Convertin all categorical variables into dummy variables
new_data_dummies = pd.get_dummies(new_data)
new_data_dummies.head()
new_data_dummies.columns

#Relationship between Monthly and total charges
sns.regplot(x='MonthlyCharges', y='TotalCharges', data = new_data_dummies, fit_reg=False)
plt.show()  #as expected total charges increases as the monthly charges increases

#Visualizing monthly charges with our target variable 'churn'
# --- Churn is high when monthly charges are high as per the visualization
m1 = sns.kdeplot(new_data_dummies.MonthlyCharges[(new_data_dummies['Churn'] == 0) ], color="blue", shade = True)
m1 = sns.kdeplot(new_data_dummies.MonthlyCharges[(new_data_dummies['Churn'] == 1) ], color="red", shade = True)
m1.legend(["No Churn","Churn"], loc = 'upper right')
m1.set_ylabel("Density")
m1.set_xlabel("Monthly Charges")
m1.set_title('Monthly charges by Churn')
plt.show()

#Visualizing total charges with our target variable 'churn'
# --- High Churn at Lower total charges as per the visualization - Surprising insight
m2 = sns.kdeplot(new_data_dummies.TotalCharges[(new_data_dummies['Churn'] == 0) ], color="blue", shade = True)
m2 = sns.kdeplot(new_data_dummies.TotalCharges[(new_data_dummies['Churn'] == 1) ], color="red", shade = True)
m2.legend(["No Churn","Churn"], loc = 'upper right')
m2.set_ylabel("Density")
m2.set_xlabel("Total Charges")
m2.set_title('Total charges by Churn')
plt.show()

#Correlation of all the predictors with the target variable
plt.figure(figsize=(20,10))
new_data_dummies.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')
plt.show()

#heatmap of correlation
plt.figure(figsize=(20,10))
sns.heatmap(new_data_dummies.corr(), cmap = 'Paired')
plt.show()

#-------------------------------------------------------------BIVARIATE ANALYSIS-------------------------------------------------------------
#Dividing the target variable into two dataframes; one for churn and one for non-churn
churn_target1 = new_data.loc[new_data['Churn'] == 1]  #We are more concerned about the customers who are churning
churn_target0= new_data.loc[new_data['Churn'] == 0]


#Writing a function to generate plots
def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()

#Making plots by calling the defined function for Churn customers
churn_target1.columns
uniplot(churn_target1,col='Partner',title='Distribution of Partner for Churned Customers',hue='gender')  
uniplot(churn_target1,col='PaymentMethod',title='Distribution of Payment method for Churned Customers',hue='gender') 
uniplot(churn_target1,col='SeniorCitizen',title='Distribution of Senior Citizen for Churned Customers',hue='gender') 
uniplot(churn_target1,col='Dependents',title='Distribution of Dependents method for Churned Customers',hue='gender') 
uniplot(churn_target1,col='PhoneService',title='Distribution of Phone Service for Churned Customers',hue='gender') 
uniplot(churn_target1,col='MultipleLines',title='Distribution of Multiple lines for Churned Customers',hue='gender') 
uniplot(churn_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender') 
uniplot(churn_target1,col='TechSupport',title='Distribution of TechSupport for Churned Customers',hue='gender') 
uniplot(churn_target1,col='tenure_group',title='Distribution of tenure_group for Churned Customers',hue='gender') 

#Making plots by calling the defined function for Non-Churn customers
churn_target0.columns
uniplot(churn_target0,col='Partner',title='Distribution of Partner for Non-Churned Customers',hue='gender')  
uniplot(churn_target0,col='PaymentMethod',title='Distribution of Payment method for Non-Churned Customers',hue='gender') 
uniplot(churn_target0,col='SeniorCitizen',title='Distribution of Senior Citizen for Non-Churned Customers',hue='gender') 
uniplot(churn_target0,col='Dependents',title='Distribution of Dependents method for Non-Churned Customers',hue='gender') 
uniplot(churn_target0,col='PhoneService',title='Distribution of Phone Service for Non-Churned Customers',hue='gender') 
uniplot(churn_target0,col='MultipleLines',title='Distribution of Multiple lines for Non-Churned Customers',hue='gender') 
uniplot(churn_target0,col='Contract',title='Distribution of Contract for Non-Churned Customers',hue='gender') 
uniplot(churn_target0,col='TechSupport',title='Distribution of TechSupport for Non-Churned Customers',hue='gender') 
uniplot(churn_target0,col='tenure_group',title='Distribution of tenure_group for Non-Churned Customers',hue='gender') 


#-------------------------------------------------------------PREPROCESSING-------------------------------------------------------------
from sklearn import metrics
from sklearn.model_selection import train_test_split

#creating a new csv file from the dummies data to use it for modeling
new_data_dummies.to_csv('model_data.csv')

#Reading the scv file
model_data = pd.read_csv('model_data.csv')
model_data.head()

#Dropping columns 
model_data = model_data.drop(columns=['Unnamed: 0'], axis = 1)
model_data.head()

#creating x and y variables 
x = model_data.drop('Churn', axis = 1)
y = model_data['Churn']

#Splitting the data into test and train set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=100)


#-------------------------------------------------------------CLASSIFICATION MODELS------------------------------------------------------------


#-------------------------------------------------------------DECISION TREE-------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

#Fitting the model
tree_clf = DecisionTreeClassifier(criterion = 'gini', random_state = 100, max_depth=4)
tree_clf.fit(x_train, y_train)

#Predicting on the test data
y_pred = tree_clf.predict(x_test)

#Comparing the predictions and the actual test data
tree_clf.score(x_test, y_test)

#Printing the classification report
print(classification_report(y_test, y_pred, labels=[0,1]))

#accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#Confusion matrix
print(confusion_matrix(y_pred, y_test))

#Plotting the tree
tree.plot_tree(tree_clf)
plt.show()


#Conclusion form this model: As the result shows, very low precision, recall, f1 score for class 1 (churn customers) and we can not trust on the accuracy as our dataset is very imbalanced 
# as we saw in the earlier part.
# Accuracy is 76%, extracted form confusion matrix results (correctly classfied values / Total values; (845+230)/(845+154+180+230))

#Let's use up-sampling and down-sampling technique to balance out the data and re build the model using SMOTEENN
from imblearn.combine import SMOTEENN
sm = SMOTEENN(random_state = 42)
x_resampled, y_resampled = sm.fit_resample(x, y)    #Resampling the model

X_train, X_test, Y_train, Y_test = train_test_split(x_resampled, y_resampled, test_size = 0.2, random_state=100)  #splitting again on resampled data

#Fitting the model
new_clf_mdl = DecisionTreeClassifier()
new_clf_mdl.fit(X_train, Y_train)

#Predicting on the test data
Y_pred_new = new_clf_mdl.predict(X_test)

#Printing the classification report
print(classification_report(Y_test, Y_pred_new, labels=[0,1]))

#accuracy
print(accuracy_score(Y_test, Y_pred_new))

#Confusion matrix
print(confusion_matrix(Y_pred_new, Y_test))

#Conclusion form the upsampling model: As the result shows, great precision, recall, f1 score and now that we balanced our data, the accuracy is 93% which is awesome. 
#Hence, this is a good model 

#-------------------------------------------------------------RANDOM FOREST-------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

#Creating a random forest classifier,fitting the model, predicting the results using the before up-sampling data to compare the results
rf_model = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=5, random_state = 100)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)

#Classification report and accuracy
print(classification_report(y_test, rf_pred))
print(accuracy_score(y_test, rf_pred))

#Again the same problem, very low results for the class 1 (churn), but still better as compared to the decision tree initial results
#Let's try with the resampled data
from imblearn.combine import SMOTEENN
rf_sm = SMOTEENN(random_state = 42)
rf_x_resampled, rf_y_resampled = rf_sm.fit_resample(x, y)    #Resampling the model

X_train_rf, X_test_rf, Y_train_rf, Y_test_rf = train_test_split(rf_x_resampled, rf_y_resampled, test_size = 0.2, random_state=100)  #splitting again on resampled data

#Fitting the model
rf_clf_mdl = RandomForestClassifier()
rf_clf_mdl.fit(X_train_rf, Y_train_rf)

#Predicting on the test data
Y_pred_rf = rf_clf_mdl.predict(X_test_rf)

#Printing the classification report
print(classification_report(Y_test_rf, Y_pred_rf))
print(accuracy_score(Y_pred_rf, Y_test_rf))

#Conclusion: upsampling method is working greT FOR RANDOM FOREST AS WELL. GIVING 94.9% ACCURACY.

#-------------------------------------------------------------LOGISTIC REGRESSION-------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
print(classification_report(y_test, lr_pred))  
print(accuracy_score(y_test, lr_pred))  #76%

#Lets try on the up-sampled data
lrs = LogisticRegression()
lrs.fit(x_resampled,y_resampled)
lr_res_pred = lr.predict(X_test)
print(classification_report(lr_res_pred, Y_test)) #Great results
print(accuracy_score(lr_res_pred, Y_test))     #92%

#-------------------------------------------------------------K-Nearest Neighbors-------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier().fit(x_train, y_train)
knn_pred = knn_model.predict(x_test)
print(classification_report(knn_pred, y_test)) #Very bad results
print(accuracy_score(y_test, knn_pred))  #76%

#Let's try with the up-sampled data
knn_res = KNeighborsClassifier().fit(X_train, Y_train)
knn_res_pred = knn_res.predict(X_test)
print(classification_report(knn_res_pred, Y_test))
print(accuracy_score(Y_test, knn_res_pred))  #94% accuracy


#-------------------------------------------------------------CONCLUSION OF ALL MODELS-------------------------------------------------------------
#Printing results with non-upsampled data
print(accuracy_score(y_test, y_pred)) #Decision tree - 76.2% Accuracy
print(accuracy_score(y_test, rf_pred)) #Random forest - 78.8% Accuracy
print(accuracy_score(y_test, lr_pred)) #Logistic regression - 79.2% Accuracy
print(accuracy_score(y_test, knn_pred)) #KNN - 76.6% Accuracy


#Printing results with up-sampled data
print(accuracy_score(Y_test, Y_pred_new)) #Decision tree - 93.8% Accuracy
print(accuracy_score(Y_pred_rf, Y_test_rf)) #Random forest - 94.9% Accuracy
print(accuracy_score(lr_res_pred, Y_test)) #logistic Regression - 92.6% Accuracy
print(accuracy_score(Y_test, knn_res_pred)) #KNN - 94.9% Accuracy

#Final Conclusion
                    #### Non- upsampled data results::::
# As per the results and the accuracies of the models, we can say on the normal data (Non Up-sampled) each of the model is not performing good in terms of class 1 which is of our interest (Chuning customers), 
# Logistic regression is still performing better than the other models in terms of accuracy, precision, recall etc. We can't trust the accuracy here because as we know our data is not balanced.

                    #### Up sampled data results::::  
#As per the results and the accuracies of the models, we can say on the up-sampled data each of the model is performing great, but Random Forest and KNN has the highest accuracy score out of all 4. 
# Also, we can trust the accuracy score of these models because the data is balanced now.


#-------------------------------------------------------------END-------------------------------------------------------------