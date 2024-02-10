#!/usr/bin/env python
# coding: utf-8

# In[171]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set(context="notebook",style = 'darkgrid' ,font_scale = 1.5, color_codes=True)
import warnings
warnings.filterwarnings('ignore')
import os
import scipy.stats
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from math import ceil
from itertools import zip_longest
import seaborn
import statsmodels.stats.api as sms
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std


# In[172]:


data = pd.read_csv("/Users/abhinavadarsh/Desktop/NEHA/WInter'22_2ndQuarter/ALY6015/W1/AmesHousing.csv")
data.shape


# In[173]:


#Exploratory data analysis
#top 5 observations
data.head()


# In[174]:


#Last 5 observations
data.tail()


# In[175]:


data.info()


# In[176]:


#descriptive statistics of data
data.describe()


# In[177]:


data.nunique()   #Unique values


# In[178]:


#Analysing the target variable (Sales price)
sns.set(rc={'figure.figsize':(12,8)})
sns.distplot(data['SalePrice'], bins = 20)

print('Skew: {:.3f} | Kurtosis: {:.3f}'.format(
    data.SalePrice.skew(), data.SalePrice.kurtosis()))

data["SalePrice"].describe()


# In[179]:


print('The lowest house price ${:,.0f} and the highest house price ${:,.0f}'.format(
    data.SalePrice.min(), data.SalePrice.max()))
print('The mean of sales price is ${:,.0f}, while median is ${:,.0f}'.format(
    data.SalePrice.mean(), data.SalePrice.median()))

data.SalePrice.hist(bins=75, rwidth=.8, figsize=(14,4))
plt.title('Highest and lowest prices of the houses?')
plt.show()


# In[180]:


#Oldest and newest houses
print('Oldest house built in {}. Newest house in {}.'.format(data.YearBuilt.min(), data.YearBuilt.max()))

data.YearBuilt.hist(bins=12, rwidth=.9, figsize=(10,4))
plt.xlabel('Year Built')
plt.show()


# In[181]:


data["YearBuilt"].describe()


# In[182]:


#Highest sales year and month wise
data.groupby(['Yr Sold','Mo Sold']).Order.count().plot(kind='bar', figsize = (14,4))
plt.xlabel('Month and Year of sales')
plt.title('Year and Month of House sales')
plt.show()


# In[183]:


#Effect on sale prices
sns.regplot(x=data['Mo Sold'], y=data['SalePrice'])


# In[184]:


#Houses range in Neighborhoods
data.groupby('Neighborhood').Order.count().    sort_values().    plot(kind='barh', figsize=(10,6))
plt.title('Houses range in Neighborhoods')
plt.show()


# In[185]:


#SalePrice vs Neighborhood
plt.figure(figsize=(20,10))
sns.boxplot(x=data['Neighborhood'], y=data['SalePrice'])
plt.xticks(rotation='vertical')


# In[186]:


#SalePrice vs Overall Quality
plt.figure(figsize=(20,10))
sns.boxplot(x=data['OverallQual'], y=data['SalePrice'])
plt.xticks(rotation='vertical')


# In[187]:


#Paired plot - SalePrice vs 'OverallQual','LotArea', 'SalePrice', 'Overall Cond', 'Bedroom AbvGr'
p1 = data[['OverallQual','LotArea', 'SalePrice', 'Overall Cond', 'Bedroom AbvGr']]


# In[188]:


sns.pairplot(p1)


# In[189]:


#YrSold vs SalePrice
sns.boxplot(x=data['Yr Sold'], y=data['SalePrice'])


# In[190]:


#Function Returns a list of numerical and categorical data
def num_cat():

    # Numerical Features
    num_data = data.select_dtypes(include=['number']).columns
    num_data = num_data.drop(['SalePrice']) # drop SalePrice

    # Categorical Features
    cat_data = data.select_dtypes(include=['object']).columns
    return list(num_data), list(cat_data)

num_data, cat_data = num_cat()


# In[191]:


#Numerical Columns plots
num1 = pd.melt(data, value_vars=sorted(num_data))
num2 = sns.FacetGrid(num1, col='variable', col_wrap=4, sharex=False, sharey=False)
num2= num2.map(sns.distplot, 'value')


# In[192]:


#Categorical Columns 
cat1 = pd.melt(data, value_vars=sorted(cat_data))
cat2 = sns.FacetGrid(cat1, col='variable', col_wrap=4, sharex=False, sharey=False)
plt.xticks(rotation='vertical')
cat2 = cat2.map(sns.countplot, 'value')
[plt.setp(ax.get_xticklabels(), rotation=60) for ax in cat2.axes.flat]
cat2.fig.tight_layout()
plt.show()


# In[193]:


#missing values
missing_val = data.isna().sum()
missing_val = missing_val[missing_val!=0]   #filtering columns with at least 1 missing value
print('Columns that are having missing values :', len(missing_val))     #Number of columns with missing values
missing_val.sort_values(ascending=False)    #sorting the columns with missing values


# In[194]:


#Imputing missing values in Numerical variables with mean values of respective variable
data['LotFrontage'] = data['LotFrontage'].fillna(69)
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(1978)
data['MasVnrArea'] = data['MasVnrArea'].fillna(101)
data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(0)
data['BsmtFullBath'] = data['BsmtFullBath'].fillna(0)
data['GarageArea'] = data['GarageArea'].fillna(472)
data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(49)
data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(442)
data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(559)
data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(1051)
data['GarageArea'] = data['GarageArea'].fillna(472)
data['GarageCars'] = data['GarageCars'].fillna(1)


# In[195]:


#Converting categorical columns to numerical
data.PoolQC.replace({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}, inplace=True)   # Pool QC
data.MiscFeature.replace({'Elev':1, 'Gar2':2, 'Shed':3, 'TenC':4, 'Othr':5}, inplace=True)   # Pool QC
data.Alley.replace({'Grvl':1, 'Pave':2}, inplace=True)   #Alley
data.Fence.replace({'GdPrv':1, 'MnPrv':2, 'GdWo':3, 'MnWw':4}, inplace=True)   #Fence
data.FireplaceQu.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)   # Fireplace Quality
data.GarageCond.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)   # Garage Condition
data.GarageQual.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)   # Garage Quality
data.GarageFinish.replace({'Unf':1, 'RFn':2, 'Fin':3}, inplace=True)  # Garage Finish
data.GarageType.replace({'2Types':1, 'Attchd':2, 'Basment':3, 'BuiltIn':4, 'CarPort':5, 'Detchd':6}, inplace=True) #Garage Type
data.BsmtExposure.replace({'No':1, 'Mn':2, 'Av':3, 'Gd':4}, inplace=True)    # Basement Exposure
data.BsmtCond.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)   # Basement Condition
data.BsmtQual.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)    # Basement Quality
data.BsmtFinType1.replace({'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}, inplace=True)   # Finished Basement 1 Rating
data.BsmtFinType2.replace({'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}, inplace=True)  # Finished Basement 2 Rating
data.MasVnrType.replace({'BrkCmn':1, 'BrkFace':2, 'CBlock':3, 'Stone':4})           #Masonry veneer type
data.Electrical.replace({'SBrkr':1, 'FuseA':2, 'FuseF':3, 'FuseP':4, 'Mix':5}, inplace=True)   #ElectricalMasonry veneer type

#Imputing missing values in categorical columns with 0 
converted_values = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageCond','GarageQual',
        'GarageFinish','GarageType','BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1', 'BsmtFinType2',
        'MasVnrType','Electrical']
data[converted_values] = data[converted_values].fillna(0)

# Update our list of numerical and categorical features
num_data, cat_data = num_cat()


# In[196]:


#Continuous variables pairplot against SalePrice
cont = sns.pairplot(data, x_vars=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','Wood Deck SF','Open Porch SF','GrLivArea','FirstFlrSF','MasVnrArea', 'GarageArea', 'TotalBsmtSF'], y_vars='SalePrice', size=4, aspect=0.6)


# In[197]:


#Numerical columns plotted against SalePrice
num_dep = pd.melt(data, id_vars=['SalePrice'], value_vars=sorted(num_data))
num_dep1 = sns.FacetGrid(num_dep, col='variable', col_wrap=4, sharex=False, sharey=False)
plt.xticks(rotation='vertical')
num_dep1 = num_dep1.map(sns.regplot, 'value', 'SalePrice', scatter_kws={'alpha':0.3})
[plt.setp(ax.get_xticklabels(), rotation=60) for ax in num_dep1.axes.flat]
num_dep1.fig.tight_layout()
plt.show()


# In[198]:


#Categorical columns plotted against SalePrice
cat_dep = pd.melt(data, id_vars=['SalePrice'], value_vars=sorted(cat_data))
cat_dep1 = sns.FacetGrid(cat_dep, col='variable', col_wrap=3, sharex=False, sharey=False, size=4)
cat_dep1 = cat_dep1.map(sns.boxplot, 'value', 'SalePrice')
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in cat_dep1.axes.flat]
cat_dep1.fig.tight_layout()
plt.show()


# In[199]:


#Extracting numeric data in a dataframe
numerics = ['int16', 'int32', 'int64']
numdata = data.select_dtypes(include=numerics)


# In[200]:


numdata1 = data.select_dtypes('number')


# In[201]:


#Correlation matrix of all numeric values
numdata = numdata1.corr()
print(numdata)


# In[202]:


#Correlation heatmap of numeric values
plt.figure(figsize=(22,18))
correlation_heatmap = sns.heatmap(numdata1.corr())


# In[203]:


from scipy import stats
from scipy.stats.stats import pearsonr


# In[204]:


#Continuous variable (GrLivArea) with the highest correlation with SalePrices

corr, _ = pearsonr(data['GrLivArea'], data['SalePrice'])
print('Pearsons correlation: %.3f' % corr)

x = data['GrLivArea']
y = data['SalePrice']
sns.regplot(x, y, scatter_kws={"color": "black"}, line_kws={"color": "red"}, marker="^").set(title='GrLivArea vs SalePrice')


# In[205]:


#Ordinal variable (OverallQual) with the highest correlation with SalePrices
x = data['OverallQual']
y = data['SalePrice']
sns.regplot(x, y, scatter_kws={"color": "black"}, line_kws={"color": "red"}, marker="^")


# In[207]:


#1. Continuous variables with the lowest correlation with SalePrices
x = data['LotArea']             #Lot Area 0.27 correlation
y = data['SalePrice']
sns.regplot(x, y, scatter_kws={"color": "black"}, line_kws={"color": "red"}, marker="^").set(title='Lot Area vs SalePrice')


# In[208]:


#2. Continuous variable (2nd Flr SF) with the lowest correlation with SalePrices
x = data['2nd Flr SF']              #2nd Flr SF 0.27 correlation
y = data['SalePrice']
sns.regplot(x, y, scatter_kws={"color": "black"}, line_kws={"color": "red"}, marker="^").set(title='2nd Floor vs SalePrice')


# In[209]:


from scipy.stats import pearsonr
corr, _ = pearsonr(data['MasVnrArea'], data['SalePrice'])
print('Pearsons correlation: %.3f' % corr)

#Continous variable with 0.5 correlation with SalePrice
sns.regplot(x = data['MasVnrArea'], y = data['SalePrice'], scatter_kws={"color": "black"}, line_kws={"color": "red"}, marker="^").set(title='Masonry veneer area vs SalePrice')


# In[210]:


corr, _ = pearsonr(data['BsmtFinSF2'], data['SalePrice'])
print('Pearsons correlation: %.3f' % corr)

#Continous variable with No correlation (0.06) with SalePrice
sns.regplot(x = data['BsmtFinSF2'], y = data['SalePrice'], scatter_kws={"color": "black"}, line_kws={"color": "red"}, marker="^").set(title='Basement Type 2 finished square feet vs SalePrice')


# In[211]:


#Checking Outliers
sns.regplot(data.GrLivArea, data.SalePrice)   #GrLivArea vs SalePrice


# In[213]:


#Removing outliers (anything above 4000 sq ft)
data.drop(data[data.GrLivArea >= 4000].index, inplace=True)


# In[214]:


#Model 1 (scikit-learn)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[215]:


#Multiple Linear regression
X = data[['GrLivArea','FirstFlrSF','MasVnrArea', 'GarageArea', 'TotalBsmtSF']]
y = data['SalePrice']


# In[216]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)


# In[217]:


#Importing linear regression model
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[218]:


#Model evaluation
# print the intercept
print(lm.intercept_)


# In[219]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[220]:


y_pred = lm.predict(X_train)


# In[221]:


predictions = lm.predict(X_test)


# In[222]:


sns.regplot(y_test, predictions)


# In[223]:


sns.distplot((y_test-predictions), bins=50)


# In[224]:


print("R squared: {}".format(r2_score(y_true=y_train,y_pred=y_pred)))


# In[225]:


#Assumptions for regression
#1. Checking the Linearity between dependent and independent variables
reg = sns.pairplot(data, x_vars=['GrLivArea','FirstFlrSF','MasVnrArea', 'GarageArea', 'TotalBsmtSF'], y_vars='SalePrice', size=4, aspect=0.6)


# In[226]:


#Dependent variable data distribution
sns.set(rc={'figure.figsize':(12,8)})
sns.distplot(data['SalePrice'], bins = 20)


# In[227]:


#Mean of residuals
residuals = y_train.values-y_pred
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))


# In[228]:


#Residual vs Fitted values plot for Homoscedasticity check
fig, ax = plt.subplots(figsize=(8,3.5))
res = sns.residplot(y_pred, residuals)
plt.title('Residuals vs fitted values for Homoscedasticity Check')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')


# In[229]:


import statsmodels.stats.api as sms
from statsmodels.compat import lzip


# In[230]:


hyp = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(residuals, X_train)
lzip(hyp, test)


# In[231]:


#Normality test
norm = sns.distplot(residuals,kde=True)
norm = plt.title('Normality of error terms/residuals')


# In[232]:


#QQ Plot for probability
fig, ax=plt.subplots(figsize=(10,6))
stats.probplot(residuals, dist='norm', plot=plt)
plt.show


# In[233]:


data1 = data[['GrLivArea','FirstFlrSF','MasVnrArea', 'GarageArea', 'TotalBsmtSF', 'SalePrice']]


# In[234]:


#Multicollinearity test
corr_m = data1.corr()
print(corr_m)


# In[235]:


plt.figure(figsize=(20,10))  
p=sns.heatmap(data1.corr(), annot=True,cmap='RdYlGn',square=True) 


# In[236]:


#Variance Inflation Factor (VIF) to detect Multicollinearity

from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor# the independent variables set

#find design matrix for linear regression model using 'SalePrice' as response variable 
y, X = dmatrices('SalePrice~MasVnrArea + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF', data=data, return_type='dataframe')

#calculate VIF for each explanatory variable
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns

#view VIF for each explanatory variable 
vif


# In[237]:


#Autocorrelation check
plt.figure(figsize=(10,5))
p = sns.lineplot(y_pred,residuals,marker='o',color='blue')
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')

p = sns.lineplot([0,400000],[0,0],color='red')
p = plt.title('Residuals vs fitted values plot for autocorrelation check')


# In[238]:


from statsmodels.stats import diagnostic as diag
min(diag.acorr_ljungbox(residuals , lags = 40)[1])


# In[239]:


#Autocorrelation
import statsmodels.api as sm
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.show()


# In[240]:


# partial autocorrelation
from statsmodels.graphics import tsaplots

fig = tsaplots.plot_acf(residuals, lags=40)
plt.show()


# In[241]:


#Diagnostic plots - Model 1

# imports
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Fit a OLS regression variable
model =smf.ols(formula='SalePrice ~ GrLivArea+MasVnrArea+GarageArea+TotalBsmtSF+FirstFlrSF', data= data)
results = model.fit()
print(results.summary())
  
# Get different Variables for diagnostic
residuals = results.resid
fitted_value = results.fittedvalues
stand_resids = results.resid_pearson
influence = results.get_influence()
leverage = influence.hat_matrix_diag


# PLot different diagnostic plots
plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(nrows=2, ncols=2)
  
plt.style.use('seaborn')

# Residual vs Fitted Plot
sns.scatterplot(x=fitted_value, y=residuals, ax=ax[0, 0])
ax[0, 0].axhline(y=0, color='grey', linestyle='dashed')
ax[0, 0].set_xlabel('Fitted Values')
ax[0, 0].set_ylabel('Residuals')
ax[0, 0].set_title('Residuals vs Fitted Fitted')

# Normal Q-Q plot
sm.qqplot(residuals, fit=True, line='45',ax=ax[0, 1], c='#4C72B0')
ax[0, 1].set_title('Normal Q-Q')

# Scale-Location Plot
sns.scatterplot(x=fitted_value, y=residuals, ax=ax[1, 0])
ax[1, 0].axhline(y=0, color='grey', linestyle='dashed')
ax[1, 0].set_xlabel('Fitted values')
ax[1, 0].set_ylabel('Sqrt(standardized residuals)')
ax[1, 0].set_title('Scale-Location Plot')

# Residual vs Leverage Plot
sns.scatterplot(x=leverage, y=stand_resids, ax=ax[1, 1])
ax[1, 1].axhline(y=0, color='grey', linestyle='dashed')
ax[1, 1].set_xlabel('Leverage')
ax[1, 1].set_ylabel('Sqrt(standardized residuals)')
ax[1, 1].set_title('Residuals vs Leverage Plot')

plt.tight_layout()
plt.show()
  
# PLot Cook's distance plot
sm.graphics.influence_plot(results, criterion="cooks")


# #Stepwise subset regression method to identify the "best" model
# from statsmodels.formula.api import ols
# import statsmodels.api as sm

# In[243]:


#define response variable
y = data['SalePrice']

#define predictor variables
x = data[['GrLivArea','FirstFlrSF','MasVnrArea','GarageArea','TotalBsmtSF']]

#add constant to predictor variables
x = sm.add_constant(x)

#fit regression model
model = sm.OLS(y, x).fit()
print(model.summary())


# In[244]:


#Removing variables with more than 0.05 p value
#define response variable
y1 = data['SalePrice']

#define predictor variables
x1 = data[['GrLivArea','MasVnrArea','GarageArea','TotalBsmtSF']]

#add constant to predictor variables
x1 = sm.add_constant(x1)

#fit regression model
model2 = sm.OLS(y1, x1).fit()
print(model2.summary())


# In[245]:


x_columns = ['GrLivArea','MasVnrArea', 'GarageArea', 'TotalBsmtSF']
y = data['SalePrice']


# In[246]:


#creating a linear model(2) and prediction
x = data[x_columns]
linear_model = LinearRegression()
linear_model.fit(x, y)

x_test = data[x_columns]
y_pred = linear_model.predict(x_test)
print("Prediction for Sale price is ", y_pred)


# In[247]:


#Diagnostic plots - Model 2

# Fit a OLS regression variable
model2 =smf.ols(formula='SalePrice ~ GrLivArea+MasVnrArea+GarageArea+TotalBsmtSF', data= data)
results2 = model2.fit()
print(results2.summary())
  
# Get different Variables for diagnostic
residuals2 = results2.resid
fitted_value2 = results2.fittedvalues
stand_resids2 = results2.resid_pearson
influence2 = results2.get_influence()
leverage2 = influence2.hat_matrix_diag

# PLot different diagnostic plots
plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(nrows=2, ncols=2)
  
plt.style.use('seaborn')

# Residual vs Fitted Plot
sns.scatterplot(x=fitted_value2, y=residuals2, ax=ax[0, 0])
ax[0, 0].axhline(y=0, color='grey', linestyle='dashed')
ax[0, 0].set_xlabel('Fitted Values')
ax[0, 0].set_ylabel('Residuals')
ax[0, 0].set_title('Residuals vs Fitted Fitted')

# Normal Q-Q plot
sm.qqplot(residuals2, fit=True, line='45',ax=ax[0, 1], c='#4C72B0')
ax[0, 1].set_title('Normal Q-Q')

# Scale-Location Plot
sns.scatterplot(x=fitted_value2, y=residuals2, ax=ax[1, 0])
ax[1, 0].axhline(y=0, color='grey', linestyle='dashed')
ax[1, 0].set_xlabel('Fitted values')
ax[1, 0].set_ylabel('Sqrt(standardized residuals)')
ax[1, 0].set_title('Scale-Location Plot')

# Residual vs Leverage Plot
sns.scatterplot(x=leverage2, y=stand_resids2, ax=ax[1, 1])
ax[1, 1].axhline(y=0, color='grey', linestyle='dashed')
ax[1, 1].set_xlabel('Leverage')
ax[1, 1].set_ylabel('Sqrt(standardized residuals)')
ax[1, 1].set_title('Residuals vs Leverage Plot')

plt.tight_layout()
plt.show()
  
# PLot Cook's distance plot
sm.graphics.influence_plot(results2, criterion="cooks")


# In[ ]:


#AIC - Model 1
print(model.aic)


# In[ ]:


#AIC - Model 2
print(model2.aic)

