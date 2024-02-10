#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[1]:


from scipy.stats import chisquare
from scipy.stats import chi2
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import scipy.stats as stats
import numpy as np
from scipy import stats


# ### Chi-Square Examples 

# In[747]:


#Creating dataframe 
Patients = pd.DataFrame([12, 8, 24, 6])
General_Population = pd.DataFrame([0.2*50, 0.28*50, 0.36*50, 0.16*50])   #multiplying with total observations


# In[521]:


#Section 11-1 Blood Types

#                Expected  |  Observed
# typeA             20%    |     12
# typeB             28%    |     8
# typeO             36%    |     24
# typeAB            16%    |     6

##############################################################
# Stating the hypothesis
# H0 : P(typeA)=0.20, P(typeB)=0.28, P(typeO)=0.36, P(typeAB)=0.16
# H1 : The distribution is not the same as stated in the null hypothesis
##############################################################

#Observed and expected values
observed = [12, 8, 24, 6]                      #Observed values
expected = [10, 14, 18, 8]                     #expected values (proportion*total number of observations)
dof = 3
alpha = 0.10

#Perform Chi-Square Goodness of Fit Test
values = stats.chisquare(f_obs=observed, f_exp=expected)      
print(values)

###############################################################

#Printing all the values
critical_val = chi2.ppf(q=1-alpha, df=3)         #critical value
print('\nCritical value:', critical_val)         
print('Significance level:', alpha)              #significance level (0.10)
print('Degree of freedom:', dof)                 #degree of freedom
p_value = 0.14035752729356485 
print('p-value:', p_value)                       #p-value
chi_square_value = 5.4714285714285715
print('Chi-Square value:', chi_square_value)      #chi square value

##############################################################

#making decision
if chi_square_value > critical_val:
    print('\nReject the Null Hypothesis')
else:
    print('\nFail to reject the Null Hypothesis')
    
if p_value > alpha:
    print('\nFail to reject the Null Hypothesis')
else:
    print('\nReject the Null Hypothesis')    

###############################################################


# In[196]:


#Section 11-1 : On time performance by Airlines

#Creating dataframe 
observed_data = pd.DataFrame([125, 10, 25, 40])
govt_data = pd.DataFrame([0.708*200, 0.082*200, 0.09*200, 0.12*200])   #multiplying with total observations
    
##############################################################
# Stating the hypothesis
# H0 : P(typeA)=0.708, P(typeB)=0.082, P(typeO)=0.09, P(typeAB)=0.12
# H1 : The results differ from the government’s statistics
##############################################################
    
#Observed and expected values
observed1 = [125, 10, 25, 40]                         #Observed values
expected1 = [141.6, 16.4, 18, 24]                     #expected values (proportion*total number of observations)
dof = 3
alpha1 = 0.05

#Perform Chi-Square Goodness of Fit Test
values1 = stats.chisquare(f_obs=observed1, f_exp=expected1)      
print(values1)

###############################################################

#Printing all the values
critical_val1 = chi2.ppf(q=1-alpha1, df=dof)         #critical value
print('\nCritical value:', critical_val1)         
print('Significance level:', alpha1)              #significance level (0.05)
print('Degree of freedom:', dof)                 #degree of freedom
p_value1 = 0.0004762587447570704 
print('p-value:', p_value1)                       #p-value
chi_square_value1 = 17.832495062238756
print('Chi-Square value:', chi_square_value1)      #chi square value

##############################################################

#making decision
if chi_square_value1 > critical_val1:
    print('\nReject the Null Hypothesis')
else:
    print('\nFail to reject the Null Hypothesis')
    
if p_value1 > alpha1:
    print('\nFail to reject the Null Hypothesis')
else:
    print('\nReject the Null Hypothesis')    

###############################################################


# In[243]:


#Section 11-2 : Ethnicity and Movie Admission

##############################################################
# Stating the hypothesis
# H0 : Movie attendance by year was independent of ethnicity
# H1 : Movie attendance by year was dependent of ethnicity
##############################################################

#Creating dataframe 
data = [[724,335,174,107],
        [370,292,152,140]]
d = 3                        #degree of freedom
alpha2 = 0.05                #significance level

##############################################################

#Chi-Square Test of Independence
chi = stats.chi2_contingency(data)
print(chi)

critical_v = chi2.ppf(q=1-alpha1, df=d) 
p_v = 5.477507399707207e-13
chiS = 60.14352474168578

##############################################################

#Printing all the values
print('\nCritical value:', critical_v)
print('Chi-Square test staistics:', chiS)
print('p-value:', p_v)
print('Degree of freedom:',d)
print('Significance level:', alpha2)

##############################################################

#making decision
if chiS > critical_v:
    print('\nReject the Null Hypothesis')
else:
    print('\nFail to reject the Null Hypothesis')
    
if p_v > alpha2:
    print('\nFail to reject the Null Hypothesis')
else:
    print('\nReject the Null Hypothesis')    

###############################################################



# In[245]:


#Section 11-2 : Women in the Millitary

##############################################################
# Stating the hypothesis
# H0 : Rank and branch of the Armed Forces are independent of each other
# H1 : Rank and branch of the Armed Forces are dependent on each other
##############################################################

#Creating dataframe 
m_data = [[10791, 62491],[7816, 42750], [932, 9525], [11819, 54344]]
deg_m = 3                        #degree of freedom
m_alpha = 0.05                #significance level

##############################################################

#Chi-Square Test of Independence
m_chi = stats.chi2_contingency(m_data)
print(m_chi)


m_critical_v = chi2.ppf(q=1-m_alpha, df=deg_m) 
p_val_m = 1.7264180110731947e-141
chiS_m = 654.2718888756281

##############################################################

#Printing all the values
print('\nCritical value:', m_critical_v)
print('Chi-Square test staistics:', chiS_m)
print('p-value:', p_val_m)
print('Degree of freedom:',deg_m)
print('Significance level:', m_alpha)

##############################################################

#making decision
if chiS_m > m_critical_v:
    print('\nReject the Null Hypothesis')
else:
    print('\nFail to reject the Null Hypothesis')
    
if p_v > m_alpha:
    print('\nFail to reject the Null Hypothesis')
else:
    print('\nReject the Null Hypothesis')    

###############################################################


# ### ANOVA Test Examples 

# In[2]:


# One-way ANOVA test (Sodium content of foods)


##############################################################
# Stating the hypothesis
# H0 : Means of all three different kinds of foods are equal
# H1 : Atleast one of food's mean value is not equal
##############################################################

#Creating dataframe 
Condiments = [270,130,230,180,80,70,200]
Cereals = [260,220,290,290,200,320,140]
Desserts = [100,180,250,250,300,360,300]

##############################################################

#Performing one-way ANOVA
from scipy.stats import f_oneway
anova_test = f_oneway(Condiments, Cereals, Desserts)
print(anova_test)

##############################################################

#Printing all the values
anova_p = 0.09238736983791047
print('\np-value:', anova_p)
anova_alpha = 0.05
print('Significance level:', anova_alpha)

##############################################################

#making decision
if anova_p > anova_alpha:
    print('\nFail to reject the Null Hypothesis')
else:
    print('\nReject the Null Hypothesis') 

##############################################################


# In[3]:


# Complete One-way ANOVA test (Sales for Leading companies)


##############################################################
# Stating the hypothesis
# H0 : Means of sales of all three companies are equal
# H1 : Atleast one of company's mean value is different
##############################################################

#Creating dataframe 
cereal = [578, 320, 264, 249, 237]
chocolate_candy = [311, 106, 109, 125, 173]
coffee = [261, 185, 302, 689]

##############################################################

#Performing one-way ANOVA
from scipy.stats import f_oneway
anova_test2 = stats.f_oneway(cereal, chocolate_candy, coffee)
print(anova_test2)

##############################################################
#Printing all the values
anova_pVal = 0.16034871320000485
print('\np-value:', anova_pVal)
anova_alpha1 = 0.01
print('Significance level:', anova_alpha1)

##############################################################

#making decision
if anova_pVal > anova_alpha:
    print('\nFail to reject the Null Hypothesis')
else:
    print('\nReject the Null Hypothesis') 

##############################################################


# In[366]:


#Tukey test - Sales for Leading companies

data_1d = [578, 320, 264, 249, 237, 311, 106, 109, 125, 173, 261, 185, 302, 689]
groups_1d = ['cereal', 'cereal', 'cereal','cereal','cereal','chocolate_candy','chocolate_candy','chocolate_candy','chocolate_candy','chocolate_candy','coffee','coffee','coffee','coffee']

# perform Tukey's test
tukey = pairwise_tukeyhsd(data_1d,
                          groups_1d,
                          alpha=0.01)
#display results
print(tukey)


# In[376]:


# Complete One-way ANOVA test (Pre pupil expenditure)


##############################################################
# Stating the hypothesis
# H0 : No difference in means
# H1 : Means are different
##############################################################

#Creating dataframe 
Eastern = [4946, 5953, 6202, 7243, 6113]
Middle = [6149, 7451, 6000, 6479]
Western = [5282, 8605, 6528, 6911]

##############################################################

#Performing one-way ANOVA
from scipy.stats import f_oneway
anova_test3 = f_oneway(Eastern, Middle, Western)
print(anova_test3)

##############################################################

#Printing all the values
anova_p_value = 0.5433264053252742
print('\np-value:', anova_p_value)
anova_alpha2 = 0.05
print('Significance level:', anova_alpha2)

##############################################################

#making decision
if anova_p_value > anova_alpha2:
    print('\nFail to reject the Null Hypothesis\n\n')
else:
    print('\nReject the Null Hypothesis\n\n') 

##############################################################


# In[377]:


#Tukey test - Pre pupil expenditure
data2_1d = [4946, 5953, 6202, 7243, 6113, 6149, 7451, 6000, 6479, 5282, 8605, 6528, 6911]
groups2_1d = ['Eastern', 'Eastern', 'Eastern','Eastern','Eastern','Middle','Middle','Middle','Middle','Western','Western','Western','Western']

# perform Tukey's test
tukey_test = pairwise_tukeyhsd(data2_1d,
                          groups2_1d,
                          alpha=0.05)
#display results
print(tukey_test)


# In[18]:


# Two Way ANOVA test : Section 12-3 (Increasing Plant growth)

##############################################################
# Stating the hypothesis
#Null Hypothesis (H0) :The means of Plant food are equal.
#Alternative Hypothesis (H1) :Means of Plant food are different

#Null Hypothesis (H0) :The means of the grow light groups are equal.
#Alternative Hypothesis (H1) :The means of the grow light groups are not equal.

#Null Hypothesis (H0) :There is no interaction between Food and Light
#Alternative Hypothesis (H1) :There is an interaction between Food and Light
##############################################################
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

#create data
df = pd.read_csv('/Users/abhinavadarsh/Downloads/plant_food.csv')

#anova test
aov = ols('Growth~Food+Light+(Food*Light)', df).fit()           #Fitted model
anova_test = anova_lm(aov, typ = 2)                #Anova test
print(anova_test)


# ### On your own part of assignment

# In[19]:


## On your own part of assignment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#Data import
baseball_data = pd.read_csv('/Users/abhinavadarsh/Desktop/NEHA/WInter_2ndQuarter/ALY6015/W2/baseball.csv')
baseball_data.shape     #Rows and columns in the dataset


# In[749]:


baseball_data.head()


# In[750]:


baseball_data.tail


# In[751]:


baseball_data.columns


# In[3]:


##Renamed confusing variables

baseball_data.rename(columns = {"RS": "Run_Scored", 'RA': 'Runs_Allowed', 'W': 'Wins', 'OBP': 'On_Base_Percentage',
                              'SLG':'Slugging_Percentage','BA':'Batting_Average', 'OOBP':'Opponents_On_Base_Percentage',
                             'OSLG':'Opponents_Slugging_Percentage', 'G':'Game_played'}, inplace = True)
baseball_data


# In[666]:


baseball_data.describe()


# In[4]:


#Missing/null values
baseball_data.isnull().sum()     


# In[5]:


#Visualizing missing values
sns.heatmap(baseball_data.isnull(),cbar=False,cmap='viridis') 


# In[6]:


## replacing missing/null values with the mean value of respective variable

baseball_data['RankSeason'] = baseball_data['RankSeason'].fillna(3)
baseball_data['RankPlayoffs'] = baseball_data['RankPlayoffs'].fillna(2.7)
baseball_data['Opponents_On_Base_Percentage'] = baseball_data['Opponents_On_Base_Percentage'].fillna(0.3)
baseball_data['Opponents_Slugging_Percentage'] = baseball_data['Opponents_Slugging_Percentage'].fillna(0.4)

baseball_data.isnull().sum()      #checking mising/null values after removing


# In[670]:


#Pairplots of the dataset
p1 = baseball_data[['Wins','Playoffs', 'RankPlayoffs', 'Year', 'RankSeason']]

sns.pairplot(p1)


# In[671]:


p2 = baseball_data[['Run_Scored', 'On_Base_Percentage','Slugging_Percentage','Batting_Average', 'Wins','Opponents_On_Base_Percentage','Opponents_Slugging_Percentage']]

sns.pairplot(p2)


# In[672]:


plt.figure(figsize=(10,10))
sns.heatmap(baseball_data.corr(),cbar=True,annot=True,cmap='Blues')


# In[673]:


def num_cat():

    # Numerical Features
    num_data = baseball_data.select_dtypes(include=['number']).columns

    # Categorical Features
    cat_data = baseball_data.select_dtypes(include=['object']).columns
    return list(num_data), list(cat_data)

num_data, cat_data = num_cat()


# In[674]:


#Numerical Columns plots
num1 = pd.melt(baseball_data, value_vars=sorted(num_data))
num2 = sns.FacetGrid(num1, col='variable', col_wrap=4, sharex=False, sharey=False)
num2= num2.map(sns.distplot, 'value')


# In[754]:


#Grouping Wins by Decade
bb = baseball_data.groupby(baseball_data.Year - (baseball_data.Year%10))['Wins'].sum()
print(bb)


# In[8]:


baseball_data.columns


# In[38]:


#Run Difference vs Wins
baseball_data.RunDiff = baseball_data.Run_Scored - baseball_data.Runs_Allowed    #run differnce
plt.scatter(baseball_data.RunDiff , baseball_data.Wins)
plt.xlabel('Run Difference')
plt.ylabel('Wins')


# In[36]:


#How many wins need to make it to the playoffs ?
plt.figure(figsize=(10,9))
ax = sns.scatterplot(x="Wins", y="Team", hue="Playoffs",data=baseball_data)
plt.plot(95, 0, color='r')


# #### Chi Square test

# In[82]:


data_baseball = baseball_data[['Year','W']]    #keeping only year and wins columns in a new variable
wins = data_baseball.groupby(["Year"])["W"].sum()   #grouped the wins by decades

# creating a new dataframe of year and wins
wins_byYear = [[1960, 13267], [1970, 17934], [1980, 18926], [1990, 17972], [2000, 24286], [2010, 7289]]
wins_byYear1 = pd.DataFrame(wins_byYear, columns = ['Year', 'Wins'])
print(wins_byYear1)


# In[105]:


expected = 16,16,16,16,16,16   #Assuming the expected frequencies are equal
observed = wins_byYear1.Wins
dt = [[expected, observed]]

##############################################################

chi_baseball = stats.chi2_contingency(dt)
print(chi_baseball)

alpha = 0.05
critical = chi2.ppf(q=1-alpha, df=5) 
p_v6 = 0.010063972216243982
chiS6 = 15.070822499302494

##############################################################

#Printing all the values
print('\nCritical value:', critical)
print('Chi-Square test staistics:', chiS6)
print('p-value:', p_v6)

##############################################################

#making decision
if chiS6 > critical:
    print('\nReject the Null Hypothesis')
else:
    print('\nFail to reject the Null Hypothesis')
    
if p_v > alpha:
    print('\nFail to reject the Null Hypothesis')
else:
    print('\nReject the Null Hypothesis')    

###############################################################

