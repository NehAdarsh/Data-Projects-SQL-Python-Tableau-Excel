#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('WMT').getOrCreate()


# In[4]:


#Loading the csv file and have Spark infer the data types.

wmt_data = spark.read.csv('/Users/abhinavadarsh/Desktop/NEHA/Spring_2022/ALY6110/Final Project/WMT.csv', inferSchema=True, header=True)


# In[5]:


# Column Names

wmt_data.columns


# In[6]:


#Checking the schema
wmt_data.printSchema()


# In[7]:


#Printing a few columns

for line in wmt_data.head(10):
 print(line, '\n')


# In[8]:


#Using describe to learn more about the DF
wmt_data.describe().show()


# In[9]:


from pyspark.sql.functions import format_number


# In[10]:


data_summary = wmt_data.describe()


# In[20]:


data_summary_cols = data_summary.select(data_summary['summary'],
                    format_number(data_summary['Open'].cast('float'), 2).alias('Open'),
                    format_number(data_summary['High'].cast('float'), 2).alias('High'), 
                    format_number(data_summary['Low'].cast('float'), 2).alias('Low'),
                    format_number(data_summary['Close'].cast('float'), 2).alias('Close'), 
                    format_number(data_summary['Volume'].cast('int'),0).alias('Volume')) 

data_summary_cols.show()


# In[24]:


#Creating a new DF to check the ratio of high prices and volume of stock traded for a day

data_hv = wmt_data.withColumn('HV Ratio', wmt_data['High']/wmt_data['Volume']).select(['HV Ratio'])
data_hv.show()


# In[25]:


#Peak prices

wmt_data.orderBy(wmt_data['High'].desc()).select(['Date']).head(1)[0]['Date']


# In[27]:


from pyspark.sql.functions import mean

#Mean of the close column
wmt_data.select(mean('Close')).show()


# In[28]:


from pyspark.sql.functions import min, max

#min and max of the volumn column
wmt_data.select(max('Volume'),min('Volume')).show()


# In[29]:


#Days close was lower than 80

wmt_data.filter(wmt_data['Close'] < 80).count()


# In[30]:


# % of the time when high was greater than 70

wmt_data.filter('High > 70').count() * 100/wmt_data.count()


# In[32]:


#correlation between High and Volume

from pyspark.sql.functions import corr
wmt_data.select(corr(wmt_data['High'], wmt_data['Volume'])).show()


# In[33]:


from pyspark.sql.functions import (dayofmonth, hour, dayofyear, month, year, weekofyear, format_number, date_format)


#Max high per year
year_wmt_data = wmt_data.withColumn('Year', year(wmt_data['Date']))
year_wmt_data.groupBy('Year').max()['Year', 'max(High)'].show()


# In[34]:


# Avg close for each month

month_wmt_data = wmt_data.withColumn('Month', month(wmt_data['Date']))   #Creating a new column Month from existing Date column

month_wmt_data = month_wmt_data.groupBy('Month').mean()    #Group by month and take average of all other columns

month_wmt_data = month_wmt_data.orderBy('Month')    #Sort by month

month_wmt_data['Month', 'avg(Close)'].show()   #Display only month and avg(Close), the desired columns


# In[ ]:




