#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#import pandas_profiling
import ydata_profiling
import os


# In[2]:


import numpy as np


# In[3]:


from datetime import datetime, date


# In[4]:


import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import seaborn as sns


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


from sklearn.preprocessing import OneHotEncoder


# In[8]:


from sklearn import metrics


# In[9]:


from sklearn.metrics import ndcg_score, make_scorer


# In[10]:


from sklearn.model_selection import GridSearchCV


# In[11]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate


# In[13]:


train_df1 = pd.read_csv('Airbnb NYC 2019.csv')


# In[14]:


train_df = train_df1.copy()
train_df.head()


# In[15]:


sessions_df  = pd.read_csv('Airbnb NYC 2019.csv')
sessions_df.head()


# In[16]:


train_df.describe()


# In[17]:


train_df.shape


# In[17]:


train_df.groupby('calculated_host_listings_count')['calculated_host_listings_count'].agg('count')


# In[20]:


df2=train_df.drop(['last_review','reviews_per_month'],axis = 'columns')


# In[21]:


df2.head()


# In[20]:


df2.isnull().sum()


# In[21]:


df2['price'] = df2.price.apply(lambda y: 1 if y == "rates" else 0)


# In[22]:


df2.price.value_counts()


# In[23]:


df2.head()


# In[24]:


df2["room_type"].unique()


# In[25]:


df2.availability_365.unique()


# In[26]:


def is.float(x):
    try
    float(x)
    except:
        return false
        return true


# In[12]:


len(df2.neighbourhood_group.unique())


# In[ ]:


df2.shape


# In[ ]:


len(df2)


# In[ ]:


df2['price'].astype(str).str.replace


# In[ ]:


df2.groupby(by=["price"]).sum()


# In[ ]:


df2.groupby("name", group_keys=False).apply(lambda x: x)


# In[ ]:


print ("Number of rows / records:",df2.count())


# In[ ]:


host_areas = df2.groupby(['host_name','neighbourhood_group'])['calculated_host_listings_count'].max().reset_index()
host_areas.sort_values(by='calculated_host_listings_count', ascending= False).head(5)


# In[ ]:


room_price_area_wise = df2.groupby(['neighbourhood_group','room_type'])['price'].max().reset_index()
room_price_area_wise.sort_values(by='price', ascending= False).head(10)


# In[ ]:


# Visualizing the data , in this Entire home room type is greater in in Manhattan and also there prices are high .
neighbourhood_group = ["Brooklyn","Manhattan" ,"Queens","Bronx","Manhattan","Staten Island","Queens","Brooklyn" ,"Brooklyn","Manhattan","Manhattan","Brooklyn","Brooklyn","Manhattan"]
room_type = ["Entire home/apt","Entire home/apt","Private room","Private room","Entire home/apt","Private room","Entire home/apt","Private room","Shared room","Entire home/apt"]
room_dict ={}
for i in room_type:
   room_dict [i] = room_dict.get(i,0)+1
plt.bar(room_dict.keys(), room_dict.values(),color = 'green', edgecolor = 'blue')
plt.title('Roomtypes')
plt.xlabel('Roomtype')
plt.ylabel('count')
plt.show()


# In[28]:


area_reviews = df2.groupby(['neighbourhood_group'])['number_of_reviews'].max().reset_index()
area_reviews


# In[29]:


area = area_reviews['neighbourhood_group']
reviews = area_reviews ['number_of_reviews']
fig = plt.figure (figsize = (15,10))

plt.bar(area,reviews,color = 'blue', width = 0.6)
plt.title('No of area in  teams of review')
plt.xlabel('reviews')
plt.ylabel('area')
plt.show()


# In[31]:


price_area = df2.groupby(['price'])['number_of_reviews'].max().reset_index()
price_area.head(10)


# In[32]:


price_list = price_area['price']
review = price_area['number_of_reviews']
fig = plt.figure(figsize = (10,5))
plt.scatter(price_list,review)
plt.title('No of reviews VS No of price')
plt.xlabel('price')
plt.ylabel('number_of_reviews')
plt.show()


# In[33]:


busy_hosts = df2.groupby(['host_id','host_name', 'room_type'])['number_of_reviews'].max().reset_index()
busy_hosts = busy_hosts.sort_values(by = 'number_of_reviews', ascending = False).head(10)
busy_hosts


# In[35]:


name_host = busy_hosts['host_name']
review_got = busy_hosts['number_of_reviews']
fig = plt.figure(figsize = (10,5))
plt.bar(name_host,review_got,color = 'purple', width = 0.6)
plt.title('busy host in  teams of review')
plt.xlabel('name of host')
plt.ylabel('review')
plt.show()


# In[22]:


Highest_price = df2.groupby(['host_id','host_name', 'neighbourhood_group'])['price'].max().reset_index()
Highest_price = Highest_price.sort_values(by = 'price', ascending = False).head(10)
Highest_price


# In[23]:


name_of_host = Highest_price['host_name']
price_charge = Highest_price['price']
fig = plt.figure(figsize = (10,5))
plt.bar(name_of_host,price_charge,color = 'orange', width = 0.6)
plt.title('Name of host')
plt.xlabel('price')
plt.ylabel('Host with maximum price charges')
plt.show()


# In[24]:


df2.head(10)


# In[26]:


traffic_areas = df2.groupby(['room_type', 'neighbourhood_group'])['minimum_nights'].max().reset_index()
traffic_areas = traffic_areas.sort_values(by = 'minimum_nights', ascending = False).head(10)
traffic_areas


# In[28]:


area_traffic = traffic_areas['room_type']
room_stayed = traffic_areas['minimum_nights']
fig = plt.figure(figsize = (10,5))
plt.bar(area_traffic,room_stayed,color = 'black', width = 0.6)
plt.title('Room_type')
plt.xlabel('minimum_nights')
plt.ylabel('Traffic area besed on minimun nights')
plt.show()


# In[29]:


corr = df2.corr(method = 'kendall')
fig = plt.figure(figsize = (12,6))
sns.heatmap(corr,annot = True)
df2.columns


# In[31]:


plt.rcParams['figure.figsize'] =(8,5)
ax = sns.countplot(y = 'room_type',hue = 'neighbourhood_group',data = df2,palette ='bright')

total = len(df2['room_type'])
for p in ax.patches:
    percentage = '[:.1f]%',format(100 * p.get_width()/total)
    x = p.get_x()+p.get_width()+0.02
    y = p.get_y()+p.get_height()/2
    ax.annotate(percentage, (x,y))
    
    
plt.title('Count of each room types in nyc')
plt.xlabel('Rooms')
plt.xticks(rotation = 90)
plt.ylabel('Room Counts')
plt.show()

