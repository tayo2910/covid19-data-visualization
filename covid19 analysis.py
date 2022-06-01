#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Desktop/covid 19.csv')
df.head()


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.index


# In[6]:


df['Population'].mean()


# In[7]:


df.info()


# In[8]:


df['Country']


# In[9]:


df.isna().sum()


# In[10]:


sns.boxplot(data = df[['Population', 'Total Cases', 'Total Deaths', 'Death percentage']]); 
#There are some outliers in the population column of this data set


# In[11]:


sns.catplot(x='Total Deaths',y ='Death percentage', data=df);


# In[12]:


sns.catplot(x = 'Total Cases', y = 'Death percentage', data= df);


# In[13]:


x = df[['Total Cases', 'Total Deaths', 'Population']]
feature_corr = sns.heatmap(x.corr(), cmap = 'brg', annot = True)


# In[17]:


df.columns.tolist()


# In[18]:


sns.histplot(data=df['Tot\xa0Cases//1M pop'], kde= True, bins =20 );       ### this distribution isheavily skewed to the right


# In[19]:


g = sns.pairplot(data=df, hue=None, hue_order=None, palette=None, kind='scatter', markers=None)
plt.show()


# In[21]:


sns.regplot(x = "Tot\xa0Cases//1M pop", 
           y = "Tot\xa0Deaths/1M pop",     ### Plotting a Regression line
                   ci = None, 
                   data = df);


# In[20]:


g = sns.pairplot(data=df, hue=None, hue_order=None, palette=None, kind='kde', markers=None)
plt.show()


# In[ ]:


### 


# In[ ]:





# In[ ]:




