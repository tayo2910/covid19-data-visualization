#!/usr/bin/env python
# coding: utf-8

# Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.
# 
# People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

# # Attribute Information
# #### Age: age of the patient [years]
# #### Sex: sex of the patient [M: Male, F: Female]
# #### ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
# #### RestingBP: resting blood pressure [mm Hg]
# #### Cholesterol: serum cholesterol [mm/dl]
# #### FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
# #### RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
# #### MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
# #### ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
# #### Oldpeak: oldpeak = ST [Numeric value measured in depression]
# #### ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
# #### HeartDisease: output class [1: heart disease, 0: Normal]

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('heart.csv')
data


# In[3]:


data.info()


# In[4]:


data.duplicated().sum()


# In[5]:


data.isna().sum()


# ## Transforming the Categorical columns

# In[6]:


data['ChestPainType'].unique()


# In[7]:


data['Sex'].unique()


# In[8]:


data['RestingECG'].unique()


# In[9]:


data['ExerciseAngina'].unique()


# In[10]:


data['ST_Slope'].unique()


# #### The Sex Column is a Nominal Data type consisting of Male and Female
# 
# #### ChestPainType Follows a particular Order: TA- Typical Angina, ATA- Atypical Angina, NAP-Non Angina Pain, ASY- Asymptomatic
# 
# #### The Sex Column is a Nominal Data type consisting of Male and Female
# 
# #### ChestPainType Follows a particular Order: TA- Typical Angina, ATA- Atypical Angina, NAP-Non Angina Pain, ASY- Asymptomatic
# 
# #### Resting ECG also follows the same order as is either Normal, ST- having ST-T wave abnormality( this is T wave inversions and/or ST elevation or depression of >0.05 mV), LVH-showing probable or definite left ventricular hypertrophy by Estes' criteria)
# 
# #### ExerciseAnigina: Exercise induced angia N-No, Y-Yes
# 
# #### STSlope: The slope of peak exercise ST segment; Up-uplsoping, Flat-flat, Down-downsloping
# 
# #### Nominal Colunmns for OneHotEncoder: SEX, ExerciseAnigina
# 
# #### Ordinal Columns for LabelEncoder: ChestPainType, RestingECG, STSlope

# In[11]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[12]:


encoder = LabelEncoder()


# In[13]:


data[['ChestPainType', 'RestingECG', 'ST_Slope']] = data[['ChestPainType', 'RestingECG', 'ST_Slope']].apply(encoder.fit_transform)
data


# In[14]:


data = pd.get_dummies(data, columns = ['Sex', 'ExerciseAngina'])
data


# In[15]:


data.info()


# In[16]:


data.describe()


# In[17]:


data.isna().sum()


# ### Data appears clean 

# ##  EDA QUESTIONS AND ANSWERS

# ### 1. Since the HeartDisease is the output column, can we know the count of those who had Heart disease based on their Sex?

# In[18]:


data.groupby(['Sex_F', 'Sex_M'])['HeartDisease'].count()


# ### This shows that 725 men were diagnosed with heart disease while only 193 women had the same disease in a sample of 918 people

# ### 2. Can we find any underaged with Heart disease in this sample data?

# In[19]:


data.groupby(data['Age']<18).count()['HeartDisease']


# #### This shows that no individual below the age of 18 who had the disease.

# In[20]:


import seaborn as sns


# In[21]:


# finding the relationship between the categories
corr = data.corr()                              #the lighter the color, the higher the correllation.
corr.style.background_gradient(cmap = 'OrRd_r')


# #### There appears to be no strong relationship between the variables.

# ### 3. Creating a dataframe and filter only those diagnosed with heartdisease

# In[116]:


df= (data["HeartDisease"] == 1)
data[df]


# ### From our analysis, there is no positive correlation between all attributes and developing a heart disease
# 
# ### 2. What is the distribution of the disease among Male and Females?
# 
# 

# In[135]:


data[df].groupby(["Sex_M", "Sex_F"])["HeartDisease"].count()


# ### Among those who were diagnosed with Heart Disease; 50 of them were Females while 458 are males

# In[48]:


sns.boxplot(data =data, x="HeartDisease", y="Age")


# ### We know from this that although no underaged was diagnosed of the disease. there are a few under 40s (between 30 and 35 years) who were diagnosed. They were the outliers. The median age of those who were diagnosed is a little above 60 years.

# In[109]:


sns.boxplot(x = 'ST_Slope', y ='Cholesterol', data = data)


# #### The cholesterol levels are represented by the points on the y-axis. For the first class of the ST_Slope, the maximum cholesterol level is around 480  and there are no outliers. But for the second and third class, there are outliers even though these last two classes seem to have the same median.

# ### There are quite a number of outliers in the RestingBP column and the MaxHR column.

# In[140]:


# Using the female and male sexes to color the graph representing the distribution of the RestingBp and Age.
sns.catplot(x = 'RestingBP', y = 'Age', hue_order = ['Sex_F', 'Sex_M'], data = data)


# In[25]:


# Using the female and male sexes to color the graph representing the distribution of the Cholesterol versus Age.
sns.stripplot(x = 'Cholesterol', y = 'Age', hue_order = ['Sex_F', 'Sex_M'], data = data)


# In[81]:


# Using the 'positive ExerciseAngina' column to color the graph representing the relationship between Old peak and Age.
sns.scatterplot(x = 'Oldpeak', y = 'Age', hue = 'ExerciseAngina_Y', data = data)


# In[90]:


# Using the 'negative ExerciseAngina' column to color the graph representing the relationship between Old peak and Age.
sns.scatterplot(x = 'RestingBP', y = 'Cholesterol', hue = 'ExerciseAngina_N', data = data)


# #### This looks clustered

# In[91]:


sns.lineplot(x = 'MaxHR', y = 'Age', hue = 'ExerciseAngina_N', data = data)


# In[29]:


sns.scatterplot(x = 'MaxHR', y = 'Age', hue = 'ExerciseAngina_Y', data = data)


# In[30]:


# The error bars are showing the uncertainty levels in these columns using the ST_Slope to color them.
sns.pointplot(x = 'FastingBS', y = 'RestingECG', hue = 'ST_Slope' , col = 'HeartDisease', data = data)


# In[95]:


ax = sns.barplot(x="FastingBS", y="ST_Slope", hue= 'RestingECG', data=data)
ax.set(ylabel="", xlabel="Blood Sugar")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# # Model Prediction

# In[141]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


# In[142]:


X= data.drop(['HeartDisease'], axis = 1).values
y= data['HeartDisease'].values              


# # Using Logistic Regression

# In[143]:


# split to 70% train dataset and 30% test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)


# In[144]:


clf = LogisticRegression()


# In[145]:


clf.fit(X_train, y_train)
Ypred = clf.predict(X_test)


# In[146]:


Ypred


# In[147]:


predictions = pd.DataFrame(Ypred).rename(columns= {0: 'predictions'})
predictions


# In[149]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# # Using Cross Validation

# In[150]:


model = KNeighborsClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=10, n_jobs= -1)
n_scores


# In[151]:


print("Cross Validation Scores: ", n_scores)
print("Average CV Score: ", n_scores.mean())
print("Standard Deviation Score: ", n_scores.std())
print("Number of CV Scores used in Average: ", len(n_scores))


# In[152]:


from sklearn import metrics


# In[153]:


confusion_matrix = metrics.confusion_matrix(y_test, predictions)


# In[154]:


ConfusionMatrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [True, False]).plot()
plt.show()


# #### From this matrix, out of 184 data points, our model predicted 77 True Positives and 81 True Negatives. 26 datapoints were predicted wrongly with a breakdown as follows: 14 False Positives and 12 False Negatives.

# In[ ]:




