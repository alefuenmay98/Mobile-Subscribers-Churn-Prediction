# Para el manejo de los datos
import sqlite3 as sq
import pandas as pd
import numpy as np


# In[3]:


# Para las visualizaciones 
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


from imblearn.over_sampling import SMOTE


# In[5]:


# Para el modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Para evaluar el modelo
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


# In[85]:


df=pd.read_csv('customer_churn.csv')
df


# In[86]:


df.info()


# In[87]:


df = df.drop(columns=['Call Failure', 'Subscription Length', 'Charge Amount', 'Tariff Plan'])
df


# In[88]:


df.describe()


# In[101]:


zero_data = df[df['Seconds of Use'] == 0]
zero_data


# In[107]:


sns.set()
fig,ax = plt.subplots(1, 2, figsize=(12,8))
sns.histplot(x='Frequency of SMS', data=zero_data, ax=ax[0], bins=10)
sns.histplot(x='Frequency of SMS', data=df, ax=ax[1], bins=10)
ax[0].set(yscale='log')
ax[1].set(yscale='log')
plt.show()


# In[89]:


plt.figure(figsize=(12,8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, mask=mask, cmap='viridis')
plt.show()


# In[90]:


df = df.drop(columns=['Frequency of use'])
df


# In[91]:


churned = df[df['Churn']==1]
not_churned = df[df['Churn']==0]

print('percentage of churned customer: {}'.format(churned.shape[0]/df.shape[0]))
print('percentage of not-churned customer: {}'.format(not_churned.shape[0]/df.shape[0]))


# In[92]:


from collections import Counter

# splitting the data as X and y
X = df.drop('Churn', axis=1)
y = df['Churn']

# making a SMOTE object
resampler = SMOTE(random_state=5)

# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# resampling the data
X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

# verifying the resampled data
print('Resampled dataset shape %s' % Counter(y_resampled))


# In[93]:


df.columns


# In[94]:


# making a ColumnTransformer object
ct = ColumnTransformer(
    [('scaler', StandardScaler(), ['Seconds of Use', 'Frequency of SMS', 'Customer Value'])], remainder='passthrough')

# transforming data
X_scaled = ct.fit_transform(X_resampled)
X_test_scaled = ct.transform(X_test)


# In[95]:


# making an object for LogisticRegression
linear_reg = LogisticRegression(max_iter=1000000)

# fitting the data
linear_reg.fit(X_scaled, y_resampled)

# predicting on x_test
y_pred = linear_reg.predict(X_test_scaled)


# In[96]:


from sklearn.metrics import classification_report, ConfusionMatrixDisplay
print(classification_report(y_test, y_pred))


# In[97]:


# this line removes the grid from the confusion matrix
sns.set_style("whitegrid", {'axes.grid' : False})

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()


# In[98]:


# the data directly scaled without resampling
X_train_scaled = ct.fit_transform(X_train)
X_test_scaled = ct.transform(X_test)

# fitting and predicting the model
linear_reg.fit(X_train_scaled, y_train)
y_pred2 = linear_reg.predict(X_test_scaled)

print(classification_report(y_test, y_pred2))


# In[100]:


# this line removes the grid from the confusion matrix
sns.set_style("whitegrid", {'axes.grid' : False})

ConfusionMatrixDisplay.from_predictions(y_test, y_pred2)
plt.show()

