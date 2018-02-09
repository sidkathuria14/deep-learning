
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[5]:


dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()


# In[14]:


x = dataset.iloc[:,3:13].values


# In[15]:


print x.shape


# In[19]:


y = dataset.iloc[:,13].values


# In[24]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_x_1 = LabelEncoder()
x[:,1] = labelEncoder_x_1.fit_transform(x[:,1])
labelEncoder_x_2 = LabelEncoder()
x[:,2] = labelEncoder_x_2.fit_transform(x[:,2])
oneHotEncoder = OneHotEncoder(categorical_features = [1])
x = oneHotEncoder.fit_transform(x).toarray()
x = x[:,1:]


# In[27]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# In[28]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[30]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[34]:


classifier = Sequential()
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim = 11))
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))


# In[36]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

