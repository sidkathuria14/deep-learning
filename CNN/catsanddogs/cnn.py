
# coding: utf-8

# In[3]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[4]:


classifier = Sequential()
classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())


# In[5]:


classifier.add(Dense(output_dim = 128,activation = 'relu'))
classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])


# In[6]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)


# In[11]:


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                               target_size = (64,64),
                                               batch_size = 32,
                                               class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                           target_size = (64,64),
                                           batch_size = 32,
                                           class_mode = 'binary')


# In[ ]:


classifier.fit_generator(training_set,steps_per_epoch = 8000,
                        nb_epoch = 25,
                        validation_data = test_set,
                        nb_val_samples = 2000)

