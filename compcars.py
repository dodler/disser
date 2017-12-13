
# coding: utf-8

# In[1]:

import helper


# In[2]:

BASE_PATH = '/home/dev/Documents/disser/'
DATA_PATH = BASE_PATH + 'compcars/data/'
IMAGE_DATA = 'image/'
IMAGE_DATA = 'cars/'
OUTPUT_DATA_PATH = BASE_PATH + 'keras_compcars_all_in_one_with_bbox/'


# In[15]:

num_classes = 1738
num_epochs = 2000
batch_size = 16


# In[16]:

train, test = helper.get_generator(OUTPUT_DATA_PATH, batch_size = batch_size)


# In[8]:

from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# In[17]:

xcep = Xception(include_top=False, weights='imagenet')


# In[18]:

x = xcep.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)


# In[19]:

model = Model(inputs=xcep.input, outputs=predictions)


# In[13]:

#model = load_model('cars_all_in_one.hdf5')


# In[20]:

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:

early_stop = EarlyStopping(monitor='val_acc', patience=50)
model_cp = ModelCheckpoint('cars_all_in_one_with_bbox.hdf5', save_best_only=True)
reduce_lr = ReduceLROnPlateau()


# In[ ]:

model.fit_generator(
    callbacks=[early_stop, model_cp, reduce_lr],
    generator=train, 
    epochs=num_epochs, 
    steps_per_epoch=int(train.samples / batch_size),
    validation_data=test,
    validation_steps=int(test.samples / batch_size))

