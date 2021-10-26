#import splitfolders
#splitfolders.ratio('./dataset', output="./split-dataset", seed=1337, ratio=(.8, .2), group_prefix=None)


TRAIN_PATH = '../datasets/split-dataset/train'
VAL_PATH = '../datasets/split-dataset/val'


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras import losses
from keras.models import Sequential




model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),activation=('relu')))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation=('relu')))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(128,(3,3),activation=('relu')))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])


train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range=0.2,horizontal_flip=True
)

test_dataset = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_PATH,
                        target_size=(224,224),
                        batch_size=32,
                        class_mode='binary'
                        )
validation_generator = test_dataset.flow_from_directory(VAL_PATH,
                        target_size=(224,224),
                        batch_size=32,
                        class_mode='binary'
                        )

hist = model.fit_generator(train_generator,
                           steps_per_epoch=8,
                           epochs = 10,
                           validation_data = validation_generator,
                           validation_steps=2
)

print(model.evaluate_generator(train_generator))

print(model.evaluate_generator(validation_generator))

model.save('radar-classifier.model')