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
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


print(model.summary())

model.compile(loss=losses.binary_crossentropy,optimizer='adam',metrics=['accuracy']) #RMSprop(lr=0.001)

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
                           epochs = 15,
                           verbose=1,
                           validation_data = validation_generator,
                           validation_steps=8
)

model.save('../trained_model/radar-classifier_2.model')


#print(model.evaluate_generator(train_generator))

#print(model.evaluate_generator(validation_generator))

#model.save('radar-classifier.model')