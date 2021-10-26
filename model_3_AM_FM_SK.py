#import splitfolders
#splitfolders.ratio('../datasets/dataset_AM_FM_SK', output="../datasets/split-dataset-AM-FM-SK", seed=1337, ratio=(.8, .2), group_prefix=None)

TRAIN_PATH = '../datasets/split-dataset-AM-FM-SK/train'
VAL_PATH = '../datasets/split-dataset-AM-FM-SK/val'


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras import losses
from keras.models import Sequential
from datetime import datetime
from tensorflow.keras.optimizers import RMSprop



start = datetime.now()


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
model.add(Dense(3,activation='softmax'))


print(model.summary())

model.compile(loss=losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy']) #RMSprop(lr=0.001)

train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range=0.2,horizontal_flip=True
)

test_dataset = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_PATH,
                        target_size=(224,224),
                        batch_size=32,
                        class_mode='categorical'
                        )
validation_generator = test_dataset.flow_from_directory(VAL_PATH,
                        target_size=(224,224),
                        batch_size=32,
                        class_mode='categorical'
                        )

hist = model.fit_generator(train_generator,
                           steps_per_epoch=8,
                           epochs = 15,
                           verbose=1,
                           validation_data = validation_generator,
                           validation_steps=8
)

model.save('../trained_model/radar-classifier_3.model')

# history = model.fit(train_data, train_labels, epochs=7, batch_size=batch_size,  validation_data=(validation_data, validation_labels))
# model.save_weights(top_model_weights_path)
# (eval_loss, eval_accuracy) = model.evaluate( 
#     validation_data, validation_labels, batch_size=batch_size,     verbose=1)
# print(“[INFO] accuracy: {:.2f}%”.format(eval_accuracy * 100)) 
# print(“[INFO] Loss: {}”.format(eval_loss)) 
# end= datetime.datetime.now()
# elapsed= end-start
# print (‘Time: ‘, elapsed)