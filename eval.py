import cv2
from keras.backend import reshape
import tensorflow as tf

CATEGORIES = ['AM','FM']

def prepare(filepath):
    IMG_SIZE = 224
    img_array = cv2.imread(filepath)
    #print(img_array.shape)
    img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    reshaped = img_array.reshape(1,IMG_SIZE,IMG_SIZE,3)
    #print(reshaped.shape)
    return reshaped

model = tf.keras.models.load_model("radar-classifier.model")

model.summary()

prediction = model.predict([prepare('AM1.png')])


print("Prediction : "+ CATEGORIES[int(prediction[0][0])])