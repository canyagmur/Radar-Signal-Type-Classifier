import cv2
from keras.backend import reshape
import os
import tensorflow as tf
import matplotlib.pyplot as plt

CATEGORIES = ['AM','FM']
PATH = "../datasets/test-set/FM-mixed/"
MODEL_PATH = "../trained_model/radar-classifier_3.model"
def prepare(filepath):
    IMG_SIZE = 224
    img_array = cv2.imread(filepath)
    #print(img_array.shape)
    img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    reshaped = img_array.reshape(1,IMG_SIZE,IMG_SIZE,3)
    #print(reshaped.shape)
    return reshaped

def data_visualization(img,img_name,label):
    plt.imshow(img)
    plt.title('Predictions')
    plt.xlabel('Predicted Label : '+ label)
    plt.ylabel('Image Name : '+ img_name )
    #plt.legend()
    plt.show()

model = tf.keras.models.load_model(MODEL_PATH)

model.summary()

testset_files_list = os.listdir(PATH)

for file_name in testset_files_list:
    abs_path = PATH+file_name
    img = prepare(abs_path)
    prediction = model.predict([img])
    prediction_label = CATEGORIES[int(prediction[0][0])]
    img = img[0,:,:,:]
    data_visualization(img,file_name,prediction_label)



#print("Prediction : "+ CATEGORIES[int(prediction[0][0])])


