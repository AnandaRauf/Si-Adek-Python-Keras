import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential,save_model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("-------------------------------------------------------------------\n")
namaprogram= "Sistem Informasi Deteksi Benda Berbahaya PC"
versi = "Version 1.0"
devdate = "Dibuat pada tanggal 24 November 2021"
print("-------------------------------------------------------------------\n")
print(namaprogram)
print(versi)
print(devdate)

model = tf.keras.models.load_model('datamodel_keras/keras_model.h5',compile = True)
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 1))) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
while True:
    _, frame = webcam.read()
    
    im = Image.fromarray(frame, 'RGB')
    im = im.resize((128, 128))
    img_array = np.array(im)

    img_array = np.expand_dims(img_array, axis=0)
    frameCopy=frame.copy()
    frameCopy = cv2.resize(frameCopy, (120, 320))
    gray = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
    img_array = np.array(gray)
    img_array = img_array.reshape(120, 320, 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = int(model.predict(img_array)[0][0])
    print(prediction)
    if prediction == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Tidak terdeteksi benda")

    cv2.imshow("Sistem Informasi Deteksi Benda Berbahaya", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
