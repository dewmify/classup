from PIL import Image
import os
import numpy as np
import tensorflow as tf
import cv2
import time 
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

video = cv2.VideoCapture(0)

model = load_model("HandGestureModel.h5")

while True:
    success, frame = video.read()

    img = Image.fromarray(frame, 'RGB')

    img = img.resize((128,128))
    img_array = np.array(img)
        
    img_array = img_array.reshape(1,128,128,3)

    prediction = model.predict(img_array)
    print(prediction)
        
    if(prediction[0][0] == 1 and prediction[0][1] == 0):
        direction = "left"
        print(direction)
        time.sleep(3)

    elif(prediction[0][0] < 0.5):
        direction = "right"
        print(direction)
        time.sleep(3)

    else:
        direction = "none"
        print(direction)
        time.sleep(3)
        
    cv2.imshow("Prediction", frame)
    cv2.waitKey(1)
