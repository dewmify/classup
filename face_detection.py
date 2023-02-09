import cv2
import os
import collections
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Lambda
from keras import backend as K
from keras_preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random


face_cascades = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_faces(img, draw_box=True):
    #converts image into grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #detect faces
    faces = face_cascades.detectMultiScale(grayscale_img, scaleFactor=1.1,
		minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
        
    face_box, face_coords = None, []

    #draw bounding box around detected faces
    for (x, y, w, h) in faces:
        if draw_box:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
        
        face_box = img[y:y+h, x:x+w]
        face_coords = [x,y,w,h]
    return img, face_box, face_coords
	
#Euclidean Distance
def euclidean_distance(vectors):
    vector1, vector2 = vectors
    sum_square = K.sum(K.square(vector1 - vector2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

#Loss Function
def contrastive_loss(Y_true, D):
    margin = 1
    return K.mean(Y_true * K.square(D) + (1 - Y_true) * K.maximum((margin-D),0))

#Accuracy Metric
def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


#Function to build the model
def create_shared_network(input_shape):
    model = Sequential(name='Shared_Conv_Network')
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='sigmoid'))
    return model

def write_on_frame(frame, text, text_x, text_y):
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
    box_coords = ((text_x, text_y), (text_x+text_width+20, text_y-text_height-20))
    cv2.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
    return frame


