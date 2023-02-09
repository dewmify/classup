import face_detection
import cv2
import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model
import collections
import os

detection_model = face_detection_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Dictionary to store true images and names
true_images = {
    'Ryo': cv2.imread('static/OnboardedImg/true_image_1.png', 0),
    'Max': cv2.imread('static/OnboardedImg/true_image_2.png', 0),
    
}
 #Preprocess true images

for name, true_img in true_images.items():
    true_img = true_img.astype('float32')/255
    true_img = cv2.resize(true_img, (92, 112))
    true_img = true_img.reshape(1, true_img.shape[0], true_img.shape[1], 1)
    print(true_img.shape)

#Load the model
input_shape = (112,92,1)
shared_network = face_detection.create_shared_network(input_shape)

#Specifying input for top and bottom layers
input_top = Input(shape = input_shape)
input_bottom = Input(shape = input_shape)

#Stacking the 2 layers.
output_top = shared_network(input_top)
output_bottom = shared_network(input_bottom)
distance = Lambda(face_detection.euclidean_distance, output_shape = (1,))([output_top, output_bottom])
model = Model(inputs = [input_top, input_bottom], outputs = distance)

model.load_weights('Siamese_nn.h5')

def predict_face(img):
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detection_model.detectMultiScale(grayscale_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) == 0:
        return None, float("inf"), img
    
    scores = []
    for (x, y, w, h) in faces:
        face = grayscale_img[y:y+h, x:x+w]
        face = cv2.resize(face, (112, 92))
        face = face.astype('float32') / 255
        face = face.reshape(1, 112, 92, 1)
        
        score_per_face = {}
        for name, true_img in true_images.items():
            true_img = true_img.astype('float32')/255
            true_img = cv2.resize(true_img, (92, 112))
            true_img = true_img.reshape(1, true_img.shape[0], true_img.shape[1], 1)
            vec = model.predict([face, true_img])[0][0]
            score_per_face[name] = vec
        
        scores.append(score_per_face)
    
    # Add the frames and names
    for i, (x, y, w, h) in enumerate(faces):
        best_score = min(scores[i].values())
        if best_score > 0.8:
            name = "Unidentified"
        else:
            best_name = [k for k, v in scores[i].items() if v == best_score][0]
            name = best_name
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return img


# img = cv2.imread('static/class_img.jpg', 1)
# predict_img = predict_face(img)
# cv2.imwrite("static/result.jpg", predict_img)
# cv2.imshow('Result', predict_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
