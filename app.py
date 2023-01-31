from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from flask_mysqldb import MySQL
from keras.models import load_model
import MySQLdb.cursors
import re

import cv2
from PIL import Image
import time

import numpy as np
import tensorflow as tf

app = Flask(__name__)

app.secret_key = 'secret'
 
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'classupdb'
 
 
mysql = MySQL(app)


@app.route("/")
def home():
    return render_template(
        "index.html"
    )

@app.route("/slides")
def slides():
    return render_template(
        "slides.html"
    )


hand_model = load_model('HandGestureModel.h5')

def gen_frames():  
    video = cv2.VideoCapture(0)
    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  


@app.route("/hand_video_feed")
def hand_video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/controlSlides", methods=['POST'])
def controlSlides():
    video = cv2.VideoCapture(0)

    while True:
        success, frame = video.read()

        img = Image.fromarray(frame, 'RGB')

        img = img.resize((128,128))
        img_array = np.array(img)
        
        img_array = img_array.reshape(1,128,128,3)

        prediction = hand_model.predict(img_array)
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

        #return jsonify({'result': direction})
        

        cv2.imshow("Prediction", frame)
        cv2.waitKey(1)
