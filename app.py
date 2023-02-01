import transformers
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences


from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
#from flask_mysqldb import MySQL
from keras.models import load_model
#import MySQLdb.cursors
import re
import cv2
from PIL import Image
import time

import numpy as np
import tensorflow as tf

import cv2
from PIL import Image
import time

import numpy as np
import tensorflow as tf

app = Flask(__name__)

#app.secret_key = 'secret'
 
#app.config['MYSQL_HOST'] = 'localhost'
#app.config['MYSQL_USER'] = 'root'
#app.config['MYSQL_PASSWORD'] = 'password'
#app.config['MYSQL_DB'] = 'classupdb'
 
 
#mysql = MySQL(app)
hand_model = load_model('models/HandGestureModel.h5')
#sentimental_model = load_model('models/sentiment_model.h5', custom_objects={"TFBertModel": transformers.TFBertModel})

def preprocess_input_data(sentence):
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)
    attention_mask = [1] * len(input_ids)
    input_ids = pad_sequences([input_ids], maxlen=512, dtype="long", padding="post", truncating="post")
    attention_mask = pad_sequences([attention_mask], maxlen=512, dtype="long", padding="post", truncating="post")
    return input_ids, attention_mask


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

@app.route("/blog")
def blog():
    return render_template(
        "blog.html"
    )

@app.route("/blog-details")
def blog_details():
    return render_template(
        "blog-details.html"
    )

@app.route("/portfolio-details")
def portfolio_details():
    return render_template(
        "portfolio-details.html"
    )

@app.route("/sample-inner-page")
def sample():
    return render_template(
        "sample-inner-page.html"
    )
       

@app.route("/reflection")
def reflection():
    return render_template('sentiment_reflection.html')

# def gen_frames():  
#     video = cv2.VideoCapture(0)
#     while True:
#         success, frame = video.read()
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  


# @app.route("/hand_video_feed")
# def hand_video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


COMMAND = None

def controlSlides():
    video = cv2.VideoCapture(0)
    
    while True:
        success, frame = video.read()
        if not success:
            break

        else:
            img = Image.fromarray(frame, 'RGB')
            ret, buffer = cv2.imencode('.jpg', frame)
            
            frame_tobytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_tobytes + b'\r\n')

            img = img.resize((128,128))
            img_array = np.array(img)
            
            img_array = img_array.reshape(1,128,128,3)

            prediction = hand_model.predict(img_array)
            print(prediction)
            
            if(prediction[0][0] == 1 and prediction[0][1] == 0):
                direction = "left"
                setcmdback()
                time.sleep(2)

            elif(prediction[0][0] < 0.5):
                direction = "right"
                setcmdnext()
                time.sleep(2)

            else:
                direction = "none"
                time.sleep(1)

            
            print(direction)
            

        cv2.imshow("Prediction", frame)
        cv2.waitKey(1)

@app.route("/controlSlides_feed")
def controlSlides_feed():
    global COMMAND
    return Response(controlSlides(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/setcmdnext/")
def setcmdnext():
    global COMMAND
    COMMAND = 'next'
    return jsonify({'command': COMMAND})
    
@app.route("/setcmdback/")
def setcmdback():
    global COMMAND
    COMMAND = 'back'
    return jsonify({'command': COMMAND})

@app.route("/prediction", methods=["POST"])
def prediction():
    new_reflection = [str(x) for x in request.form.values()]
    input_prediction = new_reflection
    print("new reflection:", new_reflection)
    print("input_prediction:", input_prediction)
    print('inputs', sentimental_model.inputs)
    in_sensor= preprocess_input_data(str(input_prediction))

    senti_prediction = sentimental_model.predict(in_sensor)[0]

    class_index = np.argmax(senti_prediction)
    print('class index', class_index)

    if class_index == 1:
        result = "Positive Sentiment"
    else:
       result = "Negative Sentiment"

    print('sentiment:', result)

    return render_template('sentiment_reflection.html', prediction_text=result)
