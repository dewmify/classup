import transformers
import pandas as pd
import numpy as np
import json
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences


from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, flash
from flask_mysqldb import MySQL
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from keras.models import load_model
#import MySQLdb.cursors
import re
import cv2
from PIL import Image
import time
import MySQLdb.cursors

from wtforms import StringField, PasswordField, BooleanField, RadioField
from wtforms.validators import InputRequired, Email, Length, Optional

import numpy as np
import tensorflow as tf

app = Flask(__name__)



app.secret_key = 'secret'
 
app.config['MYSQL_HOST'] = 'classupdb.cgdsnk6an6d3.us-east-1.rds.amazonaws.com'
app.config['MYSQL_USER'] = 'admin'
app.config['MYSQL_PASSWORD'] = 'bYaP6tsnsRFy1TIJVQAr'
app.config['MYSQL_DB'] = 'classupdb'
 
mysql = MySQL(app)
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
    if 'loggedin' in session:
        return render_template("index.html")
    return redirect(url_for('login'))
    
@app.route("/slides")
def slides():
    return render_template(
        "teacher/slides.html"
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

@app.route("/student-class")
def student_classes():
    return render_template(
        "student/student_class.html"
    )

@app.route("/student-index")
def student_index():
    return render_template(
        "student/student_index.html"
    )
       
@app.route("/reflection")
def reflection():
    return render_template('student/sentiment_reflection.html')


@app.route("/slides_list")
def slides_list():
    return render_template('teacher/slides_list.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                if user.role == "admin":
                    session["email"] = user.email
                    session["username"] = user.username
                    return redirect(url_for('viewAllUsers'))
                else:
                    session["email"] = user.email
                    session["username"] = user.username
                    return redirect(url_for('consumerUpdateUser'))
            else:
                flash("Incorrect Username or password")

                return redirect('/login')
        else:
            flash("Incorrect Username or Password")
            return redirect('/login')

    return render_template('login.html', form=form, user=current_user)

@app.route("/account")
def account():
    return render_template('account.html')



def controlSlides():
    video = cv2.VideoCapture(0)
    
    # while True:
    success, frame = video.read()
    cv2.imshow('frame', frame)

    if not success:
        return
    else:
        img = Image.fromarray(frame,'RGB')
        ret, buffer = cv2.imencode('.jpg', frame)

        img = img.resize((128,128))
        img_array = np.array(img)

        img_array = img_array.reshape(1,128,128,3)

        prediction = hand_model.predict(img_array)

        if(prediction[0][0] > 0.998):
            direction = "left"
        elif(prediction[0][0] < 0.5):
            direction = "right"
        else:
            direction = "none"

        print(direction)

    video.release()
    return direction



@app.route("/controlSlides_feed")
def controlSlides_feed():
    return Response(controlSlides(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/v1/handgesture', methods=['GET'])
def get_handgesture():
    direction = controlSlides()
    return json.dumps({'direction': direction})



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

    return render_template('student/sentiment_reflection.html', prediction_text=result)
