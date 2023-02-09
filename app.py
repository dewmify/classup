import transformers
import pandas as pd
import numpy as np
import subprocess
import os
import face_recognition
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences

from werkzeug.utils import secure_filename
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, flash
from flask_mysqldb import MySQL
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from keras.models import load_model, Model
from keras.layers import Input, Lambda

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

UPLOAD_FOLDER = 'static/AttendanceUploads/'

app.secret_key = 'secret'

# app.config['MYSQL_HOST'] = 'classupdb.cgdsnk6an6d3.us-east-1.rds.amazonaws.com'
# app.config['MYSQL_USER'] = 'admin'
# app.config['MYSQL_PASSWORD'] = 'bYaP6tsnsRFy1TIJVQAr'
# app.config['MYSQL_DB'] = 'classupdb'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 512



mysql = MySQL(app)
hand_model = load_model('models/HandGestureModel.h5')
#sentimental_model = load_model('models/sentiment_model.h5', custom_objects={"TFBertModel": transformers.TFBertModel})

allowed_extensions = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
     

def preprocess_input_data(sentence):
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)
    attention_mask = [1] * len(input_ids)
    input_ids = pad_sequences([input_ids], maxlen=512, dtype="long", padding="post", truncating="post")
    attention_mask = pad_sequences([attention_mask], maxlen=512, dtype="long", padding="post", truncating="post")
    return input_ids, attention_mask


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/test', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            filename = "class_img.jpg"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            return render_template('test.html', filename=filename)
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(request.url)
    else:
        return render_template('test.html')
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='AttendanceUploads/' + filename), code=301)


@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    filename = "class_img.jpg"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Call the predict_faces function from face_detection.py
    processed_image = face_recognition.predict_face(image_path)

    # Save the processed image to disk
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "processed_" + filename)
    cv2.imwrite(processed_image_path, processed_image)

    return redirect(url_for('display_image', filename="processed_" + filename))



    
@app.route("/slides")
def slides():
    return render_template(
        "slides.html"
    )

@app.route("/onboarding", methods=["GET", "POST"])
def onboarding():
    if request.method == "GET":
        # Return the onboarding page
        return render_template("onboarding.html")
    else:
        # Start the onboarding process by running the onboarding script
        subprocess.run(["python", "onboarding.py"])
        return "Onboarding started"

# GET/POST method for prediction
@app.route("/attendance", methods = ['GET','POST'])
def attendance():
    # When submitting
    if request.method == 'POST':
        print("Nutrition Analyser prediction ongoing ================ ")

        # Get image from form
        print("Obtaining image given.....")
        img = request.files['class_img']
        print("- Successfully obtained Image -")

        # Create Image path to store and retrieve
        # Use random number to allow same-image upload
        print("Saving image to static folder....")
        img_path = "/static/class_img.png" 
        print("Image Path: ", img_path)

        if not os.path.exists("static"):
            os.mkdir("static")

        img.save(img_path)
        return render_template("attendance.html", img_path=img_path)
    return render_template(
        "attendance.html"
    )
    

@app.route("/attendance_result", methods = ['GET','POST'])
def attendance_result():
    # When submitting
    if request.method == 'POST':

        img_path = "static/class_img.jpg" 
        result_path = "static/result.jpg"
        if not os.path.exists("static"):
            os.mkdir("static")

        return render_template("attendance_result.html", img_path=img_path,result_path=result_path)
    return render_template(
        "attendance_result.html"
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
    return render_template('slides_list.html')

@app.route("/account")
def account():
    return render_template('account.html')


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


global COMMAND
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
            
            if(prediction[0][0] > 0.998):
                direction = "left"
                # response = {
                #     'command': 'back'
                # }
                time.sleep(5)
                #return jsonify(response)

            elif(prediction[0][0] < 0.5):
                direction = "right"
                # response = {
                #     'command': 'next'
                # }
                time.sleep(5)
                #return jsonify(response)

            else:
                direction = "none"
                time.sleep(1)

            
            print(direction)
            

        cv2.imshow("Prediction", frame)
        cv2.waitKey(1)
    #fetch("https://aap-dewmify-classup-image.ayftbvf4bbhqbudp.southeastasia.azurecontainer.io/command",{
    #   method: "POST",
    #   headers: {
    #       "Content-Type": "application/json"     
    #   },
    #   body: {
    #       'command': COMMAND
    #   }
    # })
    #.then(response => response.json())
    #.then(result => {
    #   alert(result.result);
    # })

@app.route("/controlSlides_feed")
def controlSlides_feed():
    return Response(controlSlides(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/command/", methods={"POST"})
def command():
    global COMMAND
    #fetch("https://aap-dewmify-classup-image.ayftbvf4bbhqbudp.southeastasia.azurecontainer.io/command",{
    #   method: "POST",
    #   headers: {
    #       "Content-Type": "application/json"     
    #   },
    #   body: {
    #       'command': COMMAND
    #   }
    # })
    #.then(response => response.json())
    #.then(result => {
    #   alert(result.result);
    # })
    returnvalue = jsonify({'command': COMMAND})
    COMMAND = None
    
    return (returnvalue)

@app.route("/setcmdnext/")
def setcmdnext():
    COMMAND = 'next'
    return jsonify({'command': COMMAND})
    
@app.route("/setcmdback/")
def setcmdback():
    COMMAND = 'back'
    return jsonify({'command': COMMAND})
# 
# @app.route("/prediction", methods=["POST"])
# def prediction():
    # new_reflection = [str(x) for x in request.form.values()]
    # input_prediction = new_reflection
    # print("new reflection:", new_reflection)
    # print("input_prediction:", input_prediction)
    # print('inputs', sentimental_model.inputs)
    # in_sensor= preprocess_input_data(str(input_prediction))
# 
    # senti_prediction = sentimental_model.predict(in_sensor)[0]
# 
    # class_index = np.argmax(senti_prediction)
    # print('class index', class_index)
# 
    # if class_index == 1:
        # result = "Positive Sentiment"
    # else:
    #    result = "Negative Sentiment"
# 
    # print('sentiment:', result)
# 
    # return render_template('student/sentiment_reflection.html', prediction_text=result)
# 