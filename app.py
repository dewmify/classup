import transformers
import pandas as pd
import numpy as np
import subprocess
import os
import face_recognition
import face_detection
import tensorflow as tf
import json
import uuid
import math
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
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
import time
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, login_manager
from PIL import Image
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

from wtforms import StringField, PasswordField, SubmitField, BooleanField, RadioField, HiddenField, DateField, FileField
from wtforms.validators import InputRequired, Email, Length, Optional, ValidationError
from random import randrange



app = Flask(__name__)

UPLOAD_FOLDER = 'static/AttendanceUploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


bootstrap = Bootstrap(app)
mysql = MySQL(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view="login"

@login_manager.user_loader
def load_student(studentid):
    return Student.query.get(int(studentid))

@login_manager.user_loader
def load_admin(adminid):
    return Admin.query.get(int(adminid))

@login_manager.user_loader
def load_teacher(teacherid):
    return Teacher.query.get(int(teacherid))

# database

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://admin:bYaP6tsnsRFy1TIJVQAr@classupdb.cgdsnk6an6d3.us-east-1.rds.amazonaws.com/classupdb'
app.config['SECRET_KEY'] = "Guysmynameisjefferson123"
db = SQLAlchemy()
db.init_app(app)



# ai models

hand_model = load_model('models/HandGestureModel.h5')
#sentimental_model = load_model('models/sentiment_model.h5', custom_objects={"TFBertModel": transformers.TFBertModel})

allowed_extensions = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
     

# database class

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(45), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(50), nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'user',
        'polymorphic_on': type
    }

    def __init__(self, id, email, name, password):
         self.id = id
         self.email = email
         self.name = name
         self.password = password
    



class Teacher(User):
    __tablename__ = 'teachers'
    id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key = True)
    teacherSubject = db.Column(db.String(45), nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'teacher',
    }

    def __init__(self, id, name, email, password, teacherSubject):
        super().__init__(name, email, password, 'teacher')
        self.id = id
        self.teacherSubject = teacherSubject

class Student(User):
    __tablename__ = 'students'
    id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key = True)
    studentImage = db.Column(db.String(45), nullable=False)
    studentPresMath = db.Column(db.Integer, nullable=False)
    studentPresScience = db.Column(db.Integer, nullable=False)
    studentPresChinese = db.Column(db.Integer, nullable=False)
    studentPresEnglish = db.Column(db.Integer, nullable=False)
    studentisTaking = db.Column(db.Integer, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'student',
    }

    def __init__(self, id, name, email, password, studentImage, studentPresMath, studentPresScience, studentPresChinese, studentPresEnglish, studentisTaking):
        super().__init__(name, email, password, 'student')
        self.id = id
        self.studentImage = studentImage
        self.studentPresMath = studentPresMath
        self.studentPresScience = studentPresScience
        self.studentPresChinese = studentPresChinese
        self.studentPresEnglish = studentPresEnglish
        self.studentisTaking = studentisTaking

# class Student(db.Model, UserMixin):
#     id = db.Column(db.Integer, nullable=False, primary_key=True)
#     studentEmail = db.Column(db.String(100), nullable=False)
#     studentName = db.Column(db.String(45), nullable=False)
#     studentPassword = db.Column(db.String(200), nullable=False)
#     studentImage = db.Column(db.String(45), nullable=False)
#     studentPresMath = db.Column(db.Integer, nullable=False)
#     studentPresScience = db.Column(db.Integer, nullable=False)
#     studentPresChinese = db.Column(db.Integer, nullable=False)
#     studentPresEnglish = db.Column(db.Integer, nullable=False)
#     studentisTaking = db.Column(db.Integer, nullable=False)

#     def __init__(self, id, studentName, studentEmail, studentPassword, studentImage, studentPresMath, studentPresScience, studentPresChinese, studentPresEnglish, studentisTaking):
#         self.id = id
#         self.studentName = studentName
#         self.studentEmail = studentEmail
#         self.studentPassword = studentPassword
#         self.studentImage = studentImage
#         self.studentPresMath = studentPresMath
#         self.studentPresScience = studentPresScience
#         self.studentPresChinese = studentPresChinese
#         self.studentPresEnglish = studentPresEnglish
#         self.studentisTaking = studentisTaking

class Admin(db.Model, UserMixin):
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    adminEmail = db.Column(db.String(45), nullable=False, unique=True)
    adminPassword = db.Column(db.String(45), nullable=False)
    
    def __init__(self, id, adminEmail, adminPassword):
      self.id = id
      self.adminEmail = adminEmail
      self.adminPassword = adminPassword


# class Teacher(db.Model, UserMixin):
#     id = db.Column(db.Integer, nullable=False, primary_key=True)
#     teacherName = db.Column(db.String(100), nullable=False)
#     teacherEmail = db.Column(db.String(100), nullable=False, unique=True)
#     teacherPassword = db.Column(db.String(100), nullable=False)
#     teacherSubject = db.Column(db.String(45), nullable=False)

#     def __init__(self, id, teacherName, teacherEmail, teacherPassword, teacherSubject):
#         self.id = id
#         self.teacherName = teacherName
#         self.teacherEmail = teacherEmail
#         self.teacherPassword = teacherPassword
#         self.teacherSubject = teacherSubject

class Subject(db.Model, UserMixin):
    name = db.Column(db.String(100), primary_key=True)
    numofStudents = db.Column(db.Integer, nullable=False)

    def __init__(self, name, numofStudents):
        self.name = name
        self.numofStudents = numofStudents

class Topics(db.Model, UserMixin):
    subjectName = db.Column(db.String(100), primary_key=True)
    studentEmail = db.Column(db.String(100), nullable=False, unique=True)
    week = db.Column(db.Integer, nullable=False)
    sentiment = db.Column(db.String(45), nullable=False)
    reflection = db.Column(db.String(150), nullable=False)

    def __init__(self, subjectName, studentEmail, week, sentiment, reflection):
        self.subjectName = subjectName
        self.studentEmail = studentEmail
        self.week = week
        self.sentiment = sentiment
        self.reflection = reflection

class Slides(db.Model, UserMixin):
     slidesId = db.Column(db.Integer, primary_key=True)
     slidesName = db.Column(db.String(100), nullable = False)
     slidesDate = db.Column(db.Date, nullable = False)
     slidesAuthor = db.Column(db.String(100), nullable = False)
     slidesSubject = db.Column(db.String(100), nullable = False)
     slidesLink = db.Column(db.String(1000), nullable = False)
     teacherEmail = db.Column(db.String(100), db.ForeignKey('teacher.teacherEmail'))

     def __init__(self, slidesId, slidesName, slidesDate, slidesAuthor, slidesSubject, slidesLink, teacherEmail):
          self.slidesId = slidesId
          self.slidesName = slidesName
          self.slidesDate = slidesDate
          self.slidesAuthor = slidesAuthor
          self.slidesSubject = slidesSubject
          self.slidesLink = slidesLink
          self.teacherEmail = teacherEmail

with app.app_context():
    db.create_all()
    db.session.commit()


# forms

# admin forms
class adminCreateUserForm(FlaskForm):
    id = HiddenField('id')
    name = StringField('User Name', validators=[InputRequired()])
    email= StringField('User Email', validators=[InputRequired()])
    password= PasswordField('User Password', validators=[InputRequired()])

class adminCreateStudentForm(adminCreateUserForm):
    studentImage= StringField('Student Image', validators=[InputRequired()])
    studentPresMath= HiddenField('presentmath')
    studentPresScience= HiddenField('presentsci')
    studentPresChinese= HiddenField('presentchi')
    studentPresEnglish= HiddenField('presenteng')
    studentisTaking= HiddenField('istaking')

class adminCreateTeacherForm(adminCreateUserForm):
    teacherSubject= StringField('Teacher Subject', validators=[InputRequired()])

# login forms
class adminLoginForm(FlaskForm):
    adminEmail= StringField('Admin Email', validators=[InputRequired()])
    adminPass= PasswordField('Admin Password', validators=[InputRequired()])

class studLoginForm(FlaskForm):
    email= StringField('User Email', validators=[InputRequired()])
    password= PasswordField('User Password', validators=[InputRequired()])

class teachLoginForm(FlaskForm):
    email= StringField('User Email', validators=[InputRequired()])
    password= PasswordField('User Password', validators=[InputRequired()])

#add slides form
class addSlidesForm(FlaskForm):
     slidesId = HiddenField('slidesId')
     slidesName = StringField('Slides Name', validators=[InputRequired()])
     slidesDate = DateField('Slides Date', validators=[InputRequired()])
     slidesAuthor = StringField('Slides Author', validators=[InputRequired()])
     slidesSubject = StringField('Slides Subject', validators=[InputRequired()])
     slidesLink = StringField("Slides Link (Embed Link from OneDrive)", validators=[InputRequired()])

class onboardForm(FlaskForm):
    onboardSubmit = SubmitField('Start Onboarding')

# routes

@app.route("/")
def home():
        return render_template("index.html")

@app.route("/login")
def login():
        return render_template("login.html")

@app.route("/logout", methods=['GET', 'POST'])
def logout():
    session.pop('email', None)
    logout_user()
    return render_template("/index.html")

#admin routes --------------------
@app.route("/login-admin", methods =['GET', 'POST'])
def login_admin():
        form = adminLoginForm()
        if form.validate_on_submit():
            admin = Admin.query.filter_by(adminEmail=form.adminEmail.data).first()
            if admin:
                if admin.adminPassword == form.adminPass.data:
                    email = admin.adminEmail
                    session['email'] = email
                    login_user(admin)
                    return redirect(url_for('admin_dashboard'))
        return render_template("admin/admin-login.html", form=form)

@app.route("/admin-dashboard")
def admin_dashboard():
        return render_template("admin/admin-dashboard.html")

@app.route("/admin-create-student", methods=['GET', 'POST'])
def admin_create_student():
        form = adminCreateStudentForm(request.values, studentPresMath=1, studentPresScience=1, studentPresChinese=1, studentPresEnglish=1, studentisTaking=1)
        if form.validate_on_submit():
            hashed_password = bcrypt.generate_password_hash(form.studentPassword.data)
            new_student = Student(id=randrange(0,999999999),
                                    studentName=form.studentName.data,
                                    studentEmail=form.studentEmail.data, 
                                    studentPassword=hashed_password,
                                    studentImage=form.studentImage.data, 
                                    studentPresMath=form.studentPresMath.data,
                                    studentPresScience=form.studentPresScience.data,
                                    studentPresChinese=form.studentPresChinese.data,
                                    studentPresEnglish=form.studentPresEnglish.data,
                                    studentisTaking=form.studentisTaking.data)
            db.session.add(new_student)
            db.session.commit()
            # Retrieve the newly created student
            student = Student.query.filter_by(studentName=form.studentName.data).first()
            
            # Obtain the student id
            student_id = student.id
            return redirect(url_for('onboard',student_id=student_id))
        return render_template("admin/admin-create-student.html", form=form)

@app.route("/admin-create-teacher", methods=['GET', 'POST'])
def admin_create_teacher():
        form = adminCreateTeacherForm()
        if form.validate_on_submit():
            hashed_password = bcrypt.generate_password_hash(form.teacherPassword.data)
            new_teacher = Teacher(id= randrange(0,999999999),
                                    teacherName=form.teacherName.data,
                                    teacherEmail=form.teacherEmail.data,
                                    teacherPassword=hashed_password,
                                    teacherSubject=form.teacherSubject.data
                                  )
            db.session.add(new_teacher)
            db.session.commit()
            return redirect(url_for('admin_dashboard'))
        return render_template("admin/admin-create-teacher.html", form=form)


# student routes------------------
@app.route("/login-student", methods =['GET', 'POST'])
def login_student():
        form = studLoginForm()
        if form.validate_on_submit():
            student = Student.query.filter_by(studentEmail=form.studentEmail.data).first()
            hashed_password = student.studentPassword
            password = form.studentPass.data
            if student:
                if bcrypt.check_password_hash(hashed_password, password):
                    session['email'] = student.studentEmail
                    login_user(student)
                    return redirect(url_for('student_index'))
        return render_template("login_student.html", form=form)

@app.route("/student-class")
def student_classes():
    return render_template("student/student_class.html")

@app.route("/student-index")
def student_index():
    return render_template("student/student_index.html")
       
@app.route("/reflection")
def reflection():
    return render_template('student/sentiment_reflection.html')


# teacher routes------------------------
@app.route("/login-teacher", methods =['GET', 'POST'])
def login_teacher():
        form = teachLoginForm()
        if form.validate_on_submit():
            teacher = Teacher.query.filter_by(teacherEmail=form.teacherEmail.data).first()
            hashed_password = teacher.teacherPassword
            password = form.teacherPassword.data
            if teacher:
                if bcrypt.check_password_hash(hashed_password, password):
                    session['email'] = teacher.teacherEmail
                    login_user(teacher)
                    return redirect(url_for('teacher_dashboard'))
        return render_template("login_teacher.html", form=form)


# teacher pages ---------------------------
@app.route("/teacher_dashboard")
def teacher_dashboard():
    if session['email'] != "":
        slidesList = Slides.query.all()
        return render_template("teacher/teacher_dashboard.html", slidesList = slidesList)
    else:
        return render_template("login_teacher.html")
    
@app.route("/<int:slidesId>/")
def slides(slidesId):
    slides = Slides.query.get_or_404(slidesId)
    return render_template("teacher/slides.html", slides = slides)

@app.route("/slides_list")
def slides_list():
    slidesList = Slides.query.all()      
    return render_template('teacher/slides_list.html', slidesList = slidesList)

@app.route("/account")
def account():
    return render_template('account.html')


@app.route("/addSlides", methods=["GET", "POST"])
def addSlides():
    form = addSlidesForm(request.values)
    if form.validate_on_submit():
        new_slides = Slides(
                            slidesId = form.slidesId.data,
                            slidesName=form.slidesName.data,
                            slidesDate=form.slidesDate.data,
                            slidesAuthor=form.slidesAuthor.data,
                            slidesSubject=form.slidesSubject.data,
                            slidesLink=form.slidesLink.data,
                            teacherEmail=session['email']
                            )
        db.session.add(new_slides)
        db.session.commit()
        return redirect(url_for('slides_list'))
    return render_template("teacher/add_slides.html", form=form)

#joshua model-----------------------------------------
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
         

@app.route("/addSlides", methods=["POST"])
def addSlides():
    values = request.form.getlist("value")
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="test"
    )
    cursor = conn.cursor()
    sql = "INSERT INTO slides (value) VALUES (%s)"
    cursor.executemany(sql, [(value,) for value in values])
    conn.commit()
    cursor.close()
    conn.close()
    return "Slides uploaded successfully!"  



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




#Ryo stuff ------------------------
@app.route('/attendance', methods=['GET', 'POST'])
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
            return render_template('attendance.html', filename=filename)
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(request.url)
    else:
        return render_template('attendance.html')
 
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

    return render_template('attendance_result.html', filename="processed_" + filename)


def onboarding(student_id):
    video_capture = cv2.VideoCapture(0)
    counter = 8

    while True:
        _, frame = video_capture.read()
        frame, face_box, face_coords = face_detection.detect_faces(frame)
        text = 'Image will be taken in {}..'.format(math.ceil(counter))
        if face_box is not None:
            frame = face_detection.write_on_frame(frame, text, face_coords[0], face_coords[1]-10)
        cv2.imshow('Video', frame)
        cv2.waitKey(1)
        counter -= 0.1
        if counter <= 0:
            imgpath = 'static/AttendanceUploads/true_image_' + str(student_id) + '.png'
            cv2.imwrite(imgpath, face_box)
            #Update the Student object in the database
            student = Student.query.get(student_id)
            student.studentImage = imgpath
            db.session.commit()
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    print("Onboarding Image Captured")



@app.route('/onboarding/<int:student_id>', methods=["GET","POST"])
def onboard(student_id):
    student = Student.query.get(student_id)
    #form = onboardForm()
    #if form.validate_on_submit():
    onboarding(student_id)
    return render_template('onboarding.html', student=student)
    

# @app.route("/onboarding", methods=["GET", "POST"])
# def onboarding():
#     if request.method == "GET":
#         # Return the onboarding page
#         return render_template("onboarding.html")
#     else:
#         # Start the onboarding process by running the onboarding script
#         subprocess.run(["python", "onboarding.py"])
        

# max model
def preprocess_input_data(sentence):
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)
    attention_mask = [1] * len(input_ids)
    input_ids = pad_sequences([input_ids], maxlen=512, dtype="long", padding="post", truncating="post")
    attention_mask = pad_sequences([attention_mask], maxlen=512, dtype="long", padding="post", truncating="post")
    return input_ids, attention_mask


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
#@app.route("/prediction", methods=["POST"])
#def prediction():
#    new_reflection = [str(x) for x in request.form.values()]
#    input_prediction = new_reflection
#    print("new reflection:", new_reflection)
#    print("input_prediction:", input_prediction)
#    print('inputs', sentimental_model.inputs)
#    in_sensor= preprocess_input_data(str(input_prediction))
#
#    senti_prediction = sentimental_model.predict(in_sensor)[0]
#
#    class_index = np.argmax(senti_prediction)
#    print('class index', class_index)
#
#    if class_index == 1:
#        result = "Positive Sentiment"
#    else:
#       result = "Negative Sentiment"
#
#    print('sentiment:', result)
#
#    return render_template('student/sentiment_reflection.html', prediction_text=result)
