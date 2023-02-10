import transformers
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from keras_preprocessing.sequence import pad_sequences

import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, flash
from flask_mysqldb import MySQL
from keras.models import load_model
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

from wtforms import StringField, PasswordField, SubmitField, BooleanField, RadioField, HiddenField, DateField
from wtforms.validators import InputRequired, Email, Length, Optional, ValidationError

app = Flask(__name__)

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

# database class
class Student(db.Model, UserMixin):
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    studentEmail = db.Column(db.String(100), nullable=False, unique=True)
    studentName = db.Column(db.String(45), nullable=False)
    studentPassword = db.Column(db.String(200), nullable=False)
    studentImage = db.Column(db.String(45), nullable=False)
    studentPresMath = db.Column(db.Integer, nullable=False)
    studentPresScience = db.Column(db.Integer, nullable=False)
    studentPresChinese = db.Column(db.Integer, nullable=False)
    studentPresEnglish = db.Column(db.Integer, nullable=False)
    studentisTaking = db.Column(db.Integer, nullable=False)

    def __init__(self, id, studentName, studentEmail, studentPassword, studentImage, studentPresMath, studentPresScience, studentPresChinese, studentPresEnglish, studentisTaking):
        self.id = id
        self.studentName = studentName
        self.studentEmail = studentEmail
        self.studentPassword = studentPassword
        self.studentImage = studentImage
        self.studentPresMath = studentPresMath
        self.studentPresScience = studentPresScience
        self.studentPresChinese = studentPresChinese
        self.studentPresEnglish = studentPresEnglish
        self.studentisTaking = studentisTaking

class Admin(db.Model, UserMixin):
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    adminEmail = db.Column(db.String(45), nullable=False, unique=True)
    adminPassword = db.Column(db.String(45), nullable=False)
    
    def __init__(self, id, adminEmail, adminPassword):
      self.id = id
      self.adminEmail = adminEmail
      self.adminPassword = adminPassword

class Teacher(db.Model, UserMixin):
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    teacherName = db.Column(db.String(100), nullable=False)
    teacherEmail = db.Column(db.String(100), nullable=False, unique=True)
    teacherPassword = db.Column(db.String(100), nullable=False)
    teacherSubject = db.Column(db.String(45), nullable=False)

    def __init__(self, id, teacherName, teacherEmail, teacherPassword, teacherSubject):
        self.id = id
        self.teacherName = teacherName
        self.teacherEmail = teacherEmail
        self.teacherPassword = teacherPassword
        self.teacherSubject = teacherSubject

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

     def __init__(self, slidesId, slidesName, slidesDate, slidesAuthor, slidesSubject):
          self.slidesId = slidesId
          self.slidesName = slidesName
          self.slidesDate = slidesDate
          self.slidesAuthor = slidesAuthor
          self.slidesSubject = slidesSubject

with app.app_context():
    db.create_all() 
    db.session.commit()


# forms

# admin forms
class adminCreateStudentForm(FlaskForm):
    id= HiddenField('id')
    studentName= StringField('Student Name', validators=[InputRequired()])
    studentEmail= StringField('Student Email', validators=[InputRequired()])
    studentPassword= PasswordField('Student Password', validators=[InputRequired()])
    studentImage= StringField('Student Image', validators=[InputRequired()])
    studentPresMath= HiddenField('presentmath')
    studentPresScience= HiddenField('presentsci')
    studentPresChinese= HiddenField('presentchi')
    studentPresEnglish= HiddenField('presenteng')
    studentisTaking= HiddenField('istaking')

class adminCreateTeacherForm(FlaskForm):
    id= HiddenField('id')
    teacherName= StringField('Teacher Name', validators=[InputRequired()])
    teacherEmail= StringField('Teacher Email', validators=[InputRequired()])
    teacherPassword= PasswordField('Teacher Password', validators=[InputRequired()])
    teacherSubject= StringField('Teacher Subject', validators=[InputRequired()])

# login forms
class adminLoginForm(FlaskForm):
    adminEmail= StringField('Admin Email', validators=[InputRequired()])
    adminPass= PasswordField('Admin Password', validators=[InputRequired()])

class studLoginForm(FlaskForm):
    studentEmail= StringField('Student Email', validators=[InputRequired()])
    studentPass= PasswordField('Student Password', validators=[InputRequired()])

class teachLoginForm(FlaskForm):
    teacherEmail= StringField('Teacher Email', validators=[InputRequired()])
    teacherPassword= PasswordField('Teacher Password', validators=[InputRequired()])

#add slides form
class addSlidesForm(FlaskForm):
     slidesId = HiddenField('slidesId')
     slidesName = StringField('Slides Name', validators=[InputRequired()])
     slidesDate = DateField('Slides Date', validators=[InputRequired()])
     slidesAuthor = StringField('Slides Author', validators=[InputRequired()])
     slidesSubject = StringField('Slides Subject', validators=[InputRequired()])

# routes

@app.route("/")
def home():
        return render_template("index.html")

@app.route("/login")
def login():
        return render_template("login.html")

@app.route("/logout", methods=['GET', 'POST'])
def logout():
    logout_user()
    return redirect(url_for('login'))

#admin routes --------------------
@app.route("/login-admin", methods =['GET', 'POST'])
def login_admin():
        form = adminLoginForm()
        if form.validate_on_submit():
            admin = Admin.query.filter_by(adminEmail=form.adminEmail.data).first()
            if admin:
                if admin.adminPassword == form.adminPass.data:
                    login_user(admin)
                    return redirect(url_for('admin_dashboard'))
        return render_template("admin/admin-login.html", form=form)

@app.route("/admin-dashboard")
def admin_dashboard():
        return render_template("admin/admin-dashboard.html")

@app.route("/admin-create-student", methods=['GET', 'POST'])
def admin_create_student():
        form = adminCreateStudentForm(request.values, id=uuid.uuid4().int, studentPresMath=1, studentPresScience=1, studentPresChinese=1, studentPresEnglish=1, studentisTaking=1)
        if form.validate_on_submit():
            hashed_password = bcrypt.generate_password_hash(form.studentPassword.data)
            new_student = Student(id= form.id.data,
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
            return redirect(url_for('admin_dashboard'))
        return render_template("admin/admin-create-student.html", form=form)

@app.route("/admin-create-teacher", methods=['GET', 'POST'])
def admin_create_teacher():
        form = adminCreateTeacherForm(request.values, id=uuid.uuid4().int)
        if form.validate_on_submit():
            hashed_password = bcrypt.generate_password_hash(form.teacherPassword.data)
            new_teacher = Teacher(id= form.id.data,
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
                    login_user(teacher)
                    return redirect(url_for('teacher_dashboard'))
        return render_template("login_teacher.html", form=form)


# teacher pages ---------------------------
@app.route("/teacher-dashboard")
def teacher_dashboard():
    return render_template("teacher/teacher-dashboard.html")
    
@app.route("/slides")
def slides():
    return render_template("teacher/slides.html")

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
                            slidesSubject=form.slidesSubject.data
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
         



# max model
def preprocess_input_data(sentence):
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)
    attention_mask = [1] * len(input_ids)
    input_ids = pad_sequences([input_ids], maxlen=512, dtype="long", padding="post", truncating="post")
    attention_mask = pad_sequences([attention_mask], maxlen=512, dtype="long", padding="post", truncating="post")
    return input_ids, attention_mask

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
