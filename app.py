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
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, login_manager, current_user
from PIL import Image
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

from wtforms import StringField, PasswordField, SubmitField, BooleanField, RadioField, HiddenField, DateField, SelectField
from wtforms.validators import InputRequired, Email, Length, Optional, ValidationError
from random import randrange

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

class User(db.Model, UserMixin):
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

    def __init__(self, id, email, name, password, type):
        self.id = id
        self.email = email
        self.name = name
        self.password = password
        self.type = type 


class Teacher(User):
    __tablename__ = 'teachers'
    id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key = True)
    teacherSubject = db.Column(db.String(45), nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'teacher',
    }

    def __init__(self, id, email, name, password, teacherSubject):
        super().__init__(id, email, name, password, 'teacher')
        self.teacherSubject = teacherSubject

class Student(User):
    __tablename__ = 'students'
    id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key = True)
    studentImage = db.Column(db.String(45), nullable=True)
    studentPresMath = db.Column(db.Integer, nullable=False)
    studentPresScience = db.Column(db.Integer, nullable=False)
    studentPresChinese = db.Column(db.Integer, nullable=False)
    studentPresEnglish = db.Column(db.Integer, nullable=False)
    studentisTaking = db.Column(db.Integer, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'student',
    }

    def __init__(self, id, email, name, password, studentImage, studentPresMath, studentPresScience, studentPresChinese, studentPresEnglish, studentisTaking):
        super().__init__(id, email, name, password, 'student')
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


class Subject(db.Model, UserMixin):
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    name = db.Column(db.String(100))
    numofStudents = db.Column(db.Integer)

    def __init__(self, id, name, numofStudents):
        self.id = id
        self.name = name
        self.numofStudents = numofStudents

class Topic(db.Model, UserMixin):
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    topicSubject = db.Column(db.String(100))
    topicName = db.Column(db.String(100))
    topicWeek = db.Column(db.Integer, nullable=False)
    
    def __init__(self, id, topicSubject, topicName, topicWeek):
        self.id = id
        self.topicSubject = topicSubject
        self.topicName = topicName
        self.topicWeek = topicWeek

class Reflection(db.Model, UserMixin):
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    subjectName = db.Column(db.String(100))
    topicName = db.Column(db.String(100), nullable=False)
    week = db.Column(db.Integer, nullable=False)
    studentEmail = db.Column(db.String(100), nullable=False)
    reflection = db.Column(db.String(255), nullable=False)
    sentiment = db.Column(db.String(45), nullable=False)

    def __init__(self, id, subjectName, topicName, week, studentEmail, reflection, sentiment):
        self.id = id
        self.subjectName = subjectName
        self.topicName = topicName
        self.week = week
        self.studentEmail = studentEmail
        self.reflection = reflection
        self.sentiment = sentiment

class Slides(db.Model, UserMixin):
     slidesId = db.Column(db.Integer, primary_key=True)
     slidesName = db.Column(db.String(100), nullable = False)
     slidesDate = db.Column(db.Date, nullable = False)
     slidesAuthor = db.Column(db.String(100), nullable = False)
     slidesSubject = db.Column(db.String(100), nullable = False)
     slidesLink = db.Column(db.String(1000), nullable = False)
     email = db.Column(db.String(100), db.ForeignKey('users.email'))

     def __init__(self, slidesId, slidesName, slidesDate, slidesAuthor, slidesSubject, slidesLink, email):
          self.slidesId = slidesId
          self.slidesName = slidesName
          self.slidesDate = slidesDate
          self.slidesAuthor = slidesAuthor
          self.slidesSubject = slidesSubject
          self.slidesLink = slidesLink
          self.email = email

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
    type = HiddenField('type')

class adminCreateStudentForm(adminCreateUserForm):
    studentImage= StringField('Student Image', validators=[InputRequired()])
    studentPresMath= HiddenField('presentmath')
    studentPresScience= HiddenField('presentsci')
    studentPresChinese= HiddenField('presentchi')
    studentPresEnglish= HiddenField('presenteng')
    studentisTaking= HiddenField('istaking')

class adminCreateTeacherForm(adminCreateUserForm):
    teacherSubject= StringField('Teacher Subject', validators=[InputRequired()])
    
class adminCreateTopicForm(FlaskForm):
    id= HiddenField('id')
    topicSubject= RadioField('Topics Subject', validators=[InputRequired()], choices=['Math', 'Science', 'English', 'Chinese'])
    topicName= StringField('Topics Name', validators=[InputRequired()])
    topicWeek= StringField('Week', validators=[InputRequired()])


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

#reflection form
class studentReflectionForm(FlaskForm):
    id= HiddenField('id')
    subjectName=HiddenField('subjectname')
    topicName=HiddenField('topicname')
    week=HiddenField('week')
    studentEmail=HiddenField('studentemail')
    reflection= StringField('', validators=[InputRequired()])
    sentiment=HiddenField('sentiment')

#add slides form
class addSlidesForm(FlaskForm):
     slidesId = HiddenField('slidesId')
     slidesName = StringField('Slides Name', validators=[InputRequired()])
     slidesDate = DateField('Slides Date', validators=[InputRequired()])
     slidesAuthor = StringField('Slides Author', validators=[InputRequired()])
     slidesSubject = SelectField('Slides Subject', choices=[('Science', 'Science'), ('Math', 'Math'), ('Chinese', 'Chinese'), ('English', 'English')])
     slidesLink = StringField("Slides Link (Embed Link from OneDrive)", validators=[InputRequired()])

# routes

@app.route("/")
def home():
        if 'id' in session:
            user = User.query.get(session['id'])
            return render_template('index.html', user=user)
        else:
            return render_template("index.html")

@app.route("/login")
def login():
        return render_template("login.html")

@app.route("/logout", methods=['GET', 'POST'])
def logout():
    session.pop('email', None)
    logout_user()
    return render_template("/index.html")

# routes error handling

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

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
            hashed_password = bcrypt.generate_password_hash(form.password.data)
            new_student = Student(id=randrange(0,999999999),
                                    name=form.name.data,
                                    email=form.email.data, 
                                    password=hashed_password,
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
        form = adminCreateTeacherForm()
        if form.validate_on_submit():
            hashed_password = bcrypt.generate_password_hash(form.password.data)
            new_teacher = Teacher(id= randrange(0,999999999),
                                    name=form.name.data,
                                    email=form.email.data,
                                    password=hashed_password,
                                    teacherSubject=form.teacherSubject.data
                                  )
            db.session.add(new_teacher)
            db.session.commit()
            return redirect(url_for('admin_dashboard'))
        return render_template("admin/admin-create-teacher.html", form=form)

@app.route("/admin-create-topic", methods=['GET', 'POST'])
def admin_create_topic():
        form = adminCreateTopicForm()
        if form.validate_on_submit():
            new_topic = Topic(id= randrange(0,999999999),
                                topicSubject=form.topicSubject.data,
                                topicName=form.topicName.data,
                                topicWeek=form.topicWeek.data)
            db.session.add(new_topic)
            db.session.commit()
            return redirect(url_for('admin_dashboard'))
        return render_template("admin/admin-create-topic.html", form=form)
 

# student routes------------------
@app.route("/login-student", methods =['GET', 'POST'])
def login_student():
        form = studLoginForm()
        if form.validate_on_submit():

            if not re.match(r"[^@]+@[^@]+\.[^@]+", form.email.data):
                flash("Invalid email address format", "danger")
                return redirect(url_for('login_student'))

            student = User.query.filter_by(email=form.email.data).first()

            if not student:
                flash("User does not exist", "danger")
                return redirect(url_for('login_student'))

            hashed_password = student.password
            password = form.password.data

            if not password:
                flash("Password is required", "danger")
                return redirect(url_for('login_student'))

            if bcrypt.check_password_hash(hashed_password, password):
                session['email'] = student.email
                session['name'] = student.name
                login_user(student)
                return redirect(url_for('student_index'))
            
            else:
                flash("Wrong Password", "danger")
                return redirect(url_for('login_student'))
        return render_template("login_student.html", form=form)

@app.route("/student-index")
def student_index():
    name = session['name']
    return render_template("student/student_index.html", name=name)

@app.route("/student-class/<value>")
def student_classes(value):
    name = session['name']
    topics_list= []
    topics = Topic.query.all()
    topic_of_choice = value
    topic_subject = topic_of_choice
    for topic in topics:
        if topic.topicSubject == topic_of_choice:
            topics_list.append(topic)
        
    sorted_topics_list = sorted(topics_list, key=lambda x: x.topicWeek)
    
    return render_template("student/student_class.html", topics_list=sorted_topics_list, topic_subject=topic_subject, name=name)
       
@app.route("/reflection/<topic_subject>/<topicWeek>/<topicName>", methods =['GET', 'POST'])
def reflection(topic_subject, topicWeek, topicName):
    form = studentReflectionForm()
    topic_subject = topic_subject
    topicWeek = topicWeek
    topicName = topicName
    student_email = session['email']
    name = session['name']
    isAvailable = False
#
    #reflections = Reflection.query.all()
    #for reflection in reflections:
    #    if reflection.studentEmail == student_email and reflection.topicName == topicName:
    #        isAvailable = True
    #        break
    #
    #if form.validate_on_submit():
        #input_reflection = [str(x) for x in request.form.values()]
        #input_prediction = input_reflection
        #print("new reflection:", input_reflection)
        #print("input_prediction:", input_prediction)
        #print('inputs', sentimental_model.inputs)
        #in_sensor= preprocess_input_data(str(input_prediction))
#
        #senti_prediction = sentimental_model.predict(in_sensor)[0]
#
        #class_index = np.argmax(senti_prediction)
        #print('class index', class_index)
#
        #if class_index == 1:
        #    result = "Positive"
        #else:
        #    result = "Negative"
        #print('sentiment:', result)
        #new_reflection=Reflection(id= randrange(0,999999999),
        #                            subjectName=topic_subject,
        #                            topicName=topicName,
        #                            week=topicWeek,
        #                            studentEmail=student_email,
        #                            reflection= str(input_reflection[-1]),
        #                            sentiment=result)
        #db.session.add(new_reflection)
        #db.session.commit()
        #return redirect(url_for('reflection_submitted'))
    return render_template('student/sentiment_reflection.html', form=form, topic_subject=topic_subject, topicWeek=topicWeek, topicName=topicName, isAvailable=isAvailable, name=name)

@app.route("/reflection-submitted")
def reflection_submitted():
    name = session['name']
    return render_template('student/reflection-submitted.html', name=name)

@app.route("/student-grades")
def student_grades():
    name = session['name']
    return render_template('student/student_grades.html', name=name)


# teacher routes------------------------
@app.route("/login-teacher", methods =['GET', 'POST'])
def login_teacher():
        form = teachLoginForm()
        if form.validate_on_submit():

            if not re.match(r"[^@]+@[^@]+\.[^@]+", form.email.data):
                flash("Invalid email address format", "danger")
                return redirect(url_for('login_teacher'))

            teacher = User.query.filter_by(email=form.email.data).first()

            if not teacher:
                flash("User does not exist", "danger")
                return redirect(url_for('login_teacher'))

            hashed_password = teacher.password
            password = form.password.data
            if not password:
                flash("Password is required", "danger")
                return redirect(url_for('login_teacher'))

            if bcrypt.check_password_hash(hashed_password, password):
                session['email'] = teacher.email
                session['teacherSubject'] = teacher.teacherSubject
                login_user(teacher)
                return redirect(url_for('teacher_dashboard'))
            else:
                flash("Wrong Password", "danger")
                return redirect(url_for('login_teacher'))

        return render_template("login_teacher.html", form=form)


# teacher pages ---------------------------
@app.route("/teacher_dashboard")
def teacher_dashboard():
    if session['email'] != "":
        subject = session['teacherSubject']
        negativeSentimentCount = 0
        positiveSentimentCount = 0
        reflections = Reflection.query.all()
        
        for reflection in reflections:
            if subject == reflection.subjectName:
                if reflection.sentiment == 'Positive':
                    positiveSentimentCount += 1
                if reflection.sentiment == 'Negative':
                    negativeSentimentCount += 1

        totalSentimentCount = positiveSentimentCount + negativeSentimentCount
        positivePercentage = (positiveSentimentCount / totalSentimentCount) * 100
        negativePercentage = (negativeSentimentCount / totalSentimentCount) * 100
        
        slidesList = Slides.query.all()
        return render_template("teacher/teacher_dashboard.html", slidesList = slidesList, positivePercentage=positivePercentage, negativePercentage=negativePercentage)
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

@app.route("/teacher-sentiments")
def teacher_sentiments():
    subject = session['teacherSubject']
    name = session['name']
    reflections_list=[]
    reflections = Reflection.query.all()
    for reflection in reflections:
        if reflection.subjectName == subject:
            reflections_list.append(reflection)

    sorted_reflection_list = sorted(reflections_list, key=lambda x: x.week)
    return render_template('teacher/teacher-sentiments.html', sorted_reflection_list=sorted_reflection_list, name=name)


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
                            email=session['email']
                            )
        db.session.add(new_slides)
        db.session.commit()
        return redirect(url_for('slides_list'))
    else:
        flash("Error: Please fill out all the fields correctly.")
    return render_template("teacher/add_slides.html", form=form)

@app.route("/editSlides/<int:slidesId>", methods=["GET", "POST"])
def editSlides(slidesId):
    form = addSlidesForm(request.values)
    slides = Slides.query.filter_by(slidesId=slidesId).first()
    if form.validate_on_submit():
        slides.slidesId = form.slidesId.data
        slides.slidesName = form.slidesName.data
        slides.slidesDate = form.slidesDate.data
        slides.slidesAuthor = form.slidesAuthor.data
        slides.slidesSubject = form.slidesSubject.data
        slides.slidesLink = form.slidesLink.data
        db.session.commit()
        return redirect(url_for('slides_list'))
    form.slidesId.data = slides.slidesId
    form.slidesName.data = slides.slidesName
    form.slidesDate.data = slides.slidesDate
    form.slidesAuthor.data = slides.slidesAuthor
    form.slidesSubject.data = slides.slidesSubject
    form.slidesLink.data = slides.slidesLink
    return render_template("teacher/edit_slides.html", form=form, slides = slides)

@app.route("/deleteSlides/<int:slidesId>", methods=["GET", "POST"])
def deleteSlides(slidesId):
    slides = Slides.query.filter_by(slidesId=slidesId).first()
    db.session.delete(slides)
    db.session.commit()
    return redirect(url_for('slides_list'))

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

        if(prediction[0][0] == 1):
            direction = "left"
        elif(prediction[0][0] < 0.5):
            direction = "right"
        else:
            direction = "none"

        print(prediction)
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

@app.route("/prediction")
def prediction():
        return render_template('student/sentiment_reflection.html')

