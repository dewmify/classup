from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, Response
import re
import subprocess
import os
import cv2
import numpy as np
import face_recognition

app = Flask(__name__)

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
        img_path = "static/class_img" 
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

