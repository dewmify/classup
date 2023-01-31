from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

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


slides_model = tf.keras.models.load_model('slidesModel.h5')

@app.route("/controlSlides", methods=['POST'])
def controlSlides():
    input = request.get_json()
    target = ['right', 'left']

    x1 = input['x1']
    result = slides_model.predict([[x1]])

    return jsonify({'result': target[np.argmax(result[0])]})
