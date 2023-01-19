from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

app = Flask(__name__)

app.secret_key = 'secret'
 
 
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'classup'
 
 
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
