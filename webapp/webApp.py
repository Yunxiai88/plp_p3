import os
import json
import argparse
import csv
import time
import pandas as pd

from werkzeug.utils import secure_filename
from flask import render_template, jsonify, flash
from flask import Flask, request, redirect, send_from_directory

import sys
import pathlib
path = pathlib.Path(__file__)
sys.path.append(os.path.join(str(path.parent.parent), "inference/"))

from sentiment import Sentiment
from process import single_process, batch_process

# initialize a flask object
app = Flask(__name__)
app.config['SECRET_KEY'] = "plpsystemkey"
app.config['UPLOAD_FOLDER'] = 'upload'
app.config['SESSION_TYPE'] = 'filesystem'

ALLOWED_EXTENSIONS = {'csv'}

database = []
length = 0
sentiment = Sentiment()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/error")
def error():
    return render_template("error.html")

@app.route("/")
def index():
    database = []
    with open('./webapp/data/progress.csv','r', encoding="utf8") as data:
        reader = csv.reader(data)
        next(reader, None)  # skip the headers

        for line in reader:
            database.append(line)

    try:
        database[0]
    except IndexError:
        return redirect("/error")

    length = len(database)

    # return the rendered template
    return render_template("index.html", data=database, len=length)

@app.route('/data/<path:filename>')
def base_static(filename):
    image_file = os.path.join(app.root_path + '/data/', filename)
    # check whether file exist
    if os.path.isfile(image_file):
        return send_from_directory(app.root_path + '/data', filename)
    else:
        return send_from_directory(app.root_path + '/static/img', 'not_found.jpg')

#---------------------------------------------------------------------
#----------------------------Functions--------------------------------
#---------------------------------------------------------------------
@app.route('/upload',methods = ['POST', 'GET'])
def upload():
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect("/")
        
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')  
            return redirect("/")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # process batch file
            result = batch_process(sentiment, filename)
            if result == "success":
                flash('file uploaded and process successful.')
            else:
                flash('file uploaded or process failed.')
            return redirect("/")
        else:
            flash('No allow file format')
            return redirect("/")

@app.route('/process', methods = ['POST', 'GET'])
def process():
    if request.method == "POST":
        review = request.form['review']

        # Get Polarity for single review
        polarity = single_process(sentiment, review.strip())

        output = {"data": polarity}
        return jsonify(output)


# execute function
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default="127.0.0.1", help="ip address")
    ap.add_argument("-o", "--port", type=int, default=8000, help="port number of the server")
    args = vars(ap.parse_args())

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)