import os
import json
import argparse
import csv
import pandas as pd

from flask import render_template, jsonify
from flask import Flask, request, redirect, send_from_directory

import sys
import pathlib
path = pathlib.Path(__file__)
sys.path.append(os.path.join(str(path.parent.parent), "inference/"))

from sentiment import Sentiment
from process import process_review

# initialize a flask object
app = Flask(__name__)
ROWS_PER_PAGE = 10
database = []
configData = []
length = 0

sentiment = Sentiment()

@app.route("/error")
def error():
    return render_template("error.html")

@app.route("/")
def index():
    database = []
    with open('webapp/data/progress.csv','r') as data:
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
    return send_from_directory(app.root_path + '/data/', filename)

#---------------------------------------------------------------------
#----------------------------Functions--------------------------------
#---------------------------------------------------------------------
@app.route('/upload',methods = ['POST', 'GET'])
def config():
    if request.method == "POST":
        # upload file
        with open ('webapp/data/reference.csv','w') as f:
            writer = csv.writer(f)
            writer.writerows(updates)

        return redirect("/")


@app.route('/process', methods = ['POST', 'GET'])
def process():
    if request.method == "POST":
        review = request.form['review']

        # Get Polarity
        polarity = process_review(sentiment, review.strip())

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