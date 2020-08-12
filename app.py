from __future__ import division, print_function
# coding=utf-8

import os
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, flash, send_from_directory

# Define a flask app
app = Flask(__name__)

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'uploads/'


def model_predict(img_path):
    img = image.load_img(img_path, target_size=(150, 150, 3))

    x = image.img_to_array(img)
    x = x * 1./ 255
    x = np.expand_dims(x, axis=0)

    model = load_model('model.h5')
    pred = model.predict(x)
    preds = np.argmax(pred, axis=1)
    if preds == 0:
        preds = "It's Building image"
    elif preds == 1:
        preds = "It's Forest image"
    elif preds == 2:
        preds = "It's Glacier image"
    elif preds == 3:
        preds = "It's Mountain image"
    elif preds == 4:
        preds = "It's Sea image"
    else:
        preds = "It's Street image"

    return preds


@app.route('/')
def base():
    return render_template("home.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('model.html')
    else:

        target = os.path.join(APP_ROOT, 'uploads/')
        print(target)

        if not os.path.isdir(target):
            os.mkdir(target)

        file = request.files['file']
        print(file)

        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)

        file.save(destination)

        preds = model_predict(destination)
        print(preds)
        print(type(model))
        print("before done")
        return render_template('output.html', image_file_name=filename, label=preds)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/model")
def model():
    return render_template("model.html")


if __name__ == '__main__':
    app.run(debug=True)
