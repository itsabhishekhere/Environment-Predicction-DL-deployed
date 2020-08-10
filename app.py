from __future__ import division, print_function
# coding=utf-8

import os
import numpy as np


from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

MODEL_PATH ='model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

UPLOAD_FOLDER = 'uploads'

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))

    x = image.img_to_array(img)


    x = x / 255
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

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


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        return render_template('model.html')
    else:
        try:
            file = request.files['file']

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            preds = model_predict(file_path, model)

            return render_template('result.html')
        except:
            flash("Please select the image first !!")
            return redirect(url_for("model"))


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