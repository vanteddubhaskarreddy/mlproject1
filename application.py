from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

ridge = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        new_data_scaled = scaler.transform([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])

        result = ridge.predict(new_data_scaled)

        return render_template('home.html', result = result[0])
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)