from flask import Flask, render_template, url_for, request
from keras.models import model_from_json, load_model
import pandas as pd
import numpy as np
import h5py

model = load_model('titanic.h5')
#app = Flask(__name__, template_folder='tamplates' , static_folder='static')
app = Flask(__name__)
# Pclass	Sex	Age	SibSp	Parch	Fare	Embarked
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        P_Class = float(request.form['P_Class'])
        Gender = float(request.form['Gender'])
        Age = float(request.form['Age'])
        Sib_Sp = float(request.form['Sib_Sp'])
        Parch = float(request.form['Parch'])
        Fare = float(request.form['Fare'])
        Embarked = float(request.form['Embarked'])

        Fare = Fare / 34.69451400560218
        Age = Age / 24.044258373205743

        data = np.asarray([P_Class, Gender, Age, Sib_Sp, Parch ,Fare, Embarked])
        p = data.reshape(7,1).T

        prediction = model.predict(p)

        return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)