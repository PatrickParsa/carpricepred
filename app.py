import numpy as np
from flask import Flask, request, jsonify, render_template
import jsonify
import requests
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import pickle
# Creating flask app

app = Flask(__name__)

# Loading pkl model

model = pickle.load(open('lr_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index1.html')

standard_to = StandardScaler()
@app.route("/predict", methods = ['Post'])
def predict():
    Model_Camry=0
    Model_Civic=0
    Model_Corolla=0
    if request.method =='POST':
        Year = int(request.form['Year'])
        Mileage = int(request.form['Mileage'])
        Model_Camry = request.form['Model_Camry']
        if(Model_Camry =='Camry'):
            Model_Camry=1
            Model_Civic=0
            Model_Corolla=0
        elif(Model_Camry == 'Civic'):
            Model_Camry=0
            Model_Civic=1
            Model_Corolla=0
        elif(Model_Camry == 'Corolla'):
            Model_Camry=0
            Model_Civic=0
            Model_Corolla=1
        elif(Model_Camry == 'Accord'):
            Model_Camry=0
            Model_Civic=0
            Model_Corolla=0
        else:
            return render_template('index1.html',prediction_text="Please enter a valid model")
        Year = 2022-Year
        prediction = model.predict([[Mileage,Year,Model_Camry,Model_Civic,Model_Corolla]])
        output = round(prediction[0],0)
        return render_template('index1.html', prediction_text="The approximate value of this car is {}.".format(output))

    else:
        return render_template('index1.html')

if __name__ == "__main__":
    app.run(debug=True)



