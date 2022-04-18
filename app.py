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
    if request.method =='POST':
        try:
            int(request.form['Mileage'])
        except:
            return render_template('index1.html', prediction_text="Please enter a valid number for mileage")
        else:
            Year = int(request.form['Year'])
            age = 2022 - Year
            Mileage = int(request.form['Mileage'])
            modelCar = request.form['Model_Camry']
        if(modelCar =='Camry'):
            Model_Camry=1
            Model_Civic=0
            Model_Corolla=0
        elif(modelCar == 'Civic'):
            Model_Camry=0
            Model_Civic=1
            Model_Corolla=0
        elif(modelCar == 'Corolla'):
            Model_Camry=0
            Model_Civic=0
            Model_Corolla=1
        elif(modelCar == 'Accord'):
            Model_Camry=0
            Model_Civic=0
            Model_Corolla=0
        else:
            return render_template('index1.html',prediction_text="Please enter a valid model")
        prediction = model.predict([[Mileage,age,Model_Camry,Model_Civic,Model_Corolla]])
        output = round(prediction[0],0)
        return render_template('index1.html', prediction_text="The approximate value of this car is {}.".format(output))

    else:
        return render_template('index1.html')

if __name__ == "__main__":
    app.run(debug=True)



