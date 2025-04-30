# Flask predict API

from flask import Flask, request, jsonify, render_template
from main import load_model_and_features
import pandas as pd

# execute load_model_and_features
model, features = load_model_and_features()
# print(features)

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            # get data from request.form
            age     = float(request.form['age'])
            gender  = request.form['gender']
            bmi     = float(request.form['bmi'])
            children = int(request.form['children'])
            smoker  = request.form['smoker']
            region  = request.form['region'] 

            # Create a DataFrame with the input data
            input_data = pd.DataFrame({
                'age'   : [age],
                'gender': [gender],
                'bmi'   : [bmi],
                'children': [children],
                'smoker': [smoker],
                'region': [region]
            })

            # Convert categorical variables to numerical
            input_data['gender'] = input_data['gender'].map({'male': 1, 'female': 0})
            input_data['smoker'] = input_data['smoker'].map({'yes': 1, 'no': 0})

            # One-hot encode region
            regions = ['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']
            for reg in regions:
                input_data[reg] = 0
            input_data[f'region_{region}'] = 1

            # Ensure columns are in the same order as during training
            input_data = input_data[features['columns']]

            # Make prediction
            prediction = model.predict(input_data)

            return render_template('result.html', prediction=round(prediction[0], 2))
        
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html', regions=['northeast', 'northwest', 'southeast', 'southwest'])

if __name__ == '__main__':
    app.run(debug=True)
