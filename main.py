import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle


app = Flask(__name__)
data= pd.read_csv('cleaned_dataset.csv')
pipe = pickle.load(open("lr_model.pkl","rb"))

@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0] * 1e5

    return render_template("popup.html", prediction_text = "Price : {}".format(np.round(prediction,0)))


if __name__=="__main__":
    app.run(debug=True)