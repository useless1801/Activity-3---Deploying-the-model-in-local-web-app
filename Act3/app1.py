import numpy as np
from flask import Flask, request, render_template
import pickle
import sklearn



app1 = Flask(__name__)
model = pickle.load(open('models/model.pkl', 'rb'))

@app1.route('/')
def home():
    return render_template('web.html')

@app1.route('/predict',methods=['POST'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)] 
    prediction = model.predict(features) 

    output = round(prediction[0], 3)

    return render_template('web.html', Result='Profit Predicted {}'.format(output))


if __name__ == "__main__":
    app1.run()
