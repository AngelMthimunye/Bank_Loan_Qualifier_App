import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("Bank_Loan.pkl", "rb"))

@app.route("/")
def home():
    return render_template("loan.html")

@app.route("/predict", methods=["POST"])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    return render_template("loan.html", prediction_text="STATUS: {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)