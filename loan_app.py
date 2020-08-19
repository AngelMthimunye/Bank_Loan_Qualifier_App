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
    status = ""
    
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    if prediction == 0:
        status = "Thank you for your participation. Unfortunately you do not qualify for a Bank Loan."
    else:
        status = "Congratulations, you qualify for a bank loan."
    
    return render_template("loan.html", prediction_text="STATUS: {}".format(status))

if __name__ == "__main__":
    app.run(debug=True)