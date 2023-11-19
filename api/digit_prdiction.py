from flask import Flask, request, jsonify
from sklearn import svm
import numpy as np
from joblib import load
app = Flask(__name__)


model = load("models/svm_gamma:0.0001_C:10.joblib")
@app.route("/predict",methods = ["POST"])
def digit_predict():
    js = request.get_json()
    img_1 = js["input1"]
    img_1 = [float(i) for i in img_1]
    model = load("models/svm_gamma:0.0001_C:10.joblib")
    img_1 = np.array(img_1).reshape(-1,64)
    pred_1 = model.predict(img_1)
    return str(pred_1[0])

@app.route("/compare",methods = ["POST"])
def digit_prediction():
    js = request.get_json()
    img_1 = js["input1"]
    img_1 = [float(i) for i in img_1]
    img_2 = js["input2"]
    img_2 = [float(i) for i in img_2]
    model = load("models/svm_gamma:0.0001_C:10.joblib")
    img_1 = np.array(img_1).reshape(-1,64)
    img_2 = np.array(img_2).reshape(-1,64)
    pred_1 = model.predict(img_1)
    pred_2 = model.predict(img_2)
    if pred_1 == pred_2:
        return "TRUE"
    return "FALSE"


