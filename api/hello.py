from flask import Flask,request
from sklearn import svm
from joblib import load


app = Flask(__name__)

@app.route("/mohit/<name>")
def hello_world(name):
    return "<p>Hello, World!</p>" + name
# @app.route("/<a>/<b>")
# def submission(n1,n2):
#     return int(n1)+ int(n2)

@app.route("/predict",methods = ["POST"])
def digit_prediction():
    js = request.get_json()
    img_1 = js["input1"]
    img_1 = [float(i) for i in img_1]
    img_2 = js["input2"]
    img_2 = [float(i) for i in img_2]
    model = load("models/svm_gamma:0.0001_C:10.joblib")
    import numpy as np 
    img_1 = np.array(img_1).reshape(-1,64)
    img_2 = np.array(img_2).reshape(-1,64)
    pred_1 = model.predict(img_1)
    pred_2 = model.predict(img_2)
    if pred_1 == pred_2:
        return "TRUE"
    return "FALSE"


