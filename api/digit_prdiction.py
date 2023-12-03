from flask import Flask,request
from joblib import load
import numpy as np

app = Flask(__name__)

@app.route("/compare",methods=["POST"])
def compare():
    json = request.get_json()
    image1 = json["input1"]
    image1 = [float(i) for i in image1]

    image2 = json["input2"]
    image2 = [float(i) for i in image2]
    model = load("models/svm_gamma:0.0001_C:10.joblib")
    
    image1, image2 = np.array(image1).reshape(-1,64), np.array(image2).reshape(-1,64)
    
    prediction_1, prediction_2 =  model.predict(image1), model.predict(image2)

    return str(prediction_1 == prediction_2)


@app.route("/predict/<models>",methods=["POST"])
def prediction(models):
    json = request.get_json()
    image1 = json["input"]
    image1 = [float(i) for i in image1]
    trans = load('./models/transforms.joblib')
    image1 = np.array(image1).reshape(-1,64)
    image1 = trans.transform(image1)
    
    
    if models == "svm":
        model = load(f'./models/M23CSA015_svm_gamma:0.001_C:10.joblib')
    if models == 'tree':
        model = load(f'./models/M23CSA015_tree_max_depth:10.joblib')
    if models == 'lr':
        model = load(f'./models/M23CSA015_lr_solver:newton-cholesky.joblib')
    
    return str(model.predict(image1)[0])
