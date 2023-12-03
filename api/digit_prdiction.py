from flask import Flask,request
from joblib import load
import numpy as np

app = Flask(__name__)

@app.route("/compare",methods=["POST"])
def compare():
    json = request.get_json()
    image = json["input1"]
    image = [float(i) for i in image]

    image2 = json["input2"]
    image2 = [float(i) for i in image2]
    model = load("models/svm_gamma:0.0001_C:10.joblib")
    
    image, image2 = np.array(image).reshape(-1,64), np.array(image2).reshape(-1,64)
    
    prediction_1, prediction_2 =  model.predict(image), model.predict(image2)

    return str(prediction_1 == prediction_2)


@app.route("/predict/<models>",methods=["POST"])
def prediction(models):
    json = request.get_json()
    image = json["input"]
    image = [float(i) for i in image]
    trans = load('./models/transforms.joblib')
    image = np.array(image).reshape(-1,64)
    image = trans.transform(image)
    
    
    if models == "svm":
        model = load(f'./models/M23CSA015_svm_gamma:0.001_C:10.joblib')
    if models == 'tree':
        model = load(f'./models/M23CSA015_tree_max_depth:10.joblib')
    if models == 'lr':
        model = load(f'./models/M23CSA015_lr_solver:newton-cholesky.joblib')
    
    return str(model.predict(image)[0])
