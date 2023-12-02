from flask import Flask, request, jsonify
from sklearn import svm
import numpy as np
from joblib import load
app = Flask(__name__)


# model = load("models/svm_gamma:0.0001_C:10.joblib")

# @app.route("/predict",methods = ["POST"])
# def digit_predict():
#     js = request.get_json()
#     img_1 = js["input1"]
#     img_1 = [float(i) for i in img_1]
#     model = load("models/svm_gamma:0.0001_C:10.joblib")
#     img_1 = np.array(img_1).reshape(-1,64)
#     pred_1 = model.predict(img_1)
#     return str(pred_1[0])

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

@app.route("/predict/<model_name>",methods=["POST"])
def digit_predict(model_name):
    js = request.get_json()
    img_1 = js["input"]
    # img_1 = ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"]
    img_1 = [float(i) for i in img_1]
    if model_name == "svm":
        model = load(f'./models/M23CSA015_svm_gamma:0.001_C:10.joblib')
    elif model_name == 'tree':
        model = load(f'./models/M23CSA015_tree_max_depth:10.joblib')
    elif model_name == 'lr':
        model = load(f'./models/M23CSA015_lr_solver:newton-cholesky.joblib')

    trans = load('./models/transforms.joblib')

    import numpy as np
    img_1 = np.array(img_1).reshape(-1,64)
    img_1 = trans.transform(img_1)
    pred_1 = model.predict(img_1)
    return str(pred_1[0])


