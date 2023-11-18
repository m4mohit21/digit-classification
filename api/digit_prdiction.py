from flask import Flask, request, jsonify
from sklearn import svm
import numpy as np
from joblib import load
app = Flask(__name__)


model = load("models/svm_gamma:0.0001_C:10.joblib")
@app.route("/predict",methods = ["POST"])
def digit_predict():
 
    try:
        # Get the two image files from the request
        data = request.get_json()  # Parse JSON data from the request body
        image1 = data.get('image1', [])
        image2 = data.get('image2', [])

        # Preprocess the images and make predictions
        digit1 = predict_digit(image1)
        digit2 = predict_digit(image2)

        # Compare the predicted digits and return the result
        result = digit1 == digit2

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})
    
def predict_digit(image):
    try:
        # Convert the input list to a numpy array and preprocess for prediction
        img_array = np.array(image, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)

        return digit
    except Exception as e:
        return str(e)

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


