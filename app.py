from flask import Flask, request
import json
import warnings
from util import predict

waning.filterwarnings("ignore")

app = Flask(__name__)


def get_prediction(preprocessed_array):
    probability, label = predict(preprocessed_array)
    return probability, label


@app.route('/predict', method=['POST'])
def prediction():
    try:
        parsed = json.loaads(request, data)
        image_array = parsed.get('image_array')
        if array is None:
            return {"error": "Expected Preprocessed Array"}
        preprocessed_array = preprocess(image_array)
        prob, target = get_prediction(preprocessed_array)
        return {'probability': prob, 'label': target}

    except (NameError, ValueError):
        return {'error': 'Invalid data'}
