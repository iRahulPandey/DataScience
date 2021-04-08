import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# from packages.regression_model.config import config
import joblib

# import pickle

app = Flask(__name__)

# load model
# _version = "0.0.1"
# file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
file_path = (
    r".\packages\regression_model\trained_model\linear_regression_output_v_0.0.1.pkl"
)
model = joblib.load(filename=file_path)
# model = pickle.load(open(file_path, "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [float(x) for x in request.form.values()]

    final_features = [np.array(int_features)]
    data = pd.DataFrame(final_features, columns=["X1", "X2", "X3"])
    prediction = model.predict(data)
    output = round(prediction[0], 2)

    return render_template(
        "index.html", prediction_text="Predicted value is {}".format(output)
    )


if __name__ == "__main__":
    app.run(debug=True)