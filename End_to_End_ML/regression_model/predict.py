import numpy as np
import pandas as pd

import sys

# import config folder
sys.path.append("./regression_model/config")
sys.path.append("./regression_model/processing")

from regression_model.processing.data_management import load_pipeline
from regression_model.config import config
from sklearn.metrics import mean_squared_error, r2_score

_version = "0.0.1"
pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(input_data):
    """Make a prediction using the saved model pipeline."""

    data = pd.read_csv(input_data)
    print(data)
    prediction = _price_pipe.predict(data[config.FEATURES])
    output = prediction
    print(output)

    results = {"predictions": output, "version": _version}
    # determine mse and rmse
    print("test mse: {}".format(int(mean_squared_error(data[config.TARGET], (results['predictions'])))))
    print("test rmse: {}".format(int(np.sqrt(mean_squared_error(data[config.TARGET], (results['predictions']))))))
    print("test r2: {}".format(r2_score(data[config.TARGET], (results['predictions']))))

    return results

if __name__ == "__main__":
    file_path = f"{config.DATASET_PATH}\{config.TRAINING_DATA_FILE}"
    #print(file_path)
    result = make_prediction(file_path)
    print(result['predictions'])