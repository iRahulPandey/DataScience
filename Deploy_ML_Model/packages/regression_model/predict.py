import numpy as np
import pandas as pd

import sys

# import config folder
sys.path.append("./packages/regression_model/config")
sys.path.append("./packages/regression_model/processing")

from packages.regression_model.processing.data_management import load_pipeline, load_dataset
from packages.regression_model.config import config
from sklearn.metrics import mean_squared_error, r2_score

_version = "0.0.1"
pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_load_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(input_data, data_type = ""):
    """Make a prediction using the saved model pipeline."""
    if data_type == "json":
        print(input_data)
        data = pd.read_json(input_data, orient='records')
    else:    
        data = pd.DataFrame(input_data)
    prediction = _load_pipe.predict(data[config.FEATURES])
    output = prediction
    results = {"predictions": output, "version": _version}
    return results


if __name__ == "__main__":
    test_data = load_dataset(file_name='test.csv')
    single_test_json = test_data[0:1].to_json(orient='records')
    single_test = test_data[0:1]
    print(single_test)
    #print(pd.DataFrame(single_test_json))
    prediction_json = make_prediction(single_test_json, "json")
    print(prediction_json)
    prediction_df = make_prediction(single_test, "")
    print(prediction_df)

