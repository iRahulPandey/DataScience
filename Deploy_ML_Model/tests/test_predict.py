import math

from packages.regression_model.predict import make_prediction
from packages.regression_model.processing.data_management import load_dataset
from packages.regression_model.config import config


def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name="test.csv")
    single_test_json = test_data[0:1].to_json(orient="records")

    # When
    prediction_json = make_prediction(single_test_json, "json")

    # Then
    assert prediction_json is not None
    assert isinstance(prediction_json.get("predictions")[0], float)
    assert math.ceil(prediction_json.get("predictions")[0]) == 450887
