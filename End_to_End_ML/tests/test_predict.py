import math

from regression_model.predict import make_prediction
from regression_model.processing.data_management import load_dataset
from regression_model.config import config


def test_make_single_prediction():
    # Given
    file_path = f"{config.DATASET_PATH}\{config.TRAINING_DATA_FILE}"

    # When
    subject = make_prediction(file_path)

    # Then
    assert subject is not None
    assert isinstance(subject.get("predictions")[0], float)
    assert math.ceil(subject.get("predictions")[0]) == 11
