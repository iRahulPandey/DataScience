import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import sys

# import config folder
sys.path.append("./packages/regression_model/config")
sys.path.append("./packages/regression_model/processing")

import pipeline
from packages.regression_model.processing.data_management import load_dataset, save_pipeline
from packages.regression_model.config import config
from sklearn.metrics import mean_squared_error, r2_score

import logging

_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # print(data.head())

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    )  # we are setting the seed here

    pipeline.end_to_end_pipeline.fit(X_train[config.FEATURES], y_train)

    pred = pipeline.end_to_end_pipeline.predict(X_test)

    # determine mse and rmse
    print("test mse: {}".format(int(mean_squared_error(y_test, (pred)))))
    print("test rmse: {}".format(int(np.sqrt(mean_squared_error(y_test, (pred))))))
    print("test r2: {}".format(r2_score(y_test, (pred))))
    print(pipeline.end_to_end_pipeline.named_steps["Linear_model"].coef_)

    _version = "0.0.1"
    _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.end_to_end_pipeline)


if __name__ == "__main__":
    run_training()
