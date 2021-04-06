# import library
import pathlib as pl
import sys

# path to the root folder
root_path = r".\packages\regression_model"

# path to package root
PACKAGE_ROOT = pl.Path(root_path).resolve()

# path to dataset
DATASET_PATH = PACKAGE_ROOT / "datasets"
print(DATASET_PATH)

# path to trained model
TRAINED_MODEL_PATH = PACKAGE_ROOT / "trained_model"

# pipline name
PIPELINE_NAME = "linear_regression"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v_"

# file name
TRAINING_DATA_FILE = "train.csv"
TESTING_DATA_FILE = "test.csv"

# feature name
FEATURES = ["X1", "X2", "X3"]
TARGET = "TARGET"

# uniform distributed features
UNIFORM_DISTRIBUTED_FEATURES = ["X1", "X3"]

# normal distributed feature
NORMAL_DISTRIBUTED_FEATURES = ["X2"]