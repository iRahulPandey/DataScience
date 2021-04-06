import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# import config folder
import sys

sys.path.append("./regression_model/config")
from packages.regression_model.config import config
from packages.regression_model import __version__ as _version

import logging

_logger = logging.getLogger(__name__)


def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.DATASET_PATH}/{file_name}")
    return _data


def save_pipeline(*, pipeline_to_persist) -> None:
    # _version = "0.0.1"
    # Prepare versioned save file name
    save_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    save_path = config.TRAINED_MODEL_PATH / save_file_name
    
    remove_old_pipelines(files_to_keep=save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_PATH / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep) -> None:

    for model_file in config.TRAINED_MODEL_PATH.iterdir():
        if model_file.name not in [files_to_keep, "__init__.py"]:
            model_file.unlink()
