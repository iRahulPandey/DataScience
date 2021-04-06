# import library
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import sys

# import config folder
sys.path.append("./packages/regression_model/config")
sys.path.append("./packages/regression_model/processing")

from packages.regression_model.config import config
from packages.regression_model.processing import features_scaler as fs

import logging

_logger = logging.getLogger(__name__)

end_to_end_pipeline = Pipeline(
    [
        (
            "linear_scaler",
            fs.MinMaxTransformer(variables=config.UNIFORM_DISTRIBUTED_FEATURES),
        ),
        (
            "z_score_scaler",
            fs.ZTransformer(variables=config.NORMAL_DISTRIBUTED_FEATURES),
        ),
        ("Linear_model", LinearRegression()),
    ]
)
