# import library
import os
import sys

import numpy as np
import pandas as pd

# import config folder
from regression_model.config.config import DATASET_PATH

# column name
column_name = ['X1', 'X2', 'X3', 'TARGET']

# create dataset
X1 = np.random.rand(10000,1)
X2 = np.random.randn(10000,1)
X3 = np.random.uniform(0,1,10000)
Error = np.random.uniform(0,0.1,10000)

# create label
Y = 10*X1+0.2*X2.reshape(X1.shape)+9*X3.reshape(X1.shape)+Error.reshape(X1.shape)

# create dataframe
data = np.concatenate((X1, X2.reshape(X1.shape), X3.reshape(X1.shape), Y.reshape(X1.shape)), axis=1)
data_test = np.concatenate((X1, X2.reshape(X1.shape), X3.reshape(X1.shape)), axis=1)
df = pd.DataFrame(data=data, columns=column_name)
df_test = pd.DataFrame(data=data_test, columns=column_name[:-1])

# export dataframe
df.to_csv(DATASET_PATH / "train.csv", index=None)
df_test.to_csv(DATASET_PATH / "test.csv", index=None)


