from sklearn.base import BaseEstimator, TransformerMixin


class MinMaxTransformer(BaseEstimator, TransformerMixin):
    """Linear scale transformer."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        self.min_dict_ = {}
        self.max_dict_ = {}
        for feature in self.variables:
            self.min_dict_[feature] = X[feature].min()
            self.max_dict_[feature] = X[feature].max()
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = (X[feature] - self.min_dict_[feature]) / (
                self.max_dict_[feature] - self.min_dict_[feature]
            )
        return X


class ZTransformer(BaseEstimator, TransformerMixin):
    """z score transformer."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        self.mean_dict_ = {}
        self.std_dict_ = {}
        for feature in self.variables:
            self.mean_dict_[feature] = X[feature].mean()
            self.std_dict_[feature] = X[feature].std()
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = (X[feature] - self.mean_dict_[feature]) / (
                self.std_dict_[feature]
            )
        return X