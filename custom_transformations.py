from sklearn.base import TransformerMixin
import pandas as pd
from baikal import Step
from sklearn_pandas.dataframe_mapper import DataFrameMapper
from catboost import CatBoostRegressor

def default_params(*args, **kwargs):
    return {}

setattr(Step, 'get_params', default_params)

class ConcatDataFrame(TransformerMixin):

    def __init__(self):
        return super().__init__()

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        return pd.concat(X, axis=1)


class DataFrameMapperStep(Step, DataFrameMapper):

    def __init__(self, *args, name=None, n_outputs=1, **kwargs):
        super().__init__(*args, name=name, n_outputs=n_outputs, **kwargs)
        self._nodes = []

    def __getstate__(self):
        state = super().__getstate__()
        state["_name"] = self._name
        state["_nodes"] = self._nodes
        state["_n_outputs"] = self._n_outputs
        return state

    def __setstate__(self, state):
        self._name = state["_name"]
        self._nodes = state["_nodes"]
        self._n_outputs = state["_n_outputs"]
        super().__setstate__(state)


class CatBoostRegressorStep(Step, CatBoostRegressor):
    def __init__(self, *args, name=None, n_outputs=1, **kwargs):
        super().__init__(*args, name=name, n_outputs=n_outputs, **kwargs)
        self._nodes = []

    def __getstate__(self):
        state = super().__getstate__()
        state["_name"] = self._name
        state["_nodes"] = self._nodes
        state["_n_outputs"] = self._n_outputs
        # make sure to return the state
        return state

    def __setstate__(self, state):
        self._name = state["_name"]
        self._nodes = state["_nodes"]
        self._n_outputs = state["_n_outputs"]
        super().__setstate__(state)

    # include this otherwise you will get hashing error
    def __hash__(self):
        return hash(super().name)

    def __repr__(self):
        return self._name
