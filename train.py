"""
Metaflow pipeline to train house price prediction model
"""

import os
import io
import numpy as np
import pandas as pd
import functools
from metaflow import FlowSpec, step, conda_base, conda, IncludeFile
import cloudpickle
import sklearn
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.impute import MissingIndicator, SimpleImputer
from category_encoders import TargetEncoder


def pip(libraries):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            import subprocess
            import sys

            for library, version in libraries.items():
                print('Pip Install:', library, version)
                subprocess.run([sys.executable, '-m', 'pip',
                                'install', '--quiet', library + '==' + version])
            return function(*args, **kwargs)

        return wrapper

    return decorator


def script_path(filename):
    """
    A convenience function to get the absolute path to a file in this
    tutorial's directory. This allows the tutorial to be launched from any
    directory.
    """
    import os

    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


@conda_base(python='3.7', libraries={
    'pandas': '1.2.1', 
    'numpy': '1.20.0', 
    'scikit-learn': '0.24.1',
    'catboost': '0.24.4',
    'category_encoders': '2.2.2',
    'cloudpickle': '1.6.0'
})
class HousePriceFlow2(FlowSpec):
    """House Price Model"""
    train_data = IncludeFile(
        'train_data',
        help='Training File',
        default=script_path('data/train.csv'))


    def __init__(self, *args, **kwargs):
        """Constructor"""
        super().__init__(*args, **kwargs)        

    @step
    def start(self):
        """This workflow trains a model for visa transaction
        categorization.
        """
        self.trainDF = pd.read_csv(io.StringIO(self.train_data))
        self.next(self.train)

    @conda(libraries={
        
    })
    @pip(libraries={
        'xgboost': '1.3.3',
        'sklearn-pandas': '2.0.4',
        'baikal': '0.4.2',
    })
    @step
    def train(self):
        import xgboost
        from baikal import make_step, Step, Input, Model
        from baikal.steps import Stack
        from sklearn_pandas import gen_features
        import custom_transformations as ct
        from custom_transformations import DataFrameMapperStep, ConcatDataFrame, CatBoostRegressorStep        

        # these are the categorical columns in the dataset
        CATEGORICAL_COLUMNS = [
            'KitchenQual', 'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
            'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
            'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
            'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'Heating', 'HeatingQC', 'CentralAir', 'Functional', 'FireplaceQu', 'GarageType',
            'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
            'SaleCondition',
            'OverallQual', 'OverallCond',
        ]

        # these columns will be terated as a numerical columns
        NUMERICAL_COLUMNS = [
            'LotFrontage', 'LotArea', 'YearBuilt',
            'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
            'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
            'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
            '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
        ]


        # These columns have missing values and the one for which we will add missing indicator variable
        MISSING_INDICATOR = [
            'LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual',
            'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
        ]

        ## Categorical Columns for which we want One Hot Encoding
        ONEHOT_COLUMNS = [
            'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
            'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterQual',
            'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
            'SaleType', 'SaleCondition'
        ]

        ## Categorical Columns for which we want to have target encoding
        TARGET_COLUMNS = [
            'MSSubClass', 'Neighborhood', 'Exterior1st', 'Exterior2nd'
        ]

        ## Columns for that require log transformations
        LOG_COLUMNS = [
            'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
            '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'
        ]


        # Define Steps
        ElasticNetStep = make_step(ElasticNet, class_name='ElasticNet')
        ConcatStep = make_step(ConcatDataFrame, class_name='Concat')
        XGBRegressorStep = make_step(
            xgboost.XGBRegressor, class_name='XGBRegressor')
        LinearRegressionStep = make_step(
            sklearn.linear_model.LinearRegression, class_name='LinearRegression')

        # Define sklearn-pandas transformations. Here I am using gen_features utility to
        # define transformations for individual columns.
        baseProcessing = (
            gen_features(
                columns=[[x] for x in MISSING_INDICATOR],
                classes=[
                    {'class': MissingIndicator, 'features': 'all',
                        'sparse': False, 'error_on_new': False}
                ],
                prefix='na_'
            ) +
            gen_features(
                columns=LOG_COLUMNS,
                classes=[
                    {'class': FunctionTransformer, 'func': lambda x: x.astype(
                        np.float).reshape((-1, 1))},
                    {'class': SimpleImputer, 'strategy': 'mean'},
                    {'class': FunctionTransformer, 'func': np.log1p}
                ]
            ) +
            gen_features(
                columns=list(set(NUMERICAL_COLUMNS) - set(LOG_COLUMNS)),
                classes=[
                    {'class': FunctionTransformer, 'func': lambda x: x.astype(
                        np.float).reshape((-1, 1))},
                    {'class': SimpleImputer, 'strategy': 'mean'}
                ],
            ) +
            [
                # constructing new features -- age of the house
                (
                    ['YrSold', 'YearBuilt'],
                    [
                        FunctionTransformer(func=lambda x: np.clip(
                            x[:, 0] - x[:, 1], 0, 1000)),
                        FunctionTransformer(np.log1p)
                    ],
                    {'alias': 'age'}
                ),

                # constructing new feature -- remodeling age
                (
                    ['YrSold', 'YearRemodAdd'],
                    [
                        FunctionTransformer(func=lambda x: np.clip(
                            x[:, 0] - x[:, 1], 0, 1000)),
                        FunctionTransformer(np.log1p)
                    ],
                    {'alias': 'remodel_age'}
                ),

                # new feature -- total surface area
                (
                    ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF'],
                    [
                        FunctionTransformer(lambda x: np.nansum(x, axis=1)),
                        FunctionTransformer(np.log1p)
                    ],
                    {'alias': 'numerical_TotalArea'}
                )
            ]
        )

        # Since CatBoost model can handle categorical data, we don't need to encode categorical variables
        # we will simply impute missing values and let CatBoost model handle categorical data.
        catModelPreprocessing = gen_features(
            columns=CATEGORICAL_COLUMNS,
            classes=[
                {'class': FunctionTransformer, 'func': lambda x: x.astype(
                    np.object).reshape(-1, 1)},
                {'class': SimpleImputer, 'strategy': 'most_frequent'}
            ],
        )

        # for regression and XGBoost, we will need to encode categorical variables ourselfs.
        # Depending on the cardinality of the variable, I am either using one hot encoding or target encoding.
        regressionModelProcessing = (
            gen_features(
                columns=[[x] for x in ONEHOT_COLUMNS],
                classes=[
                    {'class': OneHotEncoder, 'handle_unknown': 'ignore', 'sparse': False}
                ]
            ) +
            gen_features(
                columns=[[x] for x in TARGET_COLUMNS],
                classes=[
                    {'class': TargetEncoder},
                    {'class': SimpleImputer, 'strategy': 'mean'},
                ]
            )
        )

        # Define DAG
        x = Input(name="x")
        y = Input(name='y')

        # Define feature transformations
        d0 = DataFrameMapperStep(baseProcessing, df_out=True,
                                name='BasePreprocess')(x, y)
        d1 = DataFrameMapperStep(regressionModelProcessing,
                                df_out=True, name='RegressionModelPreprocess')(x, y)
        d2 = DataFrameMapperStep(catModelPreprocessing,
                                df_out=True, name='CatModelPreprocess')(x, y)

        # Consolidate features for catboost and elasticnet
        regressionFeatures = ConcatStep(name='RegressionFeatures')([d0, d1])
        catFeatures = ConcatStep(name='CatBoostFeatures')([d0, d2])

        # Generate predictions using three different algorithms.
        m1 = ElasticNetStep(name='ElasticNet')(regressionFeatures, y)
        m2 = XGBRegressorStep(name='XGBoost')(regressionFeatures, y)
        m3 = CatBoostRegressorStep(
            name='CatBoost', cat_features=CATEGORICAL_COLUMNS, iterations=10)(catFeatures, y)

        # combine predictions from the three models
        combinedPredictions = Stack(name='CombinePredictions')([m1, m3])

        # construct an ensemble model
        ensembleModel = LinearRegressionStep()(combinedPredictions, y)
        model = Model(x, ensembleModel, y)
        model.fit(self.trainDF, self.trainDF['SalePrice'])
        self.model = {
            'model.pkl': cloudpickle.dumps(model)
        }
        self.next(self.end)
        
    @step
    def end(self):
        print("done")


if __name__ == '__main__':
    HousePriceFlow2()
