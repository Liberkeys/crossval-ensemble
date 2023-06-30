from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import Pipeline

from .custom_transformed_target_regressor import CustomTransformedTargetRegressor


class CustomPipeline(Pipeline):
    def __init__(self, pipeline, **init_param):
        """Initialize CustomPipeline instance

        Parameters
        ----------
        pipeline : sklearn Pipeline
            Pipeline to customize
        """
        super().__init__(pipeline.steps, **init_param)
        self.pipeline = pipeline
        self.preprocess_pipeline = self.pipeline[:-1]
        self.model_pipeline = self.pipeline[-1]

    def fit(self, X, y, X_valid, y_valid, **fit_params):
        """Fit a CustomPipeline instance

        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas Series
            Targets
        X_valid : pandas DataFrame
            Validation features
        y_valid : pandas Series
            Validation targets
        """
        if len(self.preprocess_pipeline) > 0:
            X_prepro = self.preprocess_pipeline.fit_transform(X)
            X_valid_prepro = self.preprocess_pipeline.transform(X_valid)
        else:
            X_prepro = X
            X_valid_prepro = X_valid
        self.model_pipeline.fit(X_prepro, y, eval_set=[(X_valid_prepro, y_valid)], **fit_params)


class CustomRegressionPipeline(CustomPipeline):
    def __init__(self, pipeline, **init_param):
        """Initialize CustomRegressionPipeline instance

        Parameters
        ----------
        pipeline : sklearn Pipeline
            The last step has to be a CustomTransformedTargetRegressor object with a regressor
            in [XGBoostRegressor, LGBMRegressor, CatBoostRegressor]
        """
        assert (
            isinstance(pipeline[-1], CustomTransformedTargetRegressor)
        ), 'The last step of a CustomPipeline object has to be a CustomTransformedTargetRegressor'
        assert (
            isinstance(pipeline[-1].regressor, XGBRegressor)
            or isinstance(pipeline[-1].regressor, CatBoostRegressor)
            or isinstance(pipeline[-1].regressor, LGBMRegressor)
        ), 'Regressor must be in [XGBRegressor, CatBoostRegressor, LGBMRegressor]'
        super().__init__(pipeline, **init_param)

    def predict(self, X):
        """Predict a fitted model

        Parameters
        ----------
        X : pandas DataFrame
            Features

        Returns
        -------
        np.array
            Predicted targets
        """
        if len(self.preprocess_pipeline) > 0:
            X_prepro = self.preprocess_pipeline.transform(X)
        else:
            X_prepro = X
        return self.model_pipeline.predict(X_prepro).flatten()


class CustomClassificationPipeline(CustomPipeline):
    def __init__(self, pipeline, **init_param):
        """Initialize CustomClassificationPipeline instance

        Parameters
        ----------
        pipeline : sklearn Pipeline
            The last step has to be a Classfier object in [XGBClassifier, LGBMClassifier, CatBoostClassifier]
        """
        assert (
            isinstance(pipeline[-1], XGBClassifier)
            or isinstance(pipeline[-1], LGBMClassifier)
            or isinstance(pipeline[-1], CatBoostClassifier)
        ), 'Classifier must be in [XGBClassifier, LGBMClassifier, CatBoostClassifier]'
        super().__init__(pipeline, **init_param)

    def predict(self, X):
        """Predict a fitted model

        Parameters
        ----------
        X : pandas DataFrame
            Features

        Returns
        -------
        np.array
            Predicted targets
        """
        if len(self.preprocess_pipeline) > 0:
            X_prepro = self.preprocess_pipeline.transform(X)
        else:
            X_prepro = X
        return self.model_pipeline.predict_proba(X_prepro)
