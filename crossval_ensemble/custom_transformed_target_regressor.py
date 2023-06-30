
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor


class CustomTransformedTargetRegressor(TransformedTargetRegressor):
    def __init__(self, regressor, transformer=None):
        """Initialize CustomTransformedTargetRegressor instance.
        /!/ This is needed to use sklearn TransformedTargetRegressor with eval_sets.

        Parameters
        ----------
        regressor : _type_
            Must be in [XGBoostRegressor, LGBMRegressor, CatBoostRegressor]
        transformer : _type_, optional
            Must have fit, transform and inverse_transform methods, by default None
        """
        if transformer is None:
            transformer = FunctionTransformer()
        super().__init__(regressor=regressor, transformer=transformer)
        self.transformer = transformer

    def fit(self, X, y, eval_set, **fit_params):
        """Fit a CustomTransformedTargetRegressor object

        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas Series
            Targets
        eval_set = [(X_valid, y_valid)] : list of a couple
            Evaluation set
        """
        X_valid, y_valid = eval_set[0]
        y_transformed = self.transformer.fit_transform(y.reshape(-1, 1))
        y_valid_transformed = self.transformer.transform(y_valid.reshape(-1, 1))
        self.regressor.fit(
            X,
            y_transformed,
            eval_set=[(X_valid, y_valid_transformed)],
            **fit_params
        )

    def predict(self, X):
        """Predict a fitted CustomTransformedTargetRegressor instance on X

        Parameters
        ----------
        X : pandas DataFrame
            Features

        Returns
        -------
        np.array
            Predicted targets
        """
        return self.transformer.inverse_transform(self.regressor.predict(X).reshape(-1, 1)).flatten()

    def get_feature_importance(self):
        """Return feature importance

        Returns
        -------
        np.array
            returns an array representing the contribution of each features in prediction
        """
        if hasattr(self.regressor, 'get_feature_importance'):
            return self.regressor.get_feature_importance()
        return
