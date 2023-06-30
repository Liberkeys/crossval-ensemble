import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone
from .custom_pipeline import CustomRegressionPipeline, CustomClassificationPipeline

ONE_FOLD_TEST_SIZE = 0.2
RANDOM_STATE = 0


class CrossvalPipeline(Pipeline):
    def __init__(self, steps, n_folds=2, **init_param):
        """Initialize CrossvalPipeline instance

        Parameters
        ----------
        steps : list of tuples
            Pipeline steps
        n_folds : int, optional
            Number of folds in the crossval, by default 2
        """
        super().__init__(steps, **init_param)
        self.n_folds = n_folds

    def fit(self, X, y, **fit_params):
        """Fit the CrossvalPipeline object

        Parameters
        ----------
        X : pandas DataFrame
            Features
        y : pandas Series
            Targets
        """
        self.fit_params = fit_params
        y_reshaped = y.values
        self.crossval_dict, self.feature_importance_dict = self.cross_validate(
            self.pipe_class,
            self.steps,
            X,
            y_reshaped,
            self.n_folds,
            **fit_params
        )

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
        y = 0
        for i in range(1, self.n_folds + 1):
            y += self.crossval_dict[f'fold{i}']['pipeline'].predict(X) / self.n_folds
        return y

    @staticmethod
    def cross_validate(pipe_class, pipe_steps, X, y, n_folds, **fit_params):
        """Separate data in n_folds folds and fit the input pipe steps on each dataset, using validation data
        as an evaluation_set to help to fit the model (early stopping)

        Parameters
        ----------
        pipe_steps : list
            List of couples (transformer name, transformer object)
        X : pandas DataFrame
            Features
        y : pandas Series
            Targets
        n_folds : int
            Number of folds in the crossval

        Returns
        -------
        tuple
            pair of dictionnaries: crossval dict and feature importance dict
        """
        if n_folds == 1:
            warning_msg = "Warning : number of folds set to 1 : splitting train dataset into "
            warning_msg += f"train {round((1 - ONE_FOLD_TEST_SIZE) * 100)}%"
            warning_msg += f" / valid {round(ONE_FOLD_TEST_SIZE * 100)}%"
            warnings.warn(warning_msg, UserWarning)
            train_idx, valid_idx = train_test_split(
                np.arange(len(X)),
                test_size=ONE_FOLD_TEST_SIZE,
                random_state=RANDOM_STATE
                )
            kf_index_list = [(train_idx, valid_idx)]
        else:
            kf = KFold(n_splits=n_folds, random_state=RANDOM_STATE, shuffle=True)
            kf_index_list = kf.split(X)

        feature_importance = np.zeros(len(X.columns))
        crossval_dict = {}
        with tqdm(total=n_folds) as pbar:
            for i, (train_idx, valid_idx) in enumerate(kf_index_list):
                i += 1
                crossval_dict[f'fold{i}'] = {}
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]
                crossval_dict[f'fold{i}']['train_idx'] = train_idx
                crossval_dict[f'fold{i}']['valid_idx'] = valid_idx
                crossval_dict[f'fold{i}']['pipeline'] = pipe_class(clone(Pipeline(steps=pipe_steps)))
                crossval_dict[f'fold{i}']['pipeline'].fit(X_train, y_train, X_valid, y_valid, **fit_params)
                crossval_dict[f'fold{i}']['features_importance'] = crossval_dict[f'fold{i}']['pipeline']\
                    .model_pipeline.get_feature_importance()
                feature_importance += crossval_dict[f'fold{i}']['features_importance'] / n_folds
                pbar.update(1)
        columns = crossval_dict['fold1']['pipeline'].preprocess_pipeline[-1].transformed_names_
        return crossval_dict, dict(zip(columns, feature_importance))

    def get_feature_importance(self, prettifier=False):
        """Get features importance figures

        Parameters
        ----------
        prettifier : bool, optional
            False: return dictionnary with feature names as dict keys and importance as dict values
            True: return pandas DataFrame --> columns : [feature names , feature importance],
                sorted byimportance in descending order
            , by default False

        Returns
        -------
        Depends on prettifier value:
            If prettifier is 'False' --> returns a dictionnary
            If prettifier is 'True' --> returns a pandas DataFrame
            Feature importances
        """
        if not prettifier:
            return self.feature_importance_dict
        else:
            df = pd.concat([
                pd.Series(self.feature_importance_dict.keys()),
                pd.Series(self.feature_importance_dict.values())
            ], axis=1)
            df.rename(columns={0: 'feature', 1: 'feature_importance'}, inplace=True)
            return df.sort_values(by=['feature_importance'], ascending=False)


class CrossvalRegressionPipeline(CrossvalPipeline):
    def __init__(self, steps, n_folds=2, **init_param):
        """
        Parameters
        ----------
        steps : list of tuples
            The last step has to be a Classfier object in [XGBClassifier, LGBMClassifier, CatBoostClassifier]
        n_folds : int, optional
            Number of folds in the crossval, by default 2
        """
        super().__init__(steps, n_folds, **init_param)
        self.pipe_class = CustomRegressionPipeline


class CrossvalClassificationPipeline(CrossvalPipeline):
    def __init__(self, steps, n_folds=2, **init_param):
        """Initialize CrossvalPipeline instance

        Parameters
        ----------
        steps : list of tuples
            Pipeline steps :
        n_folds : int, optional
            Number of folds in the crossval, by default 2
        """
        super().__init__(steps, n_folds, **init_param)
        self.pipe_class = CustomClassificationPipeline
