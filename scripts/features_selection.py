import logging

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from base_logger import VitroLogger

logger = VitroLogger()


class FeatureSelector:
    def __init__(self, model, top: int = 10, debug: bool = False):
        self.model = model
        self.top: int = top
        self.debug = debug

    # Example:
    #
    # X_train = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9],
    #     [10, 11, 12],
    #     [13, 14, 15],
    #     [16, 17, 18],
    #     [19, 20, 21],
    #     [22, 23, 24]
    # ])
    # importance factor [0.1, 0.3, 0.2]
    # result
    # [[ 2  3 1]
    # [ 5  6 4]
    # [ 8  9 7]
    # [11 12 0]
    # [14 15 3]
    # [17 18 6]
    # [20 21 9]
    # [23 24 2]]

    def select_features(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        if isinstance(self.model, RandomForestRegressor):
            feature_importances = np.array(self.model.feature_importances_)
            top_features_indices = np.argsort(feature_importances)[::-1][
                                   :(self.top if self.top < feature_importances.size else feature_importances.size)]
            X_train = [X_train[:, idx] for i, idx in enumerate(top_features_indices)]
        elif isinstance(self.model, LinearRegression):
            coefficients = self.model.coef_
            top_coefficient_indices = np.argsort(abs(coefficients))[::-1][
                                      :(self.top if self.top < coefficients.size else coefficients.size)]
            X_train = X_train[:, top_coefficient_indices]
        elif isinstance(self.model, SVR):
            result = permutation_importance(self.model, X_train, y_train, n_repeats=10, random_state=42)
            importance = result.importances_mean
            top_indices = np.argsort(importance)[::-1][:(self.top if self.top < importance.size else importance.size)]
            X_train = X_train[:, top_indices]
        elif isinstance(self.model, GradientBoostingRegressor):
            feature_importances = self.model.feature_importances_
            top_features_indices = np.argsort(feature_importances)[::-1][
                                   :(self.top if self.top < feature_importances.size else feature_importances.size)]
            X_train = X_train[:, top_features_indices]
        elif isinstance(self.model, MLPRegressor):
            explainer = LimeTabularExplainer(X_train, mode='regression')
            top_indices = []

            for instance in X_train:
                explanation = explainer.explain_instance(instance, self.model.predict, num_features=(
                    self.top if self.top < len(explainer.feature_names) else len(explainer.feature_names)))
                top_features = explanation.as_list()
                try:
                    top_feature_indices = [int(feature[0].split()[0]) for feature in
                                           top_features]
                except:
                    if self.debug:
                        logger.error(message=f'strange feature {str(top_features)}')
                    continue
                top_indices.append(top_feature_indices)
            X_train = X_train[:, top_indices]

        return X_train, y_train
