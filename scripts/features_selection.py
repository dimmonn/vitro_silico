from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer
import numpy as np


class FeatureSelector:
    def __init__(self, model, top: int = 10):
        self.model = model
        self.top: int = top

    def select_features(self, X_train: np.ndarray, y_train: np.ndarray):
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
            explanation = explainer.explain_instance(X_train[0], self.model.predict, num_features=(
                self.top if self.top < explainer.size else explainer.as_list().size))
            top_features = explanation.as_list()
            top_feature_names = [x[0] for x in top_features]
            top_indices = [list(X_train[0]).index(feature) for feature in top_feature_names]
            X_train = X_train[:, top_indices]

        return X_train, y_train
