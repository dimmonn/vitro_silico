from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from scripts.base_logger import VitroLogger

logger = VitroLogger()


class BaseImportanceFactor:

    def __init__(self, model, X: np.ndarray, y: np.ndarray, top: int = 10):
        self.model = model
        self.X = X
        self.y = y
        self.top = top

    def process(self) -> np.ndarray:
        pass


class RandomForestRegressorImportanceFactor(BaseImportanceFactor):

    def __init__(self, model, X: np.ndarray, y: np.ndarray):
        super().__init__(model, X, y)

    def process(self) -> np.ndarray:
        feature_importances = np.array(self.model.feature_importances_)
        top_features_indices = np.argsort(feature_importances)[::-1][
                               :(self.top if self.top < feature_importances.size else feature_importances.size)]
        return self.X[:, top_features_indices], self.y


class LinearRegressionImportanceFactor(BaseImportanceFactor):

    def __init__(self, model, X: np.ndarray, y: np.ndarray):
        super().__init__(model, X, y)

    def process(self) -> np.ndarray:
        coefficients = self.model.coef_
        top_coefficient_indices = np.argsort(abs(coefficients))[::-1][
                                  :(self.top if self.top < coefficients.size else coefficients.size)]
        return self.X[:, top_coefficient_indices], self.y


class SVRImportanceFactor(BaseImportanceFactor):
    def __init__(self, model, X: np.ndarray, y: np.ndarray):
        super().__init__(model, X, y)

    def process(self) -> np.ndarray:
        result = permutation_importance(self.model, self.X, self.y, n_repeats=10, random_state=42)
        importance = result.importances_mean
        top_indices = np.argsort(importance)[::-1][:(self.top if self.top < importance.size else importance.size)]
        return self.X[:, top_indices], self.y


class GradientBoostingRegressorImportanceFactor(BaseImportanceFactor):
    def __init__(self, model, X: np.ndarray, y: np.ndarray):
        super().__init__(model, X, y)

    def process(self) -> np.ndarray:
        feature_importances = self.model.feature_importances_
        top_features_indices = np.argsort(feature_importances)[::-1][
                               :(self.top if self.top < feature_importances.size else feature_importances.size)]
        return self.X[:, top_features_indices], self.y


class MLPRegressorImportanceFactor(BaseImportanceFactor):
    def __init__(self, model, X: np.ndarray, y: np.ndarray):
        super().__init__(model, X, y)

    def process(self) -> np.ndarray:
        explainer = LimeTabularExplainer(self.X, mode='regression')
        top_indices = []

        for instance in self.X:
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
            return self.X[:, top_indices], self.y
