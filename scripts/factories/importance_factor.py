from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from scripts.base_logger import VitroLogger
from collections import Counter

logger = VitroLogger()


class BaseImportanceFactor:

    def __init__(self, model, X: np.ndarray, y: np.ndarray, top: int = 3, debug: bool = False):
        self.model = model
        self.X = X
        self.y = y
        self.top = top
        self.debug = debug

    def process(self) -> dict[str, np.ndarray]:
        pass


class RandomForestRegressorImportanceFactor(BaseImportanceFactor):

    def __init__(self, model, X: np.ndarray, y: np.ndarray):
        super().__init__(model, X, y)

    def process(self) -> dict[str, np.ndarray]:
        feature_importances = np.array(self.model.feature_importances_)
        top_features_indices = np.argsort(feature_importances)[::-1][
                               :(self.top if self.top < feature_importances.size else feature_importances.size)]
        return {str(type(self).__name__): self.X[:, top_features_indices]}


class LinearRegressionImportanceFactor(BaseImportanceFactor):

    def __init__(self, model, X: np.ndarray, y: np.ndarray):
        super().__init__(model, X, y)

    def process(self) -> dict[str, np.ndarray]:
        coefficients = self.model.coef_
        top_coefficient_indices = np.argsort(abs(coefficients))[::-1][
                                  :(self.top if self.top < coefficients.size else coefficients.size)]
        return {str(type(self).__name__): self.X[:, top_coefficient_indices]}


class SVRImportanceFactor(BaseImportanceFactor):
    def __init__(self, model, X: np.ndarray, y: np.ndarray):
        super().__init__(model, X, y)

    def process(self) -> dict[str, np.ndarray]:
        result = permutation_importance(self.model, self.X, self.y, n_repeats=10, random_state=42)
        importance = result.importances_mean
        top_indices = np.argsort(importance)[::-1][:(self.top if self.top < importance.size else importance.size)]
        return {str(type(self).__name__): self.X[:, top_indices]}


class GradientBoostingRegressorImportanceFactor(BaseImportanceFactor):
    def __init__(self, model, X: np.ndarray, y: np.ndarray):
        super().__init__(model, X, y)

    def process(self) -> dict[str, np.ndarray]:
        feature_importances = self.model.feature_importances_
        top_features_indices = np.argsort(feature_importances)[::-1][
                               :(self.top if self.top < feature_importances.size else feature_importances.size)]
        return {str(type(self).__name__): self.X[:, top_features_indices]}


class MLPRegressorImportanceFactor(BaseImportanceFactor):
    def __init__(self, model, X: np.ndarray, y: np.ndarray):
        super().__init__(model, X, y)

    def process(self) -> dict[str, np.ndarray]:
        explainer = LimeTabularExplainer(self.X, mode='regression')
        top_indices = []

        for instance in self.X:
            explanation = explainer.explain_instance(instance, self.model.predict, num_features=(
                self.top if self.top < len(explainer.feature_names) else len(explainer.feature_names)))
            top_features = explanation.as_list()
            try:
                top_feature_indices = [int(feature[0].split()[0]) for feature in
                                       top_features]
            except Exception as e:
                if self.debug:
                    logger.error(message=f'strange feature {str(top_features)} failed {e}')
                continue
            top_indices.append(top_feature_indices)

        counter = Counter(map(tuple, top_indices))

        return {str(type(self).__name__): self.X[:, max(counter, key=counter.get)]}
