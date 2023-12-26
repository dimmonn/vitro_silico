from scripts.factories.importance_factor import *


class ImportanceFactorySelector:

    def __init__(self, model, X: np.ndarray, y: np.ndarray):
        self.model = model
        self.map = {
            'RandomForestRegressor': RandomForestRegressorImportanceFactor(model, X, y),
            'SVR': SVRImportanceFactor(model, X, y),
            'MLPRegressor': MLPRegressorImportanceFactor(model, X, y),
            'LinearRegression': LinearRegressionImportanceFactor(model, X, y)
        }

    def createImportanceFactor(self, key: str) -> BaseImportanceFactor:
        return self.map[key]
