import numpy as np
from scripts.factories.importance_factor import BaseImportanceFactor

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

    def select_features(self, importance_factor: BaseImportanceFactor) -> np.ndarray:
        return importance_factor.process()
