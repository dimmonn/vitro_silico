from itertools import combinations
from scipy.stats import chi2_contingency
from scripts.base_logger import VitroLogger
from typing import List
import numpy as np

logger = VitroLogger()


class Analyzer:

    def __init__(self, p: float = 0.05):
        self.p = p
        self.top_features: List[dict[str, np.ndarray]] = []

    def append_features(self, selected_feature: List[dict[str, np.ndarray]]):
        self.top_features.append(selected_feature)

    def analyze_feature_sets(self):
        feature_combinations = combinations(self.top_features, 2)
        p_values = []
        for comb in feature_combinations:
            model1_features, model2_features = comb[0], comb[1]

            set_model1 = {tuple(row) for arr in list(model1_features[0].values()) for row in arr}
            set_model2 = {tuple(row) for arr in list(model2_features[0].values()) for row in arr}

            common_features = len(set_model1.intersection(set_model2))
            unique_model1 = len(set_model1 - set_model2)
            unique_model2 = len(set_model2 - set_model1)

            observed_data = [[common_features, unique_model1], [unique_model2, 0]]
            _, p, _, _ = chi2_contingency(observed_data)
            p_values.append(p)
        significant_differences = sum(p < self.p for p in p_values)
        logger.info(f"Significant differences between feature sets in: {significant_differences}")
        return significant_differences
