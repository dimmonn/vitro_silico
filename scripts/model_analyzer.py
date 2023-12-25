from itertools import combinations
from scipy.stats import chi2_contingency
from base_logger import VitroLogger as logger
from typing import List
import numpy as np

logger = logger()


class Analyzer:

    def __init__(self, p: float = 0.05):
        self.p = p
        self.top_features: List[np.ndarray] = []

    def append_features(self, selected_feature: np.ndarray):
        self.top_features.append(selected_feature)

    def analyze_feature_sets(self):
        flattened_features = [item.flatten() for sublist in self.top_features for tup in sublist for item in tup if
                              isinstance(item, np.ndarray)]
        tuple_flattened_features = [tuple(arr.tolist()) for arr in flattened_features]
        unique_features_per_model = [set(arr) for arr in tuple_flattened_features]
        feature_combinations = list(combinations(unique_features_per_model, 2))

        p_values = []
        for comb in feature_combinations:
            if comb[0] == comb[1]:
                continue
            model1_features, model2_features = comb[0], comb[1]
            observed_data = [
                [
                    len(model1_features.intersection(model2_features)),
                    len(model1_features - model2_features)
                ],
                [
                    len(model2_features - model1_features),
                    len(model2_features - model1_features)
                ]
            ]
            try:
                chi2, p, _, _ = chi2_contingency(observed_data)
            except Exception as e:
                logger.error(e)
            p_values.append(p)

        significant_differences = [{p, p < 0.05} for p in p_values]
        logger.info(f"Significant differences between feature sets in: {significant_differences}")
        return significant_differences
