from sklearn.ensemble import RandomForestRegressor
from feature_engineering import FeatureExtractor
from strategies import CustomFoldStrategy
from strategies import CrossValidationStrategy
from base_logger import VitroLogger as logger
import json
from typing import List
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from model_analyzer import Analyzer

logger = logger()


class ModelContex:
    def __init__(self):
        self.models = self.load_models_from_config('../config.json')

    def train_model(self):
        return self.strategy.train_model()

    def load_models_from_config(self, config_file) -> List:
        with open(config_file) as f:
            config = json.load(f)
        models = []
        for model_config in config['models']:
            model_type = globals()[model_config['type']]
            model_params = model_config.get('params', {})
            model_instance = model_type(**model_params)
            models.append(model_instance)

        return models


if __name__ == "__main__":
    fe = FeatureExtractor()
    fn = fe.get_feature_extraction_lambda()

    context = ModelContex()
    custom_analyzer = Analyzer()
    for model in context.models:
        strategy = CustomFoldStrategy(model, feature_extractor=fn)
        train_model = strategy.train_model()
        custom_analyzer.append_features(strategy.top_features)
        logger.info(train_model)
    custom_significant_differences = custom_analyzer.analyze_feature_sets()
    logger.info(custom_significant_differences)

    context = ModelContex()
    cross_analyzer = Analyzer()
    for model in context.models:
        strategy = CrossValidationStrategy(model, feature_extractor=fn)
        train_model = strategy.train_model()
        cross_analyzer.append_features(strategy.top_features)
        logger.info(train_model)
    cross_significant_differences = cross_analyzer.analyze_feature_sets()
    logger.info(cross_significant_differences)
