import json
from typing import List
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


class ModelContex:
    def __init__(self):
        self.models = self.load_models_from_config('config.json')

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
