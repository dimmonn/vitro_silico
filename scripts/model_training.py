from sklearn.ensemble import RandomForestRegressor
from feature_engineering import FeatureExtractor
from strategies import CustomFoldStrategy
from strategies import BaseStrategy


class ModelTrainer:
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy

    def train_model(self):
        return self.strategy.train_model()


if __name__ == "__main__":
    data_folder = '../data/raw_data'
    fe = FeatureExtractor()
    fn = fe.get_feature_extraction_lambda()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    trainer = ModelTrainer(strategy=CustomFoldStrategy(data_folder, model=rf_model, feature_extractor=fn))
    model, val_rmse, test_rmse = trainer.train_model()
    #   def __init__(self, data_folder, model, feature_extractor):
    print(f"Validation RMSE: {val_rmse}")
    print(f"Test RMSE: {test_rmse}")
