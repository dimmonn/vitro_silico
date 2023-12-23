from sklearn.ensemble import RandomForestRegressor
from feature_engineering import FeatureExtractor
from strategies import CustomFoldStrategy
from strategies import CrossValidationStrategy
from strategies import BaseStrategy
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from base_logger import VitroLogger as logger

logger = logger()


class ModelTrainer:
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy

    def train_model(self):
        return self.strategy.train_model()


if __name__ == "__main__":
    fe = FeatureExtractor()
    fn = fe.get_feature_extraction_lambda()

    rf_model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model2 = SVR(kernel='rbf', C=1.0, gamma='scale')
    rf_model3 = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam')
    rf_model4 = LinearRegression()

    trainer1_custom = ModelTrainer(strategy=CustomFoldStrategy(model=rf_model1, feature_extractor=fn))
    trainer1_cross_validation = ModelTrainer(strategy=CrossValidationStrategy(model=rf_model1, feature_extractor=fn))

    trainer2_custom = ModelTrainer(strategy=CustomFoldStrategy(model=rf_model2, feature_extractor=fn))
    trainer2_cross_validation = ModelTrainer(strategy=CrossValidationStrategy(model=rf_model2, feature_extractor=fn))

    trainer3_custom = ModelTrainer(strategy=CustomFoldStrategy(model=rf_model3, feature_extractor=fn))
    trainer3_cross_validation = ModelTrainer(strategy=CrossValidationStrategy(model=rf_model3, feature_extractor=fn))

    trainer4_custom = ModelTrainer(strategy=CustomFoldStrategy(model=rf_model4, feature_extractor=fn))
    trainer4_cross_validation = ModelTrainer(strategy=CrossValidationStrategy(model=rf_model4, feature_extractor=fn))

    logger.info(trainer1_custom.train_model())
    logger.info(trainer1_cross_validation.train_model())
    logger.info(trainer2_custom.train_model())
    logger.info(trainer2_cross_validation.train_model())
    logger.info(trainer3_custom.train_model())
    logger.info(trainer3_cross_validation.train_model())
    logger.info(trainer4_custom.train_model())
    logger.info(trainer4_cross_validation.train_model())
