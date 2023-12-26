from scripts.features.feature_engineering import FeatureExtractor
from scripts.processor.strategies import CustomFoldStrategy
from scripts.processor.strategies import CrossValidationStrategy
from scripts.base_logger import VitroLogger
from scripts.features.model_analyzer import Analyzer
from scripts.processor.model_training import ModelContex

logger = VitroLogger()
if __name__ == "__main__":
    fe = FeatureExtractor()
    fn = fe.get_feature_descr_mol_lambda()

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
