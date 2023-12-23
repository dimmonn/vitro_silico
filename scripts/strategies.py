from sklearn.model_selection import KFold
from data_loading import DataLoader
import numpy as np
import configparser
from features_selection import FeatureSelector


class BaseStrategy:

    def __init__(self, model, feature_extractor):
        self.loader = DataLoader('../data/raw_data')
        self.fn = feature_extractor
        self.data = self.loader.load_and_process_data(self.fn)
        self.model = model
        self.selector = FeatureSelector(self.model, int(10))
        self.rmse_scores = []
        self.top_features = []
        self.rmse_val = None
        self.rmse_test = None
        self.strategy_name = None

    def __str__(self):
        info = f"Strategy Name: {self.strategy_name}\n"

        if self.rmse_val is not None and self.rmse_test is not None and self.top_features:
            info += f"RMSE (Validation): {self.rmse_val}, RMSE (Test): {self.rmse_test}, "
            info += f"Top Features: {self._format_feature_list(self.top_features)}"
        elif self.rmse_scores and self.top_features:
            info += f"RMSE Scores: {self.rmse_scores}, "
            info += f"Top Features: {self._format_feature_list(self.top_features)}"
        else:
            info += "No valid information available."
        return info + "\n" + "=" * 50 + "\n"

    def _format_feature_list(self, features, max_items=5):
        if len(features) <= max_items:
            return str(features)
        else:
            return f"{str(features[:max_items])[:-1]}, ... ({len(features) - max_items} more items)"

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def train_model(self):
        pass

    def calculate_rmse(self, predictions, targets):
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        return rmse

    def get_features_and_labels(self, data):
        X = list(data['features'])
        y = data['y_exp']
        return np.array(X), np.array(y)


class CustomFoldStrategy(BaseStrategy):

    def __init__(self, model, feature_extractor):
        super().__init__(model, feature_extractor)

    def train_model(self):
        train_data = self.data[self.data['fold'].isin([0, 1, 2])]
        val_data = self.data[self.data['fold'] == 3]
        test_data = self.data[self.data['fold'] == 4]
        X_train, y_train = self.get_features_and_labels(train_data)
        X_val, y_val = self.get_features_and_labels(val_data)
        X_test, y_test = self.get_features_and_labels(test_data)
        self.train(X_train, y_train)
        predictions_val = self.predict(X_val)
        self.rmse_val = self.calculate_rmse(predictions_val, y_val)
        predictions_test = self.predict(X_test)
        self.rmse_test = self.calculate_rmse(predictions_test, y_test)
        self.top_features = self.selector.select_features(X_train, y_train)
        return self


class CrossValidationStrategy(BaseStrategy):

    def __init__(self, model, feature_extractor):
        super().__init__(model, feature_extractor)

    def train_model(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]
            X_train, y_train = self.get_features_and_labels(train_data)
            X_test, y_test = self.get_features_and_labels(test_data)
            self.train(X_train, y_train)
            predictions_test = self.predict(X_test)
            rmse_test = self.calculate_rmse(predictions_test, y_test)
            self.rmse_scores.append(rmse_test)
            self.top_features.append(self.selector.select_features(X_train, y_train))
        return self
