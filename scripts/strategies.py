from sklearn.model_selection import KFold
from data_loading import DataLoader
import numpy as np
class BaseStrategy:

    def __init__(self, data_folder, model, feature_extractor):
        self.data_folder = data_folder
        self.loader = DataLoader(data_folder)
        self.fn = feature_extractor
        self.model = model

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
        return X, y


class CustomFoldStrategy(BaseStrategy):

    def train_model(self):
        data = self.loader.load_and_process_data(self.fn)
        train_data = data[data['fold'].isin([0, 1, 2])]
        val_data = data[data['fold'] == 3]
        test_data = data[data['fold'] == 4]

        X_train, y_train = self.get_features_and_labels(train_data)
        X_val, y_val = self.get_features_and_labels(val_data)
        X_test, y_test = self.get_features_and_labels(test_data)
        self.train(X_train, y_train)

        predictions_val = self.predict(X_val)
        rmse_val = self.calculate_rmse(predictions_val, y_val)

        predictions_test = self.predict(X_test)
        rmse_test = self.calculate_rmse(predictions_test, y_test)

        return self, rmse_val, rmse_test


class CrossValidationStrategy(BaseStrategy):
    def __init__(self, model_handler):
        self = model_handler

    def train_model(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []

        for train_index, test_index in kf.split(data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            X_train, y_train = self.get_features_and_labels(train_data)
            X_test, y_test = self.get_features_and_labels(test_data)
            self.train(X_train, y_train)

            predictions_test = self.predict(X_test)
            rmse_test = self.calculate_rmse(predictions_test, y_test)

            rmse_scores.append(rmse_test)
