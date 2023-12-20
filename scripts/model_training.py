from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from data_loading import DataLoader
from feature_engineering import FeatureExtractor
import numpy as np


class ModelTrainer:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.loader = DataLoader(data_folder)
        self.fe = FeatureExtractor()
        self.fn = self.fe.get_feature_extraction_lambda()

    def train_model(self):
        data = self.loader.load_and_process_data(self.fn)
        X = list(data['features'])
        y = data['y_exp']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = self.calculate_rmse(predictions, y_test)
        return model, rmse

    def calculate_rmse(self, predictions, targets):
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        return rmse

if __name__ == "__main__":
    data_folder = '../data/raw_data'
    trainer = ModelTrainer(data_folder)
    model, rmse = trainer.train_model()
    print(f"Root Mean Squared Error (RMSE): {rmse}")
