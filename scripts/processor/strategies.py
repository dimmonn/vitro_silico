from sklearn.model_selection import KFold
from scripts.loaders.data_loading import DataLoader
from scripts.features.features_selection import FeatureSelector
from scripts.factories.importance_factor import *
from scripts.factories.selection_factory import ImportanceFactorySelector


class BaseStrategy:

    def __init__(self, model, feature_extractor):
        self.loader = DataLoader('data/raw_data')
        self.fn = feature_extractor
        self.data = self.loader.load_and_process_data(self.fn)
        self.model = model
        self.selector = FeatureSelector(self.model, int(10))
        self.rmse_scores_exp = []
        self.rmse_scores_sim = []
        self.top_features_exp = []
        self.top_features_sim = []

        self.rmse_val_exp = None
        self.rmse_test_exp = None
        self.rmse_val_sim = None
        self.rmse_test_sim = None
        self.strategy_name = f'{type(self).__name__} {type(model).__name__}'

    def flatten_nested_arrays(self, data):
        if isinstance(data, np.ndarray):
            return [data]
        elif isinstance(data, (list, tuple)):
            flattened = []
            for item in data:
                flattened.extend(self.flatten_nested_arrays(item))
            return flattened
        else:
            return [data]

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
        X = data['features']
        y_exp = data['y_exp']
        y_sim = data['y_sim']
        return np.array(X.tolist()), np.array(y_exp.tolist()), np.array(y_sim.tolist())


class CustomFoldStrategy(BaseStrategy):

    def __init__(self, model, feature_extractor):
        super().__init__(model, feature_extractor)

    def __str__(self):
        info = f"Strategy Name: {self.strategy_name}\n"
        info += f"RMSE (Validation): exp: {self.rmse_val_exp}, sim: {self.rmse_val_sim} RMSE (Test): exp: {self.rmse_test_exp}, sim: {self.rmse_test_sim}, "
        info += f"Top Features: exp {self._format_feature_list(self.top_features_exp)}"
        info += f"Top Features: sim {self._format_feature_list(self.top_features_sim)}"
        return info + "\n" + "=" * 50 + "\n"

    def train_model(self):
        train_data = self.data[self.data['fold'].isin([0, 1, 2])]
        val_data = self.data[self.data['fold'] == 3]
        test_data = self.data[self.data['fold'] == 4]
        X_train, y_train_exp, y_train_sim = self.get_features_and_labels(train_data)
        X_val, y_val_exp, y_val_sim = self.get_features_and_labels(val_data)
        X_test, y_test_exp, y_test_sim = self.get_features_and_labels(test_data)
        self.importance_factory_exp = ImportanceFactorySelector(self.model, X_train,
                                                                y_train_exp).createImportanceFactor(
            str(type(self.model).__name__).rsplit('.', 1)[-1]
        )
        self.importance_factory_sim = ImportanceFactorySelector(self.model, X_train,
                                                                y_train_sim).createImportanceFactor(
            str(type(self.model).__name__).rsplit('.', 1)[-1]
        )

        self.train(X_train, y_train_exp)
        predictions_val_exp = self.predict(X_val)
        self.rmse_val_exp = self.calculate_rmse(predictions_val_exp, y_val_exp)
        predictions_test_exp = self.predict(X_test)
        self.rmse_test_exp = self.calculate_rmse(predictions_test_exp, y_test_exp)
        selected_exp = self.selector.select_features(self.importance_factory_exp)
        self.top_features_exp.append(selected_exp)

        self.train(X_train, y_train_sim)
        predictions_val_sim = self.predict(X_val)
        self.rmse_val_sim = self.calculate_rmse(predictions_val_sim, y_val_sim)
        predictions_test_sim = self.predict(X_test)
        self.rmse_test_sim = self.calculate_rmse(predictions_test_sim, y_test_sim)
        selected_sim = self.selector.select_features(self.importance_factory_sim)
        self.top_features_sim.append(selected_sim)
        return self


class CrossValidationStrategy(BaseStrategy):

    def __init__(self, model, feature_extractor):
        super().__init__(model, feature_extractor)

    def __str__(self):
        info = f"Strategy Name: {self.strategy_name}\n"
        info += f"RMSE (Test): exp: {self.rmse_scores_exp}, sim: {self.rmse_scores_sim}, "
        info += f"Top Features: exp {self._format_feature_list(self.top_features_exp)}"
        info += f"Top Features: sim {self._format_feature_list(self.top_features_sim)}"
        return info + "\n" + "=" * 50 + "\n"

    def train_model(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]
            X_train, y_train_exp, y_train_sim = self.get_features_and_labels(train_data)
            X_test, y_test_exp, y_test_sim = self.get_features_and_labels(test_data)

            self.importance_factory_exp = ImportanceFactorySelector(self.model, X_train,
                                                                    y_train_exp).createImportanceFactor(
                str(type(self.model).__name__).rsplit('.', 1)[-1]
            )
            self.importance_factory_sim = ImportanceFactorySelector(self.model, X_train,
                                                                    y_train_sim).createImportanceFactor(
                str(type(self.model).__name__).rsplit('.', 1)[-1]
            )

            self.train(X_train, y_train_exp)
            predictions_test_exp = self.predict(X_test)
            rmse_test_exp = self.calculate_rmse(predictions_test_exp, y_test_exp)
            self.rmse_scores_exp.append(rmse_test_exp)
            selected_exp = self.selector.select_features(self.importance_factory_exp)
            self.top_features_exp.append(selected_exp)

            self.train(X_train, y_train_sim)
            predictions_test_sim = self.predict(X_test)
            rmse_test_sim = self.calculate_rmse(predictions_test_sim, y_test_sim)
            self.rmse_scores_sim.append(rmse_test_sim)
            selected_sim = self.selector.select_features(self.importance_factory_sim)
            self.top_features_sim.append(selected_sim)

        return self
