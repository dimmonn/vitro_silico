# vitro_silico

# Model Training and Evaluation

This repository contains scripts and notebooks to train and evaluate machine learning models for chemical compound prediction tasks.

## Project Structure
```
data/
├── processed_data/
└── raw_data/
├── 2Z5Y.csv
├── 5HT2.csv
└── CDK2.csv
notebooks/
scripts/
├── data_loading.py
├── feature_engineering.py
├── model_evaluation.py
├── model_training.py
└── strategies.py
```

### Scripts

- **`data_loading.py`**: Contains the `DataLoader` class to load and process raw data.
- **`feature_engineering.py`**: Defines the `FeatureExtractor` class to extract features from chemical compounds.
- **`model_training.py`**: Includes the `ModelTrainer` class to train machine learning models.
- **`strategies.py`**: Holds strategies for model training like `BaseStrategy`, `CustomFoldStrategy`, and `CrossValidationStrategy`.

## Usage

### `data_loading.py`

- `DataLoader`: Loads and processes raw data to create feature sets.

### `feature_engineering.py`

- `FeatureExtractor`: Uses RDKit to extract molecular features from SMILES strings.

### `model_training.py`

- `ModelTrainer`: Trains models using different strategies like `CustomFoldStrategy` or `CrossValidationStrategy`.

### `strategies.py`

- `BaseStrategy`: Base class defining common methods for model training strategies.
- `CustomFoldStrategy`: Implements a custom fold strategy for model training.
- `CrossValidationStrategy`: Implements cross-validation for model training.

## How to Use

1. **Prepare Raw Data**: Place raw data files in the `raw_data` directory.
2. **Run `model_training.py`**: Update paths and configurations in the scripts and run `model_training.py` to train and evaluate models using different strategies.
