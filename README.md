# vitro_silico

# Model Training and Evaluation

This repository contains scripts and notebooks to train and evaluate machine learning models for chemical compound prediction tasks.

## Project Structure
```
data
├── processed_data
│   └── CDK2.csv
└── raw_data
    ├── 2Z5Y.csv
    └── 5HT2.csv
scripts
├── app.log
├── base_logger.py
├── data_loading.py
├── feature_engineering.py
├── features_selection.py
├── freesolv_dglgraph.bin
├── model_analyzer.py
├── model_training.py
├── strategies.py

```

### Scripts

- **`data_loading.py`**: Loads and processes raw data.
- **`feature_engineering.py`**: Extracts features from chemical compounds using RDKit.
- **`model_training.py`**: Trains machine learning models.
- **`strategies.py`**: Contains various strategies for model training.

## Usage

### `data_loading.py`

- `DataLoader`: Loads and processes raw data.

### `feature_engineering.py`

- `FeatureExtractor`: Extracts molecular features from SMILES strings using RDKit.

### `model_training.py`

- `ModelTrainer`: Trains models using different strategies (`CustomFoldStrategy`, `CrossValidationStrategy`).

### `strategies.py`

- `BaseStrategy`: Base class with common methods for model training strategies.
- `CustomFoldStrategy`: Implements a custom fold strategy.
- `CrossValidationStrategy`: Implements cross-validation for model training.

## Installation

1. **Dependencies**: Python, RDKit, scikit-learn.
2. **Environment Setup**: Create a virtual environment.
3. **Installation**: Install dependencies and run the project.

## Data Description

- Raw data files in `raw_data` directory.
- scripts.model_training.ModelTrainer.

## Usage Examples

- scripts.model_training.ModelTrainer.

## Contributing

- UJ doctoral school.

## License and Acknowledgements

- Project license information, issued for UJ
- Acknowledgements for third-party resources or libraries.

## Future Improvements

- ADD DI framework, possibly UI.

## References

- https://docs.google.com/document/d/1oqVmAtbBk4b_U5AS13IMOFcAoYgXgn_Dz7PLa86mEow/edit.

## Conclusion

- The Vitro-Silico project brings together a suite of tools and methodologies aimed at predictive modeling in chemical compound analysis. Throughout this endeavor, we have achieved several milestones:

- Model Training and Evaluation
Training Strategies: Developed various training strategies such as CustomFoldStrategy and CrossValidationStrategy to optimize model performance under different conditions.

- Feature Engineering: Implemented feature extraction techniques using RDKit to transform SMILES strings into meaningful molecular descriptors for model input.

- Model Selection: Explored diverse machine learning models including RandomForest, SVR, MLPRegressor, and LinearRegression, tailored to suit different data characteristics.
