# vitro_silico

# Molecular Activity Prediction Project

This project aims to compare predictive models trained on experimental biological activity data measured in a laboratory with models trained on simulated molecular docking data.

## Project Overview

The project focuses on the comparison of machine learning models for predicting biological activities. It involves training various machine learning models on both experimental and simulated data to predict different biological targets. The performance of these models is evaluated using root mean squared error (RMSE) and the significance of feature importance for each model is determined.

### Data

The datasets required for this project are available at the following location:
- [Google Drive Dataset](https://drive.google.com/drive/folders/15bZ-eoozHs5Msqy3NmsSfOykM9j2_xvY?usp=sharing)

### Drug Design ML Course

The course materials relevant to this project can be found at the [MLDD23 GitHub Repository](https://github.com/gmum/mldd23). Specifically, refer to the first and third notebooks in the labs folder. The fourth notebook provides details about the type of simulation used to generate training data.

## Project Plan

The project plan involves the following steps:

1. **Training Models:** Selected machine learning models will be trained to predict activity. Each model will be trained on diverse datasets targeting various biological objectives (targets) labeled with experimental and simulated data.

2. **Comparison of Prediction Effectiveness:** The effectiveness of predictions will be measured using RMSE and recorded in a table structured as follows:

3. **Feature Selection:** Significant features for each model will be determined using interpretability models. For classical models like RF or linear regression, interpreting learned regression coefficients or occurrences in trees will be attempted. Techniques like LIME will also be explored.

4. **Comparison of Feature Importance:** The project will explore if certain features are more frequently deemed important for predicting activity in experimental data. Methods for comparing sets of such features and conducting statistical tests for significance will be employed.


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
├── base_logger.py
├── factories
│   ├── importance_factor.py
│   └── selection_factory.py
├── features
│   ├── feature_engineering.py
│   ├── features_selection.py
│   └── model_analyzer.py
├── loaders
│   └── data_loading.py
├── processor
│   ├── model_training.py
│   └── strategies.py


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
