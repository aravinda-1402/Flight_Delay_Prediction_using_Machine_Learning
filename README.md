# Deciphering Air Travel Disruptions: A Machine Learning Approach ✈️

## Conda Environment 📦
Provided as `conda_environment.yml`. Create using: 
```bash
conda env create -f conda_environment.yml
```
(source: [Managing Conda Environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))

⚠️ **NOTE**: There may be differences in models produced with this environment as the models reported were trained in Google Colab. The reported time-series models can be found under `models` as `bilstm_model.h5`, `lstm_model.h5`, and `hybrid_model.h5`.

## Notebooks 📓

### `src/preprocessing.ipynb`
Notebook for preprocessing dataset sourced from: [Flight Delay and Cancellation Dataset 2019-2023](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023?select=flights_sample_3m.csv)

Creates processed label-encoded data and processed one-hot-encoded data. Splits data into train and test sets. Saves the data in CSV format. One-hot-encoded data was experimented with but ultimately not chosen due lacking computational power to handle a massive increase in features.

### `src/baseline_models.ipynb`
Trains and evaluates baseline regression models: Linear Regression, Decision Tree, Random Forest, Neural Network, XG Boost.

### `src/build_lstm_data.ipynb`
Creates datasets for time series models:
1. Combines `X_train` and `X_test`
2. Sorts by date-time
3. Temporally splits the data

This is necessary because datasets used for the baseline models are shuffled.

### `src/time_series_models.ipynb`
Trains the time-series models.

### `src/evaluate_time_series_models.ipynb`
Evaluates the time-series models.
