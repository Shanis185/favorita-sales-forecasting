<div align="center">

# Favorita Sales Forecasting

### Retail Sales Prediction with Time-Series Analysis and Machine Learning

A data science project exploring retail sales forecasting using historical sales, store information, transactions, oil prices, and holiday events. The project compares regression and time-series models and includes Power BI visualizations for data exploration.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

## Overview

Accurate sales forecasts help retailers plan inventory, allocate resources, and understand demand patterns. This project investigates the **Favorita grocery sales forecasting** problem using historical sales data and related business signals.

The notebook covers data preparation, feature engineering, categorical encoding, model training, evaluation, and Random Forest hyperparameter tuning. The repository also includes a Power BI report for visual exploration.

## Objectives

- Explore historical retail sales and related factors.
- Prepare sales, store, transaction, oil-price, and holiday data for modelling.
- Engineer date-based and grouped product-family features.
- Compare several regression and time-series approaches.
- Evaluate forecasts using RMSLE, RMSE, MSE, and MAE.
- Tune a Random Forest model using randomized hyperparameter search.
- Explore the data through a Power BI report.

## Models explored

| Model | Purpose |
|---|---|
| Linear Regression | Baseline regression model |
| Random Forest Regressor | Tree-based ensemble for non-linear relationships |
| Gradient Boosting Regressor | Sequential boosting model |
| ARIMA | Univariate time-series baseline |
| SARIMAX | Time-series model with exogenous features |
| XGBoost Regressor | Gradient-boosted decision trees |

The notebook reports Random Forest as the strongest model in its recorded comparison and includes a randomized search for its hyperparameters. Re-run the notebook to reproduce results in your environment; reported metrics depend on preprocessing and evaluation choices.

## Repository structure

```text
favorita-sales-forecasting/
├── Time Series Forecasting.ipynb   # Main analysis and modelling notebook
├── Untitled-1.py                   # Additional preprocessing code
├── train.csv                       # Training sales records
├── test.csv                        # Test records
├── train_processed.csv             # Processed training data
├── test_processed.csv              # Processed test data
├── stores.csv                      # Store metadata
├── oil.csv                         # Daily oil-price data
├── holidays_events.csv             # Holiday and event information
├── transactions.csv                # Store transaction counts
├── sample_submission.csv           # Example submission format
└── powerbi_visualization.pbix      # Power BI report
```

## Dataset

The files are based on the Favorita grocery sales forecasting dataset.

| File | Description |
|---|---|
| `train.csv` | Historical sales, including date, store, product family, and sales |
| `test.csv` | Records for which sales predictions can be generated |
| `stores.csv` | Store attributes such as location and store type |
| `oil.csv` | Oil prices by date |
| `holidays_events.csv` | Holidays and events |
| `transactions.csv` | Transaction counts by store and date |
| `train_processed.csv` | Preprocessed training dataset |
| `test_processed.csv` | Preprocessed test dataset |
| `sample_submission.csv` | Expected submission columns and format |

**Data note:** The CSV files may be large. Check available memory before loading the full training dataset. Use the original dataset's licence and terms when redistributing data.

## Workflow

1. **Load and inspect data** — read the sales, store, oil, transaction, and holiday datasets.
2. **Merge data sources** — combine sales with related business features.
3. **Preprocess** — convert dates, handle relevant columns, and create year/month/day features.
4. **Feature engineering** — group product families into broader categories and encode categorical features.
5. **Train/evaluate models** — compare regression and time-series approaches.
6. **Tune Random Forest** — use `RandomizedSearchCV` to search selected hyperparameters.
7. **Visualize** — explore the business data in the Power BI report.

## Getting started

### Requirements

- Python 3.9+ recommended
- Jupyter Notebook or JupyterLab
- A Python environment with the data-science dependencies below

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Replace the URL and folder name with your actual GitHub repository details.

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

The notebook uses common Python data-science and forecasting libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost jupyter
```

If the notebook imports additional packages in your environment, install those packages as well.

### 4. Launch the notebook

```bash
jupyter notebook
```

Open `Time Series Forecasting.ipynb` and run the cells in order. Ensure the CSV files are in the expected working directory, or update the file paths in the notebook.

### 5. Open the Power BI report

Open `powerbi_visualization.pbix` in Microsoft Power BI Desktop. If Power BI prompts for data-source paths, update them to match your local files.

## Evaluation metrics

| Metric | Meaning | Better result |
|---|---|---|
| RMSLE | Penalizes differences between logarithms of actual and predicted values | Lower |
| RMSE | Square root of mean squared prediction error; penalizes larger errors | Lower |
| MSE | Mean squared prediction error | Lower |
| MAE | Average absolute difference between predictions and actual values | Lower |

Metrics are only comparable when calculated on the same held-out records and target scale. RMSLE also requires non-negative values; avoid using absolute values to hide negative predictions, and handle them explicitly if needed.

## Important modelling considerations

This repository is an exploratory project, and the notebook should be reviewed before treating its metrics as a reliable benchmark or using it for production forecasts.

- **Avoid data leakage:** fit scalers and encoders on training data only, then apply the fitted transformations to validation and test data.
- **Align feature columns:** use one fitted encoder for training, validation, and test sets so that each matrix has the same columns in the same order.
- **Respect time order:** use chronological validation and avoid random splitting for time-dependent forecasting.
- **Check prediction alignment:** ensure each prediction is compared with the correct evaluation row. Review the XGBoost section in particular before relying on its metrics.
- **Model panel data appropriately:** store/product sales are multiple time series. A univariate ARIMA model over a concatenated sales column may not represent those separate series correctly.
- **Use the intended target scale:** if sales are scaled, inverse-transform predictions before interpreting errors in business units.
- **Reproduce results:** run all notebook cells from a clean kernel and verify metric calculations before reporting a best model.

These checks are important for obtaining a fair comparison between models and reliable estimates of future performance.

## Future improvements

- Build a reproducible preprocessing pipeline using `Pipeline` and `ColumnTransformer`.
- Add lagged sales, rolling averages, promotions, and calendar features where available.
- Use a time-based validation strategy and a clearly separated final test set.
- Correct and standardize metric calculations across all models.
- Save the complete preprocessing pipeline alongside the fitted model.
- Generate predictions for the provided test set and export a submission file.
- Add experiment tracking and visual comparison of model performance.

## Tools and technologies

- **Python** — core language
- **Pandas and NumPy** — data preparation and numerical operations
- **Matplotlib / Seaborn** — visualization
- **Scikit-learn** — preprocessing, regression, model evaluation, and hyperparameter search
- **Statsmodels** — ARIMA and SARIMAX models
- **XGBoost** — boosted-tree regression
- **Jupyter Notebook** — exploratory analysis
- **Power BI** — interactive business-data visualization

## Responsible use

Forecasts are estimates, not guarantees. Forecast quality can change with promotions, stock availability, local events, changes in customer behaviour, and other factors not represented in the data. Validate the model on recent data before using forecasts to support business decisions.

## License and attribution

This repository includes datasets associated with the Favorita sales forecasting task. Review the original dataset source and its terms before redistribution. Add a `LICENSE` file if you intend to publish this code under a specific open-source licence.

---

<div align="center">

**Favorita Sales Forecasting** · Retail analytics · Time-series modelling

</div>
