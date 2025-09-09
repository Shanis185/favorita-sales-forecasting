# ==============================
# Favorita Sales Forecasting Data Preprocessing (Final Version with Boolean Fix)
# ==============================

import pandas as pd
import numpy as np

# ==============================
# Step 1: Load CSV Files
# ==============================
print("Loading data...")

train = pd.read_csv("train.csv", parse_dates=["date"])
test = pd.read_csv("test.csv", parse_dates=["date"])
stores = pd.read_csv("stores.csv")
holidays = pd.read_csv("holidays_events.csv", parse_dates=["date"])
oil = pd.read_csv("oil.csv", parse_dates=["date"])
transactions = pd.read_csv("transactions.csv", parse_dates=["date"])

print("Data loaded successfully!")

# ==============================
# Step 2: Clean Holidays Dataset
# ==============================
print("Processing holidays...")

holidays = holidays[['date', 'type', 'locale', 'locale_name', 'description', 'transferred']]

# Keep valid holidays (exclude transferred)
holidays['is_holiday'] = ((holidays['type'] == 'Holiday') & (holidays['transferred'] == False)).astype(int)

# Some dates have multiple entries → keep max holiday flag
holidays = holidays.groupby('date').agg({'is_holiday': 'max'}).reset_index()

print("Holidays processed!")

# ==============================
# Step 3: Clean Oil Prices
# ==============================
print("Processing oil prices...")

# Fill missing values (forward fill, then backward fill for any edge cases)
oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()

print("Oil prices processed!")

# ==============================
# Step 4: Transactions Aggregation
# ==============================
print("Processing transactions...")

transactions = transactions.groupby(['date', 'store_nbr']).agg({'transactions': 'sum'}).reset_index()

print("Transactions processed!")

# ==============================
# Step 5: Merge Everything
# ==============================
print("Merging datasets...")

def merge_all(df):
    # Merge store info
    df = df.merge(stores, on="store_nbr", how="left")

    # Merge transactions
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")

    # Merge oil prices
    df = df.merge(oil, on="date", how="left")

    # Merge holidays
    df = df.merge(holidays, on="date", how="left")

    return df

train = merge_all(train)
test = merge_all(test)

print("Datasets merged successfully!")

# ==============================
# Step 6: Feature Engineering
# ==============================
print("Adding time and lag features...")

def add_features(df, is_train=True):
    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

    if is_train:
        # Lag features for training only
        df = df.sort_values(by=['store_nbr','family','date'])
        df['sales_lag_1'] = df.groupby(['store_nbr','family'])['sales'].shift(1)
        df['sales_lag_7'] = df.groupby(['store_nbr','family'])['sales'].shift(7)
        df['sales_rolling_mean_7'] = (
            df.groupby(['store_nbr','family'])['sales']
            .shift(1)
            .rolling(7)
            .mean()
        )
    return df

train = add_features(train, is_train=True)
test = add_features(test, is_train=False)

print("Feature engineering completed!")

# ==============================
# Step 7: One-Hot Encoding
# ==============================
print("Applying One-Hot Encoding...")

categorical_cols = ['family', 'city', 'state', 'type']

train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

# Align train & test to have the same columns
train, test = train.align(test, join="left", axis=1, fill_value=0)

# ==============================
# Step 8: Convert Boolean Columns to Int (0/1)
# ==============================
print("Converting boolean columns to int...")

train = train.astype({col: 'int8' for col in train.select_dtypes(include=['bool']).columns})
test = test.astype({col: 'int8' for col in test.select_dtypes(include=['bool']).columns})

print("Boolean conversion completed!")

# ==============================
# Final Output
# ==============================
print("✅ Final TRAIN dataset shape:", train.shape)
print("✅ Final TEST dataset shape:", test.shape)
print(train.head(10))
