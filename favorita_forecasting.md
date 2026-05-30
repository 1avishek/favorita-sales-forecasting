```python
print("hello") 
```


```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data") / "favorita-grocery-sales-forecasting"
TRAIN_PATH = DATA_DIR / "train.csv" / "train.csv"
TEST_PATH = DATA_DIR / "test.csv" / "test.csv"
ITEMS_PATH = DATA_DIR / "items.csv" / "items.csv"
STORES_PATH = DATA_DIR / "stores.csv" / "stores.csv"
TRANSACTIONS_PATH = DATA_DIR / "transactions.csv" / "transactions.csv"
OIL_PATH = DATA_DIR / "oil.csv" / "oil.csv"
HOLIDAYS_PATH = DATA_DIR / "holidays_events.csv" / "holidays_events.csv"

# === Memory Optimization Helper Function ===
def reduce_memory_usage(df):
    """ Iterate through columns and convert dtypes to save memory. """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    # print(f'  Memory usage of dataframe is {start_mem:.2f} MB') # Suppress this print inside function to avoid spam

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()

            if pd.isna(c_min) and pd.isna(c_max):
                 continue # Skip if no valid data to infer range

            if str(col_type)[:3] == 'int':
                # Check if column has floats that are effectively integers (e.g., from merges)
                # This part can be complex; simplify by relying on direct dtype loading first
                 # print(f"  Attempting to downcast integer column: {col}")
                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                     df[col] = df[col].astype(np.int8)
                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                     df[col] = df[col].astype(np.int16)
                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                     df[col] = df[col].astype(np.int32)
                 # No need to check int64 max, it's the default if larger

            else: # float
                 # print(f"  Attempting to downcast float column: {col}")
                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                     df[col] = df[col].astype(np.float16)
                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                     df[col] = df[col].astype(np.float32)
                 # No need to check float64 max, it's the default if larger


    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f'  Memory usage after optimization is: {end_mem:.2f} MB ({((start_mem - end_mem) / start_mem) * 100:.2f}% reduction)')
    return df

print("Libraries imported, paths defined, and memory helper function created.")
```


```python
#Load Dataframes (Subset of Train) and Apply Initial Processing/Memory Reduction

print("Loading data (subset of train.csv) and applying initial processing...")

# Define data types for initial load - Be explicit for potential problem columns
dtype_train_initial = {'id': 'int32', 'store_nbr': 'int16', 'item_nbr': 'int32', 'unit_sales': 'float32', 'onpromotion': 'float32'}
dtype_test_initial = {'id': 'int32', 'store_nbr': 'int16', 'item_nbr': 'int32', 'onpromotion': 'float32'}
dtype_stores_initial = {'store_nbr': 'int16', 'cluster': 'int8'} # Explicit for small int
dtype_items_initial = {'item_nbr': 'int32', 'class': 'int16', 'perishable': 'int8'} # Explicit for small int/bool
dtype_transactions_initial = {'store_nbr': 'int16', 'transactions': 'int32'}
dtype_oil_initial = {'dcoilwtico': 'float32'} # Explicitly load oil price as float32
dtype_holidays_initial = {} 


N_ROWS_TRAIN = 15000000

print(f"Loading the first {N_ROWS_TRAIN} rows of train.csv...")
train_df = pd.read_csv(TRAIN_PATH, dtype=dtype_train_initial, parse_dates=['date'], nrows=N_ROWS_TRAIN)

# Load the full test data
print("Loading test.csv (full)...")
test_df = pd.read_csv(TEST_PATH, dtype=dtype_test_initial, parse_dates=['date'])

# Load auxiliary dataframes (full) with explicit dtypes where sensible
print("Loading auxiliary dataframes...")
items_df = pd.read_csv(ITEMS_PATH, dtype=dtype_items_initial)
stores_df = pd.read_csv(STORES_PATH, dtype=dtype_stores_initial)
transactions_df = pd.read_csv(TRANSACTIONS_PATH, dtype=dtype_transactions_initial, parse_dates=['date'])
oil_df = pd.read_csv(OIL_PATH, dtype=dtype_oil_initial, parse_dates=['date']) # Oil loaded with float32
holidays_df = pd.read_csv(HOLIDAYS_PATH, dtype=dtype_holidays_initial, parse_dates=['date']) # Holidays let pandas infer types first


print("\nApplying initial processing and memory reduction...")


print("  Handling 'onpromotion' column NaNs...")
train_df['onpromotion'].fillna(0, inplace=True)
test_df['onpromotion'].fillna(0, inplace=True)

print("  Handling 'perishable' column NaNs in items_df...")
items_df['perishable'].fillna(0, inplace=True) # Assume not perishable if missing


print("  Handling 'transferred' column NaNs in holidays_df...")
holidays_df['transferred'].fillna(False, inplace=True) # Assume not transferred if missing
# Convert to bool *after* filling NaNs
holidays_df['transferred'] = holidays_df['transferred'].astype('bool')


# Apply memory reduction to all loaded dataframes after initial processing and targeted NaNs
print("  Reducing memory for all dataframes...")
train_df = reduce_memory_usage(train_df)
test_df = reduce_memory_usage(test_df)
items_df = reduce_memory_usage(items_df)
stores_df = reduce_memory_usage(stores_df)
transactions_df = reduce_memory_usage(transactions_df)
oil_df = reduce_memory_usage(oil_df) # Apply reduction to oil_df here too
holidays_df = reduce_memory_usage(holidays_df)


print("\nData loaded and initial processing/memory reduction complete.")

```


```python
#Handle Negative Sales and Log Transform Target

print("\nHandling negative sales and applying log transformation...")

# Replace negative unit_sales with 0 in the training data
train_df['unit_sales'] = train_df['unit_sales'].clip(lower=0)

# Transform target variable using log(1+x) for training
train_df['log_unit_sales'] = np.log1p(train_df['unit_sales'])

print("Sales processed and log-transformed.")
```


```python
#Process and Merge Store & Item Data (OHE 'type' on stores_df, then reduce memory)

print("\nProcessing and merging store data (OHE 'type' on stores_df)...")

stores_df['type'] = stores_df['type'].astype(str).fillna('MISSING_TYPE')


stores_df = pd.get_dummies(stores_df, columns=['type'], prefix='store_type', drop_first=True)

# Reduce memory of the processed stores_df
stores_df = reduce_memory_usage(stores_df)

# Now merge the processed stores_df (which now has store_type_X columns)
train_df = pd.merge(train_df, stores_df, on='store_nbr', how='left')
test_df = pd.merge(test_df, stores_df, on='store_nbr', how='left')

# Delete stores_df after merging
del stores_df
gc.collect()

print("Store data processed and merged.")

print("\nMerging item data...")

# Ensure categorical columns in items_df are strings before merging
items_df['family'] = items_df['family'].astype(str).fillna('MISSING_FAMILY')


train_df = pd.merge(train_df, items_df, on='item_nbr', how='left')
test_df = pd.merge(test_df, items_df, on='item_nbr', how='left')

# Delete items_df after merging
del items_df
gc.collect()

print("Item data merged.")

# Reduce memory of train_df and test_df after these merges
print("\nReducing memory after store and item merges...")
train_df = reduce_memory_usage(train_df)
test_df = reduce_memory_usage(test_df)
```


```python
#Merge Holiday Data and Reduce Memory

print("\nMerging holiday data...")

# Ensure holiday columns are suitable before creating simple flag
for col in ['type', 'locale', 'locale_name', 'description']:
    # Check if column exists before trying to fillna/astype
    if col in holidays_df.columns:
        holidays_df[col] = holidays_df[col].astype(str).fillna('MISSING')

# Create a simple holiday flag: 1 if any event, 0 otherwise
holidays_df['is_holiday'] = 1
# Group by date to handle multiple events on the same day
holidays_simple = holidays_df[['date', 'is_holiday']].drop_duplicates(subset=['date'])

# Reduce memory of holidays_simple
holidays_simple = reduce_memory_usage(holidays_simple)


train_df = pd.merge(train_df, holidays_simple, on='date', how='left')
test_df = pd.merge(test_df, holidays_simple, on='date', how='left')

# Fill dates without holidays with 0 after merging
train_df['is_holiday'].fillna(0, inplace=True)
test_df['is_holiday'].fillna(0, inplace=True)

# Delete holiday dataframes after merging
del holidays_df, holidays_simple
gc.collect()

# Reduce memory after holiday merge
print("\nReducing memory after holiday merge...")
train_df = reduce_memory_usage(train_df)
test_df = reduce_memory_usage(test_df)

print("Holiday data merged and flag created.")
```


```python
#Handle Oil Data NaNs on oil_df, Merge Oil, Reduce Memory (Simplified Fill)

print("\nHandling oil data NaNs on oil_df before merging (simplified fill)...")

# Sort oil_df by date - crucial for ffill
oil_df = oil_df.sort_values('date')

oil_df['dcoilwtico'] = oil_df['dcoilwtico'].astype(float) # Ensure it's a standard float type
oil_df['dcoilwtico'].fillna(method='ffill', inplace=True)


# Fill any remaining NaNs at the very beginning with 0
oil_df['dcoilwtico'].fillna(0, inplace=True) # Or use the mean of the non-NaN data: oil_df['dcoilwtico'].mean()

# Reduce memory of oil_df after filling NaNs (optional, as it's small)
oil_df = reduce_memory_usage(oil_df) # Keep reduction here

print("Oil data NaNs handled in oil_df. Merging oil data...")

# Now merge the cleaned oil_df into the main dataframes
train_df = pd.merge(train_df, oil_df, on='date', how='left')
test_df = pd.merge(test_df, oil_df, on='date', how='left')

# Delete oil_df after merging
del oil_df
gc.collect()

# Reduce memory of train_df and test_df after the merge
print("\nReducing memory after oil merge...")
train_df = reduce_memory_usage(train_df)
test_df = reduce_memory_usage(test_df)


print("Oil data merged.")
```


```python
#Merge Transactions Data and Handle Missing Values, Reduce Memory

print("\nMerging transactions data...")

# Aggregate transactions by date and store
transactions_agg = transactions_df.groupby(['date', 'store_nbr'])['transactions'].sum().reset_index()

# Reduce memory of transactions_agg
transactions_agg = reduce_memory_usage(transactions_agg)

train_df = pd.merge(train_df, transactions_agg, on=['date', 'store_nbr'], how='left')
test_df = pd.merge(test_df, transactions_agg, on=['date', 'store_nbr'], how='left')

# Impute missing transactions with 0 (assuming no entry means no transactions)
train_df['transactions'].fillna(0, inplace=True)
test_df['transactions'].fillna(0, inplace=True)

print("Transactions data merged.")

# Clean up transaction data to free memory
del transactions_df, transactions_agg
gc.collect()

# Reduce memory after transactions merge
print("\nReducing memory after transactions merge...")
train_df = reduce_memory_usage(train_df)
test_df = reduce_memory_usage(test_df)
```


```python
#Prepare Label Encoders and Create Features (No 'type', Add Memory Reduction Step)

print("\nPreparing Label Encoders and Creating Features...")

def prepare_categorical_encoders(train_df, test_df, categorical_cols):
    print(f"  Preparing encoders for: {categorical_cols}")
    # Ensure these columns are string/object and handle any NaNs before concat/fitting
    processed_train = train_df[categorical_cols].copy()
    processed_test = test_df[categorical_cols].copy()

    for col in categorical_cols:
        # Ensure column exists and fillna before converting to string
        if col in processed_train.columns:
             processed_train[col] = processed_train[col].fillna("MISSING").astype(str)
        if col in processed_test.columns:
             processed_test[col] = processed_test[col].fillna("MISSING").astype(str)


    # Create combined view - this might still peak memory, but should be manageable
    combined = pd.concat([processed_train, processed_test], axis=0, ignore_index=True)

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(combined[col])
        encoders[col] = le

    del combined, processed_train, processed_test # Free up memory from concatenation/copies
    gc.collect()
    return encoders

def create_features(df, encoders):
    print("  Creating date features...")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek # Monday=0, Sunday=6
    df['dayofyear'] = df['date'].dt.dayofyear
    try:
        # Use .isocalendar().week for pandas >= 2.0
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    except AttributeError: # Fallback for older pandas versions
        print("  Using .weekofyear fallback.")
        df['weekofyear'] = df['date'].dt.weekofyear.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int) # 1 for Saturday/Sunday

    print("  Applying Label Encoders...")
    for col, le in encoders.items():
        # Ensure column exists and fillna before transforming
        if col in df.columns:
             df[col] = df[col].fillna("MISSING").astype(str) # Fillna again just in case
             df[col] = le.transform(df[col])

    # Ensure dtypes are numeric after transformations - reduce_memory_usage will handle this
    # Manual type conversions are generally not needed here if reduce_memory_usage is applied

    print("  Reducing memory after adding/transforming features...")
    df = reduce_memory_usage(df) # Apply reduction here after adding/transforming columns

    return df



# Define columns for Label Encoding
categorical_cols_le = ['city', 'state', 'family'] # 'type' is excluded here (handled in Cell 4)

# Prepare and fit encoders
encoders = prepare_categorical_encoders(train_df, test_df, categorical_cols_le)
print("Label Encoders prepared.")
gc.collect() # Clean up after encoder prep

# Create features in train and test dataframes
print("\nApplying feature creation to train_df...")
train_df = create_features(train_df, encoders)
print("Applying feature creation to test_df...")
test_df = create_features(test_df, encoders)

print("\nFeature creation complete for Cell 8.")
```


```python
#Create Aggregate Time-Series Features (from Train data), Reduce Memory

print("\nCreating aggregate time-series features...")

# Example: Average sales for store on that day of week (calculated from train data)
store_day_avg_sales = train_df.groupby(['store_nbr', 'dayofweek'])['unit_sales'].mean().reset_index()

# Reduce memory of the aggregate table
store_day_avg_sales = reduce_memory_usage(store_day_avg_sales)

store_day_avg_sales.rename(columns={'unit_sales': 'store_day_avg_sales'}, inplace=True)

# Merge this feature into both train and test dataframes
train_df = pd.merge(train_df, store_day_avg_sales, on=['store_nbr', 'dayofweek'], how='left')
test_df = pd.merge(test_df, store_day_avg_sales, on=['store_nbr', 'dayofweek'], how='left')

# Fill NaNs introduced by merging aggregate features
test_df['store_day_avg_sales'].fillna(0, inplace=True) # Or use global mean/median

gc.collect()

# Reduce memory after aggregate feature merge
print("\nReducing memory after aggregate feature merge...")
train_df = reduce_memory_usage(train_df)
test_df = reduce_memory_usage(test_df)

print("Aggregate time-series features created.")
```


```python
# Define Feature Set, Split Data, and Align Columns, Final Memory Reduction

print("\nDefining features and target, splitting data, and aligning columns...")

# Define feature columns - exclude id, date, original unit_sales, and the log target
feature_cols = [col for col in train_df.columns if col not in ['id', 'date', 'unit_sales', 'log_unit_sales']]

X_train = train_df[feature_cols]
y_train = train_df['log_unit_sales'] # Target is the log-transformed sales

# Prepare test features - use the same feature columns
X_test = test_df[feature_cols] # Excludes id, date, unit_sales (NaN anyway)

# Keep test ids for submission
test_ids = test_df['id']


train_cols = list(X_train.columns)
test_cols = list(X_test.columns)

# Add columns missing in test (fill with 0)
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    print(f"  Adding missing column '{c}' to X_test (filling with 0)")
    X_test[c] = 0

# Add columns missing in train (fill with 0)
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    print(f"  Adding missing column '{c}' to X_train (filling with 0)")
    X_train[c] = 0

# Ensure exact same column order
X_test = X_test[train_cols]


print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Final check for NaNs in feature sets - should be 0 now
print("\nFinal missing values check in features:")
print("  NaNs in X_train:", X_train.isnull().sum().sum())
print("  NaNs in X_test:", X_test.isnull().sum().sum())


# Clean up large processed dataframes to save memory before training
del train_df, test_df 
gc.collect()

print("Data split and prepared for modeling.")
```


```python
#Define LightGBM Parameters

print("\nDefining LightGBM parameters...")


lgbm_params = {
    'objective': 'regression_l1', #'regression' (MSE) or 'regression_l1' (MAE) on log1p target often works well for RMSLE
    'metric': 'rmse',          # Metric used during training (RMSE on log1p)
    'n_estimators': 3000,      # Increased estimators, can use early stopping with validation set
    'learning_rate': 0.02,     # Reduced learning rate slightly
    'feature_fraction': 0.8,   # Fraction of features considered per iteration
    'bagging_fraction': 0.8,   # Fraction of data sampled per iteration
    'bagging_freq': 1,
    'lambda_l1': 0.1,          # L1 regularization
    'lambda_l2': 0.1,          # L2 regularization
    'num_leaves': 64,          # Increase from default 31 for more complexity
    'verbose': -1,             # Suppress verbose output during training
    'n_jobs': -1,              # Use all available CPU cores
    'seed': 42,                # Random seed for reproducibility
    'boosting_type': 'gbdt',   # Standard gradient boosting decision tree
}

print("LightGBM parameters defined.")
```


```python
#Train the LightGBM Model

print("\nTraining the LightGBM model...")

model = lgb.LGBMRegressor(**lgbm_params)

print("Fitting model...")

model.fit(X_train, y_train)

print("Model training complete.")

# Clean up
del X_train, y_train
gc.collect()
```


```python
# Make Predictions

print("\nMaking predictions on the test data...")


predictions_log = model.predict(X_test)


predictions = np.expm1(predictions_log)

predictions = np.maximum(0, predictions)

print("Predictions generated.")

# Clean up 
del X_test
gc.collect()
```


```python
# Cell 14: Create and Save Submission File

print("\nCreating submission file...")

# Create submission DataFrame
submission_df = pd.DataFrame({'id': test_ids, 'unit_sales': predictions})

# Ensure the format matches the required submission format
submission_df['id'] = submission_df['id'].astype(int)
# unit_sales should be float

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
print("Submission file sample:")
print(submission_df.head())

# Clean up remaining variables
del submission_df, test_ids, predictions
gc.collect()
```


```python


from sklearn.metrics import mean_squared_log_error

def rmsle(y_true, y_pred):

    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

print("\nRMSLE evaluation")
```
