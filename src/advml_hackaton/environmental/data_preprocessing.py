"""
This module preprocesses power consumption data for Tetuan City. It includes functions for data cleaning, feature engineering, 
data visualization, and dataset splitting. The processed data is saved in a structured format for further analysis and modeling.

Main functionalities:
- Splitting and encoding date-time information.
- Generating and saving visualizations of data distributions and correlations.
- Normalizing features and splitting data into training, validation, and test sets.
- Evaluating baseline models for comparison.
- Saving processed data and metadata in a structured format.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yaml  

def split_date_time(time_str):
    """
    Splits a date-time string into its components and converts it into a pandas Timestamp object.

    Args:
        time_str (str): A string representing date and time in the format 'MM/DD/YYYY HH:MM'.

    Returns:
        pd.Timestamp: A pandas Timestamp object representing the parsed date and time.

    Raises:
        ValueError: If the day or month values are invalid.
    """

    date_part, time_part = time_str.split(' ')
    month, day, year = date_part.split('/')
    hour, minute = time_part.split(':')

    day = int(day)
    month = int(month)
    year = int(year)
    hour = int(hour)
    minute = int(minute)

    if day>31 or day<1:
        raise ValueError(f"Invalid day: {day}")

    if month > 12 or month < 1:
        raise ValueError(f"Invalid month: {month}")

    date_time = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)

    return date_time

def save_figures(df, data_path):
    """
    Generates and saves visualizations for data analysis, including correlation matrix, pair plots, and histograms.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        data_path (str): The path to the directory where figures will be saved.

    Returns:
        None
    """

    output_path = os.path.join(data_path,"figures")

    # if doesnt exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    features = ['day_time_periodic','year_time_periodic','Temperature','Humidity',
                'Zone 1 Power Consumption']

    correlation_matrix = df[features].corr()

    # generate a figure for the correlation matrix. write to data/power_consumption_tetouan/figures
    plt.figure(figsize=(10, 8))
    plt.title("Correlation Matrix")
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.savefig(os.path.join(output_path, "correlation_matrix.png"))

    plt.figure(figsize=(10, 8))
    sns.pairplot(df[features],
             diag_kind="kde", corner=True,
             plot_kws=dict(s=12, alpha=0.5))
    plt.savefig(os.path.join(output_path, "pairplot.png"))


    # show histogram of values for temperature and humidity
    plt.figure(figsize=(10, 8))
    plt.title("Temperature Distribution")
    sns.histplot(df['Temperature'], kde=True, color='blue', label='Temperature')
    plt.savefig(os.path.join(output_path, "temperature_distribution.png"))

    plt.figure(figsize=(10, 8))
    plt.title("Humidity Distribution")
    sns.histplot(df['Humidity'], kde=True, color='green', label='Humidity')
    plt.savefig(os.path.join(output_path, "humidity_distribution.png"))


# ---------------- Baselines ----------------
def evaluate_baselines(train, val, test):
    """
    Evaluates baseline models for the dataset, including a constant baseline and a linear regression model.

    Args:
        train (dict): A dictionary containing training data with keys 'X' and 'y'.
        val (dict): A dictionary containing validation data with keys 'X' and 'y'.
        test (dict): A dictionary containing test data with keys 'X' and 'y'.

    Returns:
        tuple: Mean absolute errors (MAE) for the constant baseline and linear regression model.
    """

    baseline_constant_mae = np.mean(np.abs(train['y'] - train['y'].mean()))

    lin = LinearRegression().fit(train["X"], train["y"].ravel())
    val_pred  = lin.predict(val["X"])
    test_pred = lin.predict(test["X"])

    logistic_regression_mae = np.mean(np.abs(test_pred - test['y'].ravel()))

    return baseline_constant_mae, logistic_regression_mae

def process_and_save(df, data_path, seed):
    """
    Processes the input DataFrame by normalizing features, splitting data into train/val/test sets, and saving the results.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        data_path (str): The path to the directory where processed data will be saved.
        seed (int): Random seed for reproducibility.

    Returns:
        None
    """

    # feature/target names (also stored in YAML)
    x_feature_names = ['day_time_periodic', 'year_time_periodic', 'Temperature', 'Humidity']
    y_feature_names = ['Zone 1 Power Consumption'#, 
                        #'Zone 2 Power Consumption', 
                        #'Zone 3 Power Consumption'
                        ]

    df_processed = df[x_feature_names + y_feature_names].copy()

    X = df_processed[x_feature_names].values
    y = df_processed[y_feature_names].values

    # --- normalization on full dataset (matches earlier behavior) ---
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)

    # guard against division by zero
    X_std_safe = np.where(X_std == 0, 1, X_std)
    y_std_safe = np.where(y_std == 0, 1, y_std)

    # normalize
    X_norm = (X - X_mean) / X_std_safe
    y_norm = (y - y_mean) / y_std_safe

    print(f"X normalization parameters: mean={X_mean}, std={X_std}")
    print(f"y normalization parameters: mean={y_mean}, std={y_std}")

    # ensure output directory exists
    processed_dir = os.path.join(data_path, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_norm, y_norm, test_size=0.2, random_state=seed, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, shuffle=True
    )

    baseline_constant_mae, logistic_regression_mae = evaluate_baselines({'X': X_train, 'y': y_train},
                                {'X': X_val, 'y': y_val},
                                 {'X': X_test, 'y': y_test}
                                )

    # --- dump normalization params + metadata to YAML ---
    norm_meta = {
        "x_features": x_feature_names,
        "y_features": y_feature_names,
        "x_mean": X_mean.tolist(),
        "x_std": X_std_safe.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std_safe.tolist(),
        "n_samples": int(X.shape[0]),
        "split": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "seed": int(seed),
        "normalization_scope": "full_dataset",  # change to 'train_only' if you switch strategy
        "baseline_constant_mae": float(baseline_constant_mae),
        "logistic_regression_mae": float(logistic_regression_mae)
    }
    yaml_path = os.path.join(processed_dir, "data_meta.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(norm_meta, f, sort_keys=False)

    np.savez(os.path.join(processed_dir, f"train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(processed_dir, f"val.npz"),   X=X_val,   y=y_val)
    np.savez(os.path.join(processed_dir, f"test.npz"),  X=X_test,  y=y_test)

def main(args):
    """
    Main function to preprocess the power consumption data. It loads the raw data, applies preprocessing steps, 
    generates visualizations, and saves the processed data.

    Args:
        args (argparse.Namespace): Command-line arguments containing data path and random seed.

    Returns:
        None
    """

    # Load the data
    df = pd.read_csv(os.path.join(args.data_path, "raw/Tetuan City power consumption.csv"))

    # rename power consumption cols
    df.rename(columns={
        'Zone 1 Power Consumption': 'Zone 1 Power Consumption',
        'Zone 2  Power Consumption': 'Zone 2 Power Consumption',
        'Zone 3  Power Consumption': 'Zone 3 Power Consumption'
    }, inplace=True)

    # Preprocess the DateTime column
    df['DateTime'] = df['DateTime'].apply(split_date_time)

    # Convert DateTime to sinusoidal encoding
    df['day_time_periodic'] = -np.cos(2 * np.pi * df['DateTime'].dt.hour / 24 + df['DateTime'].dt.minute / 60 / 24)
    df['year_time_periodic'] = -np.cos(2 * np.pi * df['DateTime'].dt.dayofyear / 365 + df['DateTime'].dt.hour / 24 / 365)

    save_figures(df,args.data_path)

    process_and_save(df,args.data_path,args.seed)


if __name__ == "__main__":

    # open config from yaml
    conf = yaml.safe_load(open("config/config.yaml"))

    parser = argparse.ArgumentParser(description="Power Consumption Data Preprocessing")
    parser.add_argument("--data_path", type=str, default=conf["data_path"], help="Path to the data directory")
    parser.add_argument("--seed", type=int, default=conf['seed'], help="Random seed for reproducibility")
    args = parser.parse_args()

    main(args)