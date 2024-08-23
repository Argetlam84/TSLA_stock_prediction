import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np

def predict_future(model, X_last, start_date, n_days=30):


    """
    Makes recursive predictions for the next n days using a LightGBM model.
    
    Parameters:
    - model: Trained LightGBM model
    - X_last: DataFrame containing the features of the last observation (feature set)
    - start_date: The date from which the predictions will start (datetime.date object)
    - n_days: The number of days ahead for which predictions will be made
    
    Returns:
    - future_dates: The predicted dates
    - predictions: The predictions for each day
    """
    """
    # Usage example:
    # model: Your trained LightGBM model
    # X_last: Features of the last observation
    # start_date: The starting date for predictions (e.g., datetime.date.today())
    """
    future_dates = []
    predictions = []

    expected_features = model.booster_.feature_name()
    current_X = X_last[expected_features].copy()

    for i in range(n_days):
        # Make a daily prediction
        pred = model.predict(current_X)
        predictions.append(pred[0])  # Take the first and only prediction

        # Use the predicted value for the next day's prediction
        # For example, update the 'lag_1' column and shift other lag columns
        current_X['lag_5'] = current_X['lag_4']
        current_X['lag_4'] = current_X['lag_3']
        current_X['lag_3'] = current_X['lag_2']
        current_X['lag_2'] = current_X['lag_1']
        current_X['lag_1'] = current_X['lstm_forecast']
        current_X['lstm_forecast'] = pred  # Add the new prediction
        
        # Calculate the date for the next day
        next_date = pd.to_datetime(start_date) + timedelta(days=i + 1)
        future_dates.append(next_date)
    
    return future_dates, predictions

def predict_future_simple(model, X_last, start_date, n_days=30):


    """
    Makes recursive predictions for the next n days using a LightGBM model.
    
    Parameters:
    - model: Trained LightGBM model
    - X_last: DataFrame containing the features of the last observation (feature set)
    - start_date: The date from which the predictions will start (datetime.date object)
    - n_days: The number of days ahead for which predictions will be made
    
    Returns:
    - future_dates: The predicted dates
    - predictions: The predictions for each day
    """
    """
    # Usage example:
    # model: Your trained LightGBM model
    # X_last: Features of the last observation
    # start_date: The starting date for predictions (e.g., datetime.date.today())
    """
    future_dates = []
    predictions = []

    expected_features = model.booster_.feature_name()
    current_X = X_last[expected_features].copy()

    for i in range(n_days):
        # Make a daily prediction
        pred = model.predict(current_X)
        predictions.append(pred[0])  # Take the first and only prediction

        # Use the predicted value for the next day's prediction
        # For example, update the 'lag_1' column and shift other lag columns
        current_X['lag_5'] = current_X['lag_4']
        current_X['lag_4'] = current_X['lag_3']
        current_X['lag_3'] = current_X['lag_2']
        current_X['lag_2'] = current_X['lag_1']
        current_X['lag_1'] = pred
        
        # Calculate the date for the next day
        next_date = pd.to_datetime(start_date) + timedelta(days=i + 1)
        future_dates.append(next_date)
    
    return future_dates, predictions

def lag_features(dataframe, lags, target):
    for i in lags:
        dataframe.loc[:, f"lag_{i}"] = dataframe[target].shift(i)
    
    return dataframe


def train_split(dataframe):

    train_size = int(len(dataframe) * 0.8)

    train = dataframe[:train_size]
    valid = dataframe[train_size:]

    return train, valid


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def invert_transformation(df_orig, df_forecast):
    df_fc = df_forecast.copy()
    columns = df_orig.columns
    for col in columns:
        df_fc[col] = df_orig[col].iloc[-1] + df_fc[col].cumsum()
    return df_fc

def tsla_target_eng(dataframe):
    new_df = dataframe[["date","close"]]
    new_df.sort_values(by="date", inplace=True)
    new_df.set_index("date", inplace=True)
    
    length_df = len(new_df)
    train_size = int(length_df * 0.8)
    train = new_df[:train_size]
    valid = new_df[train_size:]

    return train, valid


def separate_features(dataframe):
    dataframe_X = dataframe.drop(columns="close")
    dataframe_y = dataframe["close"]

    return dataframe_X, dataframe_y




def predict_future_lstm(model, X_last, start_date, n_days=30):
    """
    Makes recursive predictions for the next n days using an LSTM model.
    
    Parameters:
    - model: Trained LSTM model
    - X_last: Last sequence of input data with shape (1, time_steps, features)
    - start_date: The date from which the predictions will start (datetime.date object)
    - n_days: The number of days ahead for which predictions will be made
    
    Returns:
    - future_dates: The predicted dates
    - predictions: The predictions for each day
    """
    
    future_dates = []
    predictions = []

    current_X = X_last.copy()

    for i in range(n_days):
        # Make a daily prediction
        pred = model.predict(current_X)
        predictions.append(pred[0][0])  # Take the first and only prediction

        # Use the predicted value to create input for the next day
        # Assuming your input shape for LSTM is (1, time_steps, features)

        # Reshape pred to have the same number of dimensions
        pred_reshaped = np.reshape(pred, (1, 1, 1))
        
        # Append the predicted value to the sequence
        new_input = np.append(current_X[:, 1:, :], pred_reshaped, axis=1)
        current_X = new_input

        # Calculate the date for the next day
        next_date = pd.to_datetime(start_date) + timedelta(days=i + 1)
        future_dates.append(next_date)

    return future_dates, predictions


def feature_eng(dataframe):
    dataframe.sort_values(by="date", inplace=True)
    dataframe.set_index("date", inplace=True)
    dataframe.drop(columns=["label","adjClose","unadjustedVolume"], inplace=True)

    length_df = len(dataframe)
    train_size = int(length_df * 0.8)
    train, valid = dataframe[:train_size], dataframe[train_size:]

    return train, valid


