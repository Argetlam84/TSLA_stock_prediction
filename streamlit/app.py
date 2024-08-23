import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import matplotlib.pyplot as plt
import sys
import os

# Add the helpers folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import helpers_functions

# Load the dataset
df = pd.read_csv("datasets/tsla_data.csv")
df.sort_values("date", inplace=True)
df.set_index("date", inplace=True)
real_values = df["close"]

# Initialize session state for tabs and model selection
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "About"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = ""

# Tabs at the top of the page
tabs = st.tabs(["About", "Models"])

# Logic to select tabs
if st.session_state.selected_tab == "About":
    with tabs[0]:
        st.title("About")
        st.warning("This project is not INVESTMENT ADVICE!!!")
        st.write("""
        This is a Streamlit application that demonstrates the use of machine learning models to forecast TSLA Stock prices. 
        You can navigate to different pages using the Models tab to load models, visualize them, and make predictions.
        This project aims to analyze Tesla stock from various models' perspectives, make predictions, and compare them. Therefore, it focuses only on the 'close' variable of Tesla stock. The 'open' variable has been added alongside the target variable only because the VAR model requires an additional variable. Apart from this, you can observe the predictions and graphs from each model.

        If you examine the source files of the project, you will see that I built a more complex model with LSTM and CNN, but I did not use it in Streamlit because the complexity of the model did not perform well with limited data and resulted in underfitting.

        A small write-up of my findings and observations regarding this project will be coming soon, but that's all for now. Enjoy experimenting with the models! :)))
        """)

if st.session_state.selected_tab == "Models":
    with tabs[1]:
        st.title("Models Overview")
        st.write("""
        Below is a brief description of the models you can explore:
        
        - **Sarima**: Seasonal Autoregressive Integrated Moving Average, used for time series forecasting.
        - **VAR**: Vector Autoregression, suitable for forecasting multivariate time series.
        - **Prophet**: A tool from Facebook for forecasting time series data.
        - **LGBM**: Light Gradient Boosting Machine, used for classification and regression tasks.
        - **LSTM**: Long Short-Term Memory networks, a type of recurrent neural network capable of learning order dependence in sequence prediction problems.
        - **LGBM With Prophet**: Combining the power of LGBM and Prophet for enhanced forecasting.
        - **LGBM With LSTM**: Integrating LGBM with LSTM for complex time series predictions.

        Use the sidebar to select a model and start exploring its predictions.
        """)

        # Sidebar navigation for Models
        st.sidebar.title("Navigation")
        st.session_state.selected_model = st.sidebar.selectbox(
            "Select a Model", 
            ["", "Sarima", "VAR", "Prophet", "LGBM", "LGBM With Prophet", "LSTM", "LGBM With LSTM"],
            index=["", "Sarima", "VAR", "Prophet", "LGBM", "LGBM With Prophet", "LSTM", "LGBM With LSTM"].index(st.session_state.selected_model)
        )
    if st.session_state.selected_model == "Sarima":
        st.title("Sarima: Predict and Visualize")
        
        col1, col2 = st.columns(2)

        model = joblib.load("sarima_model.pkl")
        user_input = st.number_input("Input a value for forecasting:", min_value=1, max_value=100, step=1)
        
        
        if st.button("Predict"):
            forecast_end_index = model.nobs + user_input - 1
            forecast = model.get_prediction(start=model.nobs, end=forecast_end_index)
            predicted_values = forecast.predicted_mean
            
            with col1:
                st.write("Predictions:")
                st.write(predicted_values)

            with col2:
                st.write("Real Values:")
                st.write("You Can Check it TSLA Stock Price daily for real values :)")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df.index, real_values, label='Real Values', color='blue')
            ax.plot(predicted_values.index, predicted_values, label='Predicted Values', color='red')
            ax.set_title('TSLA Stock Price Forecasting')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

    elif st.session_state.selected_model == "VAR":
        st.title("VAR: Predict and Visualize")

        col1, col2 = st.columns(2)  # Moved below the title
        user_input = st.number_input("Input a value for forecasting:", min_value=1, max_value=100, step=1)
        if st.button("Predict"):
            
             model = joblib.load("var_model.pkl")
             df.index = pd.to_datetime(df.index)
             lag_order = model.k_ar

             forecast_input = df[["close", "open"]].diff().dropna().values[-lag_order:]
             forecast = model.forecast(y=forecast_input, steps=user_input)

             forecast_df = pd.DataFrame(
                 forecast, 
                 index=pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=user_input, freq='D'), 
                 columns=df[["close", "open"]].columns
             )

             forecast_df_final = helpers_functions.invert_transformation(df[["close", "open"]], forecast_df)

             with col1:
                 st.write("Predictions:")
                 st.write(forecast_df_final)

             with col2:
                 st.write("Real Values:")
                 st.write("You Can Check it TSLA Stock Price daily for real values :)") 

             fig, ax = plt.subplots(figsize=(10, 6))
             ax.plot(df.index, df["close"], label='Real Close Values', color='blue')
             ax.plot(forecast_df_final.index, forecast_df_final["close"], label='Predicted Close Values', color='red')
             ax.set_title('TSLA Stock Price Forecasting')
             ax.set_xlabel('Date')
             ax.set_ylabel('Value')
             ax.legend()
             st.pyplot(fig)

    elif st.session_state.selected_model == "Prophet":
        st.title("Prophet: Predict and Visualize")

        col1, col2 = st.columns(2)
        user_input = st.number_input("Input a value for forecasting:", min_value=1, max_value=100, step=1)
        if st.button("Predict"):
            

            model = joblib.load("prophet_model.pkl")

            df = df.reset_index()
            df.rename(columns={"date": "ds", "close": "y"}, inplace=True)
            df = df[["ds", "y"]]

            future = model.make_future_dataframe(periods=user_input)

            forecast = model.predict(future)

            forecast["ds"] = pd.to_datetime(forecast["ds"])
            df["ds"] = pd.to_datetime(df["ds"])

            with col1:
                st.write("Predictions:")
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
            with col2:
                st.write("Real Values:")
                st.write(df[["ds","y"]]) 

            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
    
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df["ds"], df["y"], label='Real Close Values', color='blue')
            ax.plot(forecast["ds"], forecast["yhat"], label='Predicted Close Values', color='red')
            ax.set_title('TSLA Stock Price Forecasting')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

    elif st.session_state.selected_model == "LGBM":
        st.title("LGBM: Predict and Visualize")
        
        user_input = st.number_input("Input a value for forecasting:", min_value=1, max_value=100, step=1)

        if st.button("Predict"):
            
            model = joblib.load("lgbm_target_model.pkl")
            X_test = pd.read_csv("datasets/lgbm_val_x.csv")
            X_test = X_test.set_index("date")
            y_pred = model.predict(X_test)

            col1, col2 = st.columns(2)

            with col1:
                st.write("Predictions:")
                st.write(y_pred)

            y_test = pd.read_csv("datasets/lgbm_val_y.csv")
            y_test = y_test.set_index("date")

            with col2:
                st.write("Real Values:")
                st.write(y_test)

            mae = mean_absolute_error(y_test, y_pred)
            st.write("MAE:")
            st.write(mae)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df.index, df["close"], label='Real Close Values', color='blue')
            ax.plot(y_test.index, y_pred, label='Predicted Close Values', color='red')
            ax.set_title('TSLA Stock Price Forecasting')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            X_last = X_test[-1:]
            start_date = X_test.index.max()

            future_dates, predictions = helpers_functions.predict_future_simple(model, X_last, start_date, n_days=user_input)
            st.write("Forecasted Values")

            col1, col2 = st.columns(2)

            with col1:
                st.write(future_dates)
            with col2:
                st.write(predictions)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(future_dates, predictions, label='Predicted Close Values', color='red')
            ax.set_title('TSLA Stock Price Forecasting')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

    elif st.session_state.selected_model == "LGBM With Prophet":
        st.title("LGBM With Prophet: Predict and Visualize")

        col1, col2 = st.columns(2)

        if st.button("Predict"):

            model = joblib.load("lgbm_w_prop_model.pkl")
            X_test = pd.read_csv("datasets/lgbm_prop_val_x.csv")
            X_test = X_test.set_index("ds")

            y_pred = model.predict(X_test)

            real_values = pd.read_csv("datasets/lgbm_prop_val_y.csv")
            real_values.set_index("ds", inplace=True)

            with col1:
                st.write("LGBM Predictions:")
                st.write(y_pred)
            
            with col2:
                st.write("Real Values:")
                st.write(real_values)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df.index, df["close"], label='Real Close Values', color='blue')
            ax.plot(X_test.index, y_pred, label='Predicted Close Values', color='red')
            ax.set_title('TSLA Stock Price Forecasting')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)
            

    elif st.session_state.selected_model == "LSTM":
        st.title("LSTM: Predict and Visualize")

        user_input = st.number_input("Input a value for forecasting:", min_value=1, max_value=100, step=1)

        if st.button("Predict"):

            model = joblib.load("lstm_model.pkl")
            X_test = pd.read_csv("datasets/lstm_val_series.csv")
            X_test = X_test.iloc[:, 1:]
            print(X_test.columns)
            X_np = X_test.values

            # I use 6 because I knew the window_size so this number represent to window_size

            X_last = X_np[-6:]
            print(X_last)
            print(X_last.shape)
            X_last = X_last.reshape((1,6))
            print(X_last.shape)
            X_last = X_last.reshape((1,6,1))

            start_date = df.index.max()

            future_dates, predictions = helpers_functions.predict_future_lstm(model, X_last, start_date, n_days=user_input)

            col1, col2 = st.columns(2)

            with col1:
                st.write(future_dates)
            
            with col2:
                st.write(predictions)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(future_dates, predictions, label='Predicted Close Values', color='red')
            ax.set_title('TSLA Stock Price Forecasting')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)
            

    elif st.session_state.selected_model == "LGBM With LSTM":
        st.title("LGBM With LSTM: Predict and Visualize")

        user_input = st.number_input("Input a value for forecasting:", min_value=1, max_value=100, step=1)

        if st.button("Predict"):
            model = joblib.load("lgmb_lstm.pkl")
 
            X_test = pd.read_csv("datasets/lstm_lgbm_val_x.csv")
            X_test.set_index("date", inplace=True)

            y_pred = model.predict(X_test)

            y_real = pd.read_csv("datasets/lstm_lgbm_val_y.csv")
            y_real.set_index("date", inplace=True)

            col1, col2 = st.columns(2)

            with col1:
                st.write(y_pred)
            
            with col2:
                st.write(y_real)

            mae = mean_absolute_error(y_real, y_pred)

            st.write(mae)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df.index, df["close"], label='Real Close Values', color='blue')
            ax.plot(X_test.index, y_pred, label='Predicted Close Values', color='red')
            ax.set_title('TSLA Stock Price Forecasting')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)

            start_date = df.index.max()

            X_last = X_test.iloc[-1:]

            future_dates, predictions = helpers_functions.predict_future(model, X_last, start_date, n_days=user_input)

            col3, col4 = st.columns(2)

            with col3:
                st.write(future_dates)
            with col4:
                st.write(predictions)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(future_dates, predictions, label='Predicted Close Values', color='red')
            ax.set_title('TSLA Stock Price Forecasting')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            st.pyplot(fig)


def select_tab(tab_name):
    st.session_state.selected_tab = tab_name

tabs[0].button("About", on_click=select_tab, args=("About",))
tabs[1].button("Models", on_click=select_tab, args=("Models",))

            