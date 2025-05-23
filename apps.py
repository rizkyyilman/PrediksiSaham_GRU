# apps.py - versi perbaikan menyeluruh

import pandas as pd
import numpy as np
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, request
import plotly.graph_objs as go
import plotly.offline as pyo
from datetime import timedelta, datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Fungsi MAPE
# ---------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ---------------------------
# Fungsi muat dan bersihkan data
# ---------------------------
def load_and_process_data():
    st.write("Memuat dan memproses data...")
    files = {
        "BBRI": "bri_5tahun_edited.csv",
        "BBNI": "bni_5tahun_edited.csv",
        "BBCA": "bca_5tahun_edited.csv"
    }
    df_list = []
    scalers = {}
    scalers_close = {}

    for stock, filename in files.items():
        st.write(f"Memuat file: {filename}")
        if not os.path.exists(filename):
            st.error(f"File {filename} tidak ditemukan di {os.getcwd()}")
            continue

        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()
        df.drop(columns=[col for col in ['Name', 'Dividends', 'Stock Splits'] if col in df.columns], inplace=True)
        df['Stock'] = stock
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)
        df_list.append(df)

    if not df_list:
        return None, None, None, None, None

    df_combined = pd.concat(df_list, axis=0).sort_values(by=["Date", "Stock"])
    df_combined.ffill(inplace=True)
    df_combined.drop_duplicates(subset=["Date", "Stock"], keep="last", inplace=True)

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for stock, group in df_combined.groupby("Stock"):
        scaler = MinMaxScaler()
        group[numeric_cols] = scaler.fit_transform(group[numeric_cols].astype(float))
        scalers[stock] = scaler
        df_combined.loc[group.index, numeric_cols] = group[numeric_cols]

        scaler_close = MinMaxScaler()
        close_data = group[['Close']].astype(float)
        scaler_close.fit(close_data)
        scalers_close[stock] = scaler_close

    return df_combined, numeric_cols, scalers, df_list, scalers_close

# ---------------------------
# Time series windowing
# ---------------------------
def create_time_series(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step, :])
        y.append(data[i+time_step, 3])
    return np.array(X), np.array(y)

# ---------------------------
# Model GRU
# ---------------------------
def build_gru_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(128, return_sequences=True),
        Dropout(0.3),
        GRU(64),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# ---------------------------
# Prediksi masa depan
# ---------------------------
def predict_future(model, last_sequence, scaler, time_step, future_steps, numeric_cols):
    current_sequence = last_sequence.copy()
    predictions = []

    changes = np.diff(current_sequence[:, 3])
    avg_change = np.mean(changes)
    std_dev = np.std(changes)

    for step in range(future_steps):
        x_input = current_sequence.reshape((1, time_step, len(numeric_cols)))
        base_pred = model.predict(x_input, verbose=0)[0, 0]

        # Perbesar noise dan tren
        trend_adjustment = avg_change * (1 + step / future_steps) * 2
        seasonal_wave = 0.03 * np.sin(2 * np.pi * step / 30)
        dynamic_noise = np.random.normal(0, std_dev * 1.5)

        pred = base_pred + trend_adjustment + seasonal_wave + dynamic_noise
        pred = np.clip(pred, 0, 1)

        predictions.append(pred)

        next_sequence = current_sequence[1:, :].copy()
        new_row = current_sequence[-1, :].copy()
        new_row[3] = pred

        for j in [0, 1, 2]:
            drift = np.random.normal(0, 0.01)
            new_row[j] = np.clip(new_row[j] * (1 + drift), 0, 1)
        new_row[4] = np.clip(new_row[4] * (1 + np.random.normal(0, 0.03)), 0, 1)

        current_sequence = np.vstack([next_sequence, new_row])

    dummy = np.zeros((len(predictions), len(numeric_cols)))
    dummy[:, 3] = predictions
    future_prices = scaler.inverse_transform(dummy)[:, 3]
    return future_prices

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.title("📈 Prediksi Harga Saham Perbankan (BBRI, BBNI, BBCA)")
    st.write("Aplikasi ini menggunakan model GRU untuk memprediksi harga penutupan saham selama 6 bulan ke depan.")
    st.caption(f"Model dilatih pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df_combined, numeric_cols, scalers, df_list, scalers_close = load_and_process_data()
    if df_combined is None:
        st.stop()

    future_steps = 180
    time_step = 60

    for stock in ['BBRI', 'BBNI', 'BBCA']:
        st.subheader(f"🏦 {stock}")
        stock_df = df_combined[df_combined['Stock'] == stock]

        if len(stock_df) < time_step + 1:
            st.warning(f"Data {stock} kurang dari {time_step + 1} baris.")
            continue

        data = stock_df[numeric_cols].values
        X, y = create_time_series(data, time_step)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = build_gru_model((X_train.shape[1], X_train.shape[2]))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=64, callbacks=[early_stop], verbose=0)

        st.success("Model GRU dilatih!")

        predictions = model.predict(X, verbose=0).flatten()
        scaler = scalers[stock]

        dummy_y = np.zeros((len(y), len(numeric_cols)))
        dummy_y[:, 3] = y

        dummy_pred = np.zeros((len(predictions), len(numeric_cols)))
        dummy_pred[:, 3] = predictions

        actual_close = scaler.inverse_transform(dummy_y)[:, 3]
        predicted_close = scaler.inverse_transform(dummy_pred)[:, 3]

        rmse = np.sqrt(mean_squared_error(actual_close, predicted_close))
        mae = mean_absolute_error(actual_close, predicted_close)
        mape = mean_absolute_percentage_error(actual_close, predicted_close)

        st.write(f"- RMSE: {rmse:.2f} IDR")
        st.write(f"- MAE: {mae:.2f} IDR")
        st.write(f"- MAPE: {mape:.2f}%")

        fig_loss = px.line(pd.DataFrame(history.history), y=['loss', 'val_loss'], title=f'{stock} - Training vs Validation Loss')
        st.plotly_chart(fig_loss, use_container_width=True)

        last_sequence = data[-time_step:]
        future_prices = predict_future(model, last_sequence, scaler, time_step, future_steps, numeric_cols)

        last_date = stock_df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps + 1)]

        original_close = scaler.inverse_transform(stock_df[numeric_cols])[:, 3]
        actual_dates = stock_df['Date'].reset_index(drop=True)

        viz_df = pd.DataFrame({
            'Date': actual_dates.iloc[time_step:],
            'Actual': original_close[time_step:],
            'Predicted': predicted_close
        })

        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Future': future_prices
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=viz_df['Date'], y=viz_df['Actual'], mode='lines', name='Harga Aktual', line=dict(color='yellow', width=2)))
        fig.add_trace(go.Scatter(x=viz_df['Date'], y=viz_df['Predicted'], mode='lines', name='Prediksi Historis', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Future'], mode='lines', name='Prediksi Masa Depan', line=dict(color='green', dash='dot', width=3)))

        fig.update_layout(
            title=f'{stock} - Prediksi Harga Saham 6 Bulan ke Depan',
            xaxis_title='Tanggal',
            yaxis_title='Harga (IDR)',
            showlegend=True,
            yaxis=dict(rangemode='tozero')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Tambahkan grafik terpisah untuk prediksi masa depan detail
        st.subheader(f"📈 {stock} - Grafik Prediksi Masa Depan (Detail)")
        fig_future = px.line(future_df, x='Date', y='Predicted_Future', title=f'{stock} - Prediksi Harga 6 Bulan Mendatang')
        fig_future.update_traces(line_color='green')
        st.plotly_chart(fig_future, use_container_width=True)

if __name__ == '__main__':
    main()