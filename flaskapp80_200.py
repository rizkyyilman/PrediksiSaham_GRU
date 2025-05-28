import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from flask import Flask, render_template, request
import plotly.graph_objs as go
import plotly.offline as pyo
from datetime import timedelta
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Seed for reproducibility
# ---------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

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
    print("Memuat dan memproses data...")
    files = {
        "BBRI": "bri_5tahun_edited.csv",
        "BBNI": "bni_5tahun_edited.csv",
        "BBCA": "bca_5tahun_edited.csv"
    }
    df_list = []
    scalers = {}
    scalers_close = {}

    for stock, filename in files.items():
        print(f"Memuat file: {filename}")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} tidak ditemukan di {os.getcwd()}")

        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()
        df.drop(columns=[col for col in ['Name', 'Dividends', 'Stock Splits'] if col in df.columns], inplace=True)
        df['Stock'] = stock
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)

        close_data = df[['Close']].astype(float)
        scaler_close = MinMaxScaler()
        scaler_close.fit(close_data)
        scalers_close[stock] = scaler_close

        scaler = MinMaxScaler()
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
        scalers[stock] = scaler

        df_list.append(df)

    if not df_list:
        raise ValueError("Tidak ada data yang berhasil dimuat.")

    df_combined = pd.concat(df_list, axis=0).sort_values(by=["Date", "Stock"])
    df_combined.ffill(inplace=True)
    df_combined.drop_duplicates(subset=["Date", "Stock"], keep="last", inplace=True)

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
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
        GRU(256),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()]
    )
    return model

# ---------------------------
# Prediksi masa depan
# ---------------------------
def predict_future(model, last_sequence, scaler_close, time_step, future_steps, numeric_cols):
    np.random.seed(SEED)  # Supaya noise konsisten setiap prediksi
    current_sequence = last_sequence.copy()
    predictions = []

    changes = np.diff(current_sequence[:, 3])
    avg_change = np.mean(changes)
    std_dev = np.std(changes)

    for step in range(future_steps):
        x_input = current_sequence.reshape((1, time_step, len(numeric_cols)))
        base_pred = model.predict(x_input, verbose=0)[0, 0]
        trend = avg_change * (1 + step / future_steps)
        noise = np.random.normal(0, std_dev * 2)
        pred = np.clip(base_pred + trend + noise, 0, 1)

        predictions.append(pred)

        next_sequence = current_sequence[1:, :].copy()
        new_row = current_sequence[-1, :].copy()
        new_row[3] = pred
        for i in [0, 1, 2, 4]:
            new_row[i] = np.clip(new_row[i] * (1 + np.random.normal(0, 0.005)), 0, 1)
        current_sequence = np.vstack([next_sequence, new_row])

    predictions = np.array(predictions).reshape(-1, 1)
    future_prices = scaler_close.inverse_transform(predictions).flatten()
    return future_prices

# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/predict')
def predict_page():
    return render_template('index.html')

@app.route('/loading', methods=['POST'])
def loading():
    stock = request.form.get('stock')
    return render_template('loading.html', stock=stock)

@app.route('/processing')
def processing():
    stock = request.args.get('stock')
    return predict(stock)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

MODEL_DIR = "saved_models"

def predict(stock):
    df_combined, numeric_cols, scalers, df_list, scalers_close = load_and_process_data()

    if stock not in scalers:
        return f"Data untuk {stock} tidak ditemukan", 404

    stock_df = df_combined[df_combined['Stock'] == stock]
    data = stock_df[numeric_cols].values
    time_step = 60
    future_steps = 180

    model_path = os.path.join(MODEL_DIR, f"{stock}_gru_model_80_200.h5")
    X, y = create_time_series(data, time_step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_gru_model((X_train.shape[1], X_train.shape[2]))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=64, callbacks=[early_stop], verbose=0)
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(model_path)

    scaler_close = scalers_close[stock]
    y_pred_test = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)

    # --------- Efisiensi prediksi historis ---------
    X_all, _ = create_time_series(data, time_step)
    historical_predictions = model.predict(X_all, verbose=0).flatten()
    historical_pred_prices = scaler_close.inverse_transform(historical_predictions.reshape(-1, 1)).flatten()
    # ------------------------------------------------

    historical_prices = scaler_close.inverse_transform(stock_df[['Close']].values).flatten()
    historical_dates = stock_df['Date'].values

    last_sequence = data[-time_step:]
    future_prices = predict_future(model, last_sequence, scaler_close, time_step, future_steps, numeric_cols)
    last_date = stock_df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps + 1)]

    trace_actual = go.Scatter(x=historical_dates, y=historical_prices, mode='lines', name='Harga Aktual', line=dict(color='blue'))
    trace_pred_historical = go.Scatter(x=historical_dates[time_step:], y=historical_pred_prices, mode='lines', name='Prediksi Historis', line=dict(color='orange'))
    trace_future = go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Prediksi Masa Depan (6 Bulan)', line=dict(color='green'))

    layout = go.Layout(
        title={
            'text': f'Harga Saham {stock} - Historis dan Prediksi 6 Bulan ke Depan (Januari - Juni) 2025 epochs 200',
            'x': 0.5,
            'xanchor': 'center'
        },
        autosize=True,
        height=500,
        margin=dict(l=60, r=30, t=80, b=50),
        xaxis=dict(title='Tanggal'),
        yaxis=dict(title='Harga (IDR)', range=[min(historical_prices.min(), future_prices.min()) * 0.9, max(historical_prices.max(), future_prices.max()) * 1.1]),
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, font=dict(size=12))
    )
    fig = go.Figure(data=[trace_actual, trace_pred_historical, trace_future], layout=layout)
    graph_html = pyo.plot(fig, output_type='div', include_plotlyjs='cdn', config={'responsive': True})

    return render_template(
        'result.html',
        graph_html=graph_html,
        stock=stock,
        mse=round(mse, 4),
        rmse=round(rmse, 4),
        mae=round(mae, 4),
        r2=round(r2, 4),
        mape=round(mape, 2)
    )

if __name__ == '__main__':
    app.run(debug=True, port=5007, use_reloader=False)
