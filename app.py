import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import timedelta

# Fungsi untuk memuat dan memproses data
def load_and_process_data():
    st.write("Memuat dan memproses data...")
    files = {
        "BBRI": "bri_5tahun_edited.csv",
        "BBNI": "bni_5tahun_edited.csv",
        "BBCA": "bca_5tahun_edited.csv"
    }
    
    df_list = []
    scalers = {}
    
    for stock, filename in files.items():
        st.write(f"Memuat file: {filename}")
        print(f"Checking file: {filename} - Exists: {os.path.exists(filename)}")
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            print(f"{stock} - Number of rows: {len(df)}")
            print(f"{stock} - Columns: {df.columns.tolist()}")
            print(f"{stock} - Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"{stock} - First 5 rows:\n{df.head().to_string()}")
            print(f"{stock} - Last 5 rows:\n{df.tail().to_string()}")
            df.columns = df.columns.str.strip()
            cols_to_drop = ['Name', 'Dividends', 'Stock Splits']
            df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors="ignore")
            df["Stock"] = stock
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
            df_list.append(df)
        else:
            st.error(f"File {filename} tidak ditemukan di {os.getcwd()}")
            print(f"Error: File {filename} tidak ditemukan")
    
    if not df_list:
        st.error("Tidak ada data yang berhasil dimuat")
        return None, None, None, None
    
    df_combined = pd.concat(df_list, axis=0, ignore_index=True)
    st.write("Data digabungkan, shape:", df_combined.shape)
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    df_combined.sort_values(by=["Date", "Stock"], inplace=True)
    df_combined.ffill(inplace=True)
    
    # Normalisasi per saham
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for stock, group in df_combined.groupby("Stock"):
        scaler = MinMaxScaler()
        group[numeric_cols] = group[numeric_cols].astype(float)
        group[numeric_cols] = scaler.fit_transform(group[numeric_cols])
        scalers[stock] = scaler
        df_combined.loc[group.index, numeric_cols] = group[numeric_cols]
    
    df_combined = df_combined.drop_duplicates(subset=["Date", "Stock"], keep="last").reset_index(drop=True)
    st.write("Data setelah pembersihan, shape:", df_combined.shape)
    return df_combined, numeric_cols, scalers, df_list

# Fungsi untuk membuat data time series
def create_time_series(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step, :])
        y.append(data[i+time_step, 3])  # Close price
    return np.array(X), np.array(y)

# Fungsi untuk membangun model GRU
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
    model.summary()
    return model

# Fungsi untuk menghitung MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Fungsi untuk memprediksi masa depan
def predict_future(model, last_sequence, scaler, time_step, future_steps, numeric_cols):
    future_predictions = []
    current_sequence = last_sequence.copy()
    step_size = 30
    stagnant_count = 0
    previous_pred = None

    # Hitung perubahan historis relatif (persentase perubahan)
    historical_changes = np.diff(last_sequence, axis=0) / (last_sequence[:-1] + 1e-8)
    # Lebarkan batas clipping perubahan rata-rata agar variasi lebih besar
    avg_changes = np.clip(np.mean(historical_changes, axis=0), -0.01, 0.01)

    # Data harga asli (inverse transform kolom harga)
    historical_prices = scaler.inverse_transform(last_sequence)[:, 3]
    min_price = historical_prices.min() * 0.8  # Batas bawah prediksi lebih longgar
    max_price = historical_prices.max() * 1.2  # Batas atas prediksi lebih longgar
    avg_price = historical_prices.mean()

    # Volatilitas dan tren historis
    historical_volatility = np.std(historical_changes[:, 3])
    volatility_factor = min(historical_volatility / 0.04, 3.5)  # Faktor volatilitas
    recent_trend = np.mean(historical_changes[-10:, 3])
    long_term_trend = np.mean(historical_changes[:, 3])

    # Penyesuaian tren dan momentum
    trend_adjustment = max(-0.05, (-recent_trend * 7.0 + long_term_trend * 0.02))
    trend_boost = 2.0 * volatility_factor if recent_trend > 0 else 1.0
    initial_momentum = (recent_trend * 7.0 * volatility_factor)
    initial_boost = 0.2 * volatility_factor if recent_trend > 0 else -0.2 * volatility_factor
    recovery_factor = ((avg_price - historical_prices[-1]) / avg_price) * 0.3
    noise_factor = max(historical_volatility * volatility_factor * 3.0, 0.03)

    for i in range(0, future_steps, step_size):
        steps_to_predict = min(step_size, future_steps - i)
        for j in range(steps_to_predict):
            x_input = current_sequence.reshape((1, time_step, len(numeric_cols)))
            next_pred = model.predict(x_input, verbose=0)[0, 0]

            # Variasi noise dan momentum yang lebih besar
            noise = np.random.normal(0, noise_factor)
            sinusoidal_variation = 0.02 * np.sin((i + j) / 5.0)
            momentum = initial_momentum * (1 - (j / 30)) if j < 30 else 0
            boost = initial_boost * (1 - (j / 10)) if j < 10 else 0

            adjustment = (trend_adjustment * trend_boost) + (recovery_factor * ((i + j) / future_steps)) + momentum + boost + noise + sinusoidal_variation
            next_pred = np.clip(next_pred + adjustment, 0.01, 0.99)

            # Deteksi stagnasi dan paksa variasi jika perlu
            if previous_pred is not None and abs(next_pred - previous_pred) < 0.0008:
                stagnant_count += 1
            else:
                stagnant_count = 0

            if stagnant_count >= 2:
                next_pred += np.random.choice([-0.02, 0.02]) + np.random.normal(0, 0.01)
                next_pred = np.clip(next_pred, 0.01, 0.99)
                stagnant_count = 0

            future_predictions.append(next_pred)
            previous_pred = next_pred

            # Update sequence untuk prediksi berikutnya
            next_sequence = current_sequence[1:, :].copy()
            new_row = current_sequence[-1, :].copy()
            new_row[3] = next_pred  # Update harga close

            # Update fitur lain dengan drift dan rata-rata perubahan
            for k in [0, 1, 2, 4]:  # Kolom selain close (open, high, low, volume)
                drift = np.random.normal(0, 0.004)
                new_row[k] = np.clip(new_row[k] * (1 + avg_changes[k] + drift), 0, 1)

            current_sequence = np.vstack([next_sequence, new_row])

    # Inverse transform hasil prediksi harga close
    dummy = np.zeros((len(future_predictions), len(numeric_cols)))
    dummy[:, 3] = future_predictions
    future_prices = scaler.inverse_transform(dummy)[:, 3]
    future_prices = np.clip(future_prices, min_price, max_price)

    return future_prices

# Streamlit aplikasi
def main():
    try:
        st.title("Prediksi Harga Saham Perbankan (BBRI, BBNI, BBCA)")
        st.write("Aplikasi ini menampilkan prediksi harga saham menggunakan metode time series dengan GRU (Gated Recurrent Unit).")
        
        df_combined, numeric_cols, scalers, df_list = load_and_process_data()
        if df_combined is None:
            st.error("Gagal memuat data. Periksa terminal untuk detail.")
            return
        
        st.write("Mempersiapkan data untuk model...")
        time_step = 60
        X_dict, y_dict = {}, {}
        for stock, group in df_combined.groupby("Stock"):
            stock_data = group[numeric_cols].values
            X, y = create_time_series(stock_data, time_step)
            X_dict[stock] = X
            y_dict[stock] = y
            st.write(f"Data untuk {stock} - X shape: {X.shape}, y shape: {y.shape}")
        
        st.subheader("Grafik Harga Historis, Prediksi Historis, dan Prediksi Masa Depan")
        stocks = ['BBRI', 'BBNI', 'BBCA']
        future_steps = 180
        models = {}
        
        for stock in stocks:
            st.write(f"### {stock}")
            stock_df = df_combined[df_combined['Stock'] == stock].copy()
            stock_X = X_dict[stock]
            stock_y = y_dict[stock]
            
            train_size = int(len(stock_X) * 0.8)
            X_train, X_test = stock_X[:train_size], stock_X[train_size:]
            y_train, y_test = stock_y[:train_size], stock_y[train_size:]
            st.write(f"Training data for {stock} - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            
            st.write(f"Melatih model GRU untuk {stock}... (mungkin memakan waktu beberapa menit)")
            model = build_gru_model(input_shape=(time_step, len(numeric_cols)))
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=64,
                callbacks=[early_stop],
                verbose=0
            )
            models[stock] = model
            st.write(f"Model untuk {stock} selesai dilatih!")
            
            st.write(f"Membuat prediksi historis untuk {stock}...")
            stock_pred = model.predict(stock_X, verbose=0).flatten()
            
            scaler = scalers[stock]
            dummy = np.zeros((len(stock_pred), len(numeric_cols)))
            dummy[:, 3] = stock_pred
            pred_close = scaler.inverse_transform(dummy)[:, 3]
            pred_close = np.maximum(pred_close, 0)
            
            dummy[:, 3] = stock_y
            actual_close = scaler.inverse_transform(dummy)[:, 3]
            
            rmse = np.sqrt(mean_squared_error(actual_close, pred_close))
            mae = mean_absolute_error(actual_close, pred_close)
            mape = mean_absolute_percentage_error(actual_close, pred_close)
            
            st.write(f"**Metrik Evaluasi untuk {stock}:**")
            st.write(f"- RMSE: {rmse:.2f} IDR")
            st.write(f"- MAE: {mae:.2f} IDR")
            st.write(f"- MAPE: {mape:.2f}%")
            
            print(f"{stock} - First 5 normalized predictions: {stock_pred[:5]}")
            print(f"{stock} - First 5 inverse transformed predictions: {pred_close[:5]}")
            
            st.write(f"Memprediksi {future_steps} hari (6 bulan) ke depan untuk {stock}...")
            last_sequence = stock_df[numeric_cols].values[-time_step:]
            future_prices = predict_future(model, last_sequence, scaler, time_step, future_steps, numeric_cols)
            
            last_date = stock_df['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps + 1)]
            
            dates = stock_df['Date'].iloc[time_step:].reset_index(drop=True)
            viz_df = pd.DataFrame({
                'Date': dates,
                'Actual': actual_close,
                'Predicted': pred_close
            })
            print(f"Visualization data for {stock} - Number of rows: {len(viz_df)}")
            
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Future': future_prices
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=viz_df['Date'], y=viz_df['Actual'], mode='lines', name='Harga Aktual', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=viz_df['Date'], y=viz_df['Predicted'], mode='lines', name='Prediksi Historis', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Future'], mode='lines', name='Prediksi Masa Depan (6 Bulan)', line=dict(color='green', dash='dash')))
            
            fig.update_layout(
                title=f'Harga Saham {stock} - Historis dan Prediksi 6 Bulan ke Depan',
                xaxis_title='Tanggal',
                yaxis_title='Harga Penutupan (IDR)',
                template='plotly_white'
            )
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()