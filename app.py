import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
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
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Fungsi untuk memprediksi masa depan
def predict_future(model, last_sequence, scaler, time_step, future_steps, numeric_cols):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    # Hitung rata-rata perubahan relatif dari data historis
    historical_changes = np.diff(last_sequence, axis=0) / last_sequence[:-1]
    avg_changes = np.clip(np.mean(historical_changes, axis=0), -0.005, 0.005)  # Batasi perubahan ±0.5%
    
    # Ambil harga minimum dan maksimum realistis dari data historis
    historical_prices = scaler.inverse_transform(last_sequence)[:, 3]
    min_price = historical_prices.min() * 0.8  # 80% dari harga minimum
    max_price = historical_prices.max() * 1.2  # 120% dari harga maksimum
    
    # Hitung tren jangka panjang
    recent_trend = np.mean(historical_changes[-10:, 3])  # Tren 10 hari terakhir untuk Close
    trend_adjustment = 0 if abs(recent_trend) < 0.01 else -recent_trend * 0.5  # Lawan tren ekstrem
    
    for _ in range(future_steps):
        current_sequence_reshaped = current_sequence.reshape((1, time_step, len(numeric_cols)))
        next_pred = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
        # Sesuaikan prediksi dengan tren jangka panjang
        next_pred = next_pred + trend_adjustment
        next_pred = np.clip(next_pred, 0, 1)  # Batasi ke [0, 1]
        future_predictions.append(next_pred)
        
        next_sequence = current_sequence[1:, :].copy()
        new_row = current_sequence[-1, :].copy()
        new_row[3] = next_pred  # Update Close
        for i in [0, 1, 2, 4]:  # Open, High, Low, Volume
            new_row[i] = new_row[i] * (1 + avg_changes[i])
            new_row[i] = np.clip(new_row[i], 0, 1)  # Batasi ke [0, 1]
        current_sequence = np.vstack([next_sequence, new_row])
    
    # Inverse transform dengan batasan
    dummy = np.zeros((len(future_predictions), len(numeric_cols)))
    dummy[:, 3] = future_predictions
    future_prices = scaler.inverse_transform(dummy)[:, 3]
    future_prices = np.clip(future_prices, min_price, max_price)  # Batasi ke rentang realistis
    
    # Debug
    print(f"First 5 normalized predictions: {future_predictions[:5]}")
    print(f"First 5 inverse transformed prices: {future_prices[:5]}")
    
    return future_prices

# Streamlit aplikasi
def main():
    try:
        st.title("Prediksi Harga Saham Perbankan (BBRI, BBNI, BBCA)")
        st.write("Aplikasi ini menampilkan prediksi harga saham menggunakan model GRU berdasarkan data historis.")
        
        # Muat dan proses data
        df_combined, numeric_cols, scalers, df_list = load_and_process_data()
        if df_combined is None:
            st.error("Gagal memuat data. Periksa terminal untuk detail.")
            return
        
        # Persiapkan data untuk model
        st.write("Mempersiapkan data untuk model...")
        time_step = 60
        X_dict, y_dict = {}, {}
        for stock, group in df_combined.groupby("Stock"):
            stock_data = group[numeric_cols].values
            X, y = create_time_series(stock_data, time_step)
            X_dict[stock] = X
            y_dict[stock] = y
            st.write(f"Data untuk {stock} - X shape: {X.shape}, y shape: {y.shape}")
        
        # Latih model terpisah untuk setiap saham
        st.subheader("Grafik Harga Historis, Prediksi Historis, dan Prediksi Masa Depan")
        stocks = ['BBRI', 'BBNI', 'BBCA']
        future_steps = 180  # 6 bulan
        models = {}
        
        for stock in stocks:
            st.write(f"### {stock}")
            stock_df = df_combined[df_combined['Stock'] == stock].copy()
            stock_X = X_dict[stock]
            stock_y = y_dict[stock]
            
            # Split data tanpa pengacakan
            train_size = int(len(stock_X) * 0.8)
            X_train, X_test = stock_X[:train_size], stock_X[train_size:]
            y_train, y_test = stock_y[:train_size], stock_y[train_size:]
            st.write(f"Training data for {stock} - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            
            # Latih model GRU
            st.write(f"Melatih model GRU untuk {stock}... (mungkin memakan waktu beberapa menit)")
            model = build_gru_model(input_shape=(time_step, 5))
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=50,
                batch_size=64,
                callbacks=[early_stop],
                verbose=0
            )
            models[stock] = model
            st.write(f"Model untuk {stock} selesai dilatih!")
            
            # Prediksi historis
            st.write(f"Membuat prediksi historis untuk {stock}...")
            stock_pred = model.predict(stock_X, verbose=0).flatten()
            
            # Inverse transform
            scaler = scalers[stock]
            dummy = np.zeros((len(stock_pred), len(numeric_cols)))
            dummy[:, 3] = stock_pred
            pred_close = scaler.inverse_transform(dummy)[:, 3]
            pred_close = np.maximum(pred_close, 0)  # Pastikan tidak negatif
            
            dummy[:, 3] = stock_y
            actual_close = scaler.inverse_transform(dummy)[:, 3]
            
            # Debug
            print(f"{stock} - First 5 normalized predictions: {stock_pred[:5]}")
            print(f"{stock} - First 5 inverse transformed predictions: {pred_close[:5]}")
            
            # Prediksi masa depan
            st.write(f"Memprediksi {future_steps} hari (6 bulan) ke depan untuk {stock}...")
            last_sequence = stock_df[numeric_cols].values[-time_step:]
            future_prices = predict_future(model, last_sequence, scaler, time_step, future_steps, numeric_cols)
            
            # Buat tanggal untuk prediksi masa depan
            last_date = stock_df['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps + 1)]
            
            # Buat DataFrame untuk visualisasi
            dates = stock_df['Date'].iloc[time_step:].reset_index(drop=True)
            viz_df = pd.DataFrame({
                'Date': dates,
                'Actual': actual_close,
                'Predicted': pred_close
            })
            print(f"Visualization data for {stock} - Number of rows: {len(viz_df)}")
            
            # DataFrame untuk prediksi masa depan
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Future': future_prices
            })
            
            # Plot
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