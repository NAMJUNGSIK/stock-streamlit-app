import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

# --- 데이터 불러오기 함수 ---
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

# --- 데이터 전처리 ---
def preprocess_data(data, window_size=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        # 상승 여부 (다음날 종가가 오늘보다 높으면 1, 아니면 0)
        y.append(1 if scaled_data[i, 0] > scaled_data[i - 1, 0] else 0)

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# --- 모델 생성 및 학습 ---
def create_train_model(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # 확률 예측
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model

# --- Streamlit UI ---
st.title("📈 확률 예측 (LSTM 기반)")

ticker = st.text_input("종목 코드 입력 (예: 005930.KS - 삼성전자)", "005930.KS")
start_date = st.date_input("시작 날짜", datetime(2020, 1, 1))
end_date = st.date_input("종료 날짜", datetime.today())

if st.button("예측 시작"):
    with st.spinner("데이터 불러오는 중..."):
        data = load_data(ticker, start=start_date, end=end_date)

    st.line_chart(data['Close'], use_container_width=True)

    with st.spinner("모델 학습 중..."):
        X, y, scaler = preprocess_data(data.values)
        model = create_train_model(X, y)

        # 가장 최신 데이터로 예측
        latest_sequence = data.values[-60:]
        latest_scaled = scaler.transform(latest_sequence)
        input_data = np.reshape(latest_scaled, (1, 60, 1))
        prob = model.predict(input_data)[0][0]
        percent = round(prob * 100, 2)

        st.success(f"📊 내일 상승할 확률: **{percent}%**")