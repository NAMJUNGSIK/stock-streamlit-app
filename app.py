import streamlit as st
import FinanceDataReader as fdr
import ta
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 종목 사전 정의
stocks = {
    '005930': '삼성전자',
    '035420': 'NAVER',
    '035720': '카카오',
    '005380': '현대차',
    '000660': 'SK하이닉스'
}
SEQ_LEN = 20
features = ['close', 'volume', 'momentum_rsi', 'trend_macd', 'volatility_bbm']

st.title("📈 한국 주식 상승 확률 예측 (LSTM)")

selected_codes = st.multiselect("분석할 종목을 선택하세요:", list(stocks.keys()), default=list(stocks.keys()))

def preprocess_stock_data(code):
    df = fdr.DataReader(code, '2017-01-01')
    
    if df is None or df.empty:
        raise ValueError("데이터를 불러오지 못했습니다.")
    
    df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    df = df.rename(columns=str.lower)
    
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    
    # 필수 컬럼이 존재하는지 확인
    for col in features:
        if col not in df.columns:
            raise ValueError(f"'{col}' 컬럼이 없습니다.")

    # 데이터 충분한지 확인
    if df[features].shape[0] == 0:
        raise ValueError("전처리 후 데이터가 없습니다.")
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    Xs, ys = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        Xs.append(X_scaled[i:i+SEQ_LEN])
        ys.append(df['target'].iloc[i+SEQ_LEN])
    
    if len(Xs) == 0:
        raise ValueError("LSTM 학습에 사용할 시퀀스 데이터가 부족합니다.")
    
    return np.array(Xs), np.array(ys), df[features], X_scaled

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

if st.button("📊 예측 시작"):
    results = []

    for code in selected_codes:
        name = stocks[code]
        try:
            X_seq, y_seq, _, X_scaled = preprocess_stock_data(code)
            if len(X_seq) < 100:
                st.warning(f"{name}({code}): 데이터 부족")
                continue
            
            split = int(len(X_seq) * 0.8)
            X_train, y_train = X_seq[:split], y_seq[:split]
            X_test, y_test = X_seq[split:], y_seq[split:]

            model = build_lstm_model((SEQ_LEN, len(features)))
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0,
                      validation_data=(X_test, y_test),
                      callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
            
            latest_seq = X_scaled[-SEQ_LEN:]
            latest_seq = latest_seq.reshape(1, SEQ_LEN, len(features))
            prob = model.predict(latest_seq, verbose=0)[0][0]
            results.append((name, code, round(prob * 100, 2)))

        except Exception as e:
            st.error(f"[오류] {name}({code}): {e}")
    
    if results:
        results = sorted(results, key=lambda x: x[2], reverse=True)
        st.subheader("📈 상승 확률 예측 결과")
        for name, code, prob in results:
            st.write(f"**{name} ({code})** - 상승 확률: `{prob}%`")