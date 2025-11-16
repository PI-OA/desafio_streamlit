
"""Aplicación Streamlit para explorar el dataset de criptomonedas y
predecir el precio de Binance Coin (BNB) usando el precio de Ethereum
con 15 días de rezago."""

from pathlib import Path
from pickle import load as pickle_load

import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "Crypto Data Since 2015.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
ETH_SCALER_PATH = ROOT_DIR / "models" / "scaler.pkl"
BNB_SCALER_PATH = ROOT_DIR / "models" / "scaler_binance.pkl"


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Carga el modelo y los escaladores serializados."""

    with (
        MODEL_PATH.open("rb") as f_model,
        ETH_SCALER_PATH.open("rb") as f_eth,
        BNB_SCALER_PATH.open("rb") as f_bnb,
    ):
        model = pickle_load(f_model)
        eth_scaler = pickle_load(f_eth)
        bnb_scaler = pickle_load(f_bnb)
    return model, eth_scaler, bnb_scaler


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    """Carga el dataset crudo y ordena las fechas."""

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df.reset_index(drop=True)


def main():
    st.set_page_config(page_title="Crypto Explorer", layout="wide")
    st.title("Exploración y predicción de precios cripto")
    st.write(
        "Visualiza la evolución histórica de las criptomonedas del dataset y "
        "obtén una predicción del precio de Binance Coin (BNB) utilizando el "
        "modelo entrenado en este repositorio."
    )

    df = load_prices()
    price_columns = [col for col in df.columns if col != "Date"]
    st.subheader("Explora los datos históricos")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Fecha inicial",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )
    with col2:
        end_date = st.date_input(
            "Fecha final",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

    if start_date > end_date:
        st.error("La fecha inicial no puede ser mayor que la fecha final.")
        filtered = df.iloc[0:0]
    else:
        mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
        filtered = df.loc[mask]

    st.dataframe(filtered, use_container_width=True, height=360)

    if not filtered.empty:
        st.line_chart(filtered.set_index("Date")[price_columns])
    else:
        st.info("Ajusta el rango de fechas para visualizar datos.")

    st.subheader("Predicción de precio para Binance Coin (BNB)")
    model, eth_scaler, bnb_scaler = load_artifacts()

    eth_min = float(df["Ethereum (USD)"].min())
    eth_max = float(df["Ethereum (USD)"].max())
    default_value = float(df["Ethereum (USD)"].iloc[-1])

    eth_price = st.slider(
        "Precio de Ethereum (USD) hace 15 días",
        min_value=float(round(eth_min, 2)),
        max_value=float(round(eth_max, 2)),
        value=float(round(default_value, 2)),
        step=1.0,
    )
    eth_price_scaled = eth_scaler.transform(np.array([[eth_price]]))

    predicted_scaled = model.predict(eth_price_scaled)
    predicted_bnb = bnb_scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0]

    st.metric("Precio estimado de BNB", f"${predicted_bnb:,.2f}")
    st.caption(
        "El modelo usa un rezago de 15 días del precio de Ethereum para estimar el precio actual de Binance Coin."
    )


if __name__ == "__main__":
    main()
