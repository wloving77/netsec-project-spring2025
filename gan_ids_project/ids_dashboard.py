import streamlit as st
import pandas as pd
import time
import numpy as np
from datetime import datetime

# Settings
st.set_page_config(layout="wide")
st.sidebar.title("Network Stream Simulation")
delay = st.sidebar.slider("Delay between records (sec)", 0.1, 2.0, 0.5)
flagged_only = st.sidebar.checkbox("Show only suspicious rows", False)

# Load data (no timestamp in CSV)
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/NF-ToN-IoT/X_test.csv")

df = load_data()

# Define a mock anomaly detection function
def flag_anomaly(row):
    return (row.abs() > 0.5).any()

# Session state init
if "index" not in st.session_state:
    st.session_state.index = 0
    st.session_state.history = pd.DataFrame()

# Simulation
if st.button("Start Simulation"):
    while st.session_state.index < len(df):
        row = df.iloc[st.session_state.index].copy()
        row["timestamp"] = datetime.now()
        row["suspicious"] = flag_anomaly(row[:-1])  # exclude timestamp

        # Store history
        st.session_state.history = pd.concat(
            [st.session_state.history, row.to_frame().T], ignore_index=True
        )

        # Show latest data
        display_df = st.session_state.history.copy()
        if flagged_only:
            display_df = display_df[display_df["suspicious"] == True]

        st.subheader("Recent Network Records")
        st.dataframe(display_df.tail(10), use_container_width=True)

        # Suspicious trend plot
        st.subheader("Suspicious Traffic Trend")
        plot_df = st.session_state.history.copy()
        plot_df["minute"] = pd.to_datetime(plot_df["timestamp"]).dt.floor("min")
        trend = plot_df.groupby("minute")["suspicious"].mean().reset_index()
        trend["% Suspicious"] = trend["suspicious"] * 100
        st.line_chart(trend.set_index("minute")["% Suspicious"])

        st.session_state.index += 1
        time.sleep(delay)