import streamlit as st
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import torch
import plotly.graph_objects as go # Using Plotly for potentially richer charts

# === Page Setup ===
st.set_page_config(
    layout="wide",
    page_title="Real-Time IDS Dashboard",
    page_icon="🚦"
)

# --- File Paths ---
parent_dir = Path(__file__).resolve().parents[0] 
base_path = parent_dir

# data_file = base_path / "data/processed/NF-ToN-IoT/X_test.csv"
data_file = base_path / "data/processed/UNSW-NB15/X_test.csv"
bin_model_file = base_path / "models/bin_tvae.pkl" 
bin_encoder_file = base_path / "models/bin_label_encoder.pkl"
multi_model_file = base_path / "models/multi_tvae.pkl"
multi_encoder_file = base_path / "models/multi_label_encoder.pkl"

# === Load Resources ===
@st.cache_data
def load_data(file_path):
    """Loads the dataset."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model_artifacts(bin_model_p, bin_enc_p, multi_model_p, multi_enc_p):
    """Loads the trained models and label encoders."""
    try:
        # Check if files exist before loading
        if not bin_model_p.exists(): raise FileNotFoundError(bin_model_p)
        if not bin_enc_p.exists(): raise FileNotFoundError(bin_enc_p)
        if not multi_model_p.exists(): raise FileNotFoundError(multi_model_p)
        if not multi_enc_p.exists(): raise FileNotFoundError(multi_enc_p)

        binary_model = joblib.load(bin_model_p)
        multi_model = joblib.load(multi_model_p)
        binary_encoder = joblib.load(bin_enc_p)
        multi_encoder = joblib.load(multi_enc_p)

        if hasattr(binary_model, 'eval'):
            binary_model.eval()
        if hasattr(multi_model, 'eval'):
            multi_model.eval()

        return binary_model, binary_encoder, multi_model, multi_encoder
    except FileNotFoundError as e:
        st.error(f"Error: Model or encoder file not found: {e}. Please check paths.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading models/encoders: {e}")
        return None, None, None, None

# --- Load Data and Models ---
df = load_data(data_file)
if df is None:
    st.stop() # Stop execution if data loading failed

binary_model, binary_encoder, multi_model, multi_encoder = load_model_artifacts(
    bin_model_file, bin_encoder_file, multi_model_file, multi_encoder_file
)
# Check if any required artifact failed to load
if binary_model is None or binary_encoder is None or multi_model is None or multi_encoder is None:
    st.error("Stopping execution due to failure loading models or encoders.")
    st.stop()

# Get feature columns (assuming all columns in the loaded CSV are features)
# It's safer to explicitly define or check against the model's expected features if possible
feature_columns = df.columns.tolist()
st.info(f"Loaded dataset with {len(feature_columns)} features.") # Removed listing all features to save space

# === Sidebar Configuration ===
st.sidebar.title("⚙️ Simulation Controls")
delay = st.sidebar.slider("Delay between packets (seconds)", 0.05, 3.0, 0.5, 0.05)
flagged_only = st.sidebar.checkbox("Show only suspicious traffic in table", False)
max_history_display = st.sidebar.number_input("Max rows in history table", 10, 100, 20)
auto_scroll = st.sidebar.checkbox("Auto-scroll history table (requires JS hack)", True) # JS Hack needed

# Restart Button
if st.sidebar.button("🔄 Restart Simulation"):
    st.session_state.index = 0
    # Reinitialize history DataFrame with correct columns
    st.session_state.history = pd.DataFrame(columns=feature_columns +
                                             ['timestamp', 'binary_prediction', 'multi_prediction', 'suspicious'])
    st.session_state.attack_counts = {}
    st.toast("Simulation Restarted!", icon="🔄")
    # Give time for toast to show before potential rerun
    time.sleep(0.5) # Short sleep is usually enough for UI update
    st.rerun()

# === Session State Initialization ===
if "index" not in st.session_state:
    st.session_state.index = 0
if "history" not in st.session_state:
    # Add columns we will create later
    st.session_state.history = pd.DataFrame(columns=feature_columns +
                                             ['timestamp', 'binary_prediction', 'multi_prediction', 'suspicious'])
if "attack_counts" not in st.session_state:
    st.session_state.attack_counts = {} # To store counts of each attack type


# === Main Dashboard Area ===
st.title("🚦 Real-Time Network Intrusion Detection Simulation")
st.markdown("---")

# --- Top Row: KPIs & Live Status ---
kpi_col1, kpi_col2, kpi_col3, live_status_col = st.columns([1, 1, 1, 2])

# Use placeholders defined outside the loop
total_processed_placeholder = kpi_col1.empty()
suspicious_total_placeholder = kpi_col2.empty() # Renamed for clarity
suspicious_perc_placeholder = kpi_col3.empty()
live_status_placeholder = live_status_col.empty()

# --- Middle Row: Charts ---
st.markdown("### 📊 Live Analytics")
chart_col1, chart_col2 = st.columns(2)

# Use placeholders defined outside the loop
suspicious_trend_placeholder = chart_col1.empty()
attack_dist_placeholder = chart_col2.empty()

# --- Bottom Row: History Table ---
st.markdown("### 📜 Live Packet History")
# Use placeholder defined outside the loop
table_placeholder = st.empty()

# JavaScript for auto-scrolling the table (optional, can be tricky)
# This needs to be injected once, not repeatedly in the simulation loop.
# It might need careful placement or execution after table renders.
if auto_scroll:
    try:
        st.components.v1.html(
            """
            <script>
            // Function to scroll the latest dataframe to bottom
            function scrollToLatestDataFrame() {
                var dataFrames = window.parent.document.querySelectorAll('[data-testid="stDataFrame"]');
                if (dataFrames.length > 0) {
                    // Select the last DataFrame on the page
                    var latestDataFrame = dataFrames[dataFrames.length - 1];
                    // Find the scrollable div within the DataFrame
                    var scrollableElement = latestDataFrame.querySelector('[data-testid="stDataFrameContainer"] > div:nth-child(2)'); // Adjust selector if needed
                    if (scrollableElement) {
                        scrollableElement.scrollTop = scrollableElement.scrollHeight;
                    }
                }
            }

            // Use a MutationObserver to watch for changes in the DOM that might add/update the table
            // This is more reliable than setInterval watching for element existence
            const observer = new MutationObserver((mutationsList, observer) => {
                // We don't need to inspect mutations, just scroll when changes might have occurred
                scrollToLatestDataFrame();
            });

            // Start observing the body for child list changes (new elements added/removed)
            // and subtree changes (changes within elements)
            observer.observe(window.parent.document.body, { childList: true, subtree: true });

            // Also call initially in case the table is already present
            scrollToLatestDataFrame();

            // Note: Stopping the observer might be complex when the app stops,
            // but for a continuous simulation, letting it run is usually fine.
            </script>
            """,
            height=0, # Set height to 0 to make it invisible
            scrolling=False # Important for invisible components
        )
    except Exception as e:
         st.warning(f"Could not inject scrolling script: {e}")


# === Streaming Simulation Logic (processes one packet per rerun) ===

idx = st.session_state.index
simulation_active = False # Assume simulation is not active unless we proceed

# Check if we reached the end of the data
if idx < len(df):
    simulation_active = True
    # --- Get Current Data ---
    row_series = df.iloc[idx].copy()
    timestamp = datetime.now()

    # --- Prepare Data for Model ---
    # Already confirmed data is numeric and scaled - ensure it's float32 for torch
    X_features = row_series.values.astype(np.float32) # Explicitly cast to float32
    X = X_features # X is now a float32 numpy array

    # --- CRITICAL: Reshape for PyTorch Model ---
    # Assumption: Models expect [batch_size, num_features] = [1, 10]
    try:
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.from_numpy(X).unsqueeze(0) # Shape: [1, num_features]
    except Exception as e:
        st.error(f"Error creating tensor for row {idx}: {e}")
        simulation_active = False # Stop simulation on tensor creation error


    # --- Model Prediction ---
    # Initialize prediction variables
    binary_label = "Not Processed"
    multi_label = "Not Processed"
    is_suspicious = False

    if simulation_active: # Only predict if tensor creation was successful
        try:
            # --- Binary Prediction ---
            binary_model.eval() # Ensure model is in eval mode
            with torch.no_grad(): # Disable gradient calculation for inference
                output = binary_model(X_tensor)

                # Handle different output shapes (e.g., logits for 2 classes vs single logit)
                if output.shape[1] > 1:
                    # Assumes output layer has N nodes (logits for N classes)
                    binary_pred_idx = torch.argmax(output, dim=1).item()
                else:
                    # Assumes output layer has 1 node (logit), apply sigmoid and round for binary
                    binary_pred_float = torch.sigmoid(output).item()
                    binary_pred_idx = int(round(binary_pred_float))

            # Inverse transform to get label (e.g., 'Normal', 'Attack')
            # Ensure the index is within the bounds of the encoder classes
            if 0 <= binary_pred_idx < len(binary_encoder.classes_):
                 binary_label = binary_encoder.inverse_transform([binary_pred_idx])[0]
                 is_suspicious = binary_label != "Normal" # Adjust if your normal label is different
            else:
                 binary_label = f"Unknown Bin Index ({binary_pred_idx})"
                 is_suspicious = True # Treat unknown as suspicious


            # --- Multi-class Prediction ---
            if is_suspicious: # Only run multi-class if binary flagged it
                multi_model.eval() # Ensure model is in eval mode
                with torch.no_grad(): # Disable gradient calculation
                    multi_output = multi_model(X_tensor)
                    multi_pred_idx = torch.argmax(multi_output, dim=1).item() # Get index as Python int

                # Inverse transform to get label (e.g., 'DoS', 'Analysis', 'Normal')
                # Ensure the index is within the bounds of the encoder classes
                if 0 <= multi_pred_idx < len(multi_encoder.classes_):
                    multi_label = multi_encoder.inverse_transform([multi_pred_idx])[0]
                     # Update attack counts only if it's a specific attack type and not "Normal"
                    if multi_label != "Normal":
                        st.session_state.attack_counts[multi_label] = st.session_state.attack_counts.get(multi_label, 0) + 1
                else:
                     multi_label = f"Unknown Multi Index ({multi_pred_idx})"
                     # If multi-class prediction is weird, count it under unknown
                     st.session_state.attack_counts[multi_label] = st.session_state.attack_counts.get(multi_label, 0) + 1

            else:
                 # If binary prediction is Normal, multi-class is also Normal
                multi_label = "Normal"

        except Exception as e:
            st.warning(f"⚠️ Prediction error on row {idx}: {e}")
            binary_label = "Prediction Error"
            multi_label = "Prediction Error"
            is_suspicious = True # Mark as suspicious if prediction failed


    # --- Update History DataFrame ---
    # Always append the row processed (even if prediction failed, so we have a record)
    new_row_dict = row_series.to_dict()
    new_row_dict["timestamp"] = timestamp # Use the timestamp captured earlier
    new_row_dict["binary_prediction"] = binary_label
    new_row_dict["multi_prediction"] = multi_label
    new_row_dict["suspicious"] = is_suspicious

    new_row_df = pd.DataFrame([new_row_dict])
    st.session_state.history = pd.concat(
        [st.session_state.history, new_row_df],
        ignore_index=True
    )


# --- Update Dashboard Elements (based on current state in session_state) ---
# These are now called on every rerun, which happens after each packet simulation step

# KPIs
total_processed = len(st.session_state.history)
# Ensure 'suspicious' column exists and is boolean before summing
if 'suspicious' in st.session_state.history.columns and pd.api.types.is_bool_dtype(st.session_state.history['suspicious']):
     suspicious_total = st.session_state.history['suspicious'].sum()
     suspicious_perc = (suspicious_total / total_processed * 100) if total_processed > 0 else 0
else:
     suspicious_total = 0
     suspicious_perc = 0


total_processed_placeholder.metric("Total Packets Processed", f"{total_processed:,}")
suspicious_total_placeholder.metric("🚨 Total Suspicious Flagged", f"{suspicious_total:,}")
suspicious_perc_placeholder.metric("% Suspicious Traffic", f"{suspicious_perc:.2f}%")

# Live Status (Update based on the *last* processed packet if simulation was active)
if simulation_active: # Only update the status box if a packet was just processed
    # Use the results from the packet just processed at index `idx`
    last_packet_status_color = "🟥 SUSPICIOUS" if is_suspicious else "🟩 Normal"
    last_packet_status_details = f"Type: {multi_label}" if is_suspicious else "Type: Normal"
    live_status_placeholder.markdown(f"""
    <div style="border: 2px solid {'red' if is_suspicious else 'green'}; padding: 10px; border-radius: 5px;">
        <h3 style="margin: 0;">Live Status: {last_packet_status_color}</h3>
        <small>Packet Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}</small><br>
        <small>{last_packet_status_details}</small>
    </div>
    """, unsafe_allow_html=True)

    # Display a warning *below* the status box only if suspicious
    if is_suspicious:
         st.warning(f"Suspicious activity detected: **{multi_label}**")
    # Note: Clearing previous warnings automatically in this structure is harder.
    # They will persist until a new warning overwrites them or the script structure changes.
else:
     # Display a message when simulation is finished
     if idx >= len(df) and total_processed > 0: # Simulation finished, show final status of last packet? Or just end message?
         # Option 1: Show status of the last packet processed
         last_row = st.session_state.history.iloc[-1]
         last_status_color = "🟥 SUSPICIOUS" if last_row['suspicious'] else "🟩 Normal"
         last_status_details = f"Type: {last_row['multi_prediction']}" if last_row['suspicious'] else "Type: Normal"
         live_status_placeholder.markdown(f"""
            <div style="border: 2px solid gray; padding: 10px; border-radius: 5px;">
                <h3 style="margin: 0;">Simulation Ended 🏁</h3>
                <small>Last Packet ({len(df)-1}) Timestamp: {last_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}</small><br>
                <small>Status of last packet: {last_status_color} - {last_status_details}</small>
            </div>
         """, unsafe_allow_html=True)
     elif idx >= len(df): # Simulation finished, but no data processed?
         live_status_placeholder.markdown("<center>Simulation finished. No data processed.</center>", unsafe_allow_html=True)
     else: # Simulation not active (e.g. error loading tensor)
          live_status_placeholder.markdown("<center>Simulation stopped due to processing error.</center>", unsafe_allow_html=True)


# Live Table
history_view = st.session_state.history.copy()
if flagged_only:
    history_view = history_view[history_view["suspicious"] == True]

# Ensure consistent column order, putting new ones first
display_columns_order = ['timestamp', 'binary_prediction', 'multi_prediction', 'suspicious'] + feature_columns
# Filter display_columns_order to only include columns actually present in the dataframe
display_columns_filtered = [col for col in display_columns_order if col in history_view.columns]


if not history_view.empty:
    table_placeholder.dataframe(
        history_view[display_columns_filtered].tail(max_history_display).sort_index(ascending=False), # Show latest first
        use_container_width=True,
        hide_index=True,
        column_config={ # Nicer timestamp formatting
             "timestamp": st.column_config.DatetimeColumn(
                "Timestamp",
                format="YYYY-MM-DD HH:mm:ss.SSS",
            ),
             "suspicious": st.column_config.CheckboxColumn("Suspicious?", default=False)
        }
    )
else:
    # Display an empty table or a message if the history view is empty
    # Use the filtered display columns so the empty table has the right headers
     table_placeholder.dataframe(pd.DataFrame(columns=display_columns_filtered), use_container_width=True)


# --- Update Charts (updated every rerun) ---
# Suspicious Trend Chart
trend_data = st.session_state.history.copy()
# Need at least 2 points to show a trend line meaningfully
if len(trend_data) > 1:
    trend_data['time_agg'] = pd.to_datetime(trend_data["timestamp"]).dt.floor("5S") # Aggregate per 5 seconds
    # Ensure there are actual points after aggregation
    if not trend_data['time_agg'].empty:
        summary = trend_data.groupby("time_agg").agg(
            suspicious_count=("suspicious", "sum"),
            total_count=("suspicious", "size")
        ).reset_index()
        summary["% Suspicious"] = (summary["suspicious_count"] / summary["total_count"]) * 100

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=summary["time_agg"], y=summary["% Suspicious"], mode='lines+markers', name='% Suspicious'))
        fig_trend.update_layout(
            title="Suspicious Traffic Trend (% over 5 sec intervals)",
            xaxis_title="Time",
            yaxis_title="% Suspicious Packets",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(rangemode='tozero') # Start x-axis from 0 or first data point
        )
        # Key is needed, but it's now called only once per rerun cycle
        suspicious_trend_placeholder.plotly_chart(fig_trend, use_container_width=True, key="suspicious_trend_chart")
    else:
         suspicious_trend_placeholder.markdown("<center><i>Not enough data yet for Suspicious Trend chart.</i></center>", unsafe_allow_html=True) # Key for markdown placeholder
else:
    # Display a placeholder message if not enough data for the chart
    suspicious_trend_placeholder.markdown("<center><i>Not enough data yet for Suspicious Trend chart.</i></center>", unsafe_allow_html=True)


# Attack Distribution Chart (Pie Chart)
attack_counts = st.session_state.attack_counts
if attack_counts:
    labels = list(attack_counts.keys())
    values = list(attack_counts.values())

    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig_pie.update_layout(
        title="Distribution of Detected Attack Types",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Place legend outside plot area
    )
    # Key is needed, but it's now called only once per rerun cycle
    attack_dist_placeholder.plotly_chart(fig_pie, use_container_width=True)
else:
     # Display a placeholder message if no attacks detected
     attack_dist_placeholder.markdown("<center><i>No specific attack types detected yet.</i></center>", unsafe_allow_html=True)


# === Increment index and Trigger Rerun ===
# This logic comes AFTER updating the UI elements so they display the results
# of the packet that was just processed.
if simulation_active: # Only proceed if the packet was processed without critical error and not at end of data
    st.session_state.index += 1
    # Trigger a rerun to process the next packet after a delay
    time.sleep(delay)
    st.rerun()
else:
    # If simulation is not active (either finished or hit an error),
    # we don't call rerun, allowing the app to stay on the final state.
    st.info("Simulation paused or finished.")