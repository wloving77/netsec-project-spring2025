import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import numpy as np

st.set_page_config(layout="wide")

# Sidebar
st.sidebar.title("IDS Dashboard")
view = st.sidebar.radio("Choose View", ["Performance", "Confusion Matrix", "t-SNE", "Epoch Animation"])

# Data

# 1. Performance View
if view == "Performance":
    st.title("IDS Performance Comparison")

    df = pd.DataFrame({'Real': metrics_real, 'Synthetic+Real': metrics_synth}).T.reset_index().rename(columns={'index': 'Dataset'})
    df_melted = df.melt(id_vars='Dataset', var_name='Metric', value_name='Score')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Dataset', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance on Real vs Synthetic+Real Data")
    st.pyplot(fig)

# 2. Confusion Matrix
elif view == "Confusion Matrix":
    st.title("Confusion Matrix")

    y_true = ["Normal", "Normal", "DoS", "DoS", "Reconnaissance", "Normal", "DoS"]
    y_pred = ["Normal", "DoS", "DoS", "DoS", "Reconnaissance", "Normal", "Normal"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# 3. t-SNE Visualization
elif view == "t-SNE":
    st.title("t-SNE Visualization")

    real_data = np.random.rand(200, 10)
    synth_data = np.random.rand(200, 10) + 1.0  # shift to distinguish

    X_combined = np.vstack([real_data, synth_data])
    labels_tsne = np.array(['Real'] * len(real_data) + ['Synthetic'] * len(synth_data))

    tsne = TSNE(n_components=2, perplexity=30)
    X_embedded = tsne.fit_transform(X_combined)
    df_tsne = pd.DataFrame(X_embedded, columns=["Dim1", "Dim2"])
    df_tsne['Type'] = labels_tsne

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', hue='Type', alpha=0.7, ax=ax)
    ax.set_title("t-SNE of Real vs Synthetic Samples")
    st.pyplot(fig)

# 4. Animated Epoch Performance
elif view == "Epoch Animation":
    st.title("IDS Metrics Over Training Epochs")

    epochs = list(range(1, 21))
    acc = [0.75 + i * 0.01 for i in epochs]
    f1 = [0.65 + i * 0.015 for i in epochs]

    fig = go.Figure(
        frames=[
            go.Frame(
                data=[
                    go.Bar(x=["Accuracy", "F1 Score"], y=[acc[i], f1[i]], marker_color=["steelblue", "seagreen"])
                ],
                name=f"Epoch {i+1}"
            )
            for i in range(len(epochs))
        ]
    )
    fig.add_trace(go.Bar(x=["Accuracy", "F1 Score"], y=[acc[0], f1[0]]))
    fig.update_layout(
        updatemenus=[
            dict(type="buttons", buttons=[
                dict(label="Play", method="animate", args=[None]),
                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
            ])
        ],
        title="Epoch-wise Accuracy and F1 Score",
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        height=500,
    )
    st.plotly_chart(fig)