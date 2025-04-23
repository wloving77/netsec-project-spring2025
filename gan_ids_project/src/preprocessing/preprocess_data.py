import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

FEATURES = {
    "UNSW-NB15": {
        "Input_Features": [
            "dur",  # total connection duration
            "proto",  # transaction protocol
            "service",  # destination service/type
            "state",  # connection state
            "spkts",  # packets from src → dst
            "dpkts",  # packets from dst → src
            "sbytes",  # bytes from src → dst
            "dbytes",  # bytes from dst → src
            "rate",  # packets per second
            "sload",  # source bits per second
            "dload",  # dest bits per second
            "sloss",  # src packet loss count
            "dloss",  # dest packet loss count
            "sinpkt",  # src inter‑packet arrival time (ms)
            "dinpkt",  # dest inter‑packet arrival time (ms)
            "sjit",  # source jitter (ms)
            "djit",  # dest jitter (ms)
            "tcprtt",  # TCP round‑trip time (ms)
            "synack",  # time between SYN and SYN‑ACK (ms)
            "ackdat",  # time between SYN‑ACK and ACK (ms)
            "swin",  # src TCP window size
            "dwin",  # dest TCP window size
            "stcpb",  # src TCP base sequence num
            "dtcpb",  # dest TCP base sequence num
            "smean",  # (smeansz) mean src packet size
            "dmean",  # (dmeansz) mean dest packet size
            "trans_depth",  # HTTP transaction depth
            "response_body_len",  # (res_bdy_len) HTTP response body length
            "is_ftp_login",  # FTP login flag
            "is_sm_ips_ports",  # same src/dst IPs & ports flag
            "ct_state_ttl",  # count flows by state & TTL
            "ct_flw_http_mthd",  # count HTTP methods per flow
            "ct_ftp_cmd",  # count FTP commands per flow
            "ct_srv_src",  # count recent service–src pairs
            "ct_srv_dst",  # count recent service–dst pairs
            "ct_dst_ltm",  # count recent dest addresses
            "ct_src_ltm",  # count recent src addresses
            "ct_src_dport_ltm",  # count recent src–dport pairs
            "ct_dst_sport_ltm",  # count recent dst–sport pairs
            "ct_dst_src_ltm",  # count recent src–dst pairs
        ],
        "Output_Features": [
            "label",  # 0, 1 for normal, attack
            "attack_cat",  # categorical for type of attack/no attack
        ],
    },
    "NF-ToN-IoT": {
        "Input_Features": [
            "L4_SRC_PORT",
            "L4_DST_PORT",
            "PROTOCOL",
            "L7_PROTO",
            "IN_BYTES",
            "OUT_BYTES",
            "IN_PKTS",
            "OUT_PKTS",
            "TCP_FLAGS",
            "FLOW_DURATION_MILLISECONDS",
        ],
        "Output_Features": ["Label", "Attack"],
    },
}


def load_data(base_data_dir, dataset):
    train_path = os.path.join(
        base_data_dir, "raw", dataset, f"{dataset}_training-set.csv"
    )
    test_path = os.path.join(
        base_data_dir, "raw", dataset, f"{dataset}_testing-set.csv"
    )

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_test


def process_data(df_train, df_test):
    input_features = FEATURES[dataset]["Input_Features"]
    output_features = FEATURES[dataset]["Output_Features"]

    df_train = df_train.drop_duplicates().dropna(subset=output_features)
    df_test = df_test.drop_duplicates().dropna(subset=output_features)

    if dataset == "UNSW-NB15":
        df_train = df_train.rename(columns={"label": "Label", "attack_cat": "Attack"})
        df_test = df_test.rename(columns={"label": "Label", "attack_cat": "Attack"})
        output_features = ["Label", "Attack"]
        allowed_values = [0, 255]
        df_train = df_train[
            df_train["dwin"].isin(allowed_values)
            & df_train["swin"].isin(allowed_values)
        ]
        df_test = df_test[
            df_test["dwin"].isin(allowed_values) & df_test["swin"].isin(allowed_values)
        ]

    categorical_cols = list(
        df_train.select_dtypes(include=["object", "category"]).columns
    )
    for target_var in output_features:
        if target_var in categorical_cols:
            categorical_cols.remove(target_var)

    for col in categorical_cols:
        df_train[col] = df_train[col].astype(str)
        df_test[col] = df_test[col].astype(str)

    for col in categorical_cols:
        unique_train_labels = set(df_train[col].astype(str).unique())
        unique_test_labels = set(df_test[col].astype(str).unique())
        all_labels = list(unique_train_labels.union(unique_test_labels))

        le = LabelEncoder()
        le.fit(all_labels)
        df_train[col] = le.transform(df_train[col].astype(str))
        df_test[col] = le.transform(df_test[col].astype(str))

    scaler = StandardScaler()
    df_train[input_features] = scaler.fit_transform(df_train[input_features])
    df_test[input_features] = scaler.transform(df_test[input_features])

    X_train = df_train[input_features]
    y_train = df_train[output_features]
    X_test = df_test[input_features]
    y_test = df_test[output_features]

    save_dir = os.path.join(BASE_DATA_DIR, "processed", dataset)
    os.makedirs(save_dir, exist_ok=True)

    X_train.to_csv(os.path.join(save_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(save_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(save_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(save_dir, "y_test.csv"), index=False)


if __name__ == "__main__":
    BASE_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
    datasets = ["NF-ToN-IoT", "UNSW-NB15"]

    for dataset in datasets:
        df_train, df_test = load_data(BASE_DATA_DIR, dataset)
        process_data(df_train, df_test)
