import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_data(base_data_dir, train_path=None, test_path=None):
    train_path = train_path or os.path.join(base_data_dir, "data/raw_data/NUSW-NB15/Training_and_Testing_Splits/UNSW_NB15_training-set.csv")
    test_path = test_path or os.path.join(base_data_dir, "data/raw_data/NUSW-NB15/Training_and_Testing_Splits/UNSW_NB15_testing-set.csv")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    return df_train, df_test


class TVAEAugmentor:
    """Generates synthetic data for minority attack categories using TVAE"""
    
    def __init__(self, df_train, categorical_cols, target_variable, minority_threshold=10000, epochs=50):
        self.minority_threshold = minority_threshold
        self.df_train = df_train.copy()
        self.epochs = epochs
        self.categorical_cols = categorical_cols
        self.target_variable = target_variable
        
    def generate_augmented_data(self):
        # extract the attack records as labeled (assuming label==1 indicates an attack)
        train_attacks = self.df_train[self.df_train['label'] == 1].copy()
        target_variable_counts = train_attacks[self.target_variable].value_counts()
        
        minority_cats = target_variable_counts[target_variable_counts < self.minority_threshold].index.tolist()
        synthetic_dfs = []
        
        for category in minority_cats:
            real_data = train_attacks[train_attacks[self.target_variable] == category]
            if len(real_data) < 10:
                print(f"Skipping {category}. Insufficient samples (<10) for generating synthetic data.")
                continue
            
            metadata = SingleTableMetadata()
            
            try:
                metadata.detect_from_dataframe(real_data)
                for col in real_data.columns:
                    if col in self.categorical_cols:
                        metadata.update_column(col, sdtype='categorical')
                    else:
                        if pd.api.types.is_integer_dtype(real_data[col]):
                            metadata.update_column(col, sdtype='integer')
                        elif pd.api.types.is_float_dtype(real_data[col]):
                            metadata.update_column(col, sdtype='float')
            except Exception as e:
                print(f"Metadata error for {category}: {str(e)}")
                continue
                        
            # initialize and fit TVAE synthesizer
            tvae = TVAESynthesizer(metadata, epochs=self.epochs)
            tvae.fit(real_data)
            
            # generate synthetic samples to reach the minority threshold
            num_samples = self.minority_threshold - len(real_data)
            synthetic_data = tvae.sample(num_rows=num_samples)
            synthetic_dfs.append(synthetic_data)
        
        # combine datasets
        if synthetic_dfs:
            synthetic_df = pd.concat(synthetic_dfs, ignore_index=True)
            augmented_train = pd.concat([self.df_train, synthetic_df], ignore_index=True)
        else:
            augmented_train = self.df_train.copy()
        
        return augmented_train


class CTGANAugmentor:
    """Generates synthetic data for minority attack categories using CTGAN"""
    
    def __init__(self, df_train, categorical_cols, target_variable, minority_threshold=5000, epochs=50):
        self.minority_threshold = minority_threshold
        self.df_train = df_train.copy()
        self.epochs = epochs
        self.categorical_cols = categorical_cols
        self.target_variable = target_variable
        
    def generate_augmented_data(self):
        # extract the attack records as labeled (assuming label==1 indicates an attack)
        train_attacks = self.df_train[self.df_train['label'] == 1].copy()
        target_variable_counts = train_attacks[self.target_variable].value_counts()
        
        minority_cats = target_variable_counts[target_variable_counts < self.minority_threshold].index.tolist()
        synthetic_dfs = []
        
        for category in minority_cats:
            real_data = train_attacks[train_attacks[self.target_variable] == category]
            if len(real_data) < 100:
                print(f"Skipping {category}. Insufficient samples (<10) for generating synthetic data.")
                continue
            
            print("Generating synthetic data for category: ", category)

            metadata = SingleTableMetadata()
            
            try:
                metadata.detect_from_dataframe(real_data)
                for col in real_data.columns:
                    if col in self.categorical_cols:
                        metadata.update_column(col, sdtype='categorical')
                    else:
                        if pd.api.types.is_integer_dtype(real_data[col]):
                            metadata.update_column(col, sdtype='integer')
                        elif pd.api.types.is_float_dtype(real_data[col]):
                            metadata.update_column(col, sdtype='float')
            except Exception as e:
                print(f"Metadata error for {category}: {str(e)}")
                continue
                                    
            # initialize and fit CTGAN synthesizer
            ctgan = CTGANSynthesizer(metadata, epochs=self.epochs)
            ctgan.fit(real_data)
            
            # generate synthetic samples to reach the minority threshold
            num_samples = self.minority_threshold - len(real_data)
            synthetic_data = ctgan.sample(num_rows=num_samples)
            synthetic_dfs.append(synthetic_data)
        
        # combine datasets
        if synthetic_dfs:
            synthetic_df = pd.concat(synthetic_dfs, ignore_index=True)
            augmented_train = pd.concat([self.df_train, synthetic_df], ignore_index=True)
        else:
            augmented_train = self.df_train.copy()
        
        return augmented_train


def process_data(
        df_train, 
        df_test, 
        target_variable, 
        gan_augmentor=None, 
        use_synthetic=True
    ):
    df_train = df_train.drop_duplicates().dropna(subset=[target_variable])
    df_test = df_test.drop_duplicates().dropna(subset=[target_variable])
    
    # Drop rows where dwin or swin is not 0 or 255 ... only a few of them
    # don't want them memorized
    allowed_values = [0, 255]
    df_train = df_train[df_train['dwin'].isin(allowed_values) & df_train['swin'].isin(allowed_values)]
    df_test = df_test[df_test['dwin'].isin(allowed_values) & df_test['swin'].isin(allowed_values)]
    
    categorical_cols = list(df_train.select_dtypes(include=['object', 'category']).columns)
    if target_variable in categorical_cols: 
        categorical_cols.remove(target_variable)
        
    for col in categorical_cols:
        df_train[col] = df_train[col].astype(str)
        df_test[col] = df_test[col].astype(str)
    
    if use_synthetic:
        if gan_augmentor=="TVAE": 
            gan_augmentor = TVAEAugmentor(df_train, categorical_cols, target_variable)
        else: 
            gan_augmentor = CTGANAugmentor(df_train, categorical_cols, target_variable)
        df_train = gan_augmentor.generate_augmented_data()
    
    # lavel encode categorical features
    for col in categorical_cols:
        unique_train_labels = set(df_train[col].astype(str).unique())
        unique_test_labels = set(df_test[col].astype(str).unique())
        all_labels = list(unique_train_labels.union(unique_test_labels))
        
        le = LabelEncoder()
        le.fit(all_labels)
        df_train[col] = le.transform(df_train[col].astype(str))
        df_test[col] = le.transform(df_test[col].astype(str))
        
    # features to be used
    auth_features = [
        'dur',              # total connection duration
        'proto',            # transaction protocol
        'service',          # destination service/type
        'state',            # connection state
        'spkts',            # packets from src → dst
        'dpkts',            # packets from dst → src
        'sbytes',           # bytes from src → dst
        'dbytes',           # bytes from dst → src
        'rate',             # packets per second
        'sload',            # source bits per second
        'dload',            # dest bits per second
        'sloss',            # src packet loss count
        'dloss',            # dest packet loss count
        'sinpkt',           # src inter‑packet arrival time (ms)
        'dinpkt',           # dest inter‑packet arrival time (ms)
        'sjit',             # source jitter (ms)
        'djit',             # dest jitter (ms)
        'tcprtt',           # TCP round‑trip time (ms)
        'synack',           # time between SYN and SYN‑ACK (ms)
        'ackdat',           # time between SYN‑ACK and ACK (ms)
        'swin',             # src TCP window size
        'dwin',             # dest TCP window size
        'stcpb',            # src TCP base sequence num
        'dtcpb',            # dest TCP base sequence num
        'smean',            # (smeansz) mean src packet size
        'dmean',            # (dmeansz) mean dest packet size
        'trans_depth',      # HTTP transaction depth
        'response_body_len',# (res_bdy_len) HTTP response body length
        'is_ftp_login',     # FTP login flag
        'is_sm_ips_ports',  # same src/dst IPs & ports flag
        'ct_state_ttl',     # count flows by state & TTL
        'ct_flw_http_mthd', # count HTTP methods per flow
        'ct_ftp_cmd',       # count FTP commands per flow
        'ct_srv_src',       # count recent service–src pairs
        'ct_srv_dst',       # count recent service–dst pairs
        'ct_dst_ltm',       # count recent dest addresses
        'ct_src_ltm',       # count recent src addresses
        'ct_src_dport_ltm', # count recent src–dport pairs
        'ct_dst_sport_ltm', # count recent dst–sport pairs
        'ct_dst_src_ltm'    # count recent src–dst pairs
    ]
    
    scaler = StandardScaler()
    df_train[auth_features] = scaler.fit_transform(df_train[auth_features])
    df_test[auth_features] = scaler.transform(df_test[auth_features])
        
    X_train = df_train[auth_features]
    y_train = df_train[target_variable]
    X_test = df_test[auth_features]
    y_test = df_test[target_variable]
    
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    y_test = le_target.transform(y_test)

    return X_train, y_train, X_test, y_test, le_target


class NetworkAnomalyDetector(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NetworkAnomalyDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        if num_classes == 2:
            self.fc3 = nn.Linear(32, 1)
        else:
            self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DeeperNetworkAnomalyDetector(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        if num_classes == 2:
            self.fc4 = nn.Linear(32, 1)
        else:
            self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


def plot_conf_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def train_and_evaluate_model(X_train, y_train, X_test, y_test, le_target, model_label, epochs=15, batch_size=64, lr=0.001):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    num_classes = len(le_target.classes_)

    if num_classes == 2:
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    else:
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # dataloaders for training and testing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train_tensor.shape[1]
    num_classes = len(le_target.classes_)
    print(f"number of classes: {num_classes}")
    # model = NetworkAnomalyDetector(input_dim, num_classes)
    model = DeeperNetworkAnomalyDetector(input_dim, num_classes)
    
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
        loss = "Binary Cross Entropy with Logits"
    else:
        criterion = nn.CrossEntropyLoss()
        loss = "Cross Entropy"
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Training model {model_label} using {loss}")
        
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # eval on test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            if num_classes == 2:
                predicted = (torch.sigmoid(outputs) > 0.5).long()
            else:
                _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)*100
    print("\nTest Accuracy: {:.2f}%".format(accuracy))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[str(c) for c in le_target.classes_]))
    # plot_conf_matrix(all_labels, all_preds, [str(c) for c in le_target.classes_])
    
    return model


if __name__ == "__main__": 
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    BASE_DATA_DIR = os.path.dirname(os.getcwd())
    df_train, df_test = load_data(base_data_dir=BASE_DATA_DIR)
    target_variable = 'attack_cat' # 'label' for binary classification
        
    # without synthetic augmentation
    X_train_base, y_train_base, X_test_base, y_test_base, le_target = process_data(
        df_train, 
        df_test, 
        target_variable=target_variable, 
        use_synthetic=False
    )
    model_base = train_and_evaluate_model(
        X_train_base, y_train_base, X_test_base, y_test_base, le_target,
        model_label="Without Synthetic Data"
    )
    
    # with synthetic augmentation from CTGAN
    X_train_ctgan, y_train_ctgan, X_test_ctgan, y_test_ctgan, le_target = process_data(
        df_train, 
        df_test, 
        target_variable=target_variable, 
        use_synthetic=True,
        gan_augmentor="CTGAN"
    )
        
    model_syn = train_and_evaluate_model(
        X_train_ctgan, y_train_ctgan, X_test_ctgan, y_test_ctgan, le_target,
        model_label="With Synthetic Data"
    )
    
    # with synthetic augmentation from TVAE
    X_train_tvae, y_train_tvae, X_test_tvae, y_test_tvae, le_target = process_data(
        df_train, 
        df_test, 
        target_variable=target_variable, 
        use_synthetic=True,
        gan_augmentor="TVAE"
    )
    model_tvae = train_and_evaluate_model(
        X_train_tvae, y_train_tvae, X_test_tvae, y_test_tvae, le_target,
        model_label="With Synthetic Data (TVAE)"
    )
