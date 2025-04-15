import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def load_data(base_data_dir, train_path=None, test_path=None):
    train_path = train_path or os.path.join(base_data_dir, "data/raw_data/NUSW-NB15/Training_and_Testing_Splits/UNSW_NB15_training-set.csv")
    test_path = test_path or os.path.join(base_data_dir, "data/raw_data/NUSW-NB15/Training_and_Testing_Splits/UNSW_NB15_testing-set.csv")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    return df_train, df_test


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
            if len(real_data) < 10:
                print(f"Skipping {category}. Insufficient samples (<10) for generating synthetic data.")
                continue
            
            metadata = SingleTableMetadata()
            
            try:
                metadata.detect_from_dataframe(real_data)
                for col in self.categorical_cols:
                    # ensure categorical columns are treated as such by metadata object
                    if col in metadata.columns:
                        metadata.update_column(col, sdtype='categorical')
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


def process_data(df_train, df_test, target_variable, gan_augmentor=None, use_synthetic=True):
    df_train = df_train.drop_duplicates().dropna(subset=[target_variable])
    df_test = df_test.drop_duplicates().dropna(subset=[target_variable])
        
    categorical_cols = list(df_train.select_dtypes(include=['object', 'category']).columns)
    if target_variable in categorical_cols: 
        categorical_cols.remove(target_variable)

    for col in categorical_cols:
        df_train[col] = df_train[col].astype(str)
        df_test[col] = df_test[col].astype(str)
    
    if use_synthetic:
        if gan_augmentor is None:
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
    auth_features = ['dur', 'proto', 'state', 'spkts', 'dpkts', 
                     'sbytes', 'dbytes', 'rate', 'sload', 'dload', 
                     'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 
                     'djit', 'tcprtt', 'synack', 'ackdat']
    
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

    return X_train, X_test, y_train, y_test, le_target


class NetworkAnomalyDetector(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NetworkAnomalyDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


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
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # dataloaders for training and testing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train_tensor.shape[1]
    num_classes = len(le_target.classes_)
    
    print(f"number of classes: {num_classes}")
    model = NetworkAnomalyDetector(input_dim, num_classes)
    
    # cross-entropy loss for multiclass classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training model: {model_label}")
    
    # train
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
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
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)*100
    print("\nTest Accuracy: {:.2f}%".format(accuracy))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=le_target.classes_))
    
    plot_conf_matrix(all_labels, all_preds, le_target.classes_)
    
    return model


# if __name__ == "__main__": 
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
#     BASE_DATA_DIR = os.path.dirname(os.getcwd())
#     df_train, df_test = load_data(base_data_dir=BASE_DATA_DIR)
#     target_variable = 'attack_cat'
    
#     # with synthetic augmentation
#     X_train_syn, X_test_syn, y_train_syn, y_test_syn, le_target = process_data(
#         df_train, df_test, target_variable=target_variable, use_synthetic=True
#     )
        
#     model_syn = train_and_evaluate_model(
#         X_train_syn, y_train_syn, X_test_syn, y_test_syn, le_target,
#         model_label="With Synthetic Data"
#     )
    
#     # without synthetic augmentation
#     X_train_base, X_test_base, y_train_base, y_test_base, le_target = process_data(
#         df_train, df_test, target_variable=target_variable, use_synthetic=False
#     )
#     model_base = train_and_evaluate_model(
#         X_train_base, y_train_base, X_test_base, y_test_base, le_target,
#         model_label="Without Synthetic Data"
#     )