import numpy as np
import torch, os, warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from gan_ids_project.src.models import NetworkAnomalyDetector, DeeperNetworkAnomalyDetector
from gan_ids_project.src.data_augmentors import CTGANAugmentor, TVAEAugmentor

def load_data(dataset, target_variable):
    base_data_dir = os.path.dirname(os.path.abspath(__file__))

    processed_dir = os.path.join(base_data_dir, "..", "data", "processed", dataset)
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    
    unused_features = y_train.drop(columns=[target_variable])
    y_train.drop(columns=unused_features, inplace=True)
    y_test.drop(columns=unused_features, inplace=True)

    return X_train, X_test, y_train, y_test


def make_synthetic_data(
        X_train,
        y_train,
        X_test,
        y_test, 
        target_variable, 
        gan_augmentor, 
    ):
    
    base_data_dir = os.path.dirname(os.path.abspath(__file__))
    synthetic_data_dir=os.path.join(base_data_dir, "..", "data", "synthetic")

    df_train = X_train.copy()
    df_train[target_variable] = y_train
    
    categorical_cols = list(df_train.select_dtypes(include=['object', 'category']).columns)
        
    if gan_augmentor is not None: 
        if gan_augmentor.name == "TVAE":
            gan_augmentor.df_train = df_train
            gan_augmentor.categorical_cols = categorical_cols
            gan_augmentor.target_variable = target_variable
            gan_augmentor.synthetic_data_dir = synthetic_data_dir
        elif gan_augmentor.name == "CTGAN":
            gan_augmentor.df_train = df_train
            gan_augmentor.categorical_cols = categorical_cols
            gan_augmentor.target_variable = target_variable
            gan_augmentor.synthetic_data_dir = synthetic_data_dir
        new_df = gan_augmentor.generate_augmented_data()
    else: 
        print("No GAN augmentor specified. Using original data.")
        new_df = df_train.copy()
    
    X_syn_df = new_df.drop(columns=[target_variable]).squeeze()
    y_syn_df = new_df[[target_variable]].squeeze()
    
    le_target = LabelEncoder()
    y_syn_df = le_target.fit_transform(y_syn_df)
    y_test = le_target.transform(y_test)

    return X_syn_df, y_syn_df, X_test, y_test, le_target


def plot_conf_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def train_and_evaluate_model(
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        le_target, 
        model, 
        epochs=15, 
        batch_size=64, 
        lr=0.001
    ):    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    
    num_classes = len(np.unique(y_train, return_counts=True)[1])
    print(f"number of classes: {num_classes}")
    
    if num_classes == 2:
        y_train_tensor = torch.tensor(y_train.squeeze(), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.squeeze(), dtype=torch.float32)
    else:
        y_train_tensor = torch.tensor(y_train.squeeze(), dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.squeeze(), dtype=torch.long)
    
    # dataloaders for training and testing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train_tensor.shape[1]
    model = model(input_dim, num_classes)
    
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
        loss = "Binary Cross Entropy with Logits"
    else:
        criterion = nn.CrossEntropyLoss()
        loss = "Cross Entropy"
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Loss function: {loss}")
    
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
    plot_conf_matrix(all_labels, all_preds, [str(c) for c in le_target.classes_])
    
    return model


if __name__ == "__main__":     
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    dataset = "UNSW-NB15"                       # "UNSW-NB15" is smaller, "NF-ToN-IoT-v1" is larger
    target_variable = 'Label'                   #  "Label" for binary, "Attack" for multi class
    augmentor_ctgan = CTGANAugmentor()
    augmentor_ctgan.minority_threshold = 20000

    augmentor_tvae = TVAEAugmentor()
    augmentor_tvae.minority_threshold = 20000

    augmentors = [augmentor_ctgan, augmentor_tvae] 
    model_type = DeeperNetworkAnomalyDetector   # NetworkAnomalyDetector, DeeperNetworkAnomalyDetector

    X_train, X_test, y_train, y_test = load_data(dataset, target_variable)


    for augmentor in augmentors:
        X_train_tvae, y_train_tvae, X_test_tvae, y_test_tvae, le_target = make_synthetic_data(
            X_train,
            y_train,
            X_test,
            y_test, 
            target_variable, 
            gan_augmentor=augmentor, 
        )

        model_tvae = train_and_evaluate_model(
            X_train_tvae, 
            y_train_tvae, 
            X_test_tvae, 
            y_test_tvae, 
            le_target,
            model=model_type
        )