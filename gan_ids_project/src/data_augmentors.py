from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd
import os

class CTGANAugmentor:
    """Generates synthetic data for minority classes using CTGAN."""
    name = "CTGAN"
    def __init__(
            self, 
            df_train=None, 
            categorical_cols=None, 
            target_variable=None, 
            synthetic_data_dir=None, 
            minority_threshold=10000, 
            epochs=50
        ):
        self.df_train = df_train
        self.categorical_cols = categorical_cols
        self.target_variable = target_variable
        self.synthetic_data_dir = synthetic_data_dir
        self.minority_threshold = minority_threshold
        self.epochs = epochs

    def generate_augmented_data(self):
        # Find minority classes (excluding normal/majority)
        target_counts = self.df_train[self.target_variable].value_counts()
        minority_cats = target_counts[target_counts < self.minority_threshold].index.tolist()
        synthetic_dfs = []
        
        for category in minority_cats:
            real_data = self.df_train[self.df_train[self.target_variable] == category]
            if len(real_data) < 0.001*len(self.df_train):
                print(f"Skipping {category}: not enough samples ({0.001*len(self.df_train)}) for synthetic generation.")
                continue

            print(f"Generating synthetic data for category: {category}")

            metadata = SingleTableMetadata()
            try:
                metadata.detect_from_dataframe(real_data)
                for col in real_data.columns:
                    if col in self.categorical_cols:
                        metadata.update_column(col, sdtype='categorical')
                    elif pd.api.types.is_integer_dtype(real_data[col]):
                        metadata.update_column(col, sdtype='integer')
                    elif pd.api.types.is_float_dtype(real_data[col]):
                        metadata.update_column(col, sdtype='float')
            except Exception as e:
                print(f"Metadata error for {category}: {str(e)}")
                continue

            ctgan = CTGANSynthesizer(metadata, epochs=self.epochs)
            ctgan.fit(real_data)
            num_samples = self.minority_threshold - len(real_data)
            synthetic_data = ctgan.sample(num_rows=num_samples)
            synthetic_dfs.append(synthetic_data)

        if synthetic_dfs:
            synthetic_df = pd.concat(synthetic_dfs, ignore_index=True)
            print("Size of synthetic data:", synthetic_df.shape)
            os.makedirs(self.synthetic_data_dir, exist_ok=True)
            synthetic_data.to_csv(os.path.join(self.synthetic_data_dir, "synthetic_data.csv"), index=False)
            augmented_train = pd.concat([self.df_train, synthetic_df], ignore_index=True)
        else:
            print("No synthetic data generated.")
            return self.df_train

        return augmented_train


class TVAEAugmentor:
    """Generates synthetic data for minority classes using TVAE."""
    
    name = "TVAE"
    def __init__(
            self, 
            df_train=None, 
            categorical_cols=None, 
            target_variable=None, 
            synthetic_data_dir=None, 
            minority_threshold=10000, 
            epochs=50
        ):
        self.df_train = df_train
        self.categorical_cols = categorical_cols
        self.target_variable = target_variable
        self.synthetic_data_dir = synthetic_data_dir
        self.minority_threshold = minority_threshold
        self.epochs = epochs
        self.name = "TVAE"

    def generate_augmented_data(self):
        target_counts = self.df_train[self.target_variable].value_counts()
        minority_cats = target_counts[target_counts < self.minority_threshold].index.tolist()
        synthetic_dfs = []

        for category in minority_cats:
            real_data = self.df_train[self.df_train[self.target_variable] == category]
            if len(real_data) < 10:
                print(f"Skipping {category}: not enough samples (<10) for synthetic generation.")
                continue

            print(f"Generating synthetic data for category: {category}")

            metadata = SingleTableMetadata()
            try:
                metadata.detect_from_dataframe(real_data)
                for col in real_data.columns:
                    if col in self.categorical_cols:
                        metadata.update_column(col, sdtype='categorical')
                    elif pd.api.types.is_integer_dtype(real_data[col]):
                        metadata.update_column(col, sdtype='integer')
                    elif pd.api.types.is_float_dtype(real_data[col]):
                        metadata.update_column(col, sdtype='numerical')
            except Exception as e:
                print(f"Metadata error for {category}: {str(e)}")
                continue

            tvae = TVAESynthesizer(metadata, epochs=self.epochs)
            tvae.fit(real_data)
            num_samples = self.minority_threshold - len(real_data)
            synthetic_data = tvae.sample(num_rows=num_samples)
            synthetic_dfs.append(synthetic_data)

        if synthetic_dfs:
            synthetic_df = pd.concat(synthetic_dfs, ignore_index=True)
            print("Size of synthetic data:", synthetic_df.shape)
            os.makedirs(self.synthetic_data_dir, exist_ok=True)
            synthetic_data.to_csv(os.path.join(self.synthetic_data_dir, "synthetic_data.csv"), index=False)
            augmented_train = pd.concat([self.df_train, synthetic_df], ignore_index=True)
        else:
            print("No synthetic data generated.")
            return self.df_train
            
        return augmented_train