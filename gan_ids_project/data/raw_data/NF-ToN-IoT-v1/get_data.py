import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_parquet("/Users/axelgyllenhoff/Desktop/Spring_2025/Network_Security_and_Privacy/netsec-project-spring2025/gan_ids_project/data/raw_data/NF-ToN-IoT-v1/NF-ToN-IoT.parquet")

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

output_dir = "/Users/axelgyllenhoff/Desktop/Spring_2025/Network_Security_and_Privacy/netsec-project-spring2025/gan_ids_project/data/raw_data/NF-ToN-IoT-v1/Splits"
os.makedirs(output_dir, exist_ok=True)
train_df.to_csv(f"{output_dir}/NF-ToN-IoT-v1_training-set.csv", index=False)
test_df.to_csv(f"{output_dir}/NF-ToN-IoT-v1_testing-set.csv", index=False)