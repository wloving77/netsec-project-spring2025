# /data

## Subdirectories:

- `raw/`:  
  Raw, unprocessed network traffic datasets. These files are downloaded as-is and not modified.

  UNSW-NB15 was downloaded with predefined train/test splits from https://www.kaggle.com/datasets/dhoogla/unswnb15/data. 
  NF-ToN-IoT-v1 was downloaded from https://www.kaggle.com/code/rnaveensrinivas/models/input

- `processed/`:  
  Cleaned and transformed versions of the raw datasets. Includes normalized numeric features, encoded categorical variables, and train/test splits. Used for model input.

- `synthetic/`:  
  Synthetic attack and benign samples generated by the GAN (CTGAN/TVAE). These augment the original data for training and evaluation.