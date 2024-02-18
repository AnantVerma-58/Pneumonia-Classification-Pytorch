import os
# Set the Kaggle config directory
os.environ['KAGGLE_CONFIG_DIR'] = '~/.kaggle'

from kaggle.api.kaggle_api_extended import KaggleApi

# Specify the dataset name (replace <dataset-name>)
dataset_name = "paultimothymooney/chest-xray-pneumonia"

# Specify the path to save the downloaded data
save_path = "~/"


kaggle_api = KaggleApi()
# kaggle_api.dataset_list()
kaggle_api.dataset_download_files(dataset_name, path=save_path, unzip=True)