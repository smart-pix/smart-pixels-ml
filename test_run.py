# Import necessary modules

# Check all the dependencies are installed
import tensorflow as tf
from models import *
from loss import *
from OptimizedDataGenerator import *
import numpy as np
import pandas as pd
import qkeras
import shutil
import os

batch_size = 8
input_shape = (13, 21, 2)

try:
    shutil.rmtree("./testing_tmp")
except:
    pass
os.mkdir("./testing_tmp")
data_directory_path = "./testing_tmp/data/"
labels_directory_path = "./testing_tmp/labels/"
tfrecords_dir = "./testing_tmp/tfrecords/"


data_file = os.path.join(data_directory_path, "recon3D_data.parquet")
labels_file = os.path.join(labels_directory_path, "labels_data.parquet")

os.makedirs(data_directory_path, exist_ok=True)
os.makedirs(labels_directory_path, exist_ok=True)
os.makedirs(tfrecords_dir, exist_ok=True)
os.makedirs(data_directory_path, exist_ok=True)
os.makedirs(labels_directory_path, exist_ok=True)

def generate_random_data():
    """
    Generate synthetic data and labels matching the expected format
    for OptimizedDataGenerator. Saves files in the required structure.
    """
    # Create synthetic data and labels
    num_samples = 8  # Matches batch_size
    input_shape = (13, 21, 2)  # Includes time stamps
    labels_list = ['x-midplane', 'y-midplane', 'cotAlpha', 'cotBeta']

    # Generate synthetic data
    data = np.random.rand(num_samples, *input_shape).astype(np.float32)

    # Flatten each sample and create a DataFrame
    flat_data = data.reshape(num_samples, -1)
    column_names = [str(i) for i in range(flat_data.shape[1])]  # Numeric column names for use_time_stamps
    data_df = pd.DataFrame(flat_data, columns=column_names)

    # Generate synthetic labels
    labels = pd.DataFrame(
        np.random.rand(num_samples, len(labels_list)),
        columns=labels_list
    )

    # Add 'event_id' column
    labels['event_id'] = np.arange(num_samples)
    data_df['event_id'] = labels['event_id']

    # Save data and labels in parquet format without the index
    data_df.to_parquet(data_file, index=False)
    labels.to_parquet(labels_file, index=False)

    print(f"Generated synthetic data file: {data_file}")
    print(f"Generated synthetic labels file: {labels_file}")


def testing_pandas_reading():
    data_df = pd.read_parquet(data_file)
    labels_df = pd.read_parquet(labels_file)

    print("Data shape and head:")
    print(data_df.shape)
    print(data_df.head())

    print("Labels shape and head:")
    print(labels_df.shape)
    print(labels_df.head())



def initialize_data_generator():
    generator = OptimizedDataGenerator(
                    data_directory_path = data_directory_path,
                    labels_directory_path = labels_directory_path,
                    is_directory_recursive = False,
                    file_type = "parquet",
                    data_format = "3D",
                    batch_size = batch_size,
                    file_count = 1,
                    to_standardize= True,
                    include_y_local= False,
                    labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
                    input_shape = (2,13,21), # (20,13,21),
                    transpose = (0,2,3,1),
                    shuffle = False, 
                    files_from_end=True,

                    tfrecords_dir = tfrecords_dir,
                    use_time_stamps = [0, 19], #-1
                    max_workers = 1, # Don't make this too large (will use up all RAM)
                    seed = 10, 
                    quantize = True # Quantization ON
                )
    return generator



if __name__ == "__main__":
    generate_random_data()
    testing_pandas_reading()
    generator = initialize_data_generator() 