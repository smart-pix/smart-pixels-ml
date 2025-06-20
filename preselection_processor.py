#!/usr/bin/env python3
import numpy as np
import pandas as pd
import glob
import os
import shutil
from tqdm import tqdm
from pretrain_data_prep.dataset_utils import quantize_manual
from natsort import natsorted

def preselection_processing(input_directory, output_directory, file_type='parquet'):
    # Collect all simulation files
    simulation_files = natsorted(glob.glob(os.path.join(input_directory, f"*.{file_type}")))
    filenames = [os.path.basename(f) for f in simulation_files]

    # Clear or create the output directory
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
    os.mkdir(output_directory)

    # Process each file
    for i in tqdm(range(len(simulation_files)), desc="Processing files..."):
        temp_df = pd.read_parquet(simulation_files[i])
        filtered_df = temp_df[temp_df['original_atEdge'] == False]
        final_df = filtered_df.reset_index(drop=True)

        # Save to new directory
        output_path = os.path.join(output_directory, filenames[i])
        final_df.to_parquet(output_path)

if __name__ == '__main__':
    print('*** Preselection Processor ***')
    preselection_processing(input_directory='/data/dajiang/smart-pixels/dataset_3sr/shuffled/dataset_3sr_16x16_50x12P5_parquets/all/', 
                            output_directory='/data/dajiang/smart-pixels/dataset_3sr/shuffled/dataset_3sr_16x16_50x12P5_parquets/contained/')
    print('Processing done.')