#!/usr/bin/env python3
import numpy as np
import pandas as pd
import glob
import os
import shutil
from tqdm import tqdm
from pretrain_data_prep.dataset_utils import quantize_manual

# Make sure to edit the paths in preselection_processing() for each dataset you process!

def preselection_processing(pitch):
    directory_path = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_parquets/unflipped/'.format(pitch)
    data_format = '3D'
    file_type = 'parquet'
    
    # recon3D files list
    recon3D_files = glob.glob(directory_path + "recon" + data_format + "*." + file_type)
    recon3D_files.sort()
    
    # labels files list
    labels_files = [directory_path + recon3D_file.split('/')[-1].replace("recon" + data_format, "labels") for recon3D_file in recon3D_files]
    
    # join recon3D and labels lists together, sharing the same index for the corresponding file number
    joined_files = list(zip(recon3D_files, labels_files))
    
    # filter the parquet files, then save them to a new directory
    output_dir = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_quantize_manual_400_1515_2971_parquets/'.format(pitch)

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    output_path = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_quantize_manual_400_1515_2971_parquets/unflipped/'.format(pitch)
    
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    for i in tqdm(range(len(joined_files)), desc="Processing files for {}".format(pitch)):
        recon3D_filename = joined_files[i][0]
        labels_filename = joined_files[i][1]
    
        recon3D_temp_df = pd.read_parquet(recon3D_filename)
        labels_temp_df = pd.read_parquet(labels_filename)

        charge_levels = [400, 1515, 2971]
        quant_values= [0, 0.25, 0.5, 0.75]
        recon3D_df = quantize_manual(recon3D_temp_df, charge_levels=charge_levels, quant_values=quant_values)
        recon3D_df.columns = [f'{i}' for i in range(recon3D_df.shape[1])]
        labels_df = labels_temp_df

        # save to new directory
        recon3D_df.to_parquet(output_path + recon3D_filename.split('/')[-1])
        labels_df.to_parquet(output_path + labels_filename.split('/')[-1])

if __name__ == '__main__':
    print('*** Preselection Processor ***')
    print('1. Takes in recon3D, recon2D, and labels parquet files of each sensor geometry') 
    print('2. Applies preselections to recon3D/recon2D/labels files of matching file number')
    print('3. Saves the processed parquet files to a new directory.')
    preselection_processing('50x12P5')
    print('Processing done.')