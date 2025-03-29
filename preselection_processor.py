import numpy as np
import pandas as pd
import glob
import os
import shutil
from tqdm import tqdm

# Make sure to edit the paths in preselection_processing() for each dataset you process!

def preselection_processing(pitch):
    directory_path = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_parquets/unflipped/'.format(pitch)
    data_format = '3D'
    data2D_format = '2D'
    file_type = 'parquet'
    
    # recon3D files list
    recon3D_files = glob.glob(directory_path + "recon" + data_format + "*." + file_type)
    recon3D_files.sort()

    # recon2D files list
    recon2D_files = glob.glob(directory_path + "recon" + data2D_format + "*." + file_type)
    recon2D_files.sort()
    
    # labels files list
    labels_files = [directory_path + recon3D_file.split('/')[-1].replace("recon" + data_format, "labels") for recon3D_file in recon3D_files]
    
    # join recon3D and labels lists together, sharing the same index for the corresponding file number
    joined_files = list(zip(recon3D_files, recon2D_files, labels_files))
    
    # filter the parquet files, then save them to a new directory
    output_dir = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_parquets/'.format(pitch)

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    output_path = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_parquets/unflipped/'.format(pitch)
    
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    for i in tqdm(range(len(joined_files)), desc="Processing files for {}".format(pitch)):
        recon3D_filename = joined_files[i][0]
        recon2D_filename = joined_files[i][1]
        labels_filename = joined_files[i][2]
    
        recon3D_temp_df = pd.read_parquet(recon3D_filename)
        recon2D_temp_df = pd.read_parquet(recon2D_filename)
        labels_temp_df = pd.read_parquet(labels_filename)
    
        # preselections
        preselections = abs(labels_temp_df['cotBeta']) <= 1.5
        recon3D_df = recon3D_temp_df[preselections].reset_index(drop=True)
        recon2D_df = recon2D_temp_df[preselections].reset_index(drop=True)
        labels_df = labels_temp_df[preselections].reset_index(drop=True)

        # save to new directory
        recon3D_df.to_parquet(output_path + recon3D_filename.split('/')[-1])
        recon2D_df.to_parquet(output_path + recon2D_filename.split('/')[-1])
        labels_df.to_parquet(output_path + labels_filename.split('/')[-1])

if __name__ == '__main__':
    print('*** Preselection Processor ***')
    print('1. Takes in recon3D, recon2D, and labels parquet files of each sensor geometry') 
    print('2. Applies preselection cuts to recon3D/recon2D/labels files of matching file number')
    print('3. Saves the processed parquet files to a new directory.')
    preselection_processing('100x25x150')
    preselection_processing('100x25')
    preselection_processing('50x25')
    preselection_processing('50x20')
    preselection_processing('50x15')
    preselection_processing('50x12P5')
    preselection_processing('50x10')
    print('Processing done.')
