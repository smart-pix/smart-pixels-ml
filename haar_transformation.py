import os
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

haar16_matrix = np.array([
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1., -1., -1., -1.],
    [ 1.,  1.,  1.,  1., -1., -1., -1., -1.,  0.,  0.,  0.,  0., -0., -0., -0., -0.],
    [ 0.,  0.,  0.,  0., -0., -0., -0., -0.,  1.,  1.,  1.,  1., -1., -1., -1., -1.],
    [ 1.,  1., -1., -1.,  0.,  0., -0., -0.,  0.,  0., -0., -0.,  0.,  0., -0., -0.],
    [ 0.,  0., -0., -0.,  1.,  1., -1., -1.,  0.,  0., -0., -0.,  0.,  0., -0., -0.],
    [ 0.,  0., -0., -0.,  0.,  0., -0., -0.,  1.,  1., -1., -1.,  0.,  0., -0., -0.],
    [ 0.,  0., -0., -0.,  0.,  0., -0., -0.,  0.,  0., -0., -0.,  1.,  1., -1., -1.],
    [ 1., -1.,  0., -0.,  0., -0.,  0., -0.,  0., -0.,  0., -0.,  0., -0.,  0., -0.],
    [ 0., -0.,  1., -1.,  0., -0.,  0., -0.,  0., -0.,  0., -0.,  0., -0.,  0., -0.],
    [ 0., -0.,  0., -0.,  1., -1.,  0., -0.,  0., -0.,  0., -0.,  0., -0.,  0., -0.],
    [ 0., -0.,  0., -0.,  0., -0.,  1., -1.,  0., -0.,  0., -0.,  0., -0.,  0., -0.],
    [ 0., -0.,  0., -0.,  0., -0.,  0., -0.,  1., -1.,  0., -0.,  0., -0.,  0., -0.],
    [ 0., -0.,  0., -0.,  0., -0.,  0., -0.,  0., -0.,  1., -1.,  0., -0.,  0., -0.],
    [ 0., -0.,  0., -0.,  0., -0.,  0., -0.,  0., -0.,  0., -0.,  1., -1.,  0., -0.],
    [ 0., -0.,  0., -0.,  0., -0.,  0., -0.,  0., -0.,  0., -0.,  0., -0.,  1., -1.]
])

def padXH16(entry):
    values = entry[:13]
    padded_values = np.pad(values, (0, 16 - len(values)), 'constant')
    coeffs = np.dot(haar16_matrix, padded_values)
    return coeffs

def wavelet_transform_PlusOne(df):
    df_transformed = df.apply(lambda row: padXH16(row), axis=1)
    # Haar-wavelet 16 point
    df_transformed_padded = np.vstack(df_transformed.apply(lambda x: np.pad(np.append(x[:8], np.sum(x[8:16])), (0, 4), 'constant')).values)
    df.iloc[:len(df_transformed_padded), :13] = df_transformed_padded
    return df
    
def haar_transformation_processor(pitch):
    directory_path = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_parquets/unflipped/'.format(pitch)
    data_format = '3D'
    data2D_format = '2D'
    file_type = 'parquet'

    # recon3D files list
    recon3D_files = glob.glob(directory_path + "recon" + data_format + "*." + file_type)
    recon3D_files.sort()

    # recon2D files list
    recon2D_files = glob.glob(directory_path + "recon" + data2D_format + "*." + file_type)
    recon2D_files.sort()
    
    # join recon3D and labels lists together, sharing the same index for the corresponding file number
    joined_files = list(zip(recon3D_files, recon2D_files))
    
    # haar transform the parquet files, then save them to a new directory
    output_dir = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_haarTransformed_parquets/'.format(pitch)

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    output_path = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_haarTransformed_parquets/unflipped/'.format(pitch)
    
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    for i in tqdm(range(len(joined_files)), desc="Processing files for {}".format(pitch)):
        recon3D_filename = joined_files[i][0]
        recon2D_filename = joined_files[i][1]
    
        recon3D_temp_df = pd.read_parquet(recon3D_filename)
        recon2D_temp_df = pd.read_parquet(recon2D_filename)
    
        # Haar Transformation
        recon3D_df = wavelet_transform_PlusOne(recon3D_temp_df)
        recon2D_df = wavelet_transform_PlusOne(recon2D_temp_df)

        # save to new directory
        recon3D_df.to_parquet(output_path + recon3D_filename.split('/')[-1])
        recon2D_df.to_parquet(output_path + recon2D_filename.split('/')[-1])

if __name__ == '__main__':
    print('*** Haar Transformation File Processor ***')
    print('1. Takes in recon3D and recon2D parquet files of each sensor geometry') 
    print('2. Applies the Haar Transformation to each recon2D and recon3D of matching file number (row-wise)')
    print('3. Saves the processed parquet files to a new directory.')
    haar_transformation_processor('50x12P5')
    print('Processing done.')
