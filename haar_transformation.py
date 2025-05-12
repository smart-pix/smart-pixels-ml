import os
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

def to_pixel_array(parquet_file_path, format='3D'):
    temp_df = pd.read_parquet(parquet_file_path)
    temp_array = temp_df.to_numpy()
    if format == '3D':
        pixel_array = temp_array.reshape(-1,20,13,21)
        return pixel_array
    #elif format == '2D':
    #    pixel_array = temp_array.reshape(-1,13,21)
    #    return pixel_array

def haarMatrix(n, normalized=False):
    # Allow only size n of power 2
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = haarMatrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    # calculate upper haar part
    h_n = np.kron(h, [1, 1])
    # calculate lower haar part 
    if normalized:
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    return h

haar_16x16 = haarMatrix(16) # for column-wise
haar_32x32 = haarMatrix(32) # for row-wise

def haar_16x16_transform(column):
    padded_values = np.pad(column, (0, 16 - len(column)), 'constant')
    coeffs = np.dot(haar_16x16, padded_values)
    top8 = coeffs[:8]
    remaining_sum = np.array([sum(coeffs[8:])])
    padding = np.zeros(len(column) - 9)
    return np.concatenate([top8, remaining_sum, padding]) # returns array of 13 values

def haar_32x32_transform(row):
    padded_values = np.pad(row, (0, 32 - len(row)), 'constant')
    coeffs = np.dot(haar_32x32, padded_values)
    top12 = coeffs[:12]
    remaining_sum = np.array([sum(coeffs[12:])])
    padding = np.zeros(len(row) - 13)
    return np.concatenate([top12, remaining_sum, padding]) # returns array of 21 values 

def apply_transformations(pixel_arrays, row_or_column='column'):
    if row_or_column == 'column':
        return np.apply_along_axis(haar_16x16_transform, axis=2, arr=pixel_arrays)
    elif row_or_column == 'row':
        return np.apply_along_axis(haar_32x32_transform, axis=3, arr=pixel_arrays)
    
def array_to_parquet(pixel_arrays, output_path):
    flattened_arrays = pixel_arrays.reshape(pixel_arrays.shape[0], -1)
    output_df = pd.DataFrame(flattened_arrays)
    output_df.columns = [f'{i}' for i in range(output_df.shape[1])]
    output_df.to_parquet(output_path, index=False, engine='fastparquet')

def haar_transformation_processor(pitch):
    directory_path = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_parquets/unflipped/'.format(pitch)
    data_format = '3D'
    #data2D_format = '2D'
    file_type = 'parquet'

    # recon3D files list
    recon3D_files = glob.glob(directory_path + "recon" + data_format + "*." + file_type)
    recon3D_files.sort()

    # recon2D files list
    #recon2D_files = glob.glob(directory_path + "recon" + data2D_format + "*." + file_type)
    #recon2D_files.sort()
    
    # join recon3D and labels lists together, sharing the same index for the corresponding file number
    #joined_files = list(zip(recon3D_files, recon2D_files))
    
    # haar transform the parquet files, then save them to a new directory
    output_dir_colwise = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_haarTransformed_colwise_parquets/'.format(pitch)
    output_dir_rowwise = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_haarTransformed_rowwise_parquets/'.format(pitch)

    if os.path.isdir(output_dir_colwise):
        shutil.rmtree(output_dir_colwise)
    os.mkdir(output_dir_colwise)

    if os.path.isdir(output_dir_rowwise):
        shutil.rmtree(output_dir_rowwise)
    os.mkdir(output_dir_rowwise)

    output_path_colwise = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_haarTransformed_colwise_parquets/unflipped/'.format(pitch)
    output_path_rowwise = '/data/dajiang/smart-pixels/dataset_3sr/dataset_3sr_{}_cotBeta1P5_haarTransformed_rowwise_parquets/unflipped/'.format(pitch)

    if os.path.isdir(output_path_colwise):
        shutil.rmtree(output_path_colwise)
    os.mkdir(output_path_colwise)

    if os.path.isdir(output_path_rowwise):
        shutil.rmtree(output_path_rowwise)
    os.mkdir(output_path_rowwise)

    #for i in tqdm(range(len(joined_files)), desc="Processing files for {}".format(pitch)):
        #recon3D_filename = joined_files[i][0]
        #recon2D_filename = joined_files[i][1]
    
    for i in tqdm(range(len(recon3D_files)), desc="Processing files for {}".format(pitch)):
        recon3D_filename = recon3D_files[i]

        # Convert parquet files to pixel arrays of shape (num_events, 20, 13, 21) or (num_events, 2, 13, 21)
        recon3D_pixel_arrays = to_pixel_array(recon3D_filename, format='3D')
        #recon2D_pixel_arrays = to_pixel_array(recon2D_filename, format='2D')
    
        # Apply Haar Transformations
        recon3D_pixel_arrays_colwise = apply_transformations(recon3D_pixel_arrays, row_or_column='column')
        #recon2D_pixel_arrays_colwise = apply_transformations(recon2D_pixel_arrays, row_or_column='column')
        recon3D_pixel_arrays_rowwise = apply_transformations(recon3D_pixel_arrays, row_or_column='row')
        #recon2D_pixel_arrays_rowwise = apply_transformations(recon2D_pixel_arrays, row_or_column='row')

        # save to new directory
        array_to_parquet(recon3D_pixel_arrays_colwise, output_path_colwise + recon3D_filename.split('/')[-1])
        #array_to_parquet(recon2D_pixel_arrays_colwise, output_path_colwise + recon2D_filename.split('/')[-1])
        array_to_parquet(recon3D_pixel_arrays_rowwise, output_path_rowwise + recon3D_filename.split('/')[-1])
        #array_to_parquet(recon2D_pixel_arrays_rowwise, output_path_rowwise + recon2D_filename.split('/')[-1])

if __name__ == '__main__':
    print('*** Haar Transformation File Processor ***')
    print('1. Takes in recon3D and recon2D parquet files of each sensor geometry') 
    print('2. Applies the Haar Transformation to each recon2D and recon3D of matching file number (column-wise and row-wise)')
    print('3. Saves the processed parquet files to a new directory.')
    haar_transformation_processor('50x12P5')
    print('Processing done.')
