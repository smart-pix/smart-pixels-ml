# custom imports
from .utils import data_prep_quantizer

# python imports
import tensorflow as tf
from typing import Union, List, Tuple
import glob
import numpy as np
import pandas as pd
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

class CustomDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, 
                data_directory_path: str = "./",
                labels_directory_path: str = "./",
                is_directory_recursive: bool = False,
                file_type: str = "csv",
                data_format: str = "2D",
                batch_size: int = 32,
                file_count = None,
                labels_list: Union[List,str] = "cotAlpha",
                to_standardize: bool = False,
                input_shape: Tuple = (13,21),
                transpose = None,
                include_y_local: bool = False,
                files_from_end = False,
                shuffle=True,
                use_time_stamps = -1,
                current=False,
                sample_delta_t=200,
                **kwargs,
                ):
        """
        Data Generator to streamline data input to the network direct from the directory.
        Args:
        data_directory_path:
        labels_directory_path: 
        is_directory_recursive: 
        file_type: Default: "csv"
                   Adapt the data loader according to file type. For now, it only supports csv and parquet file formats.
        data_format: Default: 2D
                     Used to refer to the relevant "recon" files, 2D for 2D pixel array, 3D for time series input,
        batch_size: Default: 32
                    The no. of data points to be included in a single batch.
        file_count: Default: None
                    To limit the no. of .csv files to be used for training.
                    If set to None, all files will be considered as legitimate inputs.
        labels_list: Default: "cotAlpha"
                     Input column name or list of column names to be used as label input to the neural network.
        to_standardize: If set to True, it ensures that batches are normalized prior to being used as inputs
                        for training.
                        Default: False
        input_shape: Default: (13,21) for image input to a 2D feedforward neural network.
                    To reshape the input array per the requirements of the network training.
        current: Default False, calculate the current instead of the integrated charge
        sample_delta_t: how long an "ADC bin" is in picoseconds
        use_time_stamps: which of the 20 time stamps to train on. default -1 is to train on all of them
        """

        self.shuffle = shuffle

        # decide on which time stamps to load
        self.use_time_stamps = np.arange(0,20) if use_time_stamps == -1 else use_time_stamps
        len_xy, ntime = 13*21, 20
        idx = [[i*(len_xy),(i+1)*(len_xy)] for i in range(ntime)] # 20 time stamps of length 13*21
        self.use_time_stamps = np.array([ np.arange(idx[i][0], idx[i][1]).astype("str") for i in self.use_time_stamps]).flatten()


        if file_type not in ["csv", "parquet"]:
            raise ValueError("file_type can only be \"csv\" or \"parquet\"!")
        self.file_type = file_type
        
        # Go into the folders and extract out the specific files that meets the pattern
        self.recon_files = glob.glob(
                                    data_directory_path + "recon" + data_format + "*." + file_type, 
                                    recursive=is_directory_recursive
                                    )
        
        self.recon_files.sort()
        # To make training and valid files 
        if file_count != None:
            if not files_from_end:
                self.recon_files = self.recon_files[:file_count]
            else:
                self.recon_files = self.recon_files[-file_count:]
        
        # Make the label files corresponding to the recon files (so that they have one to one mapping) 
        # Done using similar pattern as of the recon and the label files
        self.label_files = [
                            labels_directory_path + recon_file.split('/')[-1].replace("recon" + data_format, "labels") for recon_file in self.recon_files
                        ]
    
        dataset_stats = self.parallel_process_files(self.recon_files, file_type, input_shape, transpose)

        self.dataset_mean = dataset_stats['mean']
        self.dataset_std = dataset_stats['variance'] 
        self.file_offsets = np.array(dataset_stats['file_offsets'] )

        self.batch_size = batch_size
        self.labels_list = labels_list
        self.input_shape = input_shape
        self.transpose = transpose
        self.to_standardize = to_standardize
        self.include_y_local = include_y_local

        self.current_file_index = None
        self.current_dataframes = None
        
        self.on_epoch_end()
        
    def process_file(self, afile, file_type, input_shape, transpose=None):
        
        # load data for only those time stamps
        if file_type == "csv":
            adf = pd.read_csv(afile).dropna()
        elif file_type == "parquet":
            adf = pd.read_parquet(afile, columns=self.use_time_stamps).dropna()

        # convert to values
        x = adf.values
        nonzeros = abs(x) > 0
        x[nonzeros] = np.sign(x[nonzeros]) * np.log1p(abs(x[nonzeros])) / math.log(2)
        # using np.sign(x[nonzeros])*np.log1p(abs(x[nonzeros]))/math.log(2)
        # would be better -> a smooth function near zero and also [-1,1] is close to [-1,1] (log(2) is needed)

        amean, avariance = self.get_mean_and_variance(x[nonzeros])

        centered = np.zeros_like(x)
        centered[nonzeros] = (x[nonzeros] - amean) / np.sqrt(avariance)

        x = x.reshape((-1, *input_shape))
        if transpose is not None:
            x = x.transpose(transpose)

        amin, amax = np.min(centered), np.max(centered)

        return amean, avariance, amin, amax, len(adf)

    def parallel_process_files(self, recon_files, file_type, input_shape, transpose=None):
        dataset_stats = {'mean': 0,
                         'variance': 0,
                         'min': float('inf'),
                         'max': float('-inf'),
                         'total_files': len(recon_files),
                         'file_offsets': [0]}  # Initialize file_offsets as 0

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.process_file, afile, file_type, input_shape, transpose): afile for afile in recon_files}
            for future in tqdm(as_completed(futures), total=len(recon_files), desc="Processing Files"):
                amean, avariance, amin, amax, file_length = future.result()
                dataset_stats['mean'] += amean
                dataset_stats['variance'] += avariance
                dataset_stats['min'] = min(dataset_stats['min'], amin)
                dataset_stats['max'] = max(dataset_stats['max'], amax)
                # file_offsets looks like this: [0, 100, 200, ...]. This means:
                # first file contains 100 records (0 to 99).
                # second file contains 100 records (100 to 199).
                # third file contains n records (200 to ...).
                dataset_stats['file_offsets'].append(dataset_stats['file_offsets'][-1] + file_length)

        dataset_stats['mean'] = dataset_stats['mean'] / dataset_stats['total_files']
        dataset_stats['variance'] = np.sqrt(dataset_stats['variance'] / dataset_stats['total_files'])

        return dataset_stats

        
    
    def get_mean_and_variance(self, x):
        """Applies the normalization configuration in-place to a batch of
        inputs.
        `x` is changed in-place since the function is mainly used internally
        to standardize images and feed them to your network.
        Args:
            x: Batch of inputs to be normalized.
        Returns:
            The inputs, normalized. 
        """
        return np.mean(x, keepdims=True), np.var(x, keepdims=True) + 1e-10
        
    def standardize(self, x, norm_factor_pos=1.7, norm_factor_neg=2.5):
        """Applies the normalization configuration in-place to a batch of
        inputs.
        `x` is changed in-place since the function is mainly used internally
        to standardize images and feed them to your network.
        Args:
            x: Batch of inputs to be normalized.
        Returns:
            The inputs, normalized. 
        """
        # This is carriedout inthe __getitem__ funtion 
        # done with the global min and the max
        out = (x - self.dataset_mean)/self.dataset_std
        out[out > 0] = out[out > 0]/norm_factor_pos
        out[out < 0] = out[out < 0]/norm_factor_neg
        return out
        
    def on_epoch_end(self):
        """
        Inherited from the parent class.
        Used to reset indices but not of significance in this datagenerator.
        """
        pass
            
        
    def __getitem__(self, batch_index):
        """
        Used to fetch a batch of inputs (X,y) for the network's training.
        """

        index = batch_index * self.batch_size # absolute *event* index
        
        file_index = np.arange(self.file_offsets.size)[index < self.file_offsets][0] - 1 # first index is 0!

        index = index - self.file_offsets[file_index] # relative event index in file
        batch_size = min(index + self.batch_size, self.file_offsets[file_index + 1] - self.file_offsets[file_index])
        
        if file_index != self.current_file_index:
            self.current_file_index = file_index
            if self.file_type == "csv":
                recon_df = pd.read_csv(self.recon_files[file_index])
                labels_df = pd.read_csv(self.label_files[file_index])[self.labels_list]
            elif self.file_type == "parquet":
                recon_df = pd.read_parquet(self.recon_files[file_index], columns=self.use_time_stamps)
                labels_df = pd.read_parquet(self.label_files[file_index], columns=self.labels_list)

            has_nans = np.any(np.isnan(recon_df.values), axis=1)
            has_nans = np.arange(recon_df.shape[0])[has_nans]
            recon_df_raw = recon_df.drop(has_nans)
            labels_df_raw = labels_df.drop(has_nans)

            joined_df = recon_df_raw.join(labels_df_raw)

            if self.shuffle:
                joined_df = joined_df.sample(frac=1).reset_index(drop=True)  

            recon_values = joined_df[recon_df_raw.columns].values            

            nonzeros = abs(recon_values) > 0
            
            # log normalization: modified to log(1+|x|) with np.log1p
            recon_values[nonzeros] = np.sign(recon_values[nonzeros])*np.log1p(abs(recon_values[nonzeros]))/math.log(2)            
            
            if self.to_standardize:
                recon_values[nonzeros] = self.standardize(recon_values[nonzeros])
            
            # note that this line could be problematic. the number of time stamps needs to be set before
            # this is run or else unused time stamps will be reshaped into separate training examples
            recon_values = recon_values.reshape((-1, *self.input_shape))            

            # data_prep_quantizer is inside dataprep.py
            recon_values = data_prep_quantizer(recon_values)
            
            if self.transpose is not None:
                recon_values = recon_values.transpose(self.transpose)
            
            self.current_dataframes = (
                recon_values, 
                joined_df[labels_df_raw.columns].values,
            )        
        
        recon_df, labels_df = self.current_dataframes

        X = recon_df[index:batch_size]
        y = labels_df[index:batch_size] / np.array([75., 18.75, 8.0, 0.5])

        if self.include_y_local:
            y_local = labels_df.iloc[chosen_idxs]["y-local"].values
            return [X, y_local], y
        else:
            return X, y
    
    def __len__(self):
        return self.file_offsets[-1] // self.batch_size
    

if __name__ == "__main__":

    # paths
    data_directory_path = "/net/projects/particlelab/smartpix/dataset8/unflipped/"
    labels_directory_path = "/net/projects/particlelab/smartpix/dataset8/unflipped/"
    val_batch_size = 500
    val_file_size = 2
    # generator
    generator = CustomDataGenerator(
        data_directory_path = data_directory_path,
        labels_directory_path = labels_directory_path,
        is_directory_recursive = False,
        file_type = "parquet",
        data_format = "3D",
        batch_size = val_batch_size,
        file_count = val_file_size,
        to_standardize= True,
        include_y_local= False,
        labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
        input_shape = (2,13,21), # (20,13,21),
        transpose = (0,2,3,1),
        files_from_end=True,
        use_time_stamps = [0,19],
    )