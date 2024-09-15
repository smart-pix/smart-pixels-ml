import tensorflow as tf
import keras
from typing import Union, List, Tuple
import glob
import numpy as np
import pandas as pd
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import datetime

import random

def data_prep_quantizer(data, bits=3, int_bits=0): # remember there's a secret sign bit
    frac_bits = bits - int_bits
    return np.round(data * 2**frac_bits) * 2**-frac_bits

def data_prep_quantizer(data, bits=3, int_bits=0): # remember there's a secret sign bit
    frac_bits = bits - int_bits
    return np.round(data * 2**frac_bits) * 2**-frac_bits

def diffable_quantizer(data, bits=7, int_bits=0): # remember there's a secret sign bit
    frac_bits = bits - int_bits
    return tf.math.round(data * 2**frac_bits) * 2**-frac_bits

class LearnedScale(keras.layers.Layer):
    def __init__(self, input_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.scale = self.add_weight(
            shape=(self.input_dim, ), initializer="glorot_uniform", trainable=True
        )
        #self.shift = self.add_weight(shape=(input_dim, ), initializer="zeros", trainable=True)

    def call(self, inputs):
        return inputs * tf.math.softplus(self.scale) # + self.shift
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim
        })
        return config

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
                shuffle=False,
                current=False,
                sample_delta_t=200,
                scaling_list: Union[List,float] = [75., 18.75, 10.0, 1.22],
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
        """

        self.shuffle = shuffle
        
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
        self.scaling_list = scaling_list

        self.current_file_index = None
        self.current_dataframes = None
        
        self.on_epoch_end()
        
    def process_file(self, afile, file_type, input_shape, transpose=None):
        if file_type == "csv":
            adf = pd.read_csv(afile).dropna()
        elif file_type == "parquet":
            adf = pd.read_parquet(afile).dropna()

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

        with ProcessPoolExecutor(max_workers=1) as executor:
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
                recon_df = pd.read_parquet(self.recon_files[file_index])
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
        y = labels_df[index:batch_size] / np.array(self.scaling_list)
    
        if self.include_y_local:
            y_local = labels_df.iloc[chosen_idxs]["y-local"].values
            return [X, y_local], y
        else:
            return X, y
    
    def __len__(self):
        return self.file_offsets[-1] // self.batch_size


from qkeras import quantized_bits
import os
import datetime
def QKeras_data_prep_quantizer(data, bits=4, int_bits=0, alpha = 1):
    return np.array(quantized_bits(bits, int_bits, alpha=alpha)(data))

class OptimizedDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, 
                data_directory_path: str = "./",
                labels_directory_path: str = "./",
                
                load_from_tfrecords_dir: str = None,
                tfrecords_dir: str = None,
                U_TFRecord_path: bool = True,
                 
                max_workers=1,
                 
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
                current=False,
                sample_delta_t=200,
                scaling_list = [75., 18.75, 10.0, 1.22],
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
        """


        self.shuffle = shuffle
        self.max_workers = max_workers
        if load_from_tfrecords_dir is not None:
            if not os.path.isdir(load_from_tfrecords_dir):
                raise ValueError(f"Directory {load_from_tfrecords_dir} does not exist.")
            else:
                self.tfrecords_dir = load_from_tfrecords_dir
        else:
            if file_type not in ["csv", "parquet"]:
                raise ValueError("file_type can only be \"csv\" or \"parquet\"!")
            self.file_type = file_type

            self.recon_files = glob.glob(
                                        data_directory_path + "recon" + data_format + "*." + file_type, 
                                        recursive=is_directory_recursive
                                        )
            self.recon_files.sort()
            if file_count != None:
                if not files_from_end:
                    self.recon_files = self.recon_files[:file_count]
                else:
                    self.recon_files = self.recon_files[-file_count:]
            
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
            
            if tfrecords_dir is None:
                if U_TFRecord_path:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    fingerprint = '%08x' % random.randrange(16**8)
                    self.tfrecords_dir = f"./tfrecords_f_{fingerprint}_t_{timestamp}"
                else: 
                    self.tfrecords_dir = f"./tfrecords"
            else:
                self.tfrecords_dir = tfrecords_dir    
            os.makedirs(self.tfrecords_dir, exist_ok=True)
            self.save_batches_parallel()
        
        self.tfrecord_filenames = np.array(tf.io.gfile.glob(os.path.join(self.tfrecords_dir, "*.tfrecord")))
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.tfrecord_filenames)
    def prepare_batch_data(self, batch_index):
        """
        Applies the necessary preprocessing and prepares the batch data for training or validation.
        Any changes to how the data is supposed to look and data management should be implemented here.
        
        Args:
        - batch_index: The index of the batch to prepare.
        
        Returns:
        - X: The input features for the model, processed and reshaped as necessary.
        - y: The labels/targets for the training/validation, adjusted as specified.
        - filename: The path to the TFRecord file where the batch is saved

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
                recon_df = pd.read_parquet(self.recon_files[file_index])
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
            
            recon_values = recon_values.reshape((-1, *self.input_shape))            

            # QKeras_data_prep_quantizer is inside dataprep.py
            recon_values = QKeras_data_prep_quantizer(recon_values)

            if self.transpose is not None:
                recon_values = recon_values.transpose(self.transpose)
            
            self.current_dataframes = (
                recon_values, 
                joined_df[labels_df_raw.columns].values,
            )        
        
        recon_df, labels_df = self.current_dataframes
        ## recon_df = np.delete(recon_df,np.s_[1:-1],-1) # Modify this as necessary
        
        X = recon_df[index:batch_size]
        y = labels_df[index:batch_size] / np.array(self.scaling_list)
        
        filename = f"batch_{batch_index}.tfrecord"
        filename = os.path.join(self.tfrecords_dir, filename)
    
        if self.include_y_local:
            y_local = labels_df.iloc[chosen_idxs]["y-local"].values
            return [X, y_local], y, filename
        else:
            return X, y, filename
        
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
        out = (x - self.dataset_mean)/self.dataset_std
        out[out > 0] = out[out > 0]/norm_factor_pos
        out[out < 0] = out[out < 0]/norm_factor_neg
        return out
    
    def _bytes_feature(self, value):
        """
        Converts a string/byte value into a Tf feature of bytes_list
        
        Args: 
        - string/byte value
        
        Returns:
        - tf.train.Feature object as a bytes_list containing the input value.
        """
        if isinstance(value, type(tf.constant(0))): # check if Tf tensor
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize_example(self, X, y):
        """
        Serializes a single example (featuresand labels) to TFRecord format. 
        
        Args:
        - X: Training data
        - y: labelled data
        
        Returns:
        - string (serialized TFRecord example).
        """
        # X and y are float32 (maybe we can reduce this)
        X = tf.cast(X, tf.float32)
        y = tf.cast(y, tf.float32)

        feature = {
            'X': self._bytes_feature(tf.io.serialize_tensor(X)),
            'y': self._bytes_feature(tf.io.serialize_tensor(y)),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    
    def save_batches_parallel(self):
        """
        Saves data batches as multiple TFRecord files.
        """
        # TODO: Make this parallelized
        num_batches = self.__len__() # Total num of batches
        for batch_index in tqdm(range(num_batches), desc="Saving batches as TFRecords"):
            X, y, filename = self.prepare_batch_data(batch_index)
            serialized_example = self.serialize_example(X, y)
            with tf.io.TFRecordWriter(filename) as writer:
                writer.write(serialized_example)
        
    def process_file(self, afile, file_type, input_shape, transpose=None):
        """
        Scans all the data files and return the dataset - stats
        """
        if file_type == "csv":
            adf = pd.read_csv(afile).dropna()
        elif file_type == "parquet":
            adf = pd.read_parquet(afile).dropna()

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

        with ProcessPoolExecutor(max_workers= self.max_workers) as executor:
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
        
    def on_epoch_end(self):
        """
        Inherited from the parent class.
        Used to reset indices but not of significance in this datagenerator.
        """
        pass
    
    @staticmethod
    def _parse_tfrecord_fn(example):
        """
        Parses a single TFRecord example.
        
        Returns:
        - X: as a float32 tensor.
        - y: as a float32 tensor.
        """
        feature_description = {
            'X': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        X = tf.io.parse_tensor(example['X'], out_type=tf.float32)
        y = tf.io.parse_tensor(example['y'], out_type=tf.float32)
        return X, y
        
    def __getitem__(self, batch_index):
        """
        Load the batch from a pre-saved TFRecord file instead of processing raw data.
        Each file contains exactly one batch.
        TODO: prefetching (done)
        """
        tfrecord_path = os.path.join(self.tfrecords_dir, f"batch_{batch_index}.tfrecord")
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = raw_dataset.map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        prefetch_dataset = parsed_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        for X_batch, y_batch in prefetch_dataset:
            X_batch = tf.reshape(X_batch, [-1, *X_batch.shape[1:]])
            y_batch = tf.reshape(y_batch, [-1, *y_batch.shape[1:]])
            if self.shuffle:
                X_batch = tf.random.shuffle(X_batch, seed=13)
                y_batch = tf.random.shuffle(y_batch, seed=13)
            return X_batch, y_batch
    
    def __len__(self):
        try: 
            num_batches = self.file_offsets[-1] // self.batch_size
        except:
            num_batches = len(os.listdir(self.tfrecords_dir))
        return num_batches