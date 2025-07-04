import os
import gc
import math
import glob
import random
import logging
import datetime
import numpy as np
import pandas as pd

from typing import Union, List, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tqdm import tqdm
import tensorflow as tf
from qkeras import quantized_bits

import utils



# custom quantizer

# @tf.function
def QKeras_data_prep_quantizer(data, bits=4, int_bits=0, alpha=1):
    """
    Applies QKeras quantization.
    Args:
        data (tf.Tensor): Input data (tf.Tensor).
        bits (int): Number of bits for quantization.
        int_bits (int): Number of integer bits.
        alpha (float): (don't change)
    Returns::
        tf.Tensor: Quantized data (tf.Tensor).
    """
    quantizer = quantized_bits(bits, int_bits, alpha=alpha)
    return quantizer(data)


def split_df_to_X_y_df(df: pd.DataFrame,
                       input_shape: Tuple[int, int, int],
                       labels_list: List[str],
                       recon_cols: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    X_df = df[recon_cols].copy()
    y_df = df[labels_list].copy()

    return X_df, y_df



class OptimizedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
            dataset_base_dir: str = "./",
            is_directory_recursive: bool = False,
            batch_size: int = 32,
            file_count = None,
            labels_list: Union[List,str] = "cotAlpha",
            to_standardize: bool = False,
            input_shape: Tuple = (13,21),
            transpose = None,
            include_y_local: bool = False,
            files_from_end = False,
            shuffle=False,

            # Added in Optimized datagenerators 
            load_from_tfrecords_dir: str = None,
            tfrecords_dir: str = None,
            use_time_stamps = -1,
            seed: int = None,
            quantize: bool = False,
            select_contained = False, #If true, selects only clusters with original_atEdge==False
            max_workers: int = 1,
                 
            **kwargs,
            ):
        super().__init__() 

        self.shuffle = shuffle
        if shuffle:
            self.seed = seed if seed is not None else 13
            self.rng = np.random.default_rng(seed = self.seed)
        
        # If data is already prepared load anduse that data
        if load_from_tfrecords_dir is not None:
            self.file_offsets = [None]
            if not os.path.isdir(load_from_tfrecords_dir):
                raise ValueError(f"Directory {load_from_tfrecords_dir} does not exist.")
            else:
                self.tfrecords_dir = load_from_tfrecords_dir
        else:
            n_time, height, width = input_shape
            
            if use_time_stamps == -1:
                use_time_stamps = list(np.arange(0,20))
            assert len(use_time_stamps) == n_time, f"Expected {n_time} time steps, got {len(use_time_stamps)}"
    
            len_xy = height * width
            col_indices = [
                np.arange(t * len_xy, (t + 1) * len_xy).astype(str)
                for t in use_time_stamps
            ]
            self.use_time_stamps = list(np.arange(0,20)) if use_time_stamps == -1 else use_time_stamps
            self.recon_cols = np.concatenate(col_indices).tolist()
    
    
    
            self.max_workers = max_workers
            self.shuffle = shuffle

            
            self.files = sorted(glob.glob(os.path.join(dataset_base_dir, "part.*.parquet"), recursive=False))
    
            if file_count != None:
                if not files_from_end:
                    self.files = self.files[:file_count]
                else:
                    self.files = self.files[-file_count:]
    
    
            self.file_offsets = [0]
            self.dataset_mean = None
            self.dataset_std = None

            # safe_remove_directory(tfrecords_dir)
            utils.safe_remove_directory(tfrecords_dir)
            self.batch_size = batch_size
            self.labels_list = labels_list
            self.input_shape = input_shape
            self.transpose = transpose
            self.to_standardize = to_standardize
            self.include_y_local = include_y_local
            self.select_contained = select_contained

            self.process_file_parallel()
    
            self.current_file_index = None
            self.current_dataframes = None
    
            if tfrecords_dir is None:
                raise ValueError(f"tfrecords_dir is None")
                
            self.tfrecords_dir = tfrecords_dir    
            os.makedirs(self.tfrecords_dir, exist_ok=True)
            self.save_batches_parallel() # save all the batches
            del self.current_dataframes 
            
        self.tfrecord_filenames = np.sort(np.array(tf.io.gfile.glob(os.path.join(self.tfrecords_dir, "*.tfrecord"))))
        self.quantize = quantize
        self.epoch_count = 0
        self.on_epoch_end()


    def process_file_parallel(self):
        file_infos = [(afile, self.recon_cols, self.input_shape, self.transpose, self.labels_list, self.select_contained) for afile in self.files]
        results = []
        with ProcessPoolExecutor(self.max_workers) as executor:
            futures = [executor.submit(self._process_file_single, file_info) for file_info in file_infos]
            for future in tqdm(as_completed(futures), total=len(file_infos), desc="Processing Files..."):
                results.append(future.result())

        for amean, avariance, amin, amax, num_rows in results:
            self.file_offsets.append(self.file_offsets[-1] + num_rows)

            if self.dataset_mean is None:
                self.dataset_max = amax
                self.dataset_min = amin
                self.dataset_mean = amean
                self.dataset_std = avariance
            else:
                self.dataset_max = max(self.dataset_max, amax)
                self.dataset_min = min(self.dataset_min, amin)
                self.dataset_mean += amean
                self.dataset_std += avariance

        self.dataset_mean = self.dataset_mean / len(self.files)
        self.dataset_std = np.sqrt(self.dataset_std / len(self.files)) 
            
        self.file_offsets = np.array(self.file_offsets)

    @staticmethod
    def _process_file_single(file_info):
        afile, recon_cols, input_shape, transpose, labels_list,  select_contained = file_info
        if select_contained:
            df = pd.read_parquet(afile, columns=recon_cols + labels_list +['original_atEdge'])
            df = df.loc[df['original_atEdge'] == False]
        else:
            df = pd.read_parquet(afile, columns=recon_cols + labels_list)
        
        adf, _ = split_df_to_X_y_df(df, input_shape, labels_list, recon_cols)

        x = adf.values
        nonzeros = abs(x) > 0
        x[nonzeros] = np.sign(x[nonzeros]) * np.log1p(abs(x[nonzeros])) / math.log(2)
        amean, avariance = np.mean(x[nonzeros], keepdims=True), np.var(x[nonzeros], keepdims=True) + 1e-10
        centered = np.zeros_like(x)
        centered[nonzeros] = (x[nonzeros] - amean) / np.sqrt(avariance)
        x = x.reshape((-1, *input_shape))
        if transpose is not None:
            x = x.transpose(transpose)
        amin, amax = np.min(centered), np.max(centered)
        len_adf = len(adf)
        del adf
        gc.collect()
        
        return amean, avariance, amin, amax, len_adf

    def standardize(self, x, norm_factor_pos=1.7, norm_factor_neg=2.5):
        """
        Applies the normalization configuration in-place to a batch of inputs.
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

    def save_batches_parallel(self):
        """
        Saves data batches as multiple TFRecord files.
        """
        # TODO: Make this parallelized
        num_batches = self.__len__() # Total num of batches
        paths_or_errors = []

        # The max_workers is set to 1 because processing large batches in multiple threads can significantly
        # increase RAM usage. Adjust 'max_workers' based on your system's RAM capacity and requirements.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_batch = {executor.submit(self.save_single_batch, i): i for i in range(num_batches)}
            
            for future in tqdm(as_completed(future_to_batch), total=num_batches, desc="Saving batches as TFRecords"):
                result = future.result()
                paths_or_errors.append(result)
            
        for res in paths_or_errors:
            if "Error" in res:
                print(res)  
                
    def save_single_batch(self, batch_index):
        """
        Serializes and saves a single batch to a TFRecord file.
        Args:
            batch_index (int): Index of the batch to save.
        Returns:
            str: Path to the saved TFRecord file or an error message.
        """
        
        try:
            filename = f"batch_{batch_index}.tfrecord"
            TFRfile_path = os.path.join(self.tfrecords_dir, filename)
            X, y = self.prepare_batch_data(batch_index)
            serialized_example = self.serialize_example(X, y)
            with tf.io.TFRecordWriter(TFRfile_path) as writer:
                writer.write(serialized_example)
            return TFRfile_path
        except Exception as e:
            return f"Error saving batch {batch_index}: {e}" 
        
    def prepare_batch_data(self, batch_index):
        """
        Fetch and prepare one batch (X, y).
        """
        abs_idx = batch_index * self.batch_size
        file_idx = (np.arange(self.file_offsets.size)[abs_idx < self.file_offsets][0] - 1)
        rel_idx = abs_idx - self.file_offsets[file_idx]
        stop_idx = min(rel_idx + self.batch_size, self.file_offsets[file_idx + 1] - self.file_offsets[file_idx])

        if file_idx != self.current_file_index:
            self.current_file_index = file_idx
            parquet_file = self.files[file_idx]
            if self.select_contained:
                all_columns_to_read = self.recon_cols + self.labels_list + ['original_atEdge']
                df = pd.read_parquet(parquet_file, columns = all_columns_to_read).reset_index(drop=True) 
                df = df.loc[df['original_atEdge'] == False]
            else:
                all_columns_to_read = self.recon_cols + self.labels_list
                df = pd.read_parquet(parquet_file, columns = all_columns_to_read).reset_index(drop=True)
            
            # df = pd.read_parquet(parquet_file, columns=self.use_time_stamps)
            recon_df, labels_df = split_df_to_X_y_df(df, self.input_shape, self.labels_list, self.recon_cols)

            has_nans = np.any(np.isnan(recon_df.values), axis=1)
            has_nans = np.arange(recon_df.shape[0])[has_nans]
            recon_df_raw = recon_df.drop(has_nans)
            labels_df_raw = labels_df.drop(has_nans)

            joined_df = recon_df_raw.join(labels_df_raw)

            if self.shuffle: # Changed
                joined_df = joined_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)  

            recon_values = joined_df[recon_df_raw.columns].values            

            nonzeros = abs(recon_values) > 0
            
            recon_values[nonzeros] = np.sign(recon_values[nonzeros])*np.log1p(abs(recon_values[nonzeros]))/math.log(2)
            
            if self.to_standardize:
                recon_values[nonzeros] = self.standardize(recon_values[nonzeros])
            
            recon_values = recon_values.reshape((-1, *self.input_shape))            
                        
            if self.transpose is not None:
                recon_values = recon_values.transpose(self.transpose)
            
            self.current_dataframes = (
                recon_values, 
                joined_df[labels_df_raw.columns].values,
            )        
        
        recon_df, labels_df = self.current_dataframes

        X = recon_df[rel_idx:stop_idx]
        y = labels_df[rel_idx:stop_idx] / np.array([75., 18.75, 8.0, 0.5])
    
        if self.include_y_local:
            y_local = labels_df.iloc[chosen_idxs]["y-local"].values
            return [X, y_local], y
        else:
            return X, y
    
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

    @staticmethod
    def _bytes_feature(value):
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

    def __getitem__(self, batch_index):
        """
        Load the batch from a pre-saved TFRecord file instead of processing raw data.
        Each file contains exactly one batch.
        quantization is done here: Helpful for pretraining without the quantization and the later training with quantized data.
        shuffling is also done here.
        TODO: prefetching (un-done)
        """
        tfrecord_path = self.tfrecord_filenames[batch_index]
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = raw_dataset.map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

        for X_batch, y_batch in parsed_dataset:
            ''' Add the reshaping in saving'''
            X_batch = tf.reshape(X_batch, [-1, *X_batch.shape[1:]])
            y_batch = tf.reshape(y_batch, [-1, *y_batch.shape[1:]])

            if self.quantize:
                X_batch = QKeras_data_prep_quantizer(X_batch, bits=4, int_bits=0, alpha=1)

            if self.shuffle:
                indices = tf.range(start=0, limit=tf.shape(X_batch)[0], dtype=tf.int32)
                shuffled_indices = tf.random.shuffle(indices, seed=self.seed)
                X_batch = tf.gather(X_batch, shuffled_indices)
                y_batch = tf.gather(y_batch, shuffled_indices)
                
            del raw_dataset, parsed_dataset
            return X_batch, y_batch
            
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

    def __len__(self):
        if len(self.file_offsets) != 1: # used when TFRecord files are created during initialization
            num_batches = self.file_offsets[-1] // self.batch_size
        else: # used during loading saved TFRecord files
            num_batches = len(os.listdir(self.tfrecords_dir))
        return num_batches

    def on_epoch_end(self):
        '''
        This shuffles the file ordering so that it shuffles the ordering in which the TFRecord
        are loaded during the training for each epochs.
        '''
        gc.collect()
        self.epoch_count += 1
        # Log quantization status once
        if self.epoch_count == 1:
            logging.warning(f"Quantization is {self.quantize} in data generator. This may affect model performance.")

        if self.shuffle:
            self.rng.shuffle(self.tfrecord_filenames)
            self.seed += 1 # So that after each epoch the batch is shuffled with a different seed (deterministic)
