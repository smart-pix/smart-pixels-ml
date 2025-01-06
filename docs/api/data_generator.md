# datagenerator module

This module contains the `OptimizedDataGenerator` class, which generates batches of data for training and validation during model training. This datagenerator handles the loading and processing of the data, including shuffling, standardization, and quantization of the data. It does by pre-processing the data and saving it as TFRecord files and then loading the batches on the fly during training.


### **Methods**

### `__init__(...)`

Initialize the `OptimizedDataGenerator` class with the specified parameters to configure the data generator for preprocessing and batching.

#### **Arguments**

Described in the comments of the `__init__` method of the [OptimizedDataGenerator.py](../../OptimizedDataGenerator.py) file.



### **Example Usage**


#### Initializing the Data Generators
```python
training_generator = OptimizedDataGenerator(
    data_directory_path = "path/to/data/",
    labels_directory_path = "path/to/labels/",
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
    shuffle = False, 
    files_from_end=True,

    tfrecords_dir = "path/to/tfrecords/",
    use_time_stamps = [0, 19], #-1
    max_workers = 1, # Don't make this too large (will use up all RAM)
    seed = 10, 
    quantize = True # Quantization ON
)

```
#### Loading the Data Generators

Already generated TFRecords can be reused by setting `load_from_tfrecords_dir` as  
```python
training_generator = OptimizedDataGenerator(
    load_from_tfrecords_dir = "path/to/tfrecords/",
    shuffle = True,
    seed = 13,
    quantize = True
)
```

The same goes for the `validation generator`. 

#### Using the Data Generators
The data generators can be directly passed to the fit method of a Keras model.

```python
history = model.fit(
                        x=training_generator,
                        validation_data=validation_generator,
                        #callbacks=[es, mcp, csv_logger],
                        epochs=1000,
                        shuffle=False,
                        verbose=1
 )
```


