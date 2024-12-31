#################################################################
# test_run.py - smoke test
#
# This script checks:
#  1. Required dependencies are installed.
#  2. Minimal directory structure is set up.
#  3. Basic data generation logic runs without errors.
#  4. Generates TFRecords via Optimized data generator.
#  5. Builds and trains a small model to ensure everything runs.
#################################################################

import sys
import os
import shutil
import glob
from colorama import Fore, Style


def supports_color():
    # Check if running in a supported terminal for colors
    supported_platform = sys.platform != 'win32' or 'ANSICON' in os.environ
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    return supported_platform and is_a_tty

if supports_color():
    from colorama import init
    init(autoreset=True)
else:
    class NoColor:
        def __getattr__(self, name):
            return '' 

    Fore = Style = NoColor()

def log_info(message):
     print(Fore.BLUE + "[INFO] " + Style.RESET_ALL + message)

def log_warning(message):
    print(Fore.YELLOW + "[WARNING] " + Style.RESET_ALL + message)

def log_error(message):
    print(Fore.RED + "[ERROR] " + Style.RESET_ALL + message)

def log_success(message):
    print(Fore.GREEN + "[SUCCESS] " + Style.RESET_ALL + message)


try:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.optimizers import Adam
    import qkeras

    # custom modules
    from OptimizedDataGenerator import OptimizedDataGenerator
    from loss import custom_loss
    from models import CreateModel
    
except ImportError as exc:
    log_error(f"Missing dependency: {exc}")
    sys.exit(1)

TEST_ROOT = "./testing_tmp"
DATA_DIR = "./testing_tmp/data/" 
LABELS_DIR = "./testing_tmp/labels/"
BASE_MODEL_DIR = "./testing_tmp/base_model/" 

TFRECORDS_DIR ="./testing_tmp/tfrecords/"
TFRECORDS_DIR_TRAIN = "./testing_tmp/tfrecords/train" 
TFRECORDS_DIR_VALIDATION = "./testing_tmp/tfrecords/validation"  

# Define batch size and file size for training; adjust if needed.
DATA_SET_SIZE = 100
BATCH_SIZE = 10
TRAIN_FILE_SIZE = 3
VAL_FILE_SIZE = 3
NUM_DUMMY_FILES = 5
NUM_EPOCHS = 3


def check_directories() -> bool:
    """
    Check if essential directories exist.
    """
    required = [
        TEST_ROOT, DATA_DIR, LABELS_DIR, TFRECORDS_DIR, BASE_MODEL_DIR
    ]
    for dpath in required:
        if not os.path.isdir(dpath):
            log_error(f"Directory missing: {dpath}")
            return False
    log_success("All required directories exist.")
    
    return True

def generate_dummy_data(num_files=NUM_DUMMY_FILES):
    """
    Generate multiple dummy data files for testing. Each file
    contains more data than the specified BATCH_SIZE.
    """
    if DATA_SET_SIZE <= BATCH_SIZE:
        raise ValueError(f"DATA_SET_SIZE ({DATA_SET_SIZE}) must be larger than BATCH_SIZE ({BATCH_SIZE}).")

    for file_idx in range(num_files):
        # Generate random data
        sample_data = np.random.rand(DATA_SET_SIZE, 13, 21, 20).astype(np.float32)
        flat_data = sample_data.reshape(DATA_SET_SIZE, -1)  # Flatten for Parquet

        column_names = [str(i) for i in range(flat_data.shape[1])]
        df_data = pd.DataFrame(flat_data, columns=column_names)
        df_data["event_id"] = range(DATA_SET_SIZE)

        sample_labels = np.random.rand(DATA_SET_SIZE, 4).astype(np.float32)
        df_labels = pd.DataFrame(
            sample_labels,
            columns=["x-midplane", "y-midplane", "cotAlpha", "cotBeta"]
        )
        df_labels["event_id"] = range(DATA_SET_SIZE)

        # Ensure filenames follow the expected format
        data_file = os.path.join(DATA_DIR, f"recon3D_data_{file_idx}.parquet")
        labels_file = os.path.join(LABELS_DIR, f"labels_data_{file_idx}.parquet")
        df_data.to_parquet(data_file, index=False)
        df_labels.to_parquet(labels_file, index=False)

        log_info(f"Generated dummy files: {data_file}, {labels_file}")
    
    log_success(f"Generated {num_files} dummy data files.")


def validate_parquet_files(directory):
    """
    Checks if there are any valid Parquet files in a given directory.
    """
    files = glob.glob(os.path.join(directory, "*.parquet"))
    if not files:
        raise ValueError(f"No valid parquet files found in {directory}.")
    log_info(f"Found {len(files)} valid parquet files in {directory}.")

def generate_tfrecords():
    """
    Tests the first initialization of the generators, generating
    TFRecords in the specified directories if needed.
    """
    validate_parquet_files(DATA_DIR)
    validate_parquet_files(LABELS_DIR)

    recon_files = glob.glob(
            DATA_DIR + "recon" + "3D" + "*." + "parquet", 
            recursive=True
        )
    log_info(f"Found " + str(len(recon_files)) + " recon files.")
    log_info(f"Generating TFRecords...")
    log_info(f"Iitializing generators...")

    training_generator = OptimizedDataGenerator(
        data_directory_path = DATA_DIR,
        labels_directory_path = LABELS_DIR,
        is_directory_recursive = False,
        file_type = "parquet",
        data_format = "3D",
        batch_size = BATCH_SIZE,
        file_count = TRAIN_FILE_SIZE,
        to_standardize = True,
        include_y_local = False,
        labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
        scaling_list = [75.0, 18.75, 10.0, 1.22],
        input_shape = (2,13,21),
        transpose = (0,2,3,1),
        files_from_end = False,
        shuffle = True,
        tfrecords_dir = TFRECORDS_DIR_TRAIN,
        use_time_stamps = [0, 19], 
        max_workers = 1,
        seed = 10,
        quantize = True
    )

    validation_generator = OptimizedDataGenerator(
        data_directory_path = DATA_DIR,
        labels_directory_path = LABELS_DIR,
        is_directory_recursive = False,
        file_type = "parquet",
        data_format = "3D",
        batch_size = BATCH_SIZE,
        file_count = VAL_FILE_SIZE,
        to_standardize = True,
        include_y_local = False,
        labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
        scaling_list = [75.0, 18.75, 10.0, 1.22],
        input_shape = (2,13,21),
        transpose = (0,2,3,1),
        files_from_end = True,
        shuffle = True,
        tfrecords_dir = TFRECORDS_DIR_VALIDATION,
        use_time_stamps = [0, 19], 
        max_workers = 1,
        seed = 10,
        quantize = True
    )

    log_success(f"TFRecord generation completed.")
    return training_generator, validation_generator

def load_tfrecords():
    """
    Load pre-generated TFRecords for training and validation.
    """
    log_info(f"Loading TFRecords...")

    # Initialize training data generator using TFRecords
    training_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir=TFRECORDS_DIR_TRAIN,
        max_workers=1,
        seed=10,
        quantize=True
    )

    # Initialize validation data generator using TFRecords
    validation_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir=TFRECORDS_DIR_VALIDATION,
        max_workers=1,
        seed=10,
        quantize=True
    )

    log_success(f"TFRecord loading completed.")
    return training_generator, validation_generator

def test_model_generation():
    """
    Test the generation of the model.
    """
    
    log_info(f"Building model...")
    try:
        model = CreateModel((13,21,2), n_filters=5, pool_size=3)
        model.summary()
        model.compile(
            optimizer = Adam(learning_rate=0.001),
            loss = custom_loss
        )
        log_success(f"Model built successfully.")
        return model
    
    except Exception as exc:
        log_error(f"Error building model: {exc}")
        sys.exit(1)

def test_train_model():
    """
    Test the training of the model.
    """
    es = tf.keras.callbacks.EarlyStopping(
        patience = 2,  # small patience for quick test
        restore_best_weights = True
    )

    checkpoint_filepath = os.path.join(
        BASE_MODEL_DIR,
        'weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5'
    )

    mcp = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        save_weights_only = True,
        monitor = 'val_loss',
        save_best_only = False
    )

    model = test_model_generation()
    training_gen, validation_gen = generate_tfrecords()
    del training_gen, validation_gen
    training_gen, validation_gen = load_tfrecords()

    log_info(f"Training model...")
    history = model.fit(
        x = training_gen,
        validation_data = validation_gen,
        epochs = NUM_EPOCHS, 
        shuffle = False, 
        verbose = 1,
        callbacks = [mcp, es]
    )

    log_success(f"Training completed successfully.")

    final_loss = model.evaluate(validation_gen, verbose=0)
    log_info(f"Final validation loss: {final_loss}")


def run_smoke_test():
    """
    Run all smoke tests in sequence.
    """
    log_info(f"Checking directories...")
    if not check_directories():
        sys.exit(1)

    log_info(f"Generating dummy data...")
    generate_dummy_data(num_files=NUM_DUMMY_FILES) 

    test_train_model()
    log_success(f"All smoke tests passed successfully.")



if __name__ == "__main__":
    # Clean up old test directory if it exists.
    try:
        shutil.rmtree(TEST_ROOT)
    except FileNotFoundError:
        pass

    # Create required directories.
    os.makedirs(TEST_ROOT, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(TFRECORDS_DIR, exist_ok=True)
    os.makedirs(TFRECORDS_DIR_TRAIN, exist_ok=True)
    os.makedirs(TFRECORDS_DIR_VALIDATION, exist_ok=True)
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)

    run_smoke_test()


    input("\n[END OF TEST]\n To clean up temporary files and exit: Press [ENTER] ")
    shutil.rmtree(TEST_ROOT)
