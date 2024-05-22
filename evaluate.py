# fix for keras v3.0 update
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' 

# python based
import tensorflow as tf
import random
from pathlib import Path
import pandas as pd
import argparse
import submitit
import json 
import numpy as np
import shutil

# custom code
from dataloaders.OptimizedDataGenerator import OptimizedDataGenerator
from models import CreateModel

minval=1e-9

def evaluate(config):

    # update %j with actual job number
    try:
        job_env = submitit.JobEnvironment()
        config["outFileName"] = Path(str(config["outFileName"]).replace("%j", str(job_env.job_id)))
    except:
        config["outFileName"] = Path(str(config["outFileName"]).replace("%j", "%08x" % random.randrange(16**8)))
    output_directory = config["outFileName"].parent
    os.makedirs(output_directory, exist_ok=True)
    print(output_directory)

    # create tf records directory
    tfrecords_dir = Path(output_directory, f"tfrecords_{'%08x' % random.randrange(16**8)}").resolve()
    
    # data generator
    test_generator = OptimizedDataGenerator(
        data_directory_path = config["data_directory_path"],
        labels_directory_path = config["labels_directory_path"],
        is_directory_recursive = False,
        file_type = "parquet",
        data_format = "3D",
        batch_size = config["val_batch_size"],
        file_count = config["val_file_size"],
        to_standardize= True,
        include_y_local= False,
        labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
        input_shape = (2,13,21), # (20,13,21),
        transpose = (0,2,3,1),
        files_from_end=True,
        use_time_stamps = [0,19],
        tfrecords_dir = tfrecords_dir,
    )

    # build model, load weights, predict
    model=CreateModel((13,21,2), n_filters=config["n_filters"], pool_size=config["pool_size"])
    model.load_weights(config["weightsPath"])
    p_test = model.predict(test_generator)

    complete_truth = None
    for _, y in test_generator:
        if complete_truth is None:
            complete_truth = y
        else:
            complete_truth = np.concatenate((complete_truth, y), axis=0)

    # creates df with all predicted values and matrix elements - 4 predictions, all 10 unique matrix elements
    df = pd.DataFrame(p_test,columns=['x','M11','y','M22','cotA','M33','cotB','M44','M21','M31','M32','M41','M42','M43'])

    # stores all true values in same matrix as xtrue, ytrue, etc.
    df['xtrue'] = complete_truth[:,0]
    df['ytrue'] = complete_truth[:,1]
    df['cotAtrue'] = complete_truth[:,2]
    df['cotBtrue'] = complete_truth[:,3]
    df['M11'] = minval+tf.math.maximum(df['M11'], 0)
    df['M22'] = minval+tf.math.maximum(df['M22'], 0)
    df['M33'] = minval+tf.math.maximum(df['M33'], 0)
    df['M44'] = minval+tf.math.maximum(df['M44'], 0)

    # calculates residuals for x, y, cotA, cotB
    residuals = df['xtrue'] - df['x']
    residualsy = df['ytrue'] - df['y']
    residualsA = df['cotAtrue'] - df['cotA']
    residualsB = df['cotBtrue'] - df['cotB']

    # stores results as csv
    df.to_csv(config["outFileName"], header=True, index=False)

    # clean up tf records
    shutil.rmtree(tfrecords_dir)

if __name__ == "__main__":
    
    # set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inFolder", help="path to the training results", default=None)
    parser.add_argument("--query", help="path to json file containing query", default=None)
    parser.add_argument("--njobs", help="number of jobs to actually launch. default is all", default=-1, type=int)
    parser.add_argument("--doOverwrite", help="overwrite any existing evaluation files", action="store_true")
    args = parser.parse_args()

    # read in query
    if Path(args.query).resolve().exists():
        query_path = Path(args.query).resolve()
    else:
        # throw
        raise ValueError(f"Could not locate {args.query} in query directory or as absolute path")
    with open(query_path) as f:
        query = json.load(f)

    # data paths and configs
    data_directory_path = "/net/projects/particlelab/smartpix/dataset8/unflipped/" # "/net/scratch/badea/dataset8/unflipped/"
    labels_directory_path = "/net/projects/particlelab/smartpix/dataset8/unflipped/" # "/net/scratch/badea/dataset8/unflipped/"
    val_batch_size = 5000
    val_file_size = 16

    # figure out which weight to use
    inFolder = Path(args.inFolder).resolve()
    weightsFolders = list(inFolder.glob("*/weights*"))
    top_dir = Path(inFolder, "eval", f'{"%08x" % random.randrange(16**8)}', "%j").resolve()
    # weightsFolder = Path("/home/badea/smartpix/semiparametric/timeslices-2/neurips-3x3-2conv/results/training-a9a85a9d/16367_0/weights-nFilters1-poolSize1-checkpoints").resolve()
    
    # configurations 
    confs = []
    for weightsFolder in weightsFolders:
        n_filters = int(weightsFolder.parts[-1].split("-")[1].split("nFilters")[1])
        pool_size = int(weightsFolder.parts[-1].split("-")[2].split("poolSize")[1])

        # files = os.listdir(weightsFolder)
        files = [str(f) for f in weightsFolder.glob("*.hdf5")]
        vlosses = [float(f.split("-v")[1].split(".hdf5")[0]) for f in files]
        bestfile = files[np.argmin(vlosses)]
        weightsPath = Path(weightsFolder, bestfile).resolve()
        # outFileName = Path(str(weightsPath).replace(".hdf5", "_eval.csv")).resolve()
        outFileName = Path(top_dir, weightsPath.parts[-3], weightsPath.parts[-1].replace(".hdf5", "_eval.csv")).resolve()

        if outFileName.exists() and not args.doOverwrite:
            print(f"Warning: {outFileName} exists. If you want to overwrite it pass in --doOverwrite.")
            continue
        
        confs.append({
                    "weightsPath" : weightsPath,
                    "outFileName" : outFileName,
                    "data_directory_path" : data_directory_path,
                    "labels_directory_path" : labels_directory_path,
                    "n_filters" : n_filters,
                    "pool_size" : pool_size,
                    "val_batch_size" : val_batch_size,
                    "val_file_size" : val_file_size
                })

    # if submitit false then just launch job
    if not query.get("submitit", False):
        for iC, conf in enumerate(confs):
            # only launch a single job
            if args.njobs != -1 and (iC+1) > args.njobs:
                continue
            print(conf)
            evaluate(conf)
    else:
        # submission
        executor = submitit.AutoExecutor(folder=top_dir)
        executor.update_parameters(**query.get("slurm", {}))
        # the following line tells the scheduler to only run at most 2 jobs at once. By default, this is several hundreds
        # executor.update_parameters(slurm_array_parallelism=2)
        
        # loop over configurations
        jobs = []
        with executor.batch():
            for iC, conf in enumerate(confs):
                
                # only launch a single job
                if args.njobs != -1 and (iC+1) > args.njobs:
                    continue
                
                print(conf)
                job = executor.submit(evaluate, conf)
                jobs.append(job)