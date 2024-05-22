# python imports
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

if __name__ == "__main__":

    # set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inFolder", help="top level directory", default=None)
    parser.add_argument("-t", "--trainingID", help="training folder id", default=None)
    parser.add_argument("-e", "--evaluationID", help="evaluation folder id", default=None)
    args = parser.parse_args()

    # get file lists
    training_dir = Path(args.inFolder, args.trainingID).resolve()
    training_list = list(training_dir.glob("*/weights*"))
    eval_dir = Path(training_dir, "eval", args.evaluationID).resolve()
    eval_list = list(eval_dir.glob("*/*/*.csv"))

    # entries
    stats = []
    nf = [1,2,3,4,5] # number filters
    ps = [1,2,3,4,5] # pool size

    # loop over
    for n_filters in nf:
        for pool_size in ps:
            # identify training file
            t = [i for i in training_list if f"nFilters{n_filters}-poolSize{pool_size}" in str(i)][0]
            # identify evaluation file
            e = [i for i in eval_list if t.parts[-2] in str(i)][0]
            # get loss
            losses = str(e).split("-t")[1].split("-v") # float(str(e).split("-v")[1].split("_eval.csv")[0])
            training_loss = float(losses[0])
            validation_loss = float(losses[1].split("_eval.csv")[0])
            # append to stats
            stats.append([n_filters, pool_size, training_loss, validation_loss])

    # make numpy matrix
    stats = np.array(stats)

    # make plot for training and validation loss
    for name in ["Training Loss", "Validation Loss"]:
        
        # know column index in stats
        idx = 2 if name == "Training Loss" else 3

        # make quick plot
        fig, ax = plt.subplots(figsize=(6, 5))
        # binning for scan
        nf_bins = np.linspace(nf[0]-0.5, nf[-1]+0.5, len(nf)+1)
        ps_bins = np.linspace(ps[0]-0.5, ps[-1]+0.5, len(ps)+1)
        # plot
        hist, xbins, ybins, im = plt.hist2d(stats[:,0], stats[:,1], bins=[nf_bins, ps_bins], weights=stats[:,idx], cmap="viridis")
        plt.xlabel("Number of Filters")
        plt.ylabel("Pool Size")
        cbar = plt.colorbar(label=name)
        # draw loss text on each bin
        for i in range(len(ybins)-1):
            for j in range(len(xbins)-1):
                ax.text(xbins[j]+0.5, ybins[i]+0.5, hist.T[i,j], color="w", ha="center", va="center") #, fontweight="bold")
                if hist.T[i,j] == np.min(hist):
                    ax.text(xbins[j]+0.5, ybins[i]+0.2, "min", color="w", ha="center", va="center", fontweight="bold")
        
        # stamps
        plt.text(0.12, 0.9, f"{args.trainingID}, evaluation {args.evaluationID}", fontsize=12, transform=plt.gcf().transFigure)

        # save fig
        outFileName = f"{args.trainingID}_evaluation-{args.evaluationID}_{''.join(name.split(' '))}.pdf"
        print(outFileName)
        plt.savefig(outFileName)