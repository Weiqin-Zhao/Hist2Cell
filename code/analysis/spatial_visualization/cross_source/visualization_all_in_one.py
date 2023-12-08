import math
from IPython.core.pylabtools import figsize
import joblib
import pandas as pd
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def process_cell_type(i, coordinates, cell2location_count2, hist2cell_counts, stnet_counts, X, Y, case, slide, cell_names):
    # Calculate the minimum distance between points
    min_distance = np.min([math.sqrt((coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2) for i in range(len(coordinates)) for j in range(i+1, len(coordinates))])
    # Set the node size as a fraction of the minimum distance (e.g., 25%)
    node_size = min_distance * 0.220

    figsize(17, 5)
    
    # Create subplots for cell2location_count2, hist2cell_counts, and stnet_counts
    fig, axs = plt.subplots(1, 3)
    # Reduce the space between subplots
    plt.subplots_adjust(wspace=0.02)
    
    # Set the background color of each scatter plot to light gray
    axs[0].set_facecolor('lightgray')
    axs[1].set_facecolor('lightgray')
    axs[2].set_facecolor('lightgray')
    
    # cell2location_count2 plot
    scatter_plot1 = axs[0].scatter(X, Y, c=cell2location_count2[:, i], cmap='magma', s=node_size)
    axs[0].invert_yaxis()
    axs[0].set_xlim(min(coordinates[:, 0]), max(coordinates[:, 0]))
    axs[0].set_ylim(max(coordinates[:, 1]), min(coordinates[:, 1]))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("Cell2location")
    cbar1 = fig.colorbar(scatter_plot1, ax=axs[0], shrink=1.0, aspect=35)
    
    # hist2cell_counts plot
    scatter_plot2 = axs[1].scatter(X, Y, c=hist2cell_counts[:, i], cmap='magma', s=node_size)
    axs[1].invert_yaxis()
    axs[1].set_xlim(min(coordinates[:, 0]), max(coordinates[:, 0]))
    axs[1].set_ylim(max(coordinates[:, 1]), min(coordinates[:, 1]))
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title("Hist2Cell")
    cbar2 = fig.colorbar(scatter_plot2, ax=axs[1], shrink=1.0, aspect=35)
    
    # stnet_counts plot
    scatter_plot3 = axs[2].scatter(X, Y, c=stnet_counts[:, i], cmap='magma', s=node_size)
    axs[2].invert_yaxis()
    axs[2].set_xlim(min(coordinates[:, 0]), max(coordinates[:, 0]))
    axs[2].set_ylim(max(coordinates[:, 1]), min(coordinates[:, 1]))
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].set_title("STNet")
    cbar3 = fig.colorbar(scatter_plot3, ax=axs[2], shrink=1.0, aspect=35)

    # Save the figure without white margins
    plt.savefig(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/cross_source/all_in_one", case, slide, cell_names[i]+".png"), bbox_inches='tight')
    plt.close()




tif_list = glob('/data1/r20user3/shared_project/Hist2Cell/code/training/train_test_splits/stnet/test*')
tif_list.sort()
test_slides = list()
for tif in tif_list:
    tif_path = tif.split('_')[-1].split('.')[0]
    test_slides.append(tif_path)
    
cell_names = [cell[23:] for cell in list(pd.read_csv("/data1/r20user3/shared_project/Hist2Cell/data/stnet/23209_C1/cell_ratio.csv").columns)[1:]]

os.mkdir("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/cross_source/all_in_one")
for case in test_slides:
    os.mkdir(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/cross_source/all_in_one", case))
    
    save_path = "/data1/r20user3/shared_project/Hist2Cell/code/analysis/inference/breast_cross_source/breast_cross_source_epoch100_lr1e-4_2hop_ensemble_Trans1layer_GNNoutput50_onlycell_"+case+"_best_cell_all_abundance_average.pkl"
    pred_and_label = joblib.load(save_path)
    
    for slide in pred_and_label:
        os.mkdir(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/cross_source/all_in_one", case, slide))
        
        save_path = "/data1/r20user3/shared_project/Hist2Cell/code/analysis/inference/breast_cross_source/breast_cross_source_epoch100_lr1e-4_2hop_ensemble_Trans1layer_GNNoutput50_onlycell_"+case+"_best_cell_all_abundance_average.pkl"
        pred_and_label = joblib.load(save_path)
        pred_and_label[slide]['cell_abundance_predictions'] = np.clip(pred_and_label[slide]['cell_abundance_predictions'], a_min=0, a_max=None)
        pred_and_label[slide]['cell_abundance_labels'] = np.clip(pred_and_label[slide]['cell_abundance_labels'], a_min=0, a_max=None)

        coordinates = pred_and_label[slide]['coords']
        X = coordinates[:, 0]
        Y = coordinates[:, 1]

        cell2location_count2 = pred_and_label[slide]['cell_abundance_labels']
        hist2cell_counts = pred_and_label[slide]['cell_abundance_predictions']

        save_path = "/data1/r20user3/shared_project/Hist2Cell/code/analysis/inference/breast_cross_source/breast_cross_source_epoch100_lr1e-4_2hop_densenet_onlycell_"+case+"_best_cell_all_abundance_average.pkl"
        pred_and_label = joblib.load(save_path)
        pred_and_label[slide]['cell_abundance_predictions'] = np.clip(pred_and_label[slide]['cell_abundance_predictions'], a_min=0, a_max=None)

        stnet_counts = pred_and_label[slide]['cell_abundance_predictions']

        print(case)
        print(slide)
        
        # Number of threads to use
        n_threads = 16
        # Run the process_cell_type function in parallel with n_threads
        Parallel(n_jobs=n_threads)(delayed(process_cell_type)(i, coordinates, cell2location_count2, hist2cell_counts, stnet_counts, X, Y, case, slide, cell_names) for i in range(39))