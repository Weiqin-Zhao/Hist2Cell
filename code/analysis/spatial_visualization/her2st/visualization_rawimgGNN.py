# import matplotlib.pyplot as plt 
# import matplotlib
# from matplotlib import rcParams
# rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
# import numpy as np
# from scipy.spatial.distance import jensenshannon
# import matplotlib as mpl
# import joblib
# import pandas as pd
# import os
# from anndata import AnnData
# import scanpy as sc
# import cv2
# from glob import glob


# tif_list = glob('/data1/r20user3/shared_project/Hist2Cell/code/training/train_test_splits/her2st/test*')
# tif_list.sort()
# test_slides = list()
# for tif in tif_list:
#     tif_path = tif.split('_')[-1].split('.')[0]
#     test_slides.append(tif_path)

# cell_names = [cell[23:] for cell in list(pd.read_csv("/data1/r20user3/shared_project/Hist2Cell/data/her2st/A1/cell_ratio.csv").columns)[1:]]
# for case in test_slides:
#     os.mkdir(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case))
    
#     save_path = "/data1/r20user3/shared_project/Hist2Cell/code/analysis/inference/her2st/her2st_epoch100_lr1e-4_2hop_ensemble_onlycell_"+case+"_best_cell_all_abundance_average.pkl"
#     pred_and_label = joblib.load(save_path)
    
#     case_pred = list()
#     case_label = list()
    
#     for slide in pred_and_label:
#         os.mkdir(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide))
        
#         pred_and_label[slide]['cell_abundance_predictions'] = np.clip(pred_and_label[slide]['cell_abundance_predictions'], a_min=0, a_max=None)
#         pred_ratio = pred_and_label[slide]['cell_abundance_predictions'] / pred_and_label[slide]['cell_abundance_predictions'].sum(axis=1, keepdims=True) * 10.0
#         real_ratio = pred_and_label[slide]['cell_abundance_labels'] / pred_and_label[slide]['cell_abundance_labels'].sum(axis=1, keepdims=True) * 10.0
        
#         os.mkdir(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide, "cell proportion"))
#         counts = pred_ratio
#         coordinates = pred_and_label[slide]['coords']
#         image = cv2.imread("/data1/r20user3/shared_project/Hist2Cell/data/her2st/"+slide+"/"+slide+".jpg")
#         adata = AnnData(counts, obsm={"spatial": coordinates}, dtype=np.int64)
#         adata.var_names = cell_names
#         # sc.pp.normalize_total(adata)
#         # sc.pp.log1p(adata)
#         spatial_key = "spatial"
#         library_id = slide
#         adata.uns[spatial_key] = {library_id: {}}
#         adata.uns[spatial_key][library_id]["images"] = {"hires": image}
#         adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 224.0}
#         for i in range(39):
#             # plot with nice names
#             with mpl.rc_context({'figure.figsize': (10, 10), "font.size": 18}):
#                 img = sc.pl.spatial(adata, 
#                             color=[cell_names[i]], # limit size in this notebook
#                             ncols=1, 
#                             cmap='magma',
#                             size=0.8, img_key=None, 
#                             # alpha_img=0.9,
#                             # vmin='p5.0', vmax='p95.0',
#                             return_fig=True
#                             )
#                 fig = img[0].figure  # Get the figure associated with the axes object
#                 fig.savefig(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide, "cell proportion", cell_names[i]+"_Hist2cell.png"))
#                 plt.close(fig)
            
#         counts = real_ratio
#         coordinates = pred_and_label[slide]['coords']
#         image = cv2.imread("/data1/r20user3/shared_project/Hist2Cell/data/her2st/"+slide+"/"+slide+".jpg")
#         adata = AnnData(counts, obsm={"spatial": coordinates}, dtype=np.int64)
#         adata.var_names = cell_names
#         # sc.pp.normalize_total(adata)
#         # sc.pp.log1p(adata)
#         spatial_key = "spatial"
#         library_id = slide
#         adata.uns[spatial_key] = {library_id: {}}
#         adata.uns[spatial_key][library_id]["images"] = {"hires": image}
#         adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 224.0}
#         for i in range(39):
#             # plot with nice names
#             with mpl.rc_context({'figure.figsize': (10, 10), "font.size": 18}):
#                 img = sc.pl.spatial(adata, 
#                             color=[cell_names[i]], # limit size in this notebook
#                             ncols=1, 
#                             cmap='magma',
#                             size=0.8, img_key=None, 
#                             # alpha_img=0.9,
#                             # vmin='p5.0', vmax='p95.0',
#                             return_fig=True
#                             )
                
#                 fig = img[0].figure  # Get the figure associated with the axes object
#                 fig.savefig(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide, "cell proportion", cell_names[i]+"_Cell2location.png"))
#                 plt.close(fig)  
            
        
#         os.mkdir(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide, "cell abundance"))
#         counts = pred_and_label[slide]['cell_abundance_predictions']
#         coordinates = pred_and_label[slide]['coords']
#         image = cv2.imread("/data1/r20user3/shared_project/Hist2Cell/data/her2st/"+slide+"/"+slide+".jpg")
#         adata = AnnData(counts, obsm={"spatial": coordinates}, dtype=np.int64)
#         adata.var_names = cell_names
#         # sc.pp.normalize_total(adata)
#         # sc.pp.log1p(adata)
#         spatial_key = "spatial"
#         library_id = slide
#         adata.uns[spatial_key] = {library_id: {}}
#         adata.uns[spatial_key][library_id]["images"] = {"hires": image}
#         adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 224.0}
#         for i in range(39):
#             # plot with nice names
#             with mpl.rc_context({'figure.figsize': (10, 10), "font.size": 18}):
#                 img = sc.pl.spatial(adata, 
#                             color=[cell_names[i]], # limit size in this notebook
#                             ncols=1, 
#                             cmap='magma',
#                             size=0.8, img_key=None, 
#                             # alpha_img=0.9,
#                             # vmin='p5.0', vmax='p95.0',
#                             return_fig=True
#                             )
                
#                 fig = img[0].figure  # Get the figure associated with the axes object
#                 fig.savefig(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide, "cell abundance", cell_names[i]+"_Hist2cell.png"))
#                 plt.close(fig)
            
#         counts = pred_and_label[slide]['cell_abundance_labels']
#         coordinates = pred_and_label[slide]['coords']
#         image = cv2.imread("/data1/r20user3/shared_project/Hist2Cell/data/her2st/"+slide+"/"+slide+".jpg")
#         adata = AnnData(counts, obsm={"spatial": coordinates}, dtype=np.int64)
#         adata.var_names = cell_names
#         sc.pp.normalize_total(adata)
#         sc.pp.log1p(adata)
#         spatial_key = "spatial"
#         library_id = slide
#         adata.uns[spatial_key] = {library_id: {}}
#         adata.uns[spatial_key][library_id]["images"] = {"hires": image}
#         adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 224.0}
#         for i in range(39):
#             # plot with nice names
#             with mpl.rc_context({'figure.figsize': (10, 10), "font.size": 18}):
#                 img = sc.pl.spatial(adata, 
#                             color=[cell_names[i]], # limit size in this notebook
#                             ncols=1, 
#                             cmap='magma',
#                             size=0.8, img_key=None, 
#                             # alpha_img=0.9,
#                             # vmin='p5.0', vmax='p95.0',
#                             return_fig=True
#                             )
                
#                 fig = img[0].figure  # Get the figure associated with the axes object
#                 fig.savefig(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide, "cell abundance", cell_names[i]+"_Cell2location.png"))
#                 plt.close(fig)   




import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt 
import matplotlib
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
import numpy as np
from scipy.spatial.distance import jensenshannon

    
from glob import glob
tif_list = glob('/data1/r20user3/shared_project/Hist2Cell/code/training/train_test_splits/her2st/test*')
tif_list.sort()
test_slides = list()
for tif in tif_list:
    tif_path = tif.split('_')[-1].split('.')[0]
    test_slides.append(tif_path)
    
import joblib
import pandas as pd
import os

cell_names = [cell[23:] for cell in list(pd.read_csv("/data1/r20user3/shared_project/Hist2Cell/data/her2st/A1/cell_ratio.csv").columns)[1:]]
for case in test_slides:
    # os.mkdir(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case))
    
    save_path = "/data1/r20user3/shared_project/Hist2Cell/code/analysis/inference/her2st/her2st_epoch100_lr1e-4_2hop_rawimgGNN_onlycell_"+case+"_best_cell_all_abundance_average.pkl"
    pred_and_label = joblib.load(save_path)
    
    case_pred = list()
    case_label = list()
    
    for slide in pred_and_label:
        # os.mkdir(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide))
        
        pred_and_label[slide]['cell_abundance_predictions'] = np.clip(pred_and_label[slide]['cell_abundance_predictions'], a_min=0, a_max=None)
        pred_ratio = pred_and_label[slide]['cell_abundance_predictions'] / pred_and_label[slide]['cell_abundance_predictions'].sum(axis=1, keepdims=True)
        real_ratio = pred_and_label[slide]['cell_abundance_labels'] / pred_and_label[slide]['cell_abundance_labels'].sum(axis=1, keepdims=True)
        
        coordinates = pred_and_label[slide]['coords']
        X = coordinates[:, 0]
        Y = coordinates[:, 1]
        
        # os.mkdir(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide, "cell abundance"))
        # counts = pred_and_label[slide]['cell_abundance_labels']
        # for i in range(39):
        #     A = counts[:, i]
        #     # Calculate the minimum distance between points
        #     min_distance = np.min([math.sqrt((coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2) for i in range(len(coordinates)) for j in range(i+1, len(coordinates))])
        #     # Set the node size as a fraction of the minimum distance (e.g., 25%)
        #     node_size = min_distance * 0.90
        #     # set figure size
        #     figsize(10, 10)
        #     # Create a scatter plot with a color map
        #     plt.scatter(X, Y, c=A, cmap='magma', s=node_size)
        #     # Add a color bar with the spot values
        #     cbar = plt.colorbar()
        #     cbar.set_label('Cell Abundance', fontsize=25)
        #     # Add title
        #     plt.suptitle(cell_names[i], fontsize=25, y=0.95)
        #     # Flip y-axis
        #     plt.gca().invert_yaxis()
        #     # Adjust the plot size to match the range of X and Y coordinates
        #     plt.xlim(min(coordinates[:, 0]), max(coordinates[:, 0]))
        #     plt.ylim(max(coordinates[:, 1]), min(coordinates[:, 1]))
        #     # Show the plot
        #     plt.axis('equal')
        #     plt.axis('off')
        #     plt.savefig(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide, "cell abundance", cell_names[i]+"_STNet.png"))
        #     plt.close()
        #     # break
            
        counts = pred_and_label[slide]['cell_abundance_predictions']
        for i in range(39):
            A = counts[:, i]
            # Calculate the minimum distance between points
            min_distance = np.min([math.sqrt((coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2) for i in range(len(coordinates)) for j in range(i+1, len(coordinates))])
            # Set the node size as a fraction of the minimum distance (e.g., 25%)
            node_size = min_distance * 0.90
            # set figure size
            figsize(10, 10)
            # Create a scatter plot with a color map
            plt.scatter(X, Y, c=A, cmap='magma', s=node_size)
            # Add a color bar with the spot values
            cbar = plt.colorbar()
            cbar.set_label('Cell Abundance', fontsize=25)
            # Add title
            plt.suptitle(cell_names[i], fontsize=25, y=0.95)
            # Flip y-axis
            plt.gca().invert_yaxis()
            # Adjust the plot size to match the range of X and Y coordinates
            plt.xlim(min(coordinates[:, 0]), max(coordinates[:, 0]))
            plt.ylim(max(coordinates[:, 1]), min(coordinates[:, 1]))
            # Show the plot
            plt.axis('equal')
            plt.axis('off')
            plt.savefig(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide, "cell abundance", cell_names[i]+"_rawimgGNN.png"))
            plt.close()
            # break

        # os.mkdir(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide, "cell proportion"))
        # counts = real_ratio
        # for i in range(39):
        #     A = counts[:, i]
        #     # Calculate the minimum distance between points
        #     min_distance = np.min([math.sqrt((coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2) for i in range(len(coordinates)) for j in range(i+1, len(coordinates))])
        #     # Set the node size as a fraction of the minimum distance (e.g., 25%)
        #     node_size = min_distance * 0.90
        #     # set figure size
        #     figsize(10, 10)
        #     # Create a scatter plot with a color map
        #     plt.scatter(X, Y, c=A, cmap='magma', s=node_size)
        #     # Add a color bar with the spot values
        #     cbar = plt.colorbar()
        #     cbar.set_label('Cell Abundance', fontsize=25)
        #     # Add title
        #     plt.suptitle(cell_names[i], fontsize=25, y=0.95)
        #     # Flip y-axis
        #     plt.gca().invert_yaxis()
        #     # Adjust the plot size to match the range of X and Y coordinates
        #     plt.xlim(min(coordinates[:, 0]), max(coordinates[:, 0]))
        #     plt.ylim(max(coordinates[:, 1]), min(coordinates[:, 1]))
        #     # Show the plot
        #     plt.axis('equal')
        #     plt.axis('off')
        #     plt.savefig(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide, "cell proportion", cell_names[i]+"_STNet.png"))
        #     plt.close()
        #     # break        

        counts = pred_ratio
        for i in range(39):
            A = counts[:, i]
            # Calculate the minimum distance between points
            min_distance = np.min([math.sqrt((coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2) for i in range(len(coordinates)) for j in range(i+1, len(coordinates))])
            # Set the node size as a fraction of the minimum distance (e.g., 25%)
            node_size = min_distance * 0.90
            # set figure size
            figsize(10, 10)
            # Create a scatter plot with a color map
            plt.scatter(X, Y, c=A, cmap='magma', s=node_size)
            # Add a color bar with the spot values
            cbar = plt.colorbar()
            cbar.set_label('Cell Abundance', fontsize=25)
            # Add title
            plt.suptitle(cell_names[i], fontsize=25, y=0.95)
            # Flip y-axis
            plt.gca().invert_yaxis()
            # Adjust the plot size to match the range of X and Y coordinates
            plt.xlim(min(coordinates[:, 0]), max(coordinates[:, 0]))
            plt.ylim(max(coordinates[:, 1]), min(coordinates[:, 1]))
            # Show the plot
            plt.axis('equal')
            plt.axis('off')
            plt.savefig(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/spatial_visualization/her2st", case, slide, "cell proportion", cell_names[i]+"_rawimgGNN.png"))
            plt.close()
            # break  
        
        # break