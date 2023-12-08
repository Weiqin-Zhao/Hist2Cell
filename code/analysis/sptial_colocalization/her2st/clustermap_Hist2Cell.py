import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


cell_names = [cell[23:] for cell in list(pd.read_csv("/data1/r20user3/shared_project/Hist2Cell/data/her2st/A1/cell_ratio.csv").columns)[1:]]

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

combinations = []
for i in range(len(cell_names)):
    for j in range(i+1, len(cell_names)):
        combinations.append((cell_names[i], cell_names[j]))
        
for case in test_slides:
    # os.mkdir(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/sptial_colocalization/her2st", case))
    
    save_path = "/data1/r20user3/shared_project/Hist2Cell/code/analysis/inference/her2st/her2st_epoch100_lr1e-4_2hop_ensemble_onlycell_"+case+"_best_cell_all_abundance_average.pkl"
    pred_and_label = joblib.load(save_path)
    
    for slide in pred_and_label:
        # os.mkdir(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/sptial_colocalization/humanlung", case, slide))
        
        df_label = pd.read_csv(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/sptial_colocalization/her2st", case, slide, "MoranR_Cell2location.csv"), index_col=0)
        df_figure = df_label[['A', 'B', 'R_val']]
        correlation_df = df_figure.pivot(index="A", columns="B", values="R_val")
        for var in correlation_df.index:
            correlation_df.at[var, var] = 1
            correlation_df = correlation_df.combine_first(correlation_df.T)
        correlation_df.iloc[-1, -1] = 1.0
        cell2location_cluster = sns.clustermap(correlation_df, figsize=(20, 20))
        row_linkage = cell2location_cluster.dendrogram_row.linkage
        col_linkage = cell2location_cluster.dendrogram_col.linkage
        plt.savefig(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/sptial_colocalization/her2st", case, slide, "MoranR_clustermap_Cell2location.png"))
        plt.close()
        
        
        df_label = pd.read_csv(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/sptial_colocalization/her2st", case, slide, "MoranR_Hist2Cell.csv"), index_col=0)
        df_figure = df_label[['A', 'B', 'R_val']]
        correlation_df = df_figure.pivot(index="A", columns="B", values="R_val")
        for var in correlation_df.index:
            correlation_df.at[var, var] = 1
            correlation_df = correlation_df.combine_first(correlation_df.T)
        correlation_df.iloc[-1, -1] = 1.0
        sns.clustermap(correlation_df, figsize=(20, 20), row_linkage=row_linkage, col_linkage=col_linkage)
        plt.savefig(os.path.join("/data1/r20user3/shared_project/Hist2Cell/code/analysis/sptial_colocalization/her2st", case, slide, "MoranR_clustermap_Hist2Cell.png"))
        plt.close()