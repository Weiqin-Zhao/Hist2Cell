import argparse


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Classification')

        parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate of model training")
        parser.add_argument("--num_epochs", type=int, default=100, help="Cycle times of model training")
        parser.add_argument('--batch_size', type=int, default=1, help='batch size for origin global image (without downsampling)')

        parser.add_argument('--log', type=bool, default=False, help='log the experiment or not')
        parser.add_argument('--task_name', type=str, default="test", help='name of current task')

        parser.add_argument('--patch_dir', type=str,
                            default='/home/r15user3/Documents/shared_project/Hist2Cell/data/human_lung_cell2location',
                            help='path of all data')
        parser.add_argument('--train_set', type=str,
                            default="/home/r15user3/Documents/shared_project/Hist2Cell/code/training/train_test_splits/humanlung_cell2location/train_leave_A37.txt",
                            help='path of train set .txt')
        parser.add_argument('--test_set', type=str,
                            default="/home/r15user3/Documents/shared_project/Hist2Cell/code/training/train_test_splits/humanlung_cell2location/test_leave_A37.txt",
                            help='path of test set .txt')

        parser.add_argument('--gpu_list', type=list,
                            default=[0],
                            help='use which gpu')

        parser.add_argument('--graph_data_path', type=str,
                            default="/data2/r10user3/Spatial_Gene_Cell_Ratio/code/graphs/human_lung_cell2location_high250gene_celltype80_KimiaNet_graphs",
                            help='path of graph data')
        
        parser.add_argument("--resume_from_Kimia", type=bool, default=False, help="initialize with KimiaNet weights")
        
        parser.add_argument('--celltype_num', type=int, default=80, help='number of cell types being predicted')
        
        parser.add_argument('--vit_depth', type=int, default=3, help='number of ViT layers')
        parser.add_argument('--seed', type=int, default=3, help='initialization seed')
        
        parser.add_argument('--rawimg_graph_path', type=str, default="/home/r15user3/Documents/shared_project/Hist2Cell/code/data_preprocessing/rawimg_graph/human_lung_cell2location", help='rawimg_graph_path')
        parser.add_argument('--hop', type=int, default=2, help='hop of subgraph from center node')
        parser.add_argument('--subgraph_bs', type=int, default=16, help='batch size of neighborloader of subgraphs')
        parser.add_argument("--proto", type=bool, default=False, help="add proto token or not")
        parser.add_argument("--ensemble", type=bool, default=True, help="use ensemble or not")
        
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args
    
    def parse_known_args(self):
        args, unknown = self.parser.parse_known_args()
        return args, unknown