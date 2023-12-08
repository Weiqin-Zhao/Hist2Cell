import os
import pickle
import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
import time
from torch.utils.tensorboard import SummaryWriter
import sys
from option import Options
from utils import setup_seed, Logger
import torchvision.models as models


args, _ = Options().parse_known_args()

setup_seed(args.seed)
task_name = args.task_name

if args.log:
    sys.stdout = Logger(sys.stdout, task_name=task_name)
    sys.stderr = Logger(sys.stderr, task_name=task_name)
    writer = SummaryWriter(log_dir=os.path.join(os.path.split(os.path.realpath(__file__))[0], "runs"), comment=task_name)
    
print(task_name)

lr = args.lr
num_epochs = args.num_epochs
batch_size = args.batch_size

gpu_list = args.gpu_list
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_slides = open(args.train_set).read().split('\n')
test_slides = open(args.test_set).read().split('\n')

from torch_geometric.nn import GATv2Conv, LayerNorm
# from ViT import Mlp, VisionTransformer
from model.ViT import Mlp, VisionTransformer
from torch_geometric.data import Batch
from torch_geometric.loader import NeighborLoader
import torch_geometric
torch_geometric.typing.WITH_PYG_LIB = False
from tqdm import tqdm
data_path = args.rawimg_graph_path

train_graph_list = list()
for item in train_slides:
    train_graph_list.append(torch.load(os.path.join(data_path, item+'.pt')))
train_dataset = Batch.from_data_list(train_graph_list)

test_graph_list = list()
for item in test_slides:
    test_graph_list.append(torch.load(os.path.join(data_path, item+'.pt')))
test_dataset = Batch.from_data_list(test_graph_list)    

if args.hop <= 2:
    num_neighbors = [-1]*args.hop
else:
    num_neighbors = [-1]+[2]*(args.hop-1)

train_loader = NeighborLoader(
    train_dataset,
    num_neighbors=num_neighbors,
    batch_size=args.subgraph_bs,
    directed=False,
    input_nodes=None,
    shuffle=True,
    # num_workers=8,
    # pin_memory=True, 
    # prefetch_factor=2,
)

test_loader = NeighborLoader(
    test_dataset,
    num_neighbors=[-1]*args.hop,
    batch_size=args.subgraph_bs,
    directed=False,
    input_nodes=None,
    shuffle=False,
    # num_workers=8,
    # pin_memory=True, 
    # prefetch_factor=2,
)

from torch.nn import Linear
from model.layers import Add, Clone
from model.ViT import Attention
import torch.nn.functional as F
from einops import rearrange
class Transformer(nn.Module):
    def __init__(self, cell_dim=80, vit_depth=3):
        super(Transformer, self).__init__()
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet18 = torch.nn.Sequential(*list(self.resnet18.children())[:-1])
        
        self.embed_dim = 32 * 8
        self.head = 8
        self.dropout = 0.3
        
        self.downdim = torch.nn.Linear(in_features=512, out_features=self.embed_dim)
        
        self.cell_transformer = VisionTransformer(num_classes=cell_dim, embed_dim=self.embed_dim, depth=vit_depth,
                                                  mlp_head=True, drop_rate=self.dropout, attn_drop_rate=self.dropout)

        
        
        
    def forward(self, x, edge_index):
        x_spot = self.resnet18(x)
        
        x_local = x_spot.squeeze()
        x_local = x_local.unsqueeze(0)
        
        x_local = self.downdim(x_local)
        
        cell_prediction, _ = self.cell_transformer(x_local)
        cell_prediction = cell_prediction.squeeze()
        
        return cell_prediction


model = Transformer(vit_depth=args.vit_depth, cell_dim=args.celltype_num)
model = model.to(device)

params = model.parameters()
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5, last_epoch=-1, verbose=False)

# best_gene_loss = 100.0
# best_gene_pos_average = 0.0
# best_gene_all_average = 0.0
# best_gene_pos_pearson_count = 0.0
# best_gene_pearson_list = []
best_cell_abundance_loss = 100.0
best_cell_abundance_pos_average = 0.0
best_cell_abundance_all_average = 0.0
best_cell_abundance_pos_pearson_count = 0.0
best_cell_pearson_list = []

since = time.time()
for epoch in range(num_epochs):
    model.train()
    print("---------------------------------------"*4)
    print('Epoch: {} \t'.format(epoch + 1))
    print('lr = ',optimizer.param_groups[0]["lr"])
    
    train_sample_num = 0
    # train_gene_pred_array = []
    train_cell_pred_array = []
    # train_gene_label_array = []
    train_cell_label_array = []
    # train_loss, train_gene_loss, train_cell_abundance_loss = 0, 0, 0
    train_loss, train_cell_abundance_loss = 0, 0
    batch_loss = 0.0
    # for graph in tqdm(train_loader):
    for graph in train_loader:
        x = graph.x.to(device)
        y = graph.y.to(device)
        edge_index = graph.edge_index.to(device)
        # gene_label = y[:, :250]
        cell_label = y[:, 250:]
        
        # gene_pred, cell_pred = model(x=x, edge_index=edge_index)
        cell_pred = model(x=x, edge_index=edge_index)

        # gene_loss = criterion(gene_pred, gene_label) 
        cell_loss = criterion(cell_pred, cell_label)

        # loss = gene_loss + cell_loss
        loss = cell_loss
        batch_loss += loss
        
        if train_sample_num % batch_size == 0:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_loss = 0.0
            
        center_num = len(graph.input_id)
        # center_gene_label = gene_label[:center_num, :]
        center_cell_label = cell_label[:center_num, :]
        # center_gene_pred = gene_pred[:center_num, :]
        center_cell_pred = cell_pred[:center_num, :]
        
        # train_gene_label_array.append(center_gene_label.squeeze().cpu().detach().numpy())
        # train_gene_pred_array.append(center_gene_pred.squeeze().cpu().detach().numpy())
        train_cell_label_array.append(center_cell_label.squeeze().cpu().detach().numpy())
        train_cell_pred_array.append(center_cell_pred.squeeze().cpu().detach().numpy())
        train_sample_num = train_sample_num + center_num
        
        train_loss += loss.item() * center_num
        # train_gene_loss += gene_loss.item() * center_num
        train_cell_abundance_loss += cell_loss.item() * center_num
    
    scheduler.step()
    
    train_loss = train_loss / train_sample_num
    # train_gene_loss = train_gene_loss / train_sample_num
    train_cell_abundance_loss = train_cell_abundance_loss / train_sample_num
        
    # train_gene_pred_array = np.concatenate(train_gene_pred_array)
    # train_gene_label_array = np.concatenate(train_gene_label_array)
    if len(train_cell_pred_array[-1].shape) == 1:
        train_cell_pred_array[-1] = np.expand_dims(train_cell_pred_array[-1], axis=0)
    train_cell_pred_array = np.concatenate(train_cell_pred_array)
    if len(train_cell_label_array[-1].shape) == 1:
        train_cell_label_array[-1] = np.expand_dims(train_cell_label_array[-1], axis=0)
    train_cell_label_array = np.concatenate(train_cell_label_array)

    # train_gene_pos_pearson_average = 0.0
    # train_gene_all_pearson_average = 0.0
    # train_gene_pos_pearson_count = 0
    # for i in range(250):
    #     r, p = pearsonr(train_gene_pred_array[:, i], train_gene_label_array[:, i])
    #     if r > 0.0 and p <= 0.05:
    #         train_gene_pos_pearson_count = train_gene_pos_pearson_count + 1
    #         train_gene_pos_pearson_average = train_gene_pos_pearson_average + r
    #     train_gene_all_pearson_average = train_gene_all_pearson_average + r
    # if train_gene_pos_pearson_count >= 1:
    #     train_gene_pos_pearson_average = train_gene_pos_pearson_average / train_gene_pos_pearson_count
    # else:
    #     train_gene_pos_pearson_average = 0.0
    # train_gene_all_pearson_average = train_gene_all_pearson_average / 250.0

    train_cell_abundance_pos_pearson_average = 0.0
    train_cell_abundance_all_pearson_average = 0.0
    train_cell_abundance_pos_pearson_count = 0
    for i in range(args.celltype_num):
        r, p = pearsonr(train_cell_pred_array[:, i], train_cell_label_array[:, i])
        if r > 0.0 and p <= 0.05:
            train_cell_abundance_pos_pearson_count = train_cell_abundance_pos_pearson_count + 1
            train_cell_abundance_pos_pearson_average = train_cell_abundance_pos_pearson_average + r
        if np.isnan(r):
            r = 0.0
        train_cell_abundance_all_pearson_average = train_cell_abundance_all_pearson_average + r
    if train_cell_abundance_pos_pearson_count >= 1:
        train_cell_abundance_pos_pearson_average /= train_cell_abundance_pos_pearson_count
    else:
        train_cell_abundance_pos_pearson_average = 0.0
    train_cell_abundance_all_pearson_average = train_cell_abundance_all_pearson_average / args.celltype_num


    with torch.no_grad():
        model.eval()

        test_sample_num = 0
        # test_gene_pred_array = []
        test_cell_pred_array = []
        # test_gene_label_array = []
        test_cell_label_array = []
        # test_loss, test_gene_loss, test_cell_abundance_loss = 0, 0, 0
        test_loss, test_cell_abundance_loss = 0, 0
        # for graph in tqdm(test_loader):
        for graph in test_loader:
            x = graph.x.to(device)
            y = graph.y.to(device)
            edge_index = graph.edge_index.to(device)
            # gene_label = y[:, :250]
            cell_label = y[:, 250:]
            
            # gene_pred, cell_pred = model(x=x, edge_index=edge_index)
            cell_pred = model(x=x, edge_index=edge_index)

            # gene_loss = criterion(gene_pred, gene_label) 
            cell_loss = criterion(cell_pred, cell_label)

            # loss = gene_loss + cell_loss
            loss = cell_loss
                
            center_num = len(graph.input_id)
            # center_gene_label = gene_label[:center_num, :]
            center_cell_label = cell_label[:center_num, :]
            # center_gene_pred = gene_pred[:center_num, :]
            center_cell_pred = cell_pred[:center_num, :]
            
            # test_gene_label_array.append(center_gene_label.squeeze().cpu().detach().numpy())
            # test_gene_pred_array.append(center_gene_pred.squeeze().cpu().detach().numpy())
            test_cell_label_array.append(center_cell_label.squeeze().cpu().detach().numpy())
            test_cell_pred_array.append(center_cell_pred.squeeze().cpu().detach().numpy())
            test_sample_num = test_sample_num + center_num
            
            test_loss += loss.item() * center_num
            # test_gene_loss += gene_loss.item() * center_num
            test_cell_abundance_loss += cell_loss.item() * center_num
    
        # test_gene_loss = test_gene_loss / test_sample_num
        # if test_gene_loss < best_gene_loss:
        #     best_gene_loss = test_gene_loss
        #     torch.save(model.state_dict(), os.path.join(os.path.split(os.path.realpath(__file__))[0], "model_weights", task_name + "_best_gene_loss.pth"))
        #     print("saving best gene loss " + str(test_gene_loss))

        test_cell_abundance_loss = test_cell_abundance_loss / test_sample_num
        if test_cell_abundance_loss < best_cell_abundance_loss:
            best_cell_abundance_loss = test_cell_abundance_loss
            torch.save(model.state_dict(), os.path.join(os.path.split(os.path.realpath(__file__))[0], "model_weights", task_name + "_best_cell_loss.pth"))
            print("saving best cell loss " + str(test_cell_abundance_loss))
 
    # test_gene_pred_array = np.concatenate(test_gene_pred_array)
    # test_gene_label_array = np.concatenate(test_gene_label_array)
    if len(test_cell_pred_array[-1].shape) == 1:
        test_cell_pred_array[-1] = np.expand_dims(test_cell_pred_array[-1], axis=0)
    test_cell_pred_array = np.concatenate(test_cell_pred_array)
    if len(test_cell_label_array[-1].shape) == 1:
        test_cell_label_array[-1] = np.expand_dims(test_cell_label_array[-1], axis=0)
    test_cell_label_array = np.concatenate(test_cell_label_array)

    # test_gene_pos_pearson_average = 0.0
    # test_gene_all_pearson_average = 0.0
    # test_gene_pos_pearson_count = 0
    # test_gene_pearson_list = []
    # for i in range(250):
    #     r, p = pearsonr(test_gene_pred_array[:, i], test_gene_label_array[:, i])
    #     if r > 0.0 and p <= 0.05:
    #         test_gene_pos_pearson_count = test_gene_pos_pearson_count + 1
    #         test_gene_pos_pearson_average = test_gene_pos_pearson_average + r
    #     test_gene_all_pearson_average = test_gene_all_pearson_average + r
    #     test_gene_pearson_list.append(r)
    # test_gene_all_pearson_average = test_gene_all_pearson_average / 250.0
    # if test_gene_pos_pearson_count >= 1:
    #     test_gene_pos_pearson_average = test_gene_pos_pearson_average / test_gene_pos_pearson_count
    # else:
    #     test_gene_pos_pearson_average = 0.0
    # if test_gene_pos_pearson_average > best_gene_pos_average:
    #     best_gene_pos_average = test_gene_pos_pearson_average
    #     torch.save(model.state_dict(), os.path.join(os.path.split(os.path.realpath(__file__))[0], "model_weights", task_name + "_best_gene_pos_average.pth"))
    #     print("saving " + "best gene pos average " + str(test_gene_pos_pearson_average))
    # if test_gene_all_pearson_average > best_gene_all_average:
    #     best_gene_all_average = test_gene_all_pearson_average
    #     torch.save(model.state_dict(), os.path.join(os.path.split(os.path.realpath(__file__))[0], "model_weights", task_name + "_best_gene_all_average.pth"))
    #     print("saving " + "best gene all average " + str(test_gene_all_pearson_average))
    #     best_gene_pearson_list = test_gene_pearson_list
    # if test_gene_pos_pearson_count > best_gene_pos_pearson_count:
    #     best_gene_pos_pearson_count = test_gene_pos_pearson_count
    #     torch.save(model.state_dict(), os.path.join(os.path.split(os.path.realpath(__file__))[0], "model_weights", task_name + "_best_gene_pos_count.pth"))
    #     print("saving " + "best gene pos count " + str(test_gene_pos_pearson_count))
        
    test_cell_abundance_pos_pearson_average = 0.0
    test_cell_abundance_all_pearson_average = 0.0
    test_cell_abundance_pos_pearson_count = 0   
    test_cell_pearson_list = []
    for i in range(args.celltype_num):
        r, p = pearsonr(test_cell_pred_array[:, i], test_cell_label_array[:, i])
        if r > 0.0 and p <= 0.05:
            test_cell_abundance_pos_pearson_count = test_cell_abundance_pos_pearson_count + 1
            test_cell_abundance_pos_pearson_average = test_cell_abundance_pos_pearson_average + r
        if np.isnan(r):
            r = 0.0
        test_cell_abundance_all_pearson_average = test_cell_abundance_all_pearson_average + r
        test_cell_pearson_list.append(r)
    if test_cell_abundance_pos_pearson_count >= 1:
        test_cell_abundance_pos_pearson_average /= test_cell_abundance_pos_pearson_count
    else:
        test_cell_abundance_pos_pearson_average = 0.0
    test_cell_abundance_all_pearson_average = test_cell_abundance_all_pearson_average / args.celltype_num
    if test_cell_abundance_pos_pearson_average > best_cell_abundance_pos_average:
        best_cell_abundance_pos_average = test_cell_abundance_pos_pearson_average
        torch.save(model.state_dict(), os.path.join(os.path.split(os.path.realpath(__file__))[0], "model_weights", task_name + "_best_cell_pos_abundance_average.pth"))
        print("saving " + "best cell pos abundance average " + str(test_cell_abundance_pos_pearson_average))
    if test_cell_abundance_all_pearson_average > best_cell_abundance_all_average:
        best_cell_abundance_all_average = test_cell_abundance_all_pearson_average
        torch.save(model.state_dict(), os.path.join(os.path.split(os.path.realpath(__file__))[0], "model_weights", task_name + "_best_cell_all_abundance_average.pth"))
        print("saving " + "best cell all abundance average " + str(test_cell_abundance_all_pearson_average))
        best_cell_pearson_list = test_cell_pearson_list
    if test_cell_abundance_pos_pearson_count > best_cell_abundance_pos_pearson_count:
        best_cell_abundance_pos_pearson_count = test_cell_abundance_pos_pearson_count
        torch.save(model.state_dict(), os.path.join(os.path.split(os.path.realpath(__file__))[0], "model_weights", task_name + "_best_cell_abundance_pos_count.pth"))
        print("saving " + "best cell abundance pos count " + str(test_cell_abundance_pos_pearson_count))
        
    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    # print(f'Epoch: {(epoch + 1)} \tTraining Gene Loss: {train_gene_loss:.6f}')
    print(f'Epoch: {(epoch + 1)} \tTraining Cell abundance Loss: {train_cell_abundance_loss:.6f}')
    # print(f'Epoch: {(epoch + 1)} \tTraining Gene pearson positive count: {train_gene_pos_pearson_count:.6f}')
    print(f'Epoch: {(epoch + 1)} \tTraining Cell abundance pearson positive count: '
          f'{train_cell_abundance_pos_pearson_count:.6f}')
    # print(f'Epoch: {(epoch + 1)} \t'
    #       f'Training Gene pearson positive average: {train_gene_pos_pearson_average:.6f}')
    print(f'Epoch: {(epoch + 1)} \t'
          f'Training Cell abundance pearson positive average: {train_cell_abundance_pos_pearson_average:.6f}')
    # print(f'Epoch: {(epoch + 1)} \t'
    #       f'Training Gene pearson all average: {train_gene_all_pearson_average:.6f}')
    print(f'Epoch: {(epoch + 1)} \t'
          f'Training Cell abundance pearson all average: {train_cell_abundance_all_pearson_average:.6f}')

    # print(f'Epoch: {(epoch + 1)} \tTest Gene Loss: {test_gene_loss:.6f}')
    print(f'Epoch: {(epoch + 1)} \tTest Cell abundance Loss: {test_cell_abundance_loss:.6f}')
    # print(f'Epoch: {(epoch + 1)} \tTest Gene pearson positive count: {test_gene_pos_pearson_count:.6f}')
    print(f'Epoch: {(epoch + 1)} \tTest Cell abundance pearson positive count: '
          f'{test_cell_abundance_pos_pearson_count:.6f}')
    # print(f'Epoch: {(epoch + 1)} \tTest Gene pearson positive average: {test_gene_pos_pearson_average:.6f}')
    print(f'Epoch: {(epoch + 1)} \tTest Cell abundance pearson positive average: '
          f'{test_cell_abundance_pos_pearson_average:.6f}')
    # print(f'Epoch: {(epoch + 1)} \tTest Gene pearson all average: {test_gene_all_pearson_average:.6f}')
    print(f'Epoch: {(epoch + 1)} \tTest Cell abundance pearson all average: '
          f'{test_cell_abundance_all_pearson_average:.6f}')


    if args.log:
        # writer.add_scalar('loss/train_gene_loss',
        #                   train_gene_loss, global_step=epoch)
        writer.add_scalar('loss/train_cell_abundance_loss',
                          train_cell_abundance_loss, global_step=epoch)
        # writer.add_scalar('loss/test_gene_loss',
        #                   test_gene_loss, global_step=epoch)
        writer.add_scalar('loss/test_cell_abundance_loss',
                          test_cell_abundance_loss, global_step=epoch)
        # writer.add_scalar('pearson positive average/train gene pearson positive average',
        #                   train_gene_pos_pearson_average, global_step=epoch)
        # writer.add_scalar('pearson positive average/test gene pearson positive average',
        #                   test_gene_pos_pearson_average, global_step=epoch)
        writer.add_scalar('pearson positive average/train cell abundance pearson positive average',
                          train_cell_abundance_pos_pearson_average, global_step=epoch)
        writer.add_scalar('pearson positive average/test cell abundance pearson positive average',
                          test_cell_abundance_pos_pearson_average, global_step=epoch)
        # writer.add_scalar('pearson positive average/train gene pearson all average',
        #                   train_gene_all_pearson_average, global_step=epoch)
        # writer.add_scalar('pearson positive average/test gene pearson all average',
        #                   test_gene_all_pearson_average, global_step=epoch)
        writer.add_scalar('pearson positive average/train cell abundance pearson all average',
                          train_cell_abundance_all_pearson_average, global_step=epoch)
        writer.add_scalar('pearson positive average/test cell abundance pearson all average',
                          test_cell_abundance_all_pearson_average, global_step=epoch)
        # writer.add_scalar('pearson positive count/train gene pearson positive count',
        #                   train_gene_pos_pearson_count, global_step=epoch)
        # writer.add_scalar('pearson positive count/test gene pearson positive count',
        #                   test_gene_pos_pearson_count, global_step=epoch)
        writer.add_scalar('pearson positive count/train cell abundance pearson positive count',
                          train_cell_abundance_pos_pearson_count, global_step=epoch)
        writer.add_scalar('pearson positive count/test cell abundance pearson positive count',
                          test_cell_abundance_pos_pearson_count, global_step=epoch)


if args.log:
    with open(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'pearson_log', task_name+'_cell_pearson_list.pkl'), 'wb') as f:
        pickle.dump(best_cell_pearson_list, f)

    # with open(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'pearson_log', task_name+'_gene_pearson_list.pkl'), 'wb') as f:
    #     pickle.dump(best_gene_pearson_list, f)    


