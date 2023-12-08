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
from model.GCN_models import GCN, GCN_multi_task
from option import Options
from utils import setup_seed, Logger

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

data_path = args.graph_data_path
graphs = joblib.load(data_path)

train_slides = open(args.train_set).read().split('\n')
test_slides = open(args.test_set).read().split('\n')

train_loader = dict()
for item in train_slides:
    train_loader[item] = graphs[item]

test_loader = dict()
for item in test_slides:
    test_loader[item] = graphs[item]

# model = GCN_multi_task(input_dim=1024, output_dim=250+80)
model = GCN(input_dim=1024, output_dim=args.celltype_num)
model = model.to(device)
params = model.parameters()

criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5, last_epoch=-1, verbose=False)

best_cell_abundance_loss = 100.0
best_cell_abundance_pos_average = 0.0
best_cell_abundance_all_average = 0.0
best_cell_abundance_pos_pearson_count = 0.0
best_cell_pearson_list = []

since = time.time()
for epoch in range(num_epochs):
    model.train()
    print("---------------------------------------"*4)
    train_loss, test_loss = 0, 0
    print('Epoch: {} \t'.format(epoch + 1))
    print('lr = ',optimizer.param_groups[0]["lr"])

    train_prediction_array = []
    train_label_array = []
    train_sample_num = 0
    test_sample_num = 0

    train_cell_abundance_loss = 0
    test_cell_abundance_loss = 0

    for item in train_loader:
        feature = torch.from_numpy(graphs[item]['features']).to(device).unsqueeze(0)
        label = torch.from_numpy(graphs[item]['labels']).to(device).unsqueeze(0)
        adj = torch.from_numpy(graphs[item]['adj']).to(torch.float32).to(device).unsqueeze(0)
        mask = torch.ones((feature.shape[1])).to(device).unsqueeze(0)

        gene_label = label[:, :, :250]
        cell_abundance_label = label[:, :, 250:]

        train_label_array.append(label.squeeze().cpu().detach().numpy())
        output = model(node_feat=feature, adj=adj, mask=mask)

        cell_abundance_output = output

        optimizer.zero_grad()
        cell_abundance_loss = criterion(cell_abundance_output, cell_abundance_label)

        loss = cell_abundance_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * feature.size(1)
        train_cell_abundance_loss += cell_abundance_loss.item() * feature.size(1)

        train_sample_num = train_sample_num + feature.size(1)
        train_prediction_array.append(output.squeeze().cpu().detach().numpy())

    scheduler.step()

    train_loss = train_loss / train_sample_num
    train_cell_abundance_loss = train_cell_abundance_loss / train_sample_num

    train_prediction_array = np.concatenate(train_prediction_array)
    train_label_array = np.concatenate(train_label_array)

    train_cell_abundance_pos_pearson_average = 0.0
    train_cell_abundance_all_pearson_average = 0.0
    train_cell_abundance_pos_pearson_count = 0
    train_cell_abundance_prediction_array = train_prediction_array
    train_cell_abundance_label_array = train_label_array[:, 250:]
    for i in range(args.celltype_num):
        r, p = pearsonr(train_cell_abundance_prediction_array[:, i], train_cell_abundance_label_array[:, i])
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

        test_prediction_array = []
        test_label_array = []
        for item in test_loader:
            feature = torch.from_numpy(graphs[item]['features']).to(device).unsqueeze(0)
            label = torch.from_numpy(graphs[item]['labels']).to(device).unsqueeze(0)
            adj = torch.from_numpy(graphs[item]['adj']).to(torch.float32).to(device).unsqueeze(0)
            mask = torch.ones((feature.shape[1])).to(device).unsqueeze(0)

            gene_label = label[:, :, :250]
            cell_abundance_label = label[:, :, 250:]

            test_label_array.append(label.squeeze().cpu().detach().numpy())
            output = model(node_feat=feature, adj=adj, mask=mask)

            cell_abundance_output = output

            cell_abundance_loss = criterion(cell_abundance_output, cell_abundance_label)
            loss = cell_abundance_loss

            test_loss += loss.item() * feature.size(1)
            test_cell_abundance_loss += cell_abundance_loss.item() * feature.size(1)

            test_sample_num = test_sample_num + feature.size(1)
            test_prediction_array.append(output.squeeze().cpu().detach().numpy())

        test_cell_abundance_loss = test_cell_abundance_loss / test_sample_num
        if test_cell_abundance_loss < best_cell_abundance_loss:
            best_cell_abundance_loss = test_cell_abundance_loss
            torch.save(model.state_dict(), os.path.join(os.path.split(os.path.realpath(__file__))[0], "model_weights", task_name + "_best_cell_loss.pth"))
            print("saving best cell loss " + str(test_cell_abundance_loss))

    test_prediction_array = np.concatenate(test_prediction_array)
    test_label_array = np.concatenate(test_label_array)

    test_cell_pearson_list = []
 
    test_cell_abundance_pos_pearson_count = 0
    test_cell_abundance_pos_pearson_average = 0.0
    test_cell_abundance_all_pearson_average = 0.0
    test_cell_abundance_prediction_array = test_prediction_array
    test_cell_abundance_label_array = test_label_array[:, 250:]
    for i in range(args.celltype_num):
        r, p = pearsonr(test_cell_abundance_prediction_array[:, i], test_cell_abundance_label_array[:, i])
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
    print(f'Epoch: {(epoch + 1)} \tTraining Cell abundance Loss: {train_cell_abundance_loss:.6f}')
    print(f'Epoch: {(epoch + 1)} \tTraining Cell abundance pearson positive count: '
          f'{train_cell_abundance_pos_pearson_count:.6f}')
    print(f'Epoch: {(epoch + 1)} \t'
          f'Training Cell abundance pearson positive average: {train_cell_abundance_pos_pearson_average:.6f}')
    print(f'Epoch: {(epoch + 1)} \t'
          f'Training Cell abundance pearson all average: {train_cell_abundance_all_pearson_average:.6f}')

    print(f'Epoch: {(epoch + 1)} \tTest Cell abundance Loss: {test_cell_abundance_loss:.6f}')
    print(f'Epoch: {(epoch + 1)} \tTest Cell abundance pearson positive count: '
          f'{test_cell_abundance_pos_pearson_count:.6f}')
    print(f'Epoch: {(epoch + 1)} \tTest Cell abundance pearson positive average: '
          f'{test_cell_abundance_pos_pearson_average:.6f}')
    print(f'Epoch: {(epoch + 1)} \tTest Cell abundance pearson all average: '
          f'{test_cell_abundance_all_pearson_average:.6f}')


    if args.log:
        writer.add_scalar('loss/train_cell_abundance_loss',
                          train_cell_abundance_loss, global_step=epoch)
        writer.add_scalar('loss/test_cell_abundance_loss',
                          test_cell_abundance_loss, global_step=epoch)
        writer.add_scalar('pearson positive average/train cell abundance pearson positive average',
                          train_cell_abundance_pos_pearson_average, global_step=epoch)
        writer.add_scalar('pearson positive average/test cell abundance pearson positive average',
                          test_cell_abundance_pos_pearson_average, global_step=epoch)
        writer.add_scalar('pearson positive average/train cell abundance pearson all average',
                          train_cell_abundance_all_pearson_average, global_step=epoch)
        writer.add_scalar('pearson positive average/test cell abundance pearson all average',
                          test_cell_abundance_all_pearson_average, global_step=epoch)
        writer.add_scalar('pearson positive count/train cell abundance pearson positive count',
                          train_cell_abundance_pos_pearson_count, global_step=epoch)
        writer.add_scalar('pearson positive count/test cell abundance pearson positive count',
                          test_cell_abundance_pos_pearson_count, global_step=epoch)
        
if args.log:
    with open(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'pearson_log', task_name+'_cell_pearson_list.pkl'), 'wb') as f:
        pickle.dump(best_cell_pearson_list, f)
