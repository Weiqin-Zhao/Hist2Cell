import sys
import os
import pandas as pd
from option import Options
import collections
import csv
from glob import glob
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from PIL import Image, ImageFile
import torchvision.models as models
from scipy.stats import pearsonr
import pickle
from utils import setup_seed, Logger, fully_connected
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PTC_cell(torch.utils.data.Dataset):
    def __init__(self, root, slide='11_breast_cancer3',transform=None, stain_norm=False):
        super(PTC_cell, self).__init__()
        self.root = root
        self.slide = slide
        self.transform = transform
        self.stain_norm = stain_norm

        patch_path = os.path.join(root, slide, 'patches')
        patch = os.listdir(patch_path)
        patch_list = [x.split('.')[0] for x in patch]

        cell_label = pd.read_csv(os.path.join(root, slide, 'cell_ratio.csv'), index_col=0)
        gene_label = pd.read_csv(os.path.join(root, slide, 'high_250_stdata.csv'), index_col=0)
        label_df = pd.merge(gene_label, cell_label, left_index=True, right_index=True)

        label_index_set = set(label_df.index)
        patch_index_set = set(patch_list)
        and_set = label_index_set & patch_index_set

        patch_list = list(and_set)
        self.label_df = label_df.loc[patch_list]
        self.patch = patch_list


    def __getitem__(self, index):
        patch_id = self.patch[index]
        patch_path = os.path.join(self.root, self.slide, 'patches', patch_id)
        patch = Image.open(patch_path+'.jpg').convert('RGB')
        data = transforms.Resize((224, 224))(patch)
        if self.transform is not None:
            data = self.transform(data)

        label = self.label_df.loc[patch_id].values
        label = torch.Tensor(label)

        return patch_id, data, label

    def __len__(self):
        return len(self.patch)


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

#get the train and test dataset ready
train_transform_pcam = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation((90, 90))]),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        #transforms.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5, resample=PIL.Image.BILINEAR, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
test_transform_pcam = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_slides = open("/data1/r20user3/shared_project/Hist2Cell/code/training/train_test_splits/cross_source/train.txt").read().split('\n')
test_slides = open("/data1/r20user3/shared_project/Hist2Cell/code/training/train_test_splits/cross_source/test.txt").read().split('\n')
train_data_list = list()
for slide in train_slides:
    train_data = PTC_cell(root="/data1/r20user3/shared_project/Hist2Cell/data/her2st", slide=slide, transform=train_transform_pcam)
    train_data_list.append(train_data)
train_data = torch.utils.data.ConcatDataset(train_data_list)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

test_data_list = list()
for slide in test_slides:
    test_data = PTC_cell(root="/data1/r20user3/shared_project/Hist2Cell/data/stnet", slide=slide,transform=test_transform_pcam)
    test_data_list.append(test_data)
test_data = torch.utils.data.ConcatDataset(test_data_list)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)


gpu_list = args.gpu_list
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=True)
model.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
num_ftrs = model.classifier.in_features
model_final = fully_connected(model.features, num_ftrs, args.celltype_num)

# load weights from KimiaNet
if args.resume_from_Kimia:
    KimiaNetPyTorchWeights_path = "KimiaNetPyTorchWeights.pth"

    checkpoint = torch.load(KimiaNetPyTorchWeights_path)
    new_state_dict = collections.OrderedDict()

    for k, v in checkpoint.items():
        if 'fc_4' in k:
            continue
        name = k[7:]  # remove "module."
        new_state_dict[name] = v

    model2_dict = model_final.state_dict()
    state_dict = {k:v for k,v in new_state_dict.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    model_final.load_state_dict(model2_dict)

model = model_final
model = torch.nn.DataParallel(model).to(device)
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

    for name, data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        label = label.float()
        label = label.squeeze()
        gene_label = label[:, :250]
        cell_abundance_label = label[:, 250:]
        train_label_array.append(label.cpu().detach().numpy())

        _, output = model(data)

        cell_abundance_output = output

        optimizer.zero_grad()
        cell_abundance_loss = criterion(cell_abundance_output, cell_abundance_label)

        loss = cell_abundance_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        train_cell_abundance_loss += cell_abundance_loss.item() * data.size(0)

        train_sample_num = train_sample_num + data.size(0)
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
        for name, data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            label = label.float()
            label = label.squeeze()
            gene_label = label[:, :250]
            cell_abundance_label = label[:, 250:]
            test_label_array.append(label.cpu().detach().numpy())

            _, output = model(data)

            cell_abundance_output = output

            cell_abundance_loss = criterion(cell_abundance_output, cell_abundance_label)
            loss = cell_abundance_loss

            test_loss += loss.item() * data.size(0)
            test_cell_abundance_loss += cell_abundance_loss.item() * data.size(0)

            test_sample_num = test_sample_num + data.size(0)
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