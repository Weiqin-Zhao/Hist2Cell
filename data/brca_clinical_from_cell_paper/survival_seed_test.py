import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sksurv.metrics import concordance_index_censored
from torch import nn
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


y_df = pd.read_csv('/data1/r20user3/shared_project/Hist2Cell/data/brca_clinical_from_cell_paper/brca_survival_label_merge.csv')
y_df = y_df[['Case.ID', 'Slide.ID', 'survival_days', 'censor', 'survival_interval']]

X_df = pd.read_csv('/data1/r20user3/shared_project/Hist2Cell/data/brca_clinical_from_cell_paper/bulk_seq_largest250_df.csv', index_col=0)

scaler = MinMaxScaler()
X_df = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns, index=X_df.index)

y_df = y_df.set_index('Slide.ID')

dataset_df = X_df.merge(y_df[['survival_days', 'censor', 'survival_interval']], left_index=True, right_index=True, how='outer')
dataset_df = dataset_df.dropna()

x = dataset_df
y = dataset_df['survival_interval']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=2000)


class TCGA_Gene_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_df):
        super(TCGA_Gene_Dataset, self).__init__()
        self.dataset_df = dataset_df

    def __getitem__(self, index):
        sample = self.dataset_df.iloc[index]
        case_id = self.dataset_df.index[0]
        gene_data = torch.Tensor(sample.values[0:-3])
        censor = torch.Tensor([sample['censor']])
        interval = torch.Tensor([sample['survival_interval']])
        event_time = torch.Tensor([sample['survival_days']])

        return case_id, gene_data, censor, interval, event_time
    
    def __len__(self):
        return len(self.dataset_df)
    

# you can fine tune the batch_size and num_workers

train_dataset = TCGA_Gene_Dataset(dataset_df=x_train)
test_dataset = TCGA_Gene_Dataset(dataset_df=x_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss

class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)

# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)
class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)
    # h_padded = torch.cat([torch.zeros_like(c), hazards], 1)
    #reg = - (1 - c) * (torch.log(torch.gather(hazards, 1, Y)) + torch.gather(torch.cumsum(torch.log(1-h_padded), dim=1), 1, Y))


class CoxSurvLoss(object):
    def __call__(hazards, S, c, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).cuda()
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        return loss_cox
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
    
setup_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP(input_dim=250, hidden_dim=512, output_dim=4)
model = model.to(device)

# you can fine tune the learning rate and weight_decay, 
# And use different optimizer, scheduler and loss function defined above
lr = 5e-5
# lr = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
loss_fn = CrossEntropySurvLoss()


best_test_c_index = 0.0
epoch = 200
for i in range(epoch):
    model.train()
    all_risk_scores_for_train = []
    all_censorships_for_train = []
    all_event_times_for_train = []
    total_loss_for_train = 0.0
    for case_id, gene_data, censor, interval, event_time in train_loader:
        gene_data = gene_data.to(device)
        censor = censor.to(device).to(torch.long)
        interval = interval.to(device).to(torch.long)
        event_time = event_time.to(device)
        
        logits = model(gene_data)
        
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        loss = loss_fn(hazards=hazards, S=S, Y=interval, c=censor, alpha=0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # record the labels and predictions for later evaluation metric calculation
        total_loss_for_train += loss
        risk = -torch.sum(S, dim=1).cpu().detach().numpy()
        all_risk_scores_for_train.append(risk)
        all_censorships_for_train.append(censor.cpu().detach().numpy())
        all_event_times_for_train.append(event_time.cpu().detach().numpy())
    
    # flatten the recorded labels and predictions
    all_risk_scores_for_train = np.concatenate([arr.ravel() for arr in all_risk_scores_for_train])
    all_censorships_for_train = np.concatenate([arr.ravel() for arr in all_censorships_for_train])
    all_event_times_for_train = np.concatenate([arr.ravel() for arr in all_event_times_for_train])
    c_index_for_train = concordance_index_censored((1 - all_censorships_for_train).astype(bool), all_event_times_for_train, all_risk_scores_for_train, tied_tol=1e-08)[0]
    
    print("epoch：{:2d}:\t train_loss：{:.4f}\t train_c_index：{:.4f}".format(i, total_loss_for_train / len(train_loader), c_index_for_train))
    
    
    with torch.no_grad():
        model.train()
        all_risk_scores_for_test = []
        all_censorships_for_test = []
        all_event_times_for_test = []
        total_loss_for_test = 0.0
        for case_id, gene_data, censor, interval, event_time in test_loader:
            gene_data = gene_data.to(device)
            censor = censor.to(device).to(torch.long)
            interval = interval.to(device).to(torch.long)
            event_time = event_time.to(device)
            
            logits = model(gene_data)
            
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            loss = loss_fn(hazards=hazards, S=S, Y=interval, c=censor, alpha=0)
            
            total_loss_for_test += loss
            risk = -torch.sum(S, dim=1).cpu().detach().numpy()
            all_risk_scores_for_test.append(risk)
            all_censorships_for_test.append(censor.cpu().detach().numpy())
            all_event_times_for_test.append(event_time.cpu().detach().numpy())
        
        
        all_risk_scores_for_test = np.concatenate([arr.ravel() for arr in all_risk_scores_for_test])
        all_censorships_for_test = np.concatenate([arr.ravel() for arr in all_censorships_for_test])
        all_event_times_for_test = np.concatenate([arr.ravel() for arr in all_event_times_for_test])
        c_index_for_test = concordance_index_censored((1 - all_censorships_for_test).astype(bool), all_event_times_for_test, all_risk_scores_for_test, tied_tol=1e-08)[0]
        
        print("epoch：{:2d}:\t test_loss：{:.4f}\t test_c_index：{:.4f}".format(i, total_loss_for_test / len(test_loader), c_index_for_test))
        
        if c_index_for_test > best_test_c_index:
            best_test_c_index = c_index_for_test
            
print("best_test_c_index：{:.4f}".format(best_test_c_index))