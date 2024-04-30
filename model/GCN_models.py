import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn=0,add_self=0, normalize_embedding=0,
            dropout=0.0,relu=0, bias=True):
        super(GCNBlock,self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        self.relu=relu
        self.bn=bn
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        if self.bn:
            self.bn_layer = torch.nn.BatchNorm1d(output_dim)

        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
            # self.bias = nn.Parameter(torch.zeros(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj, mask):
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        if self.bn:
            index=mask.sum(dim=1).long().tolist()
            bn_tensor_bf=mask.new_zeros((sum(index),y.shape[2]))
            bn_tensor_af=mask.new_zeros(*y.shape)
            start_index=[]
            ssum=0
            for i in range(x.shape[0]):
                start_index.append(ssum)
                ssum+=index[i]
            start_index.append(ssum)
            for i in range(x.shape[0]):
                bn_tensor_bf[start_index[i]:start_index[i+1]]=y[i,0:index[i]]
            bn_tensor_bf=self.bn_layer(bn_tensor_bf)
            for i in range(x.shape[0]):
                bn_tensor_af[i,0:index[i]]=bn_tensor_bf[start_index[i]:start_index[i+1]]
            y=bn_tensor_af
        if self.dropout > 0.001:
            y = self.dropout_layer(y)
        if self.relu=='relu':
            y=torch.nn.functional.relu(y)
            print('hahah')
        elif self.relu=='lrelu':
            y=torch.nn.functional.leaky_relu(y,0.1)
        return y


class GCN(nn.Module):
    def __init__(self, input_dim=1024, output_dim=250):
        super(GCN, self).__init__()

        self.embed_dim = 512
        self.num_layers = 3
        self.node_cluster_num = 100
        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(input_dim, self.embed_dim ,self.bn ,self.add_self ,self.normalize_embedding, 0., 0)
        self.pool1 = Linear(self.embed_dim, 512)
        self.conv2 = GCNBlock(512, output_dim, self.bn ,self.add_self ,self.normalize_embedding, 0., 0)
        self.pool2 = Linear(output_dim, output_dim)


    def forward(self ,node_feat ,adj ,mask):
        X = node_feat
        X = mask.unsqueeze(2)*X
        X = self.conv1(X, adj, mask)
        X = self.pool1(X)
        X = self.conv2(X, adj, mask)
        X = self.pool2(X)
        X = torch.relu(X)

        return X


class GCN_multi_task(nn.Module):
    def __init__(self, input_dim=1024, output_dim=250):
        super(GCN_multi_task, self).__init__()

        self.embed_dim = 256
        self.num_layers = 3
        self.node_cluster_num = 100
        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1

        self.down_dim = Linear(input_dim, self.embed_dim)

        self.conv1 = GCNBlock(self.embed_dim, self.embed_dim ,self.bn ,self.add_self ,self.normalize_embedding, 0., 0)
        self.pool1 = Linear(self.embed_dim, self.embed_dim)

        self.conv11 = GCNBlock(self.embed_dim, self.embed_dim ,self.bn ,self.add_self ,self.normalize_embedding, 0., 0)

        self.conv2 = GCNBlock(self.embed_dim, output_dim, self.bn,self.add_self ,self.normalize_embedding, 0., 0)
        self.pool2 = Linear(output_dim, output_dim)

        self.conv22 = GCNBlock(self.embed_dim, output_dim, self.bn,self.add_self ,self.normalize_embedding, 0., 0)

    def forward(self, node_feat, adj, mask):
        X = node_feat
        X = mask.unsqueeze(2)*X
        X = self.down_dim(X)

        sim_matrix1 = torch.cosine_similarity(X.detach().squeeze(0).unsqueeze(1), X.detach().squeeze(0).unsqueeze(0), dim=-1)
        sim_matrix1[sim_matrix1 > 0.88] = 1.0
        sim_matrix1[sim_matrix1 <= 0.88] = 0.0
        sim_matrix1 = sim_matrix1.unsqueeze(0)
        # a = sim_matrix1.sum()

        X1 = self.conv1(X, adj, mask)
        X2 = self.conv11(X, sim_matrix1, mask)
        X = X1 + X2
        X = self.pool1(X)

        sim_matrix2 = torch.cosine_similarity(X.detach().squeeze(0).unsqueeze(1), X.detach().squeeze(0).unsqueeze(0), dim=-1)
        sim_matrix2[sim_matrix2 > 0.88] = 1.0
        sim_matrix2[sim_matrix2 <= 0.88] = 0.0
        sim_matrix2 = sim_matrix2.unsqueeze(0)
        # a = sim_matrix1.sum()

        X1 = self.conv2(X, adj, mask)
        X2 = self.conv22(X, sim_matrix2, mask)
        X = X1 + X2
        X = self.pool2(X)
        X = torch.relu(X)

        return X


class GCN_KG(nn.Module):
    def __init__(self, input_dim=1024, KG=None):
        super(GCN_KG, self).__init__()

        self.embed_dim = 1024
        self.num_layers = 3
        self.node_cluster_num = 100
        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(input_dim, self.embed_dim, self.bn, self.add_self, self.normalize_embedding, 0., 0)
        self.pool1 = Linear(self.embed_dim, 1024)
        self.conv2 = GCNBlock(1024, 1024, self.bn, self.add_self, self.normalize_embedding, 0., 0)
        self.pool2 = Linear(1024, 1024)

        self.KG = KG


    def forward(self ,node_feat ,adj ,mask):
        X = node_feat
        X = mask.unsqueeze(2)*X
        X = self.conv1(X, adj, mask)
        X = self.pool1(X)
        X = self.conv2(X, adj, mask)
        X = self.pool2(X)

        X = self.KG(X)

        X = torch.relu(X)

        return X


class KG(nn.Module):
    def __init__(self, kg_adj, kg_fuse="concate", trainable_adj=False):
        super(KG, self).__init__()

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.kg_fuse = kg_fuse

        if self.kg_fuse == "concate":
            self.conv1 = GCNBlock(256, 256, self.bn, self.add_self, self.normalize_embedding, 0., 0)
            self.conv2 = GCNBlock(256, 256, self.bn, self.add_self, self.normalize_embedding, 0., 0)
            self.pool1 = Linear(256, 256)
            self.pool2 = Linear(256, 256)
            self.gene_kg = torch.nn.Parameter(torch.zeros(1, 250, 256))
            nn.init.kaiming_uniform_(self.gene_kg, mode='fan_in', nonlinearity='relu')
            self.kg_adj = torch.nn.Parameter(kg_adj.unsqueeze(0), requires_grad=trainable_adj)
            self.kg_mask = torch.nn.Parameter(torch.ones((kg_adj.shape[0])).unsqueeze(0), requires_grad=False)

            self.fc = torch.nn.Parameter(torch.zeros(1024, 250))
            nn.init.kaiming_uniform_(self.fc, mode='fan_in', nonlinearity='relu')
            self.bias = torch.nn.Parameter(torch.zeros(1, 250))
            nn.init.kaiming_uniform_(self.bias, mode='fan_in', nonlinearity='relu')

            self.fc2 = Linear(1024+256, 1)

        elif self.kg_fuse == "dot_product":
            self.conv1 = GCNBlock(256, 512, self.bn, self.add_self, self.normalize_embedding, 0., 0)
            self.conv2 = GCNBlock(512, 1024, self.bn, self.add_self, self.normalize_embedding, 0., 0)
            self.pool1 = Linear(512, 512)
            self.pool2 = Linear(1024, 1024)
            self.gene_kg = torch.nn.Parameter(torch.zeros(1, 250, 256))
            nn.init.kaiming_uniform_(self.gene_kg, mode='fan_in', nonlinearity='relu')
            self.kg_adj = torch.nn.Parameter(kg_adj.unsqueeze(0), requires_grad=trainable_adj)
            self.kg_mask = torch.nn.Parameter(torch.ones((kg_adj.shape[0])).unsqueeze(0), requires_grad=False)

            self.fc = torch.nn.Parameter(torch.zeros(1024, 250))
            nn.init.kaiming_uniform_(self.fc, mode='fan_in', nonlinearity='relu')
            self.bias = torch.nn.Parameter(torch.zeros(1, 250))
            nn.init.kaiming_uniform_(self.bias, mode='fan_in', nonlinearity='relu')

            self.fc2 = Linear(1024, 1)


    def forward(self, feature):
        kg = self.conv1(self.gene_kg, self.kg_adj, self.kg_mask)
        kg = self.pool1(kg)
        kg = self.conv2(kg, self.kg_adj, self.kg_mask)
        kg = self.pool2(kg)

        if self.kg_fuse == "concate":
            feature = feature.repeat(250, 1, 1)
            feature = feature.transpose(0, 1)
            kg = kg.repeat(feature.shape[0], 1, 1)
            feature = torch.cat([feature, kg], dim=2)
            output = self.fc2(feature)
            output = output.squeeze(2)

        elif self.kg_fuse == "dot_product":
            kg = kg.squeeze(0)
            kg = kg.transpose(0, 1)
            output = torch.matmul(feature, kg*self.fc)
            output = output + self.bias

        output = output.unsqueeze(0)
        return output


class KG(nn.Module):
    def __init__(self, kg_adj, kg_fuse="concate", trainable_adj=False):
        super(KG, self).__init__()

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.kg_fuse = kg_fuse

        if self.kg_fuse == "concate":
            self.conv1 = GCNBlock(256, 256, self.bn, self.add_self, self.normalize_embedding, 0., 0)
            self.conv2 = GCNBlock(256, 256, self.bn, self.add_self, self.normalize_embedding, 0., 0)
            self.pool1 = Linear(256, 256)
            self.pool2 = Linear(256, 256)
            self.gene_kg = torch.nn.Parameter(torch.zeros(1, 250, 256))
            nn.init.kaiming_uniform_(self.gene_kg, mode='fan_in', nonlinearity='relu')
            self.kg_adj = torch.nn.Parameter(kg_adj.unsqueeze(0), requires_grad=trainable_adj)
            self.kg_mask = torch.nn.Parameter(torch.ones((kg_adj.shape[0])).unsqueeze(0), requires_grad=False)

            self.fc = torch.nn.Parameter(torch.zeros(1024, 250))
            nn.init.kaiming_uniform_(self.fc, mode='fan_in', nonlinearity='relu')
            self.bias = torch.nn.Parameter(torch.zeros(1, 250))
            nn.init.kaiming_uniform_(self.bias, mode='fan_in', nonlinearity='relu')

            self.fc2 = Linear(1024+256, 1)

        elif self.kg_fuse == "dot_product":
            self.conv1 = GCNBlock(256, 512, self.bn, self.add_self, self.normalize_embedding, 0., 0)
            self.conv2 = GCNBlock(512, 1024, self.bn, self.add_self, self.normalize_embedding, 0., 0)
            self.pool1 = Linear(512, 512)
            self.pool2 = Linear(1024, 1024)
            self.gene_kg = torch.nn.Parameter(torch.zeros(1, 250, 256))
            nn.init.kaiming_uniform_(self.gene_kg, mode='fan_in', nonlinearity='relu')
            self.kg_adj = torch.nn.Parameter(kg_adj.unsqueeze(0), requires_grad=trainable_adj)
            self.kg_mask = torch.nn.Parameter(torch.ones((kg_adj.shape[0])).unsqueeze(0), requires_grad=False)

            self.fc = torch.nn.Parameter(torch.zeros(1024, 250))
            nn.init.kaiming_uniform_(self.fc, mode='fan_in', nonlinearity='relu')
            self.bias = torch.nn.Parameter(torch.zeros(1, 250))
            nn.init.kaiming_uniform_(self.bias, mode='fan_in', nonlinearity='relu')

            self.fc2 = Linear(1024, 1)


    def forward(self, feature):
        kg = self.conv1(self.gene_kg, self.kg_adj, self.kg_mask)
        kg = self.pool1(kg)
        kg = self.conv2(kg, self.kg_adj, self.kg_mask)
        kg = self.pool2(kg)

        if self.kg_fuse == "concate":
            feature = feature.repeat(250, 1, 1)
            feature = feature.transpose(0, 1)
            kg = kg.repeat(feature.shape[0], 1, 1)
            feature = torch.cat([feature, kg], dim=2)
            output = self.fc2(feature)
            output = output.squeeze(2)

        elif self.kg_fuse == "dot_product":
            kg = kg.squeeze(0)
            kg = kg.transpose(0, 1)
            output = torch.matmul(feature, kg*self.fc)
            output = output + self.bias

        output = output.unsqueeze(0)
        return output


class Dense_KG(nn.Module):
    def __init__(self, DenseNet, kg_adj, kg_fuse="concate", trainable_adj=False):
        super(Dense_KG, self).__init__()
        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.densenet = DenseNet
        self.kg_fuse = kg_fuse

        if self.kg_fuse == "concate":
            self.conv1 = GCNBlock(256, 256, self.bn, self.add_self, self.normalize_embedding, 0., 0)
            self.conv2 = GCNBlock(256, 256, self.bn, self.add_self, self.normalize_embedding, 0., 0)
            self.pool1 = Linear(256, 256)
            self.pool2 = Linear(256, 256)
            self.gene_kg = torch.nn.Parameter(torch.zeros(1, 250, 256))
            nn.init.kaiming_uniform_(self.gene_kg, mode='fan_in', nonlinearity='relu')
            self.kg_adj = torch.nn.Parameter(kg_adj.unsqueeze(0), requires_grad=trainable_adj)
            self.kg_mask = torch.nn.Parameter(torch.ones((kg_adj.shape[0])).unsqueeze(0), requires_grad=False)

            self.fc = torch.nn.Parameter(torch.zeros(1024, 250))
            nn.init.kaiming_uniform_(self.fc, mode='fan_in', nonlinearity='relu')
            self.bias = torch.nn.Parameter(torch.zeros(1, 250))
            nn.init.kaiming_uniform_(self.bias, mode='fan_in', nonlinearity='relu')

            self.fc2 = Linear(1024+256, 1)

        elif self.kg_fuse == "dot_product":
            self.conv1 = GCNBlock(256, 512, self.bn, self.add_self, self.normalize_embedding, 0., 0)
            self.conv2 = GCNBlock(512, 1024, self.bn, self.add_self, self.normalize_embedding, 0., 0)
            self.pool1 = Linear(512, 512)
            self.pool2 = Linear(1024, 1024)
            self.gene_kg = torch.nn.Parameter(torch.zeros(1, 250, 256))
            nn.init.kaiming_uniform_(self.gene_kg, mode='fan_in', nonlinearity='relu')
            self.kg_adj = torch.nn.Parameter(kg_adj.unsqueeze(0), requires_grad=trainable_adj)
            self.kg_mask = torch.nn.Parameter(torch.ones((kg_adj.shape[0])).unsqueeze(0), requires_grad=False)

            self.fc = torch.nn.Parameter(torch.zeros(1024, 250))
            nn.init.kaiming_uniform_(self.fc, mode='fan_in', nonlinearity='relu')
            self.bias = torch.nn.Parameter(torch.zeros(1, 250))
            nn.init.kaiming_uniform_(self.bias, mode='fan_in', nonlinearity='relu')

            self.fc2 = Linear(1024, 1)


    def forward(self, X):
        kg = self.conv1(self.gene_kg, self.kg_adj, self.kg_mask)
        kg = self.pool1(kg)
        kg = self.conv2(kg, self.kg_adj, self.kg_mask)
        kg = self.pool2(kg)

        feature, _ = self.densenet(X)

        if self.kg_fuse == "concate":
            feature = feature.repeat(250, 1, 1)
            feature = feature.transpose(0, 1)
            kg = kg.repeat(feature.shape[0], 1, 1)
            feature = torch.cat([feature, kg], dim=2)
            output = self.fc2(feature)
            output = output.squeeze(2)

        elif self.kg_fuse == "dot_product":
            kg = kg.squeeze(0)
            kg = kg.transpose(0, 1)
            output = torch.matmul(feature, kg*self.fc)
            output = output + self.bias

        return output


class GCN_AutoEncoder(nn.Module):
    def __init__(self, input_dim=1024):
        super(GCN_AutoEncoder, self).__init__()
        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(input_dim, 256, self.bn, self.add_self, self.normalize_embedding, 0., 0)
        self.conv2 = GCNBlock(256, 64, self.bn, self.add_self, self.normalize_embedding, 0., 0)
        self.conv3 = GCNBlock(64, 256, self.bn, self.add_self, self.normalize_embedding, 0., 0)
        self.conv4 = GCNBlock(256, input_dim, self.bn, self.add_self, self.normalize_embedding, 0., 0)


    def forward(self ,node_feat ,adj ,mask):
        X = node_feat
        X = mask.unsqueeze(2)*X
        X = self.conv1(X, adj, mask)
        X = self.conv2(X, adj, mask)
        X = self.conv3(X, adj, mask)
        X = self.conv4(X, adj, mask)
        return X