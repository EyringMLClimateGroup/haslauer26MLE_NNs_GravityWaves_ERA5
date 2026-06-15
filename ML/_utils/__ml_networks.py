import torch
from torch import nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torcheval.metrics import R2Score

import torch.nn.functional as F


device = (
    "cuda"
    if torch.cuda.is_available()
      else "cpu"
)



# DATATSET AND DATALOADER -------------------------------------------------------------------------

class FluxData(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat = self.features[idx]
        targ = self.targets[idx]
    
        return feat, targ
    
def load_data(train_dataset, test_dataset, batch_size = 1024, num_workers=16):
    train_load = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_load = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_load, test_load



# NETWORKS ---------------------------------------------------------------------------------------- 

# Fully connected neural network

class FCNN(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden, n_layers, activation=F.relu, bn=False, dr=0.0):
        super(FCNN, self).__init__()
        self.activation = activation
        self.flatten = nn.Flatten()

        self.fc_in = nn.Linear(in_dim, n_hidden)
        self.fcs = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers-1)])

        if dr > 0.0:
            self.drs = nn.ModuleList([nn.Dropout(p=0.0) for _ in range(n_layers)])
        else:
            self.drs = nn.ModuleList([nn.Identity() for _ in range(n_layers)])

        if bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])
        else:
            self.bns = nn.ModuleList([nn.Identity() for _ in range(n_layers)])

        self.fc_out = nn.Linear(n_hidden, out_dim)

    def forward(self, x):
        x = self.flatten(x)

        x = self.activation(self.fc_in(x))

        for fc, dr, bn in zip(self.fcs, self.drs, self.bns):
            x = self.activation(dr(bn(fc(x))))

        x = self.fc_out(x)

        return x
    

# Convolutional neural network

class ConvNet(nn.Module):
    def __init__(self, vec_f, sca_f, levels, fc_hidden, activation=nn.ReLU):
        super(ConvNet, self).__init__()
        self.levels = levels
        self.activation = activation
        self.kernel_size=11
        self.padding=4

        self.conv1 = nn.Conv1d(in_channels=vec_f, out_channels=16, kernel_size=self.kernel_size, padding=self.padding, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, padding=self.padding, padding_mode='zeros')

        
        self.flattened_conv_output_size1 = 16 * (levels + (2*self.padding)-(self.kernel_size-1))
        self.flattened_conv_output_size2 = 32 * (int((self.flattened_conv_output_size1/16)) + (2*self.padding)-(self.kernel_size-1))


        self.fc1 = nn.Linear(self.flattened_conv_output_size2 + sca_f, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc = nn.Linear(fc_hidden, 2 * levels)

    def forward(self, x_levels, x_scalar):

        x = F.relu(self.conv1(x_levels))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        x = torch.cat([x, x_scalar], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc(x)

        x = x.view(-1, 2 * self.levels)
        
        return x
    


# U-Net with scalar features in bottleneck

class UNet_2(nn.Module):
    def __init__(self, vec_f, sca_f, levels=37, fc_hidden=128, base_filters=4, activation='ReLU', bn=False):
        super().__init__()

        self.vec_f = vec_f
        self.sca_f = sca_f
        self.levels = levels

        if activation == 'ReLU':
            self.activation = nn.ReLU()

        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()

        self.enc01 = self.conv_block(vec_f, base_filters)
        self.enc02 = self.conv_block(base_filters, base_filters * 2)

        self.bottle = self.conv_block(base_filters * 2, base_filters * 4, bn=bn)

        self.scalar_fc = nn.Sequential(
            nn.Linear(sca_f, fc_hidden),
            nn.ReLU(),
            # nn.Linear(fc_hidden, fc_hidden),
            # nn.ReLU(),
            nn.Linear(fc_hidden, base_filters * 4),
            self.activation
        )

        self.up01 = self.up_block(base_filters * 8, base_filters * 2)
        self.up02 = self.up_block(base_filters * 4, base_filters)

        self.final_conv = nn.Conv1d(base_filters * 2, 2, kernel_size=1)

        self.pool = nn.MaxPool1d(2)

    def conv_block(self, in_channels, out_channels, bn=False):
        if bn is True:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                self.activation,
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                self.activation,
                )
        else:
             return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                self.activation,
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                self.activation,
                )


    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            self.activation,
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            self.activation,
        )

    def forward(self, X):

        X_col, X_sca = X[:,:self.vec_f*self.levels].view(-1, self.vec_f, self.levels).to(device), X[:,-self.sca_f:].to(device)

        # Encoder
        e01 = self.enc01(X_col)               # (B, base_filters, 37)
        e02 = self.enc02(self.pool(e01))   # (B, base_filters*2, 18)

        # Bottleneck
        b = self.bottle(self.pool(e02))  # (B, base_filters*4, 9)

        # Scalar processing: map to (B, base_filters*4, 9)
        scalar_proj = self.scalar_fc(X_sca)  # (B, base_filters*4)
        scalar_proj = scalar_proj.unsqueeze(-1).expand(-1, -1, b.shape[2])  # (B, base_filters*4, 9)

        # Combine bottleneck with scalar features
        b = torch.cat([b, scalar_proj], dim=1)  # (B, base_filters*8, 9)

        # Decoder
        d01 = self.up01(b)       # → (B, base_filters*2, 18)
        d01 = torch.cat([d01, e02], dim=1)

        d02 = self.up02(d01)      # → (B, base_filters, 37)
        if d02.shape[2] != e01.shape[2]:
            diff = e01.shape[2] - d02.shape[2]
            d02 = F.pad(d02, (0, diff))

        d02 = torch.cat([d02, e01], dim=1)

        out = self.final_conv(d02)  # → (B, 2, 37)
        return out.view(-1,2*self.levels)
    


# Deeper U-Net with scalar features in bottleneck

class UNet_3(nn.Module):
    def __init__(self, vec_f, sca_f, levels=37, fc_hidden=128, base_filters=4, activation='ReLU', bn=False):
        super().__init__()

        self.vec_f = vec_f
        self.sca_f = sca_f
        self.levels = levels

        if activation == 'ReLU':
            self.activation = nn.ReLU()

        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()

        self.enc01 = self.conv_block(vec_f, base_filters)
        self.enc02 = self.conv_block(base_filters, base_filters * 2)
        self.enc03 = self.conv_block(base_filters*2, base_filters * 4)


        self.bottle = self.conv_block(base_filters * 4, base_filters * 8, bn=bn)

        self.scalar_fc = nn.Sequential(
            nn.Linear(sca_f, fc_hidden),
            nn.ReLU(),
            # nn.Linear(fc_hidden, fc_hidden),
            # nn.ReLU(),
            nn.Linear(fc_hidden, base_filters * 8),
            self.activation
        )

        self.up01 = self.up_block(base_filters * 16, base_filters * 4)
        self.up02 = self.up_block(base_filters * 8, base_filters * 2)
        self.up03 = self.up_block(base_filters * 4, base_filters)
      #  self.up04 = self.up_block(base_filters * 2, base_filters)


        self.final_conv = nn.Conv1d(base_filters * 2, 2, kernel_size=1)

        self.pool = nn.MaxPool1d(2)

    def conv_block(self, in_channels, out_channels, bn=False):
        if bn is True:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                self.activation,
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                self.activation,
                )
        else:
             return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                self.activation,
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                self.activation,
                )


    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            self.activation,
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            self.activation,
        )

    def forward(self, X):

        X_col, X_sca = X[:,:self.vec_f*self.levels].view(-1, self.vec_f, self.levels).to(device), X[:,-self.sca_f:].to(device)

        # Encoder
        e01 = self.enc01(X_col)               # (B, base_filters, 37)
        e02 = self.enc02(self.pool(e01))   # (B, base_filters*2, 18)
        e03 = self.enc03(self.pool(e02))   # (B, base_filters*2, 18)


        # Bottleneck
        b = self.bottle(self.pool(e03))  # (B, base_filters*4, 9)

        # Scalar processing: map to (B, base_filters*4, 9)
        scalar_proj = self.scalar_fc(X_sca)  # (B, base_filters*4)
        scalar_proj = scalar_proj.unsqueeze(-1).expand(-1, -1, b.shape[2])  # (B, base_filters*4, 9)

        # Combine bottleneck with scalar features
        b = torch.cat([b, scalar_proj], dim=1)  # (B, base_filters*8, 9)

        # Decoder
        d01 = self.up01(b)       # → (B, base_filters*2, 18)
        if d01.shape[2] != e03.shape[2]:
            diff = e03.shape[2] - d01.shape[2]
            d01 = F.pad(d01, (0, diff))
        d01 = torch.cat([d01, e03], dim=1)

        d02 = self.up02(d01)      # → (B, base_filters, 37)
        if d02.shape[2] != e02.shape[2]:
            diff = e02.shape[2] - d02.shape[2]
            d02 = F.pad(d02, (0, diff))
        d02 = torch.cat([d02, e02], dim=1)

        d03 = self.up03(d02)
        if d03.shape[2] != e01.shape[2]:
            diff = e01.shape[2] - d03.shape[2]
            d03 = F.pad(d03, (0,diff))
        d03 = torch.cat([d03, e01], dim=1)

        out = self.final_conv(d03)  # → (B, 2, 37)
        return out.view(-1,2*self.levels)
    




# EVALUATION --------------------------------------------------------------------------------------


# Calculation of R2 scores

# Standard evaluation function for training procedure
def eval(model, train_dataset, test_dataset, batch_size=4096, mode='FCNN', levels=0, vector_feat=0, scalar_feat=0, fc_hidden=0):

    model.eval()

    train_load, test_load = load_data(train_dataset, test_dataset, batch_size)

    r2_value_train = R2Score()
    r2_value_test = R2Score()

    with torch.no_grad():
        for features, targets in train_load:
            features = features.to(device)
            targets = targets.to(device)

            pred = model(features)
            
            r2_value_train.update(pred, targets)

        r2_train = r2_value_train.compute()

        for features, targets in test_load:
            features = features.to(device)
            targets = targets.to(device)
           
            pred = model(features)

            r2_value_test.update(pred, targets)

        r2_test = r2_value_test.compute()


    print('\n-----------------------')
    print(f'TRAIN  |   R2: {r2_train:.6f}')
    print(f'TEST   |   R2: {r2_test:.6f}')
    print('-----------------------\n')


# R2 score on one single set
def eval_1_set(model, test_dataset, batch_size=4096, mode='FCNN', levels=0, vector_feat=0, scalar_feat=0, fc_hidden=0):
    model.eval()
    test_load = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=16)

    r2_value_test = R2Score()

    with torch.no_grad():
    
        for features, targets in test_load:
            features = features.to(device)
            targets = targets.to(device)
           
            pred = model(features)

            r2_value_test.update(pred, targets)

        r2_test = r2_value_test.compute()
    
    return r2_test



# R2 score on range of levels
def eval_levs(model, train_dataset, test_dataset, lev_range=None, batch_size=4096, mode='FCNN', levels=0, vector_feat=0, scalar_feat=0, fc_hidden=0, print_results=False):
    model.to(device)
    model.eval()

    train_load, test_load = load_data(train_dataset, test_dataset, batch_size)

    r2_value_train = R2Score()
    r2_value_test = R2Score()

    with torch.no_grad():
        for features, targets in train_load:
            features = features.to(device)
            targets = targets.to(device)

            pred = model(features)
            
            r2_value_train.update(pred[:,lev_range], targets[:,lev_range])

        r2_train = r2_value_train.compute()

        for features, targets in test_load:
            features = features.to(device)
            targets = targets.to(device)

            pred = model(features)

            r2_value_test.update(pred[:,lev_range], targets[:,lev_range])

        r2_test = r2_value_test.compute()

    if print_results is True:
        print('\n-----------------------')
        print(f'TRAIN  |   R2: {r2_train:.6f}')
        print(f'TEST   |   R2: {r2_test:.6f}')
        print('-----------------------\n')
    
    return r2_train, r2_test


# R2 score for grid points
def eval_points(model, train_dataset, test_dataset, points=None, print_results=False):
    batch_size=4096
    model.to('cpu')
    model.eval()

    train_load, test_load = load_data(train_dataset, test_dataset, batch_size)

    r2_value_train = R2Score()
    r2_value_test = R2Score()

    with torch.no_grad():
        for features, targets in train_load:
            features = features.to('cpu')
            targets = targets.to('cpu')
            pred = model(features)
            r2_value_train.update(pred[points,:], targets[points,:])

        r2_train = r2_value_train.compute()

        for features, targets in test_load:
            features = features.to('cpu')
            targets = targets.to('cpu')
            pred = model(features)
            r2_value_test.update(pred[points,:], targets[points,:])

        r2_test = r2_value_test.compute()
    
    if print_results is True:
        print('\n-----------------------')
        print(f'TRAIN  |   R2: {r2_train:.6f}')
        print(f'TEST   |   R2: {r2_test:.6f}')
        print('-----------------------\n')

    return r2_train, r2_test



# R2 score on single levels
def eval_single_levs(model, train_dataset, test_dataset, nr_sets=0, lev_range=None, batch_size=4096, mode='FCNN', levels=0, vector_feat=0, scalar_feat=0, fc_hidden=0, print_results=False):
    model.to(device)
    model.eval()

    r2_list_train = [R2Score().to(device) for _ in range(levels)]
    r2_list_test = [R2Score().to(device) for _ in range(levels)]


    train_load, test_load = load_data(train_dataset, test_dataset, batch_size)

    from tqdm import tqdm

    with torch.no_grad():
        for features, targets in tqdm(train_load):
            features = features.to(device)
            targets = targets.to(device)

            pred = model(features)

            for i in range(levels): r2_list_train[i].update(pred[:,i], targets[:,i])

        r2_train = [0.0] * levels
        for i in range(levels): r2_train[i] = r2_list_train[i].compute().cpu()

        for features, targets in tqdm(test_load):
            features = features.to(device)
            targets = targets.to(device)

            pred = model(features)

            for i in range(levels): r2_list_test[i].update(pred[:,i].to(device), targets[:,i].to(device))

        r2_test = [0.0] * levels
        for i in range(levels): r2_test[i] = r2_list_test[i].compute().cpu()
    
    # if print_results is True:
    #     print('\n-----------------------')
    #     print(f'TRAIN  |   R2: {r2_train:.6f}')
    #     print(f'TEST   |   R2: {r2_test:.6f}')
    #     print('-----------------------\n')

    return r2_train, r2_test





def eval_single_cols(model, train_dataset, test_dataset, nr_sets=0, lev_range=None, batch_size=4096, mode='FCNN', levels=0, vector_feat=0, scalar_feat=0, fc_hidden=0, print_results=False, points=0):
    model.to(device)
    model.eval()

    #r2_list_train = [R2Score().to(device) for _ in range(levels)]
    r2_list_test = [R2Score().to(device) for _ in range(points)]


    train_load, test_load = load_data(train_dataset, test_dataset, batch_size)

    from tqdm import tqdm

    with torch.no_grad():
        # for features, targets in tqdm(train_load):
        #     features = features.to(device)
        #     targets = targets.to(device)

        #    # if mode == 'FCNN':
        #     pred = model(features)

        # #     if mode == 'UNet_2' or mode == 'UNet_3':
        # #         pred = model(features)

        # #   #  elif mode == 'CNN' or mode == 'UNet':
        # #     else:
        # #         X_col, X_sca = features[:,:vector_feat*levels].to(device), features[:,-scalar_feat:].to(device)
        # #         X_col = X_col.view(-1, vector_feat, levels)
        # #         pred = model(X_col, X_sca).view(-1, levels*2)
            
        #     for i in range(levels): r2_list_train[i].update(pred[:,i], targets[:,i])

        # r2_train = [0.0] * levels
        # for i in range(levels): r2_train[i] = r2_list_train[i].compute().cpu()

        for features, targets in tqdm(test_load):
            features = features.to(device)
            targets = targets.to(device)

         #   if mode == 'FCNN':
            pred = model(features)

        #     if mode == 'UNet_2' or mode == 'UNet_3':
        #         pred = model(features)

        #   #  elif mode == 'CNN' or mode == 'UNet':
        #     else:
        #         X_col, X_sca = features[:,:vector_feat*levels].to(device), features[:,-scalar_feat:].to(device)
        #         X_col = X_col.view(-1, vector_feat, levels)
        #         pred = model(X_col, X_sca).view(-1, levels*2)

            for pt in range(points): r2_list_test[pt].update(pred[pt,:].to(device), targets[pt,:].to(device))

        r2_test = [0.0] * points
        for pt in range(points): r2_test[pt] = r2_list_test[pt].compute().cpu()

 #   if print_results is True:
 #       print('\n-----------------------')
   #     print(f'TRAIN  |   R2: {r2_train:.6f}')
   #     print(f'TEST   |   R2: {r2_test:.6f}')
  #      print('-----------------------\n')

    print(len(r2_test))
    
    return r2_test




def eval_cols_de(model, r2_feat_lsm, r2_targ_lsm, active=0, lev_range=None, batch_size=4096, mode='FCNN', levels=0, vector_feat=0, scalar_feat=0, fc_hidden=0, print_results=False):
    model.to('cpu')
    model.eval()

    r2_list_test = [R2Score() for _ in range(active)]

    with torch.no_grad():
        if mode == 'FCNN':
            pred = model(r2_feat_lsm)

          #  elif mode == 'CNN' or mode == 'UNet':
        else:
            X_col, X_sca = r2_feat_lsm[:,:vector_feat*levels].to(device), r2_feat_lsm[:,-scalar_feat:].to(device)
            X_col = X_col.view(-1, vector_feat, levels)
            pred = model(X_col, X_sca).view(-1, active, levels*2)
            
            for i in range(active): r2_list_test[i].update(pred[:,i,:], r2_targ_lsm.view(-1, active, levels*2)[:,i,:])

        r2_test = [0.0] * active
        for i in range(active): r2_test[i] = r2_list_test[i].compute()


    if print_results is True:
        print('\n-----------------------')
        print(f'TEST   |   R2: {r2_test:.6f}')
        print('-----------------------\n')

    return r2_test