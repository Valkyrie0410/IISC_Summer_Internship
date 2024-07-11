import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from torch.distributions import Categorical, Normal, MixtureSameFamily
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyemma
import torch.distributions as dist
from scipy.signal import correlate
import pandas as pd
import math, itertools
from math import cos, sin, sqrt, acos, atan2, fabs, pi

def load_full_dataset(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("SELECT * from data")
    coordinates = cur.fetchall()
    coordinates = np.array(coordinates).reshape(200001, 384)[1:]
    coordinates = coordinates.reshape(200000, 128, 3)
    # Load coordinates
    cur.execute("SELECT * FROM bond_lengths")
    bond_lengths= cur.fetchall()
    bond_lengths = np.array(bond_lengths).reshape(200001, 127)[1:]  

    cur.execute("SELECT * FROM bond_angles")
    bond_angles = cur.fetchall()
    bond_angles = np.array(bond_angles).reshape(200001, 188)[1:]

    cur.execute("SELECT * FROM dihedral_angles")
    dihedral_angles = cur.fetchall()
    dihedral_angles = np.array(dihedral_angles).reshape(200001, 125)[1:]
    
    # cur.execute("SELECT * FROM pairwise_distances")
    # pairwise = cur.fetchall()
    # pairwise = np.array(pairwise).reshape(200001, 1830)[1:]

    conn.close()
    bond_angles = np.radians(bond_angles)
    dihedral_angles = np.radians(dihedral_angles)
    dihedral_angles = dihedral_angles + 2 * np.pi * (dihedral_angles < 0)
    features = np.hstack([bond_lengths, bond_angles, dihedral_angles])
    print(bond_angles)
    print(dihedral_angles)
    return features, coordinates

class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_components, multivariate=False, cov_scaling_factor=0.8):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.softplus = nn.Softplus()
        self.multivariate = multivariate
        self.cov_scaling_factor = cov_scaling_factor
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.fc_pi = nn.Linear(hidden_dim,1320)
        self.fc_mu = nn.Linear(hidden_dim, 945)
        self.fc_sigma = nn.Linear(hidden_dim, 945)
        self.fc_loc = nn.Linear(hidden_dim, 375)
        self.fc_concentration = nn.Linear(hidden_dim, 375)
        self.dims_bonds_ = list(np.arange(0, 127, 1))
        self.dims_angles_ = list(
            np.arange(127, 253, 1))
        self.scaling_dims = self.dims_bonds_ + self.dims_angles_
        self.dims_total = 440

    def forward(self, x):
        fc = self.fc1(x)
        fc_1 = fc
        pi_ = self.fc_pi(fc_1)
        pi_ = pi_.view(-1,440,3)
        pi = torch.softmax(pi_, dim=-1)
        mu = self.fc_mu(fc)
        mu = mu.view(-1, 315, 3)
        sigma = torch.sigmoid(self.fc_sigma(fc))*self.cov_scaling_factor
        sigma = sigma.view(-1, 315, 3)
        loc = self.fc_loc(fc)
        loc = loc.view(-1, 125, 3)
        concentration = self.softplus(self.fc_concentration(fc))
        concentration = concentration.view(-1, 125, 3)
        
        return pi, mu, sigma, loc, concentration

    def repeatAlongDim(self, var, axis, repeat_times):
        repeat_idx = len(var.size()) * [1]
        repeat_idx[axis] = repeat_times
        var = var.repeat(*repeat_idx)
        return var
        
        
    def processTargets(self, targets):
        assert (targets.size()[1] == self.dims_total)
        assert (torch.all(targets[:, self.scaling_dims, :] < 1))
        assert (torch.all(targets[:, self.scaling_dims, :] > 0))
        targets[:, self.scaling_dims, :] = torch.log(
            targets[:, self.scaling_dims, :] /
            (1.0 - targets[:, self.scaling_dims, :]))
        return targets
        
    def log_prob(self, y, pi, mu, sigma, loca, conc):
        m_normal = torch.distributions.Normal(loc=mu, scale=sigma)  # Unsqueezing mu and sigma
        m_angles = torch.distributions.VonMises(loc = loca, concentration = conc)
        y = y[:,:, None]
            
        y = self.repeatAlongDim(y, axis = 2, repeat_times = 3)
        y = self.processTargets(y)
        log_prob_normal = m_normal.log_prob(y[:, :315])
        log_prob_angles = m_angles.log_prob(y[:, 315:])
        log_prob = torch.cat((log_prob_normal, log_prob_angles), dim=1)
        pi = torch.clamp(pi, min=1e-15)
        log_pi = torch.log(pi)
        sum_logs = log_prob + log_pi
        sum_logs_max = torch.max(sum_logs, 2)[0]
        sum_logs_max = sum_logs_max[:, :, None]
        loss = torch.exp(sum_logs - sum_logs_max)
        loss = torch.sum(loss, dim=2)
        
        return torch.mean(loss)

class MDNL(nn.Module):
    def __init__(self, input_dim = 40, hidden_dim = 20, num_components = 6, output_dim = 2, multivariate=False, cov_scaling_factor=0.2):
        super(MDNL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.softplus = nn.Softplus()
        self.multivariate = multivariate
        self.cov_scaling_factor = cov_scaling_factor
        self.output_dim = output_dim
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 500), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(500, 500), nn.Tanh())
        self.fc3 = nn.Sequential(nn.Linear(500, 500), nn.Tanh())
        self.fc4 = nn.Sequential(nn.Linear(500, 500), nn.Tanh())
        self.fc5 = nn.Sequential(nn.Linear(500, 500), nn.Tanh())
        self.fc_pi = nn.Linear(500, 8)
        self.fc_mu = nn.Linear(500, 8)
        self.fc_sigma = nn.Linear(500, 8)


    def forward(self, x):
        h = (self.fc1(x))
        h = (self.fc2(h))
        h = (self.fc3(h))
        # h = (self.fc4(h))
        # h = self.fc5(h)
        pi_ = self.fc_pi(h)
        pi_ = pi_.view(-1,2,4)
        pi = torch.softmax(pi_, dim=-1)
        mu = self.fc_mu(h)
        mu = mu.view(-1, 2, 4)
        sigma = torch.sigmoid(self.fc_sigma(h))*self.cov_scaling_factor
        sigma = sigma.view(-1, 2, 4)
        
        return pi, mu, sigma

    def sample(self, pi, mu, sigma):
        batch_size, latent_dim, num_components = mu.size()

        # Sample a component from the categorical distribution
        categorical = dist.Categorical(pi)
        component = categorical.sample()  # Shape: (batch_size, latent_dim)

        # Gather the selected mu and sigma based on the sampled component
        selected_mu = torch.gather(mu, 2, component.unsqueeze(1).expand(batch_size, 1, latent_dim))  # Shape: (batch_size, 1, latent_dim)
        selected_sigma = torch.gather(sigma, 2, component.unsqueeze(1).expand(batch_size, 1, latent_dim))  # Shape: (batch_size, 1, latent_dim)

        # Sample from the normal distribution
        normal = dist.Normal(selected_mu, selected_sigma)
        sample = normal.sample().squeeze(1)  # Shape: (batch_size, latent_dim)

        return sample

    def repeatAlongDim(self, var, axis, repeat_times):
        repeat_idx = len(var.size()) * [1]
        repeat_idx[axis] = repeat_times
        var = var.repeat(*repeat_idx)
        return var
        
        
    def processTargets(self, targets):
        assert (torch.all(targets < 1))
        assert (torch.all(targets > 0))
        targets = torch.log(targets / (1 - targets))
        return targets

    def scale_targets(self, targets):
        # Flatten the 3D tensor to 2D
        targets_cpu = targets.cpu()
        targets_np = targets_cpu.detach().numpy()
        original_shape = targets_np.shape
        targets_reshaped = targets_np.reshape(targets_np.shape[0], -1)
        
        # Scale the targets to the range (0.0001, 0.9999)
        feature_range = (0.0001, 0.9999)
        scaler = MinMaxScaler(feature_range=feature_range)
        targets_scaled = scaler.fit_transform(targets_reshaped)
        targets_scaled[targets_scaled == 0] = feature_range[0]
        targets_scaled[targets_scaled == 1] = feature_range[1]
        # Reshape back to original 3D shape
        targets_scaled_3d = targets_scaled.reshape(original_shape)
        targets_scaled_torch = torch.tensor(targets_scaled_3d).to(targets.device)
        
        return targets_scaled_torch
        
    def log_prob(self, y, pi, mu, sigma, loca=None, conc=None):
        m = torch.distributions.Normal(loc=mu, scale=sigma) # Unsqueezing mu and sigma
        y = y[:,:, None]
        y = self.repeatAlongDim(y, axis = 2, repeat_times = 4)
        # y = self.scale_targets(y)
        y = self.processTargets(y)
        log_prob = m.log_prob(y)
        pi = torch.clamp(pi, min=1e-15)
        log_pi = torch.log(pi)
        sum_logs = log_prob + log_pi
        sum_logs_max = torch.max(sum_logs, 2)[0]
        sum_logs_max = sum_logs_max[:, :, None]
        loss = torch.exp(sum_logs - sum_logs_max)
        loss = torch.sum(loss, dim=2)
        
        return torch.mean(loss)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 500)
        self.fc6 = nn.Linear(500, latent_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x.float()

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_components, input_dim, multivariate=False, cov_scaling_factor=0.8, mdn_bool=True):
        super(Decoder, self).__init__()
        self.mdn_bool = mdn_bool
        self.fc1 = nn.Linear(latent_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc6 = nn.Linear(500, input_dim)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 500)
        self.tanh = nn.Tanh()
        self.mdn = MDN(500, hidden_dim, num_components, multivariate, cov_scaling_factor)

    def forward(self, x, mdn_bool = True):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
        if mdn_bool:
            pi, mu, sigma, loc, conc = self.mdn(x)
            return pi, mu, sigma, loc, conc
        else:
            x = torch.sigmoid(self.fc6(x))
            return x

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_components, mdn_bool=True):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, num_components, input_dim, mdn_bool)
        self.mdn_bool = mdn_bool
        print(mdn_bool)
    def forward(self, x, mdn_bool = True):
        x = x.to(self.encoder.fc1.weight.dtype)
        latent_states = self.encoder(x)
        z = latent_states
        if mdn_bool is True:
            pi, mu, sigma, loc, conc = self.decoder(z)
            return latent_states, pi, mu, sigma, loc, conc
        else:
            traj = self.decoder(z, mdn_bool)
            return traj

class MDN_LSTM(nn.Module):
    def __init__(self, latent_dim = 2, hidden_dim = 40, num_components = 4, hidden_units=20, num_layers=1, multivariate=False, cov_scaling_factor=0.2):
        super(MDN_LSTM, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.num_components = num_components
        self.multivariate = multivariate
        self.cov_scaling_factor = cov_scaling_factor
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.mdn = MDNL()

    def forward(self, x, hidden_state = None):
        batch_size = x.size(0)
        if hidden_state is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            hidden_state = (h0, c0)

        lstm_out, hidden_state = self.lstm(x, hidden_state)
        # lstm_out = lstm_out[:, -1, :]
        # print(lstm_out.shape)
        pi, mu, sigma = self.mdn(lstm_out)  # Select the output of the last time step
        return pi, mu, sigma, hidden_state

class MyDataset(Dataset):
    def __init__(self, data, target_data, device):
        self.data = data
        self.device = device
        self.target_data = target_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.target_data[idx+1]
        sample = torch.tensor(sample, dtype=torch.float32).to(self.device)
        target = torch.tensor(target, dtype=torch.float32).to(self.device)
        return sample, target

def collate_fn(batch):
    samples, targets = zip(*batch)
    samples = torch.stack(samples)
    targets = [t if t is not None else torch.zeros_like(samples[0]) for t in targets]
    targets = torch.stack(targets)
    return samples, targets

class scalerBAD(object):
    # Bonds/Angles/Dihedrals scaler
    def __init__(
        self,
        dims_total,
        dims_bonds,
        dims_angles,
        dims_dehedrals,
        data_min_bonds,
        data_max_bonds,
        data_min_angles,
        data_max_angles,
        # data_min_dihedrals,
        # data_max_dihedrals,
        slack=0.05,
    ):
        
        range_bonds = data_max_bonds - data_min_bonds
        self.data_min_bonds = data_min_bonds - slack * range_bonds
        self.data_max_bonds = data_max_bonds + slack * range_bonds

        range_angles = data_max_angles - data_min_angles
        self.data_min_angles = data_min_angles - slack * np.abs(range_angles)
        self.data_max_angles = data_max_angles + slack * np.abs(range_angles)
        
        # range_dihedrals = data_max_dihedrals - data_min_dihedrals
        # self.data_min_dihedrals = data_min_dihedrals - slack * np.abs(range_dihedrals)
        # self.data_max_dihedrals = data_max_dihedrals + slack * np.abs(range_dihedrals)

        self.data_min = np.concatenate(
            (self.data_min_bonds, self.data_min_angles), axis=0)
        self.data_max = np.concatenate(
            (self.data_max_bonds, self.data_max_angles), axis=0)

        self.dims_total = dims_total
        self.dims_bonds = dims_bonds
        self.dims_angles = dims_angles
        self.dims_dehedrals = dims_dehedrals

        self.dims_bonds_ = list(np.arange(0, self.dims_bonds, 1))
        self.dims_angles_ = list(
            np.arange(self.dims_bonds, self.dims_bonds + self.dims_angles, 1))
        self.dims_dehedrals_ = list(
            np.arange(self.dims_bonds + self.dims_angles,
                      self.dims_bonds + self.dims_angles + self.dims_dehedrals,
                      1))

        self.scaling_dims = self.dims_bonds_ + self.dims_angles_

    def scaleData(self, batch_of_sequences, single_sequence=False):
        if single_sequence:
            batch_of_sequences = batch_of_sequences[np.newaxis]
        
        self.data_shape = np.shape(batch_of_sequences)
        self.data_shape_length = len(self.data_shape)

        batch_of_sequences_scaled = batch_of_sequences.copy()
        data_min = self.repeatScalerParam(self.data_min, self.data_shape)
        data_max = self.repeatScalerParam(self.data_max, self.data_shape)
        assert np.all(np.shape(batch_of_sequences_scaled[:, :, self.scaling_dims]) == np.shape(data_min))
        assert np.all(np.shape(batch_of_sequences_scaled[:, :, self.scaling_dims]) == np.shape(data_max))
        batch_of_sequences_scaled[:, :, self.scaling_dims] = np.array(
            (batch_of_sequences_scaled[:, :, self.scaling_dims] - data_min) / (data_max - data_min))

        if single_sequence:
            batch_of_sequences_scaled = batch_of_sequences_scaled[0]
        return batch_of_sequences_scaled

    def descaleData(self, batch_of_sequences_scaled, single_sequence=True):
        if single_sequence:
            batch_of_sequences_scaled = batch_of_sequences_scaled[np.newaxis]
        
        batch_of_sequences = batch_of_sequences_scaled.copy()
        self.data_shape = np.shape(batch_of_sequences)
        self.data_shape_length = len(self.data_shape)
        data_min = self.repeatScalerParam(self.data_min, self.data_shape)
        data_max = self.repeatScalerParam(self.data_max, self.data_shape)
        assert np.all(np.shape(batch_of_sequences[:, :, self.scaling_dims]) == np.shape(data_min))
        assert np.all(np.shape(batch_of_sequences[:, :, self.scaling_dims]) == np.shape(data_max))
        batch_of_sequences[:, :, self.scaling_dims] = np.array(
            batch_of_sequences[:, :, self.scaling_dims] * (data_max - data_min) + data_min)
        
        if single_sequence:
            batch_of_sequences = batch_of_sequences[0]
        return np.array(batch_of_sequences)

    def repeatScalerParam(self, data, shape):
        T = shape[1]
        data = np.repeat(data[np.newaxis], T, 0)
        K = shape[0]
        data = np.repeat(data[np.newaxis], K, 0)
        return data



db_path = "/home/smart/Documents/IISC/sqlite_1.db"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
features,coordinates = load_full_dataset(db_path)

dims_total = 440
dims_bonds = 127
dims_angles = 188
dims_dehedrals = 125

sequence_length = 400
num_sequences = len(features) // sequence_length
sequences = np.split(features[:num_sequences * sequence_length], num_sequences)

num_train_sequences = 96
num_val_sequences = 96
num_test_sequences = 154
train_sequences = sequences[:96]
val_sequences = sequences[:]
test_sequences = sequences[499:]
data = np.array(sequences)

bond_lengths = data[:96, :, :127]
bond_angles = data[:96, :, 127:127+188]
dihedral_angles = data[:96, :, 127+188:]

data_min_bonds = np.min(bond_lengths, axis=(0,1))
data_max_bonds = np.max(bond_lengths, axis=(0,1))
data_min_angles = np.min(bond_angles, axis=(0,1))
data_max_angles = np.max(bond_angles, axis=(0,1))
data_min_dihedrals = np.min(dihedral_angles, axis=(0,1))
data_max_dihedrals = np.max(dihedral_angles, axis=(0,1))

scaler = scalerBAD(
    dims_total=dims_total,
    dims_bonds=dims_bonds,
    dims_angles=dims_angles,
    dims_dehedrals=dims_dehedrals,
    data_min_bonds=data_min_bonds,
    data_max_bonds=data_max_bonds,
    data_min_angles=data_min_angles,
    data_max_angles=data_max_angles,
    # data_min_dihedrals = data_min_dihedrals,
    # data_max_dihedrals=data_max_dihedrals
)

train_data = np.array(train_sequences)
test_data = np.array(test_sequences)
scaled_train_data = scaler.scaleData(train_data, single_sequence=False)
scaled_test_data = scaler.scaleData(test_data, single_sequence=False)
val_data = np.array(val_sequences)
scaled_val_data = scaler.scaleData(val_data, single_sequence=False)
batch_size = 8

# not_scaled = scaled_train_data[:,:,315:]
# not_scaled = (not_scaled+ np.pi) / (2 * np.pi)
# scaled_train_data = np.concatenate((scaled_train_data[:,:,:315], not_scaled), axis = 2)
# not_scaled = scaled_test_data[:,:,315:]
# not_scaled = (not_scaled+ np.pi) / (2 * np.pi)
# scaled_test_data = np.concatenate((scaled_test_data[:,:,:315], not_scaled), axis = 2)

# not_scaled = scaled_val_data[:,:,315:]
# not_scaled = (not_scaled+ np.pi) / (2 * np.pi)
# scaled_val_data = np.concatenate((scaled_val_data[:,:,:315], not_scaled), axis = 2)

train_dataset = MyDataset(scaled_train_data, scaled_val_data, device=device)
val_dataset = MyDataset(val_data, scaled_val_data, device=device)
test_dataset = MyDataset(scaled_test_data, scaled_val_data, device=device)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Define model parameters
output_dim_normal = 315
output_dim_von_mises = 125
input_dim = features.shape[1]
latent_dim = 2
hidden_dim = 50
num_components = 4
learning_rate = 0.001

# Create models
autoencoder = Autoencoder(input_dim, latent_dim, hidden_dim, num_components, mdn_bool = True).to(device)
mdn_lstm = MDN_LSTM().to(device)

# Optimizers
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=1e-3)
optimizer_lstm = optim.Adam(mdn_lstm.parameters(), lr=1e-3)

# Pretraining phase
pretrain_epochs = 4

for param in autoencoder.decoder.mdn.parameters():
    param.requires_grad = True

pretrain_optimizer = optim.Adam(autoencoder.decoder.mdn.parameters(), lr=0.003)
scheduler = optim.lr_scheduler.StepLR(pretrain_optimizer, step_size=10, gamma=0.1)
for epoch in range(pretrain_epochs):
    autoencoder.train()
    pretrain_loss = 0.0
    for batch_data, _ in train_dataloader:
        batch_data = batch_data.to(device).float()
        _, pi_ae, mu_ae, sigma_ae, loc_ae, conc_ae = autoencoder(batch_data)
        
        K, T, N = batch_data.size()
        D = 1
        batch_data = batch_data.view(K * T, N * D)
        mu_ae = mu_ae.view(K * T, 315, 3)
        sigma_ae = sigma_ae.view(K * T, 315, 3)
        loc_ae = loc_ae.view(K * T, 125, 3)
        conc_ae = conc_ae.view(K * T, 125, 3)
        pi_ae = pi_ae.view(K * T, 440, 3)
        pretrain_loss = -autoencoder.decoder.mdn.log_prob(batch_data, pi_ae, mu_ae, sigma_ae, loc_ae, conc_ae).mean()
        
        pretrain_optimizer.zero_grad()
        pretrain_loss.backward()
        pretrain_optimizer.step()
        
        pretrain_loss += pretrain_loss.item()

    scheduler.step()
    pretrain_loss /= len(train_dataloader)
    print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs}, Loss: {pretrain_loss:.4f}")

for param in autoencoder.decoder.mdn.parameters():
    param.requires_grad = False

for name, param in autoencoder.named_parameters():
    if 'decoder.mdn' not in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer_ae = optim.Adam(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer_ae, step_size=10, gamma=0.1)
num_epochs_ae = 50
for epoch in range(num_epochs_ae):
    autoencoder.train()
    train_loss_ae = 0.0
    
    for batch_data, targets in train_dataloader:
        batch_data = batch_data.to(device).float()
        targets = targets.to(device).float()
        latent_states_ae, pi_ae, mu_ae, sigma_ae, loc_ae, conc_ae = autoencoder(batch_data)
        
        K, T, N = batch_data.size()
        D = 1
        batch_data = batch_data.view(K * T, N * D)
        mu_ae = mu_ae.view(K * T, 315, 3)
        sigma_ae = sigma_ae.view(K * T, 315, 3)
        loc_ae = loc_ae.view(K * T, 125, 3)
        conc_ae = conc_ae.view(K * T, 125, 3)
        pi_ae = pi_ae.view(K * T, 440, 3)

        loss_ae = -autoencoder.decoder.mdn.log_prob(batch_data, pi_ae, mu_ae, sigma_ae, loc_ae, conc_ae).mean()
        
        optimizer_ae.zero_grad()
        loss_ae.backward()
        optimizer_ae.step()
    
        train_loss_ae += loss_ae.item()

    scheduler.step()
    train_loss_ae /= len(train_dataloader)
    print(f"AE Epoch {epoch+1}/{num_epochs_ae}, Loss: {train_loss_ae:.4f}")

for param in autoencoder.parameters():
    param.requires_grad = False

optimizer_lstm = optim.Adam(mdn_lstm.parameters(), lr=0.0001)

num_epochs_lstm = 25
for epoch in range(num_epochs_lstm):
    autoencoder.eval()
    mdn_lstm.train()
    train_loss_lstm = 0.0

    for batch_data, targets in train_dataloader:
        batch_data = batch_data.to(device).float()
        targets = targets.to(device).float()

        with torch.no_grad():
            latent_states_ae, _, _, _,_,_ = autoencoder(batch_data)
            latent_states_pred, _, _, _, _, _= autoencoder(targets)
        latent_states_ae = latent_states_ae.transpose(1, 0)

        pi_lstm, mu_lstm, sigma_lstm, _ = mdn_lstm(latent_states_ae)
        
        K, T, N = latent_states_ae.size()
        D = 1
        latent_states_pred = latent_states_pred.view(K * T, N * D)
        mu_lstm = mu_lstm.view(K * T, 2, 4)
        sigma_lstm = sigma_lstm.view(K * T, 2, 4)
        pi_lstm = pi_lstm.view(K * T, 2, 4)
        
        loss_lstm = -mdn_lstm.mdn.log_prob(latent_states_pred, pi_lstm, mu_lstm, sigma_lstm).mean()
        
        optimizer_lstm.zero_grad()
        loss_lstm.backward()
        optimizer_lstm.step()
        
        train_loss_lstm += loss_lstm.item()
    train_loss_lstm /= len(train_dataloader)
    print(f"LSTM Epoch {epoch+1}/{num_epochs_lstm}, Loss: {train_loss_lstm:.4f}")

torch.save(autoencoder, 'autoencoder.pth')
torch.save(mdn_lstm, 'mdn_lstm.pth')
# autoencoder = torch.load('autoencoder.pth')
# mdn_lstm = torch.load('mdn_lstm.pth')

autoencoder.eval()
mdn_lstm.eval()
test_loss_lstm = 0.0
all_latent_states_test = []
T_mu = 10
T_m = 4000

for batch_data, _ in test_dataloader:
    batch_data = batch_data.to(device).float()

    with torch.no_grad():
        latent_states_ae, _, _, _, _, _ = autoencoder(batch_data[:, :T_mu, :])
        latent_states_ae = latent_states_ae.transpose(1, 0)
        hidden_state = None
        for t in range(T_mu):
            _, _, _, hidden_state = mdn_lstm(latent_states_ae[t, :, :].unsqueeze(1), hidden_state)
    predicted_latent_states = []
    latent_states_ae = latent_states_ae.transpose(1, 0)
    input_state = latent_states_ae[:, -1, :].unsqueeze(1)
    for t in range(T_m):
        input_state = input_state.transpose(1,0)
        pi_lstm, mu_lstm, sigma_lstm, hidden_state = mdn_lstm(input_state, hidden_state)
        sampled_latent_state = mdn_lstm.mdn.sample(pi_lstm, mu_lstm, sigma_lstm)
        predicted_latent_states.append(sampled_latent_state)
        input_state = sampled_latent_state.unsqueeze(1)
    
    predicted_latent_states = torch.cat(predicted_latent_states, dim=1)
    all_latent_states_test.append(predicted_latent_states.cpu().numpy())
    

all_latent_states_test = np.concatenate(all_latent_states_test, axis=0)
file_path = 'latent_states.csv'
df = pd.DataFrame(all_latent_states_test)
df.to_csv(file_path, index = False)
all_latent_states_test = all_latent_states_test.reshape(T_m, 2)


# Decode the predicted latent states to obtain high-dimensional state configurations
dihedral = []
bl = []
ba = []
for latent_states in all_latent_states_test:
    latent_states_tensor = torch.tensor(latent_states).to(device).float()
    with torch.no_grad():
        decoded_states_tuple = autoencoder.decoder(latent_states_tensor, mdn_bool=False)
        decoded_states_numpy = np.array([element.cpu().numpy() for element in decoded_states_tuple])
        decoded_states_numpy = decoded_states_numpy.reshape((1, 440))
        descaled_data = scaler.descaleData(decoded_states_numpy, single_sequence = True)
        descaled_data = descaled_data.reshape(440)
        dihedral.append(descaled_data[315:])
        bl.append(descaled_data[:127])
        ba.append(descaled_data[127:315])
for i in range(125):
    print(dihedral[0][i]," ",features[199610][315+i])

for i in range(188):
    print(ba[0][i]," ",features[199610][127+i])
df = pd.DataFrame(bl)
df.to_csv('bond_lengths_result.csv', index = False) 
df = pd.DataFrame(ba)
df.to_csv('bond_angles_result.csv', index = False)
df = pd.DataFrame(dihedral)
df.to_csv('dihedral_result.csv', index = False)


dih = [(1, 0, 2, 3), (1, 0, 2, 4), (3, 2, 4, 5), (3, 2, 4, 6), (5, 4, 6, 7), (5, 4, 6, 8),
 (7, 6, 8, 9), (7, 6, 8, 10), (9, 8, 10, 11), (9, 8, 10, 12), (11, 10, 12, 13), (11, 10, 12, 14),
 (13, 12, 14, 15), (13, 12, 14, 16), (15, 14, 16, 17), (15, 14, 16, 18), (17, 16, 18, 19), (17, 16, 18, 20),
 (19, 18, 20, 21), (19, 18, 20, 22), (21, 20, 22, 23), (21, 20, 22, 24), (23, 22, 24, 25), (23, 22, 24, 26),
 (25, 24, 26, 27), (25, 24, 26, 28), (27, 26, 28, 29), (27, 26, 28, 30), (29, 28, 30, 31), (29, 28, 30, 32),
 (31, 30, 32, 33), (31, 30, 32, 34), (33, 32, 34, 35), (33, 32, 34, 36), (35, 34, 36, 37), (35, 34, 36, 38),
 (37, 36, 38, 39), (37, 36, 38, 40), (39, 38, 40, 41), (39, 38, 40, 42), (41, 40, 42, 43), (41, 40, 42, 44),
 (43, 42, 44, 45), (43, 42, 44, 46), (45, 44, 46, 47), (45, 44, 46, 48), (47, 46, 48, 49), (47, 46, 48, 50),
 (49, 48, 50, 51), (49, 48, 50, 52), (51, 50, 52, 53), (51, 50, 52, 54), (53, 52, 54, 55), (53, 52, 54, 56),
 (55, 54, 56, 57), (55, 54, 56, 58), (57, 56, 58, 59), (57, 56, 58, 60), (59, 58, 60, 61), (59, 58, 60, 62),
 (61, 60, 62, 63), (61, 60, 62, 64), (63, 62, 64, 65 ), (63, 62, 64, 66), (65, 64, 66, 67), (65, 64, 66, 68),
 (67, 66, 68, 69), (67, 66, 68, 70), (69, 68, 70, 71), (69, 68, 70, 72), (71, 70, 72, 73), (71, 70, 72, 74),
 (73, 72, 74, 75), (73, 72, 74, 76), (75, 74, 76, 77), (75, 74, 76, 78), (77, 76, 78, 79), (77, 76, 78, 80),
 (79, 78, 80, 81), (79, 78, 80, 82), (81, 80, 82, 83), (81, 80, 82, 84), (83, 82, 84, 85), (83, 82, 84, 86),
 (85, 84, 86, 87), (85, 84, 86, 88), (87, 86, 88, 89), (87, 86, 88, 90), (89, 88, 90, 91), (89, 88, 90, 92),
 (91, 90, 92, 93), (91, 90, 92, 94), (93, 92, 94, 95), (93, 92, 94, 96), (95, 94, 96, 97), (95, 94, 96, 98),
 (97, 96, 98, 99), (97, 96, 98, 100), (99, 98, 100, 101), (99, 98, 100, 102), (101, 100, 102, 103), (101, 100, 102, 104),
 (103, 102, 104, 105), (103, 102, 104, 106), (105, 104, 106, 107), (105, 104, 106, 108), (107, 106, 108, 109), (107, 106, 108, 110),
 (109, 108, 110, 111), (109, 108, 110, 112), (111, 110, 112, 113), (111, 110, 112, 114), (113, 112, 114, 115), (113, 112, 114, 116),
 (115, 114, 116, 117), (115, 114, 116, 118), (117, 116, 118, 119), (117, 116, 118, 120), (119, 118, 120, 121), (119, 118, 120, 122),
 (121, 120, 122, 123), (121, 120, 122, 124), (123, 122, 124, 125), (123, 122, 124, 126), (125,124,126,127)]
bonds = [(1,0), (0,2), (2,3), (2, 4), (4, 5), (4, 6), (6, 7), (6, 8), (8, 9), (8, 10), (10, 11), (10, 12), (12, 13), (12, 14), (14, 15), (14, 16), (16, 17), (16, 18), (18, 19), (18, 20), (20, 21), (20, 22), (22, 23), (22, 24), (24, 25), (24, 26), (26, 27), (26, 28), (28, 29), (28, 30), (30, 31), (30, 32), (32, 33), (32, 34), (34, 35), (34, 36), (36, 37), (36, 38), (38, 39), (38, 40), (40, 41), (40, 42), (42, 43), (42, 44), (44, 45), (44, 46), (46, 47), (46, 48), (48, 49), (48, 50), (50, 51), (50, 52), (52, 53), (52, 54), (54, 55), (54, 56), (56, 57), (56, 58), (58, 59), (58, 60), (60, 61), (60, 62), (62, 63), (62, 64), (64, 65), (64, 66), (66, 67), (66, 68), (68, 69), (68, 70), (70, 71), (70, 72), (72, 73), (72, 74), (74, 75), (74, 76), (76, 77), (76, 78), (78, 79), (78, 80), (80, 81), (80, 82), (82, 83), (82, 84), (84, 85), (84, 86), (86, 87), (86, 88), (88, 89), (88, 90), (90, 91), (90, 92), (92, 93), (92, 94), (94, 95), (94, 96), (96, 97), (96, 98), (98, 99), (98, 100), (100, 101), (100, 102), (102, 103), (102, 104), (104, 105), (104, 106), (106, 107), (106, 108), (108, 109), (108, 110), (110, 111), (110, 112), (112, 113), (112, 114), (114, 115), (114, 116), (116, 117), (116, 118), (118, 119), (118, 120), (120, 121), (120, 122), (122, 123), (122, 124), (124, 125), (124, 126), (126, 127)]
angles = [(1, 0, 2), (0, 2, 3), (0, 2, 4), (3, 2, 4), (2, 4, 5), (2, 4, 6), (5, 4, 6), (4, 6, 7), (4, 6, 8),
 (7, 6, 8), (6, 8, 9), (6, 8, 10), (9, 8, 10), (8, 10, 11), (8, 10, 12), (11, 10, 12), (10, 12, 13), (10, 12, 14),
 (13, 12, 14), (12, 14, 15), (12, 14, 16), (15, 14, 16), (14, 16, 17), (14, 16, 18), (17, 16, 18), (16, 18, 19), (16, 18, 20),
 (19, 18, 20), (18, 20, 21), (18, 20, 22), (21, 20, 22), (20, 22, 23), (20, 22, 24), (23, 22, 24), (22, 24, 25), (22, 24, 26),
 (25, 24, 26), (24, 26, 27), (24, 26, 28), (27, 26, 28), (26, 28, 29), (26, 28, 30), (29, 28, 30), (28, 30, 31), (28, 30, 32),
 (31, 30, 32), (30, 32, 33), (30, 32, 34), (33, 32, 34), (32, 34, 35), (32, 34, 36), (35, 34, 36), (34, 36, 37), (34, 36, 38),
 (37, 36, 38), (36, 38, 39), (36, 38, 40), (39, 38, 40), (38, 40, 41), (38, 40, 42), (41, 40, 42), (40, 42, 43), (40, 42, 44),
 (43, 42, 44), (42, 44, 45), (42, 44, 46), (45, 44, 46), (44, 46, 47), (44, 46, 48), (47, 46, 48), (46, 48, 49), (46, 48, 50),
 (49, 48, 50), (48, 50, 51), (48, 50, 52), (51, 50, 52), (50, 52, 53), (50, 52, 54), (53, 52, 54), (52, 54, 55), (52, 54, 56),
 (55, 54, 56), (54, 56, 57), (54, 56, 58), (57, 56, 58), (56, 58, 59), (56, 58, 60), (59, 58, 60), (58, 60, 61), (58, 60, 62),
 (61, 60, 62), (60, 62, 63), (60, 62, 64), (63, 62, 64), (62, 64, 65), (62, 64, 66), (65, 64, 66), (64, 66, 67), (64, 66, 68),
 (67, 66, 68), (66, 68, 69), (66, 68, 70), (69, 68, 70), (68, 70, 71), (68, 70, 72), (71, 70, 72), (70, 72, 73), (70, 72, 74),
 (73, 72, 74), (72, 74, 75), (72, 74, 76), (75, 74, 76), (74, 76, 77), (74, 76, 78), (77, 76, 78), (76, 78, 79), (76, 78, 80),
 (79, 78, 80), (78, 80, 81), (78, 80, 82), (81, 80, 82), (80, 82, 83), (80, 82, 84), (83, 82, 84), (82, 84, 85), (82, 84, 86),
 (85, 84, 86), (84, 86, 87), (84, 86, 88), (87, 86, 88), (86, 88, 89), (86, 88, 90), (89, 88, 90), (88, 90, 91), (88, 90, 92),
 (91, 90, 92), (90, 92, 93), (90, 92, 94), (93, 92, 94), (92, 94, 95), (92, 94, 96), (95, 94, 96), (94, 96, 97), (94, 96, 98),
 (97, 96, 98), (96, 98, 99), (96, 98, 100), (99, 98, 100), (98, 100, 101), (98, 100, 102), (101, 100, 102), (100, 102, 103), (100, 102, 104),
 (103, 102, 104), (102, 104, 105), (102, 104, 106), (105, 104, 106), (104, 106, 107), (104, 106, 108), (107, 106, 108), (106, 108, 109), (106, 108, 110),
 (109, 108, 110), (108, 110, 111), (108, 110, 112), (111, 110, 112), (110, 112, 113), (110, 112, 114), (113, 112, 114), (112, 114, 115), (112, 114, 116),
 (115, 114, 116), (114, 116, 117), (114, 116, 118), (117, 116, 118), (116, 118, 119), (116, 118, 120), (119, 118, 120), (118, 120, 121), (118, 120, 122),
 (121, 120, 122), (120, 122, 123), (120, 122, 124), (123, 122, 124), (122, 124, 125), (122, 124, 126), (125, 124, 126), (124, 126, 127)]



def find_BA(dd1, dd2, dd3, dd4):

    angleID = -1
    for aa in range(len(angles)):
        if (dd2 == angles[aa][0] and dd3 == angles[aa][1] and dd4 == angles[aa][2]):
            angleID = aa
            break
    if (angleID == -1):
        print("angle not found", dd2, dd3, dd4)
        exit()
        #find bond
    bondID = -1
    for bb in range(len(bonds)):
        if (dd3 == bonds[bb][0] and dd4 == bonds[bb][1]):
            bondID = bb
            break
    if (bondID == -1):
        print("bond not found")
        exit()

    return bondID, angleID

def place_atom(atom_a, atom_b, atom_c, angle, torsion, bond):

    R = bond
    ab = np.subtract(atom_b, atom_a)
    bc = np.subtract(atom_c, atom_b)
    bcn = bc / np.linalg.norm(bc)

    case = 1
    okinsert = False
    while (okinsert == False):
        #case 1
        if (case == 2):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                R * sin(torsion) * sin(angle)
            ])
            n = np.cross(bcn, ab)
            n = n / np.linalg.norm(n)
            nbc = np.cross(bcn, n)
        #case 2
        elif (case == 1):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                R * sin(torsion) * sin(angle)
            ])
            n = np.cross(ab, bcn)
            n = n / np.linalg.norm(n)
            nbc = np.cross(bcn, n)
        #case 3
        elif (case == 3):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                -R * sin(torsion) * sin(angle)
            ])
            n = np.cross(ab, bcn)
            n = n / np.linalg.norm(n)
            nbc = np.cross(bcn, n)
        #case 4
        elif (case == 4):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                R * sin(torsion) * sin(angle)
            ])
            n = np.cross(ab, bcn)
            n = n / np.linalg.norm(n)
            nbc = np.cross(n, bcn)

        m = np.array([[bcn[0], nbc[0], n[0]], [bcn[1], nbc[1], n[1]],
                        [bcn[2], nbc[2], n[2]]])
        d = m.dot(d)
        atom_d = d + atom_c

        #test dihedral
        r21 = np.subtract(atom_b, atom_a)
        r23 = np.subtract(atom_c, atom_b)
        r43 = np.subtract(atom_d, atom_c)
       
        n1 = np.cross(r21, r23)
        n2 = np.cross(r23, r43)
        
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        r23 = r23 / np.linalg.norm(r23)
        m = np.cross(n1, r23)
        x = np.dot(n1, n2)
        y = np.dot(m, n2)
        phi = atan2(y, x)

        #test angle
        d12 = np.subtract(atom_b, atom_c)
        d32 = np.subtract(atom_d, atom_c)
        d12 = d12 / np.linalg.norm(d12)
        d32 = d32 / np.linalg.norm(d32)
        cos_theta = np.dot(d12, d32)
        m = np.linalg.norm(np.cross(d12, d32))
        theta = atan2(m, cos_theta)

        if (fabs(theta - angle) < 0.001 and fabs(phi - torsion) < 0.001):
            # print("no case found", case, theta, angle, phi, torsion, atom_d)
            okinsert = True
        else:
            if (case < 4): case += 1
            else:
                break
    return atom_d

    ########################################################
def test_angle(atoms, angleID):
    ii, jj, kk = angles[angleID]
    d12 = np.subtract(atoms[ii], atoms[jj])
    d32 = np.subtract(atoms[kk], atoms[jj])
    d12 = d12 / np.linalg.norm(d12)
    d32 = d32 / np.linalg.norm(d32)
    cos_theta = np.dot(d12, d32)
    m = np.linalg.norm(np.cross(d12, d32))
    theta = np.arctan2(m, cos_theta)
    theta = np.degrees(theta)


    return theta

    ########################################################
def test_dihedral(atoms, dihedralID):

    ii, jj, kk, ll = dih[dihedralID]
    r21 = np.subtract(atoms[jj], atoms[ii])
    r23 = np.subtract(atoms[jj], atoms[kk])
    r43 = np.subtract(atoms[ll], atoms[kk])

    n1 = np.cross(r21, r23)
    n2 = np.cross(r23, r43)

    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    r23 = r23 / np.linalg.norm(r23)

    m = np.cross(n1, r23)
    x = np.dot(n1, n2)
    y = np.dot(m, n2)

    phi = atan2(y, x)

    return phi
    ########################################################
def new_config(CVsB, CVsA, CVsD):
    ang = CVsA[0]
    an = -1.0 * ang

    # Define rotation matrices
    R1 = np.array([[cos(an), -sin(an), 0.0],
                   [sin(an), cos(an), 0.0],
                   [0.0, 0.0, 1.0]])

    R2 = np.array([[1.0, 0.0, 0.0],
                   [0.0, cos(-math.pi / 4), -sin(-math.pi / 4)],
                   [0.0, sin(-math.pi / 4), cos(-math.pi / 4)]])

    R3 = np.array([[cos(-math.pi / 4), 0.0, sin(-math.pi / 4)],
                   [0.0, 1.0, 0.0],
                   [-sin(-math.pi / 4), 0.0, cos(-math.pi / 4)]])

    atoms = np.zeros((128, 3), float)

    # Initial vectors for atoms 1, 0, and 2
    vec10 = [1.0 / sqrt(2), 1.0 / sqrt(2), 0.0]
    vec20 = np.dot(R1, vec10)
    vec10 = np.dot(R2, vec10)
    vec20 = np.dot(R2, vec20)
    vec10 = np.dot(R3, vec10)
    vec20 = np.dot(R3, vec20)

    # Set positions for atoms 1, 0, and 2
    atoms[1] = [CVsB[0] * vec10[0], CVsB[0] * vec10[1], CVsB[0] * vec10[2]]
    atoms[0] = [0.0, 0.0, 0.0]  # Atom at index 0
    atoms[2] = [CVsB[1] * vec20[0], CVsB[1] * vec20[1], CVsB[1] * vec20[2]]

    # Iteratively place all other atoms based on dihedral angles
    for dd in range(len(dih)):
        dd1, dd2, dd3, dd4 = dih[dd]
        bondID, angleID = find_BA(dd1, dd2, dd3, dd4)
        
        coord = place_atom(atoms[dd1], atoms[dd2], atoms[dd3],
                           CVsA[angleID], CVsD[dd], CVsB[bondID])
        
        atoms[dd4] = coord
    
    return atoms

def rotationmatrix(coordref, coord):

    assert (coordref.shape[1] == 3)
    assert (coordref.shape == coord.shape)
    correlation_matrix = np.dot(np.transpose(coordref), coord)
    vv, ss, ww = np.linalg.svd(correlation_matrix)
    is_reflection = (np.linalg.det(vv) * np.linalg.det(ww)) < 0.0
        #if is_reflection:
        #print "is_reflection"
        #vv[-1,:] = -vv[-1,:]
        #ss[-1] = -ss[-1]
        #vv[:, -1] = -vv[:, -1]
    rotation = np.dot(vv, ww)

    confnew = []
    for i in range(len(coord)):
        xx = rotation[0][0] * coord[i][0] + rotation[0][1] * coord[i][
                1] + rotation[0][2] * coord[i][2]
        yy = rotation[1][0] * coord[i][0] + rotation[1][1] * coord[i][
                1] + rotation[1][2] * coord[i][2]
        zz = rotation[2][0] * coord[i][0] + rotation[2][1] * coord[i][
                1] + rotation[2][2] * coord[i][2]
        confnew.append((xx, yy, zz))

    return confnew


atoms = []
for i in range(4000):
    atom = new_config(bl[i], ba[i], dihedral[i])
    if i>0: atom = rotationmatrix(np.array(atoms[i-1]),atom)
    atoms.append(atom)

title = "test"
name = "C"
with open("file.xyz", 'w') as xyz_file:
    xyz_file.write("%d\n%s\n" % (128, title))
    
    for atom in atoms:
        for coord in atom: 
            xyz_file.write("{:4} {:11.6f} {:11.6f} {:11.6f}\n".format(
            name, coord[0], coord[1], coord[2]))



def radius_of_gyration(coords):
    center_of_mass = np.mean(coords, axis=0)
    rg_squared = np.mean(np.sum((coords - center_of_mass)**2, axis=1))
    return np.sqrt(rg_squared)

def autocorrelation_function(data):
    n = len(data)
    data_mean = np.mean(data)
    acf = np.correlate(data - data_mean, data - data_mean, mode='full') / (n * np.var(data))
    return acf[n-1:]

def plot_autocorrelation(acf, max_lag):
    plt.figure(figsize=(10, 5))
    plt.plot(acf[:max_lag], marker='o')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function of Radius of Gyration')
    plt.grid(True)
    plt.show()

rg_values = np.array([radius_of_gyration(atoms[i]) for i in range(4000)])
acf_rg = autocorrelation_function(rg_values[:])
plot_autocorrelation(acf_rg, max_lag=500)

# fe, bins = np.histogram(rg_values, bins=100, density=True)
# fe = fe+1e-12
# free_energy = -np.log(fe)

# plt.figure(figsize=(12, 8))
# plt.plot(bins[:-1], free_energy, linestyle='-')
# plt.xlabel('Latent dim 1')
# plt.ylabel('Free Energy')
# plt.title('Free Energy along Latent Dimension 1')
# plt.grid(True)
# plt.show()
# def auto_correlation_function(all_predicted_states):
#     # Remove mean to calculate correlation properly
#     data_mean = np.mean(all_predicted_states, axis=0)
#     data_zero_mean = all_predicted_states - data_mean

#     # Compute auto-correlation using FFT for efficiency
#     acf = correlate(data_zero_mean, data_zero_mean, mode='full', method='fft')

#     # Compute normalization factor
#     # variances = np.var(data_zero_mean, axis=0)
    
#     # # Reshape variances to match the shape of acf for broadcasting
#     # norm_factor = variances.reshape(1, -1) * np.arange(len(all_predicted_states), 0, -1)[:, np.newaxis]

#     # # Normalize the auto-correlation function
#     # acf /= norm_factor # Transpose norm_factor to align with acf shape

#     return acf

# data_mean = np.mean(all_predicted_states, axis=0)
# data_zero_mean = all_predicted_states - data_mean
# variances = np.var(data_zero_mean, axis=0)
# norm_factor = variances[:, np.newaxis] * np.arange(len(all_predicted_states), 0, -1)[np.newaxis, :]
# bond_angles_acf = auto_correlation_function(data_zero_mean) / norm_factor

# time_lags = np.arange(len(bond_angles_acf))
# plt.plot(time_lags, bond_angles_acf)
# plt.xlabel('Time lag')
# plt.ylabel('Auto-correlation')
# plt.show()

# print(all_predicted_states)
# for batch_data, targets in val_dataloader:
#     batch_data = batch_data.to(device).float()
#     targets = targets.to(device).float()

#     with torch.no_grad():
#         latent_states_ae, _, _, _,_ ,_= autoencoder(batch_data)
#         pi_lstm, mu_lstm, sigma_lstm = mdn_lstm(latent_states_ae)
#         latent_states_pred, _, _, _, _, _= autoencoder(targets)
#     K, T, N = latent_states_ae.size()
#     D = 1
#     latent_states_pred = latent_states_pred.view(K * T, N * D)
#     mu_lstm = mu_lstm.view(K * T, 2, 4)
#     sigma_lstm = sigma_lstm.view(K * T, 2, 4)
#     pi_lstm = pi_lstm.view(K * T, 2, 4)
#     all_latent_states.append(latent_states_ae.cpu().numpy())
#     loss_lstm = -mdn_lstm.mdn.log_prob(latent_states_pred, pi_lstm, mu_lstm, sigma_lstm).mean()
#     val_loss_lstm += loss_lstm.item()

# all_latent_states = np.concatenate(all_latent_states, axis=0) 
# val_loss_lstm /= len(val_dataloader)
# print(f"Validation Loss: {val_loss_lstm:.4f}")

# all_latent_states = all_latent_states.reshape(-1, 2)
# num_test_sequences = len(test_sequences)
latent_trajectories = all_latent_states_test.reshape(-1, sequence_length, latent_dim)
all_latent_points = latent_trajectories.reshape(-1, 2)

pyemma.plots.plot_free_energy(all_latent_points[:,0], all_latent_points[:,1])
plt.show()
def compute_2d_kde(bottleneck_values):
        kde = gaussian_kde(bottleneck_values.T)
        x_min, y_min = bottleneck_values.min(axis=0)
        x_max, y_max = bottleneck_values.max(axis=0)
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
        kde_values = kde(grid_coords).reshape(100, 100)
        return x_grid, y_grid, kde_values

def compute_1d_kde(bottleneck_values, num_points=100):
    kde = gaussian_kde(bottleneck_values)
    x_min = bottleneck_values.min()
    x_max = bottleneck_values.max()
    x_grid = np.linspace(x_min, x_max, num_points)
    kde_values = kde(x_grid)
    return x_grid, kde_values

def compute_free_energy(pdf, temperature=1):
    k_B = 1  # Assume k_B = 1 for simplicity
    free_energy = -k_B * temperature * np.log(pdf + 1e-10)  # Add a small constant to avoid log(0)
    return free_energy

x_grid, kde_values = compute_1d_kde(rg_values)
free_energy = compute_free_energy(kde_values)

plt.figure(figsize=(10, 6))
plt.plot(x_grid, free_energy, label='Free Energy', color='blue')
plt.xlabel('Bottleneck Dimension')
plt.ylabel('Free Energy (-log(PDF))')
plt.title('Free Energy Distribution')
plt.legend()
plt.grid(True)
plt.show()

x_grid, y_grid, kde_values = compute_2d_kde(all_latent_points)
free_energy = compute_free_energy(kde_values)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, free_energy, cmap='viridis')
ax.set_xlabel('Bottleneck Dimension 1')
ax.set_ylabel('Bottleneck Dimension 2')
ax.set_zlabel('Free Energy')
ax.set_title(f'Free Energy Distribution')
plt.show()

# # Plot Free Energy Distribution
# plt.figure(figsize=(8, 6))
# plt.contourf(xx, yy, free_energy, levels=50, cmap='viridis')
# plt.colorbar(label='Free Energy')
# plt.title('Free Energy Distribution in the Latent Space')
# plt.xlabel('Latent Dimension 1')
# plt.ylabel('Latent Dimension 2')
# plt.grid(True)
# plt.show()

# # Plot Free Energy
# plt.subplot(1, 2, 2)
# plt.scatter(all_latent_states[:, 0], all_latent_states[:, 1], c=free_energy, cmap='viridis')
# plt.colorbar(label='Free Energy')
# plt.title('Free energy of trajectories sampled from LED')
# plt.xlabel('Latent Dimension 1')
# plt.ylabel('Latent Dimension 2')
# plt.grid(True)

# plt.tight_layout()
# plt.show()
