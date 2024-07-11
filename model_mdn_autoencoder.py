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
import torch.nn.functional as F

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
    features = np.hstack([bond_lengths, bond_angles, dihedral_angles])
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

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 500)
        self.fc6 = nn.Linear(8, latent_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # x = torch.selu(self.fc4(x))
        # x = torch.selu(self.fc5(x))
        x = (self.fc6(x))
        return x.float()

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_components, input_dim, multivariate=False, cov_scaling_factor=0.8, mdn_bool=True):
        super(Decoder, self).__init__()
        self.mdn_bool = mdn_bool
        self.fc1 = nn.Linear(latent_dim, 8)
        self.fc2 = nn.Linear(8, 64)
        self.fc3 = nn.Linear(64, 256)
        self.fc6 = nn.Linear(256, input_dim)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 500)
        self.tanh = nn.Tanh()
        self.mdn = MDN(256, hidden_dim, num_components, multivariate, cov_scaling_factor)

    def forward(self, x, mdn_bool = True):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        # x = self.tanh(self.fc4(x))
        # x = self.tanh(self.fc5(x))
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
target = sequences[:]
train_sequences = sequences[:360]
val_sequences = sequences[360:450]
test_sequences = sequences[450:]
data = np.array(sequences)

bond_lengths = data[:450, :, :127]
bond_angles = data[:450, :, 127:127+188]
dihedral_angles = data[:450, :, 127+188:]

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
target = np.array(target)
scaled_val_data = scaler.scaleData(val_data, single_sequence=False)
scaled_target = scaler.scaleData(target, single_sequence=False)
batch_size = 8

train_dataset = MyDataset(scaled_train_data, scaled_target, device=device)
val_dataset = MyDataset(scaled_val_data, scaled_target[360:], device=device)
test_dataset = MyDataset(scaled_test_data, scaled_val_data, device=device)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Define model parameters
# Define model parameters
output_dim_normal = 315
output_dim_von_mises = 125
input_dim = features.shape[1]
latent_dim = 2
hidden_dim = 50
num_components = 4
learning_rate = 0.001

# Create models
autoencoder = Autoencoder(input_dim, latent_dim, hidden_dim, num_components, mdn_bool=True).to(device)

for param in autoencoder.parameters():
    param.requires_grad = False
for param in autoencoder.decoder.mdn.parameters():
    param.requires_grad = True

pretrain_optimizer = optim.Adam(autoencoder.decoder.mdn.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(pretrain_optimizer, step_size=10, gamma=0.1)

pretrain_epochs = 20

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

for param in autoencoder.parameters():
    if param.requires_grad is False:
        param.requires_grad = True

optimizer_ae = optim.Adam(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer_ae, step_size=10, gamma=0.1)

num_epochs_ae = 100

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
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
        optimizer_ae.step()
        
        train_loss_ae += loss_ae.item()

    autoencoder.eval()
    val_loss_ae = 0.0
    with torch.no_grad():
        for batch_data_val, _ in val_dataloader:
            batch_data_val = batch_data_val.to(device).float()
            _, pi_ae_val, mu_ae_val, sigma_ae_val, loc_ae_val, conc_ae_val = autoencoder(batch_data_val)
            
            K, T, N = batch_data_val.size()
            D = 1
            batch_data_val = batch_data_val.view(K * T, N * D)
            mu_ae_val = mu_ae_val.view(K * T, 315, 3)
            sigma_ae_val = sigma_ae_val.view(K * T, 315, 3)
            loc_ae_val = loc_ae_val.view(K * T, 125, 3)
            conc_ae_val = conc_ae_val.view(K * T, 125, 3)
            pi_ae_val = pi_ae_val.view(K * T, 440, 3)

            loss_val = -autoencoder.decoder.mdn.log_prob(batch_data_val, pi_ae_val, mu_ae_val, sigma_ae_val, loc_ae_val, conc_ae_val).mean()
            
            val_loss_ae += loss_val.item()

    val_loss_ae /= len(val_dataloader)
    train_loss_ae /= len(train_dataloader)

    scheduler.step()

    print(f"AE Epoch {epoch+1}/{num_epochs_ae}, Train Loss: {train_loss_ae:.4f}, Val Loss: {val_loss_ae:.4f}")



