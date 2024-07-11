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
import pandas as pd

def load_full_dataset(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # cur.execute("SELECT * from data")
    # coordinates = cur.fetchall()
    # coordinates = np.array(coordinates).reshape(200001, 384)[1:]
    # Load coordinates
    cur.execute("SELECT * FROM bond_lengths_1")
    bond_lengths= cur.fetchall()
    bond_lengths = np.array(bond_lengths).reshape(-1, 92)[1:]  

    cur.execute("SELECT * FROM bond_angles_1")
    bond_angles = cur.fetchall()
    bond_angles = np.array(bond_angles).reshape(-1, 92)[1:]

    cur.execute("SELECT * FROM dihedral_angles_1")
    dihedral_angles = cur.fetchall()
    dihedral_angles = np.array(dihedral_angles).reshape(-1, 92)[1:]
    
    # cur.execute("SELECT * FROM pairwise_distances")
    # pairwise = cur.fetchall()
    # pairwise = np.array(pairwise).reshape(200001, 1830)[1:]

    conn.close()

    features = np.hstack([bond_lengths, bond_angles, dihedral_angles])
    
    return features

class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_components, multivariate=False, cov_scaling_factor=0.8):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.softplus = nn.Softplus()
        self.multivariate = multivariate
        self.cov_scaling_factor = cov_scaling_factor
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.fc_pi = nn.Linear(hidden_dim,828)
        self.fc_mu = nn.Linear(hidden_dim, 552)
        self.fc_sigma = nn.Linear(hidden_dim, 552)
        self.fc_loc = nn.Linear(hidden_dim, 276)
        self.fc_concentration = nn.Linear(hidden_dim, 276)
        self.dims_bonds_ = list(np.arange(0, 92, 1))
        self.dims_angles_ = list(
            np.arange(92, 184, 1))
        self.scaling_dims = self.dims_bonds_ + self.dims_angles_
        self.dims_total = 276

    def forward(self, x):
        fc = self.fc1(x)
        fc_1 = fc
        pi_ = self.fc_pi(fc_1)
        pi_ = pi_.view(-1,276,3)
        pi = torch.softmax(pi_, dim=-1)
        mu = self.fc_mu(fc)
        mu = mu.view(-1, 184, 3)
        sigma = torch.sigmoid(self.fc_sigma(fc))*self.cov_scaling_factor
        sigma = sigma.view(-1, 184, 3)
        loc = self.fc_loc(fc)
        loc = loc.view(-1, 92, 3)
        concentration = self.softplus(self.fc_concentration(fc))
        concentration = concentration.view(-1, 92, 3)
        
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
        
    def scale_targets(self, targets):
        # Flatten the 3D tensor to 2D
        targets_cpu = targets.cpu()
        targets_np = targets_cpu.numpy()
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
        m_normal = torch.distributions.Normal(loc=mu, scale=sigma)  # Unsqueezing mu and sigma
        if loca is not None and conc is not None:
            m_angles = torch.distributions.VonMises(loc = loca, concentration = conc)
            y = y[:,:, None]
            y = self.repeatAlongDim(y, axis = 2, repeat_times = 3)
            y = self.scale_targets(y)
            y = self.processTargets(y)
            log_prob_normal = m_normal.log_prob(y[:, :184])
            log_prob_angles = m_angles.log_prob(y[:, 184:])
            log_prob = torch.cat((log_prob_normal, log_prob_angles), dim=1)
        else:
            y = y[:,:, None]
            y = self.repeatAlongDim(y, axis = 2, repeat_times = 3)
            y = self.processTargets(y)
            log_prob_normal = m_normal.log_prob(y[:, :125])
            log_prob = torch.cat(log_prob_normal, dim = 1)
        pi = torch.clamp(pi, min=1e-15)
        log_pi = torch.log(pi)
        sum_logs = log_prob + log_pi
        sum_logs_max = torch.max(sum_logs, 2)[0]
        sum_logs_max = sum_logs_max[:, :, None]
        loss = torch.exp(sum_logs - sum_logs_max)
        loss = torch.sum(loss, dim=2)
        
        return torch.mean(loss)

class MDNO(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_components, multivariate=False, cov_scaling_factor=0.8):
        super(MDNO, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.softplus = nn.Softplus()
        self.multivariate = multivariate
        self.cov_scaling_factor = cov_scaling_factor
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.fc_pi = nn.Linear(hidden_dim, 1152)
        self.fc_mu = nn.Linear(hidden_dim, 1152)
        self.fc_sigma = nn.Linear(hidden_dim, 1152)


    def forward(self, x):
        h = (self.fc1(x))
        pi_ = self.fc_pi(h)
        pi_ = pi_.view(-1,384,3)
        pi = torch.softmax(pi_, dim=-1)
        mu = self.fc_mu(h)
        mu = mu.view(-1, 384, 3)
        sigma = torch.sigmoid(self.fc_sigma(h))*self.cov_scaling_factor
        sigma = sigma.view(-1, 384, 3)
        
        return pi, mu, sigma

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
        y = self.repeatAlongDim(y, axis = 2, repeat_times = 3)
        y = self.scale_targets(y)
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
        # self.fc4 = nn.Sequential(nn.Linear(500, 500), nn.Tanh())
        # self.fc5 = nn.Sequential(nn.Linear(500, 500), nn.Tanh())
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
        y = self.scale_targets(y)
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
        # self.fc4 = nn.Linear(500, 500)
        # self.fc5 = nn.Linear(500, 500)
        self.fc6 = nn.Linear(500, latent_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # x = torch.tanh(self.fc4(x))
        # x = torch.tanh(self.fc5(x))
        x = self.fc6(x)
        return x.float()

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_components, input_dim, multivariate=False, cov_scaling_factor=0.8, mdn_bool=True):
        super(Decoder, self).__init__()
        self.mdn_bool = mdn_bool
        self.fc1 = nn.Linear(latent_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        # self.fc4 = nn.Linear(500, 500)
        # self.fc5 = nn.Linear(500, 500)
        self.tanh = nn.Tanh()
        if mdn_bool is True:
            self.mdn = MDN(500, hidden_dim, num_components, multivariate, cov_scaling_factor)
        else:
            self.fc6 = nn.Linear(500, input_dim, bias = True)
    def forward(self, x, mdn_bool = True):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        # x = self.tanh(self.fc4(x))
        # x = self.tanh(self.fc5(x))
        if self.mdn_bool is True:
            pi, mu, sigma, loc, conc = self.mdn(x)
            return pi, mu, sigma, loc, conc
        else:
            x = self.tanh(self.fc6(x))
            return x

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_components, mdn_bool=True):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, num_components, input_dim, mdn_bool)
        self.mdn_bool = mdn_bool

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
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = None
        if idx + 1 != len(self.data):
            target = self.data[idx + 1]
        sample = torch.tensor(sample, dtype=torch.float32).to(self.device)
        if target is not None:
            target = torch.tensor(target, dtype=torch.float32).to(self.device)
        return sample, target

def collate_fn(batch):
    samples, targets = zip(*batch)
    samples = torch.stack(samples)
    targets = [t if t is not None else torch.zeros_like(samples[0]) for t in targets]
    targets = torch.stack(targets)
    return samples, targets

db_path = "/home/smart/Documents/IISC/sqlite_2.db"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
features = load_full_dataset(db_path)

feature_range = (0.0001, 0.9999)
data = MinMaxScaler(feature_range=feature_range)
input_data = data.fit_transform(features)
sequence_length = 4000
num_sequences = len(features) // sequence_length
sequences = np.split(features[:num_sequences * sequence_length], num_sequences)

num_train_sequences = 96
num_val_sequences = 96
num_test_sequences = 154
train_sequences = sequences[167:]
val_sequences = sequences[num_train_sequences:num_train_sequences + num_val_sequences]
test_sequences = sequences[166:167]
data = np.array(sequences)
train_data = np.array(train_sequences)
val_data = np.array(val_sequences)
test_data = np.array(test_sequences)

batch_size = 8

train_dataset = MyDataset(train_data, device=device)
val_dataset = MyDataset(val_data, device=device)
test_dataset = MyDataset(test_data, device=device)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Define model parameters
output_dim_normal = 92 + 92
output_dim_von_mises = 92
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

for epoch in range(pretrain_epochs):
    autoencoder.train()
    pretrain_loss = 0.0
    for batch_data, _ in train_dataloader:
        batch_data = batch_data.to(device).float()
        
        _, pi_ae, mu_ae, sigma_ae, loc_ae, conc_ae = autoencoder(batch_data)
        
        K, T, N = batch_data.size()
        D = 1
        batch_data = batch_data.view(K * T, N * D)
        mu_ae = mu_ae.view(K * T, 184, 3)
        sigma_ae = sigma_ae.view(K * T, 184, 3)
        loc_ae = loc_ae.view(K * T, 92, 3)
        conc_ae = conc_ae.view(K * T, 92, 3)
        pi_ae = pi_ae.view(K * T, 276, 3)
        
        pretrain_loss = -autoencoder.decoder.mdn.log_prob(batch_data, pi_ae, mu_ae, sigma_ae, loc_ae, conc_ae).mean()
        
        pretrain_optimizer.zero_grad()
        pretrain_loss.backward()
        pretrain_optimizer.step()
        
        pretrain_loss += pretrain_loss.item()
    
    pretrain_loss /= len(train_dataloader)
    print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs}, Loss: {pretrain_loss:.4f}")

for param in autoencoder.decoder.mdn.parameters():
    param.requires_grad = False

for name, param in autoencoder.named_parameters():
    if 'decoder.mdn' not in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer_ae = optim.Adam(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=learning_rate)

num_epochs_ae = 15
for epoch in range(num_epochs_ae):
    autoencoder.train()
    train_loss_ae = 0.0
    
    for batch_data, targets in train_dataloader:
        batch_data = batch_data.to(device).float()
        targets = targets.to(device).float()

        latent_states_ae, pi_ae, mu_ae, sigma_ae, loc_ae, conc_ae = autoencoder(batch_data)
        
        K, T, N = batch_data.size()
        D = 1
        targets = targets.view(K * T, N * D)
        mu_ae = mu_ae.view(K * T, 184, 3)
        sigma_ae = sigma_ae.view(K * T, 184, 3)
        loc_ae = loc_ae.view(K * T, 92, 3)
        conc_ae = conc_ae.view(K * T, 92, 3)
        pi_ae = pi_ae.view(K * T, 276, 3)
        
        loss_ae = -autoencoder.decoder.mdn.log_prob(targets, pi_ae, mu_ae, sigma_ae, loc_ae, conc_ae).mean()
        
        optimizer_ae.zero_grad()
        loss_ae.backward()
        optimizer_ae.step()
        
        train_loss_ae += loss_ae.item()
    
    train_loss_ae /= len(train_dataloader)
    print(f"AE Epoch {epoch+1}/{num_epochs_ae}, Loss: {train_loss_ae:.4f}")

for param in autoencoder.parameters():
    param.requires_grad = False

optimizer_lstm = optim.Adam(mdn_lstm.parameters(), lr=0.001)

num_epochs_lstm = 11
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

# torch.save(autoencoder, 'autoencoder.pth')
# torch.save(mdn_lstm, 'mdn_lstm.pth')
# autoencoder = torch.load('autoencoder.pth')
# mdn_lstm = torch.load('mdn_lstm.pth')

# autoencoder.eval()
# mdn_lstm.eval()
test_loss_lstm = 0.0
all_latent_states_test = []
T_mu = 10
T_m = 20000

for batch_data, _ in test_dataloader:
    batch_data = batch_data.to(device).float()

    with torch.no_grad():
        latent_states_ae, _, _, _, _, _ = autoencoder(batch_data[:, :T_mu, :])
        hidden_state = None
        for t in range(T_mu):
            _, _, _, hidden_state = mdn_lstm(latent_states_ae[:, t, :].unsqueeze(1), hidden_state)
    predicted_latent_states = []
    input_state = latent_states_ae[:, -1, :].unsqueeze(1)
    for t in range(T_m):
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
all_latent_states_test = all_latent_states_test.reshape(-1,2)

# Decode the predicted latent states to obtain high-dimensional state configurations
all_predicted_states = []
for latent_states in all_latent_states_test:
    latent_states_tensor = torch.tensor(latent_states).to(device).float()
    with torch.no_grad():
        decoded_states = autoencoder.decoder(latent_states_tensor, mdn_bool = False)
    all_predicted_states.append(decoded_states[0].cpu().numpy())

all_predicted_states = np.concatenate(all_predicted_states, axis=0)


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

x_data = all_latent_points[:,1]
fe, bins = np.histogram(x_data, bins=100, density=True)
fe = fe+1e-12
free_energy = -np.log(fe)

plt.figure(figsize=(12, 8))
plt.plot(bins[:-1], free_energy, linestyle='-')
plt.xlabel('Latent dim 1')
plt.ylabel('Free Energy')
plt.title('Free Energy along Latent Dimension 1')
plt.grid(True)
plt.show()

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

def compute_free_energy(pdf, temperature=1):
    k_B = 1  # Assume k_B = 1 for simplicity
    free_energy = -k_B * temperature * np.log(pdf + 1e-10)  # Add a small constant to avoid log(0)
    return free_energy

x_grid, y_grid, kde_values = compute_2d_kde(all_latent_points)
free_energy = compute_free_energy(kde_values)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, free_energy, cmap='viridis')
ax.set_xlabel('Bottleneck Dimension 1')
ax.set_ylabel('Bottleneck Dimension 2')
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
