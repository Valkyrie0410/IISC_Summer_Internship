import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from torch.distributions import Categorical, Normal, MixtureSameFamily
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_full_dataset(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Load coordinates
    cur.execute("SELECT * FROM bond_lengths")
    bond_lengths= cur.fetchall()
    bond_lengths = np.array(bond_lengths).reshape(100001, 63)[1:]  

    cur.execute("SELECT * FROM bond_angles")
    bond_angles = cur.fetchall()
    bond_angles = np.array(bond_angles).reshape(100001, 62)[1:]

    cur.execute("SELECT * FROM dihedral_angles")
    dihedral_angles = cur.fetchall()
    dihedral_angles = np.array(dihedral_angles).reshape(100001, 61)[1:]
    
    conn.close()

    features = np.hstack([bond_lengths, bond_angles, dihedral_angles])
    
    return features

class MyDataset(Dataset):
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if torch.is_tensor(sample):
            sample = torch.tensor(sample, dtype=torch.float32).to(self.device)
        return sample


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
        self.fc_pi = nn.Linear(hidden_dim,930)
        self.fc_mu = nn.Linear(hidden_dim, 625)
        self.fc_sigma = nn.Linear(hidden_dim, 625)
        self.fc_loc = nn.Linear(hidden_dim, 305)
        self.fc_concentration = nn.Linear(hidden_dim, 305)
        self.dims_bonds_ = list(np.arange(0, 63, 1))
        self.dims_angles_ = list(
            np.arange(63, 63+ 62, 1))
        self.scaling_dims = self.dims_bonds_ + self.dims_angles_
        self.dims_total = 186

    def forward(self, x):
        fc = self.fc1(x)
        fc_1 = fc
        pi_ = self.fc_pi(fc_1)
        pi_ = pi_.view(-1,186,5)
        pi = torch.softmax(pi_, dim=-1)
        mu = self.fc_mu(fc)
        mu = mu.view(-1, 125, 5)
        sigma = torch.sigmoid(self.fc_sigma(fc))*self.cov_scaling_factor
        sigma = sigma.view(-1, 125, 5)
        loc = self.fc_loc(fc)
        loc = loc.view(-1, 61, 5)
        concentration = self.softplus(self.fc_concentration(fc))
        concentration = concentration.view(-1, 61, 5)
        
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
            y = self.repeatAlongDim(y, axis = 2, repeat_times = 5)
            y = self.scale_targets(y)
            y = self.processTargets(y)
            log_prob_normal = m_normal.log_prob(y[:, :125])
            log_prob_angles = m_angles.log_prob(y[:, 125:])
            log_prob = torch.cat((log_prob_normal, log_prob_angles), dim=1)
        else:
            y = y[:,:, None]
            y = self.repeatAlongDim(y, axis = 2, repeat_times = 5)
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

class MDNL(nn.Module):
    def __init__(self, input_dim = 40, hidden_dim = 20, num_components = 4, multivariate=False, cov_scaling_factor=0.2):
        super(MDNL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.softplus = nn.Softplus()
        self.multivariate = multivariate
        self.cov_scaling_factor = cov_scaling_factor
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
        h = (self.fc4(h))
        h = self.fc5(h)
        pi_ = self.fc_pi(h)
        pi_ = pi_.view(-1,2,4)
        pi = torch.softmax(pi_, dim=-1)
        mu = self.fc_mu(h)
        mu = mu.view(-1, 2, 4)
        sigma = torch.sigmoid(self.fc_sigma(h))*self.cov_scaling_factor
        sigma = sigma.view(-1, 2, 4)
        
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
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 500)
        self.fc6 = nn.Linear(500, latent_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)
        return x.float()

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_components, multivariate=False, cov_scaling_factor=0.8):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 500)
        self.tanh = nn.Tanh()
        self.mdn = MDN(500, hidden_dim, num_components, multivariate, cov_scaling_factor)
    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
        pi, mu, sigma, loc, conc = self.mdn(x)
        return pi, mu, sigma, loc, conc

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_components):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, num_components)

    def forward(self, x):
        x = x.to(self.encoder.fc1.weight.dtype)
        latent_states = self.encoder(x)
        z = latent_states
        pi, mu, sigma, loc, conc = self.decoder(z)
        return latent_states, pi, mu, sigma, loc, conc

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

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, 32, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, 32, self.hidden_dim).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        # lstm_out = lstm_out[:, -1, :]
        # print(lstm_out.shape)
        pi, mu, sigma = self.mdn(lstm_out)  # Select the output of the last time step
        return pi, mu, sigma

db_path = "/home/smart/Documents/IISC/sqlite_1.db"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = load_full_dataset(db_path)

feature_range = (0.0001, 0.9999)
data = MinMaxScaler(feature_range=feature_range)
input_data = data.fit_transform(features)
sequence_length = 40  # Number of snapshots per sequence
num_sequences = len(features) // sequence_length
sequences = np.split(features[:num_sequences * sequence_length], num_sequences)

num_train_sequences = 96
num_val_sequences = 96
num_test_sequences = 96
train_sequences = sequences[:num_train_sequences]
val_sequences = sequences[num_train_sequences:num_train_sequences + num_val_sequences]
test_sequences = sequences[:]

train_data = np.array(train_sequences)
val_data = np.array(val_sequences)
test_data = np.array(test_sequences)
# train_val_data, test_data = train_test_split(input_data, test_size=(1 - train_val_split), random_state=42)
# train_data, val_data = train_test_split(train_val_data, test_size=(1 - train_split), random_state=42)

delta_t_values = np.arange(1, 11)
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = MyDataset(train_data, device=device)
val_dataset = MyDataset(val_data, device=device)
test_dataset = MyDataset(test_data, device=device)
    
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

output_dim_normal = 63+62
output_dim_von_mishes = 61
input_dim = features.shape[1]  # Example dimension of the state space
latent_dim = 2  # Latent space dimension
hidden_dim = 50
num_components = 5  # Number of MDN components
learning_rate = 0.003

timesteps = 40
# Create models
autoencoder = Autoencoder(input_dim, latent_dim, hidden_dim, num_components).to(device)
mdn_lstm = MDN_LSTM().to(device)

# Optimizers
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=1e-3)
optimizer_lstm = optim.Adam(mdn_lstm.parameters(), lr=1e-3)

pretrain_epochs = 4
num_epochs_ae = 25
num_epochs_lstm = 25


#PRETRAINING PHASE

for param in autoencoder.decoder.mdn.parameters():
    param.requires_grad = True

pretrain_optimizer = torch.optim.Adam(autoencoder.decoder.mdn.parameters(), lr=0.003)

for epoch in range(pretrain_epochs):
    autoencoder.train()
    pretrain_loss = 0.0
    for batch_data in train_dataloader:
        batch_data = batch_data.to(device).float()
        
        # Forward pass through autoencoder
        _, pi_ae, mu_ae, sigma_ae, loc_ae, conc_ae = autoencoder(batch_data)
        
        # Reshape tensors for loss calculation
        K, T, N = batch_data.size()
        D = 1
        batch_data = batch_data.view(K * T, N * D)
        mu_ae = mu_ae.view(K * T, 125, 5)
        sigma_ae = sigma_ae.view(K * T, 125, 5)
        loc_ae = loc_ae.view(K * T, 61, 5)
        conc_ae = conc_ae.view(K * T, 61, 5)
        pi_ae = pi_ae.view(K * T, 186, 5)
        
        # Calculate loss for pretraining
        pretrain_loss = -autoencoder.decoder.mdn.log_prob(batch_data, pi_ae, mu_ae, sigma_ae, loc_ae, conc_ae).mean()
        
        # Backward pass and optimization
        pretrain_optimizer.zero_grad()
        pretrain_loss.backward()
        pretrain_optimizer.step()
        
        pretrain_loss += pretrain_loss.item()
    
    # Calculate average pretraining loss for the epoch
    pretrain_loss /= len(train_dataloader)
    print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs}, Loss: {pretrain_loss:.4f}")

# Fix kernels μk and σk of the MDN-AE
for param in autoencoder.decoder.mdn.parameters():
    param.requires_grad = False



#Now we will train MDN-AE while keeping kernels fixed
for name, param in autoencoder.named_parameters():
    if 'decoder.mdn' not in name:  # Keep mdn parameters fixed
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer_ae = torch.optim.Adam(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=learning_rate)

for epoch in range(num_epochs_ae):
    autoencoder.train()
    train_loss_ae = 0.0
    
    for batch_data in train_dataloader:
        batch_data = batch_data.to(device).float()
        
        # Forward pass through autoencoder
        latent_states_ae, pi_ae, mu_ae, sigma_ae, loc_ae, conc_ae = autoencoder(batch_data)
        
        # Reshape tensors for loss calculation
        K, T, N = batch_data.size()
        D = 1
        batch_data = batch_data.view(K * T, N * D)
        mu_ae = mu_ae.view(K * T, 125, 5)
        sigma_ae = sigma_ae.view(K * T, 125, 5)
        loc_ae = loc_ae.view(K * T, 61, 5)
        conc_ae = conc_ae.view(K * T, 61, 5)
        pi_ae = pi_ae.view(K * T, 186, 5)
        
        # Calculate loss for autoencoder
        loss_ae = -autoencoder.decoder.mdn.log_prob(batch_data, pi_ae, mu_ae, sigma_ae, loc_ae, conc_ae).mean()
        
        # Backward pass and optimization
        optimizer_ae.zero_grad()
        loss_ae.backward()
        optimizer_ae.step()
        
        train_loss_ae += loss_ae.item()
    
    # Calculate average training loss for the epoch
    train_loss_ae /= len(train_dataloader)
    print(f"AE Epoch {epoch+1}/{num_epochs_ae}, Loss: {train_loss_ae:.4f}")

#now we will keep all parameters of autoencoder fixed and run MDN-LSTM model
for param in autoencoder.parameters():
    param.requires_grad = False

optimizer_lstm = torch.optim.Adam(mdn_lstm.parameters(), lr=learning_rate)

for epoch in range(num_epochs_lstm):
    autoencoder.eval()  # Set MDN-AE to evaluation mode
    mdn_lstm.train()    # Set MDN-LSTM to training mode
    train_loss_lstm = 0.0

    for batch_data in train_dataloader:
        batch_data = batch_data.to(device).float()

        # Forward pass through autoencoder (with frozen weights)
        with torch.no_grad():
            latent_states_ae, _, _, _, _, _ = autoencoder(batch_data)
        
        # Forward pass through MDN-LSTM
        pi_lstm, mu_lstm, sigma_lstm = mdn_lstm(latent_states_ae)
        
        # Reshape tensors for loss calculation
        K, T, N = latent_states_ae.size()
        D = 1
        latent_states_ae = latent_states_ae.view(K * T, N * D)
        mu_lstm = mu_lstm.view(K * T, 2, 4)
        sigma_lstm = sigma_lstm.view(K * T, 2, 4)
        pi_lstm = pi_lstm.view(K * T, 2, 4)
        
        # Calculate loss for MDN-LSTM
        loss_lstm = -mdn_lstm.mdn.log_prob(latent_states_ae, pi_lstm, mu_lstm, sigma_lstm).mean()
        
        # Backward pass and optimization for MDN-LSTM
        optimizer_lstm.zero_grad()
        loss_lstm.backward()
        optimizer_lstm.step()
        
        train_loss_lstm += loss_lstm.item()

    # Calculate average training loss for the epoch
    train_loss_lstm /= len(train_dataloader)
    print(f"LSTM Epoch {epoch+1}/{num_epochs_lstm}, Loss: {train_loss_lstm:.4f}")
# autoencoder.eval()
# mdn_lstm.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for batch_data in test_dataloader:
#         # Move batch_data to device if necessary
#         batch_data = batch_data.to(device)

#         # Forward pass through autoencoder
#         pi, mu, sigma = autoencoder(batch_data)
#         loss_ae = -autoencoder.decoder.mdn.log_prob(batch_data, pi, mu, sigma).mean()

#         # Forward pass through LSTM
#         latent_states = autoencoder.encoder(states).detach()
#         latent_sequences = torch.stack([latent_states[i:i + timesteps] for i in range(len(latent_states) - timesteps)])
#         next_latent_states = latent_states[timesteps:]
#         pi_lstm, mu_lstm, sigma_lstm = mdn_lstm(latent_sequences)
#         loss_lstm = -mdn_lstm.mdn.log_prob(next_latent_states, pi_lstm, mu_lstm, sigma_lstm).mean()

#         test_loss += loss_ae.item() + loss_lstm.item()

# # Calculate average testing loss
# test_loss /= len(test_dataloader)
# print(f'Test Loss: {test_loss:.4f}')