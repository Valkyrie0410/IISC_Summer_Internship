import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sqlite3
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde

def load_full_dataset(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Load coordinates
    cur.execute("SELECT distance FROM pairwise_distances")
    distances = cur.fetchall()
    distances = np.array(distances).reshape(100000, 8252) # Adjust shape if needed
    conn.close()
    
    return distances 

def nn_distance(coordinates):
    distances = []
    for i in range(len(coordinates)):
        distance = np.linalg.norm(coordinates[i, :3] - coordinates[i, 189:192])
        distances.append(distance)
    distances = np.array(distances)
    distances = distances.reshape(200000, 1)
    return distances


def radius_of_gyration(coordinates):
    coordinates_array = np.array(coordinates).reshape(200000, 64, 3)
    radius_of_gyrations = []
    for i in range(len(coordinates_array)):
        center_of_mass = np.mean(coordinates_array[i], axis=0)       
        distances_squared = np.sum((coordinates_array[i] - center_of_mass)**2, axis=1)        
        average_distance = np.mean(distances_squared)
        radius_of_gyration = np.sqrt(average_distance) 
        radius_of_gyrations.append(radius_of_gyration)

    radius_of_gyrations = np.array(radius_of_gyrations)

    radius_of_gyrations = radius_of_gyrations.reshape(200000, 1)
    
    return radius_of_gyrations

class TimeDelayedDataset(Dataset):
    def __init__(self, data, mean, std, delta_t, device):
        self.data = data
        self.mean = mean
        self.std = std
        self.delta_t = delta_t
        self.device = device
    
    def __len__(self):
        return len(self.data) - self.delta_t
    
    def __getitem__(self, idx):
        features = self.data[idx]
        target = self.data[idx + self.delta_t]
        
        normalized_features = (features - self.mean) / self.std
        normalized_target = (target - self.mean) / self.std
        
        input_data = torch.tensor(normalized_features, dtype=torch.float32).to(self.device)
        target_data = torch.tensor(normalized_target, dtype=torch.float32).to(self.device)
        return input_data, target_data

class LinearEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(LinearEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, bottleneck_dim)
        init.uniform_(self.linear1.weight, -0.005, 0.005)

    def forward(self, x):
        x = self.linear1(x)
        return x

class StochasticDecoder(nn.Module):
    def __init__(self, bottleneck_dim, output_dim, variance = 0.005):
        super(StochasticDecoder, self).__init__()
        self.fc1 = nn.Linear(bottleneck_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.log_var = nn.Parameter(torch.zeros(output_dim))
        self.elu = nn.ELU()
        self.variance = variance

        init.uniform_(self.fc1.weight, -0.005, 0.005)
        init.uniform_(self.fc2.weight, -0.005, 0.005)
        init.uniform_(self.fc3.weight, -0.005, 0.005)
        init.uniform_(self.fc4.weight, -0.005, 0.005)

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc4(x))
        mean = self.fc3(x)
        return mean 

class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim, noise_variance=0.005):
        super(EncoderDecoder, self).__init__()
        self.encoder = LinearEncoder(input_dim, bottleneck_dim)
        self.decoder = StochasticDecoder(bottleneck_dim, output_dim)
        self.noise_variance = noise_variance

    def forward(self, x):
        bottleneck = self.encoder(x)
        noise = torch.randn_like(bottleneck) * self.noise_variance
        bottleneck_noisy = bottleneck + noise
        reconstructed = self.decoder(bottleneck_noisy)
        return bottleneck, reconstructed

db_path = "/home/smart/Documents/IISC/sqlite_1.db"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_dataset = load_full_dataset(db_path)
# distance_nn = nn_distance(full_dataset)
# rg_distance = radius_of_gyration(full_dataset)

# features = np.hstack([distance_nn, rg_distance])
train_val_split = 0.8  
train_split = 0.75  

train_val_data, test_data = train_test_split(full_dataset, test_size=(1 - train_val_split), random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=(1 - train_split), random_state=42)

train_mean = np.mean(train_data, axis=0)
train_std = np.std(train_data, axis=0)
delta_t_values = [1]
results = []
batch_size = 32 

input_dim = 8252 # Dimensionality of the input features
bottleneck_dim = 1 # Bottleneck dimension (can be tuned)
output_dim = input_dim  # Output dimension should match input dimension for reconstruction

def test_model(model, test_dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    bottlenecks = []

    with torch.no_grad():
        for input_data, target_data in test_dataloader:
            input_data = input_data.to(device)
            target_data = target_data.to(device)

            bottleneck, output = model(input_data)
            loss = criterion(output, target_data)
            test_loss += loss.item()

            bottlenecks.append(bottleneck.cpu().numpy())

    test_loss /= len(test_dataloader)
    bottlenecks = np.concatenate(bottlenecks)
    return test_loss, bottlenecks

absolute_weights_per_dt = []

for delta_t in delta_t_values:
    print(f"Training with time delay: {delta_t}")
    
    train_dataset = TimeDelayedDataset(train_data, train_mean, train_std, delta_t, device = device)
    val_dataset = TimeDelayedDataset(val_data, train_mean, train_std, delta_t, device = device)
    test_dataset = TimeDelayedDataset(test_data, train_mean, train_std, delta_t, device = device)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = EncoderDecoder(input_dim, bottleneck_dim, output_dim)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.003)
    
    train_losses = []
    val_losses = []
    
    num_epochs = 11
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for input_data, target_data in train_dataloader:
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            
            bottleneck, output = model(input_data)
            loss = criterion(output, target_data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        
        # if epoch == 0:  # Capture the absolute weights after the first epoch
        #     absolute_weights = []
        #     with torch.no_grad():
        #         for name, param in model.named_parameters():
        #             if 'weight' in name:
        #                 absolute_weights.append(torch.abs(param).item())
        #     absolute_weights_per_dt.append(absolute_weights)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_data, target_data in val_dataloader:
                input_data = input_data.to(device)
                target_data = target_data.to(device)
                
                bottleneck, output = model(input_data)
                loss = criterion(output, target_data)
                val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Δt: {delta_t}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    test_loss, test_bottlenecks = test_model(model, test_dataloader, criterion, device = device)
    print(f'Test Loss for Δt={delta_t}: {test_loss:.4f}')
    
    results.append((delta_t, train_losses, val_losses, test_loss, test_bottlenecks))
