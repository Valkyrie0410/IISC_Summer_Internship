import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mdtraj as md
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
    cur.execute("SELECT * FROM data")
    coordinates = cur.fetchall()
    coordinates = np.array(coordinates).reshape(200001, 384)[1:]  
    conn.close()
    
    return coordinates


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
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, bottleneck_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(bottleneck_dim)

    def forward(self, x):
        x = self.activation(self.batch_norm1(self.linear1(x)))
        x = self.dropout(x)
        x = self.batch_norm2(self.linear2(x))
        return x


class StochasticDecoder(nn.Module):
    def __init__(self, bottleneck_dim, output_dim, variance=0.005):
        super(StochasticDecoder, self).__init__()
        self.fc1 = nn.Linear(bottleneck_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(output_dim)
        self.variance = variance

    def forward(self, x):
        x = self.activation(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        mean = self.batch_norm2(self.fc2(x))
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

train_val_split = 0.8  
train_split = 0.75  

train_val_data, test_data = train_test_split(full_dataset, test_size=(1 - train_val_split), random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=(1 - train_split), random_state=42)

train_mean = np.mean(train_data, axis=0)
train_std = np.std(train_data, axis=0)
delta_t_values = [10500]
results = []
batch_size = 32 

input_dim = 384 # Dimensionality of the input features
bottleneck_dim = 2 # Bottleneck dimension (can be tuned)
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

def compute_2d_kde(bottleneck_values):
        kde = gaussian_kde(bottleneck_values.T)
        x_min, y_min = bottleneck_values.min(axis=0)
        x_max, y_max = bottleneck_values.max(axis=0)
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
        kde_values = kde(grid_coords).reshape(100, 100)
        return x_grid, y_grid, kde_values

def compute_kde_pdf(bottleneck_values):
    """Compute the PDF using kernel density estimation (KDE)."""
    bottleneck_values = bottleneck_values.ravel()  # Ensure bottleneck_values is a 1D array
    kde = gaussian_kde(bottleneck_values)
    x_values = np.linspace(np.min(bottleneck_values), np.max(bottleneck_values), 1000)
    pdf = kde(x_values)
    return x_values, pdf

def compute_free_energy(pdf, temperature=1):
    k_B = 1  # Assume k_B = 1 for simplicity
    free_energy = -k_B * temperature * np.log(pdf + 1e-10)  # Add a small constant to avoid log(0)
    return free_energy

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
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    
    num_epochs = 50
    early_stopping_patience = 10
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
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
        
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Δt: {delta_t}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), f'best_model_delta_t_{delta_t}.pth')
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
    
    model.load_state_dict(torch.load(f'best_model_delta_t_{delta_t}.pth'))
    test_loss, test_bottlenecks = test_model(model, test_dataloader, criterion, device)
    print(f'Test Loss for Δt={delta_t}: {test_loss:.4f}')
    
    results.append((delta_t, train_losses, val_losses, test_loss, test_bottlenecks))
