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
    
    cur.execute("SELECT * from data")
    coordinates = cur.fetchall()
    coordinates = np.array(coordinates).reshape(200001, 384)[1:]
    # Load coordinates
    cur.execute("SELECT * FROM bond_lengths")
    bond_lengths= cur.fetchall()
    bond_lengths = np.array(bond_lengths).reshape(200001, 127)[1:]  

    cur.execute("SELECT * FROM bond_angles")
    bond_angles = cur.fetchall()
    bond_angles = np.array(bond_angles).reshape(200001, 188)[1:]
    bond_angles = np.radians(bond_angles)

    cur.execute("SELECT * FROM dihedral_angles")
    dihedral_angles = cur.fetchall()
    dihedral_angles = np.array(dihedral_angles).reshape(200001, 125)[1:]
    dihedral_angles = np.radians(dihedral_angles)
    # cur.execute("SELECT * FROM pairwise_distances")
    # pairwise = cur.fetchall()
    # pairwise = np.array(pairwise).reshape(200001, 1830)[1:]

    conn.close()

    features = np.hstack([bond_lengths, bond_angles, dihedral_angles])
    
    return features, coordinates

class PolymerDataset(Dataset):
    def __init__(self, data, mean, std, device):
        self.data = data
        self.mean = mean
        self.std = std
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data[idx]
        normalized_features = (features - self.mean) / self.std
        input_data = torch.tensor(normalized_features, dtype=torch.float32).to(self.device)
        return input_data



class LinearEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(LinearEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 32)
        self.linear2 = nn.Linear(32, bottleneck_dim)
        init.uniform_(self.linear1.weight, -0.005, 0.005)
        init.uniform_(self.linear3.weight, -0.005, 0.005)
        init.uniform_(self.linear4.weight, -0.005, 0.005)
        init.uniform_(self.linear2.weight, -0.005, 0.005)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear3(x)
        x = self.linear2(x)
        return x


class StochasticDecoder(nn.Module):
    def __init__(self, bottleneck_dim, output_dim, variance = 0.005):
        super(StochasticDecoder, self).__init__()
        self.fc1 = nn.Linear(bottleneck_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)
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
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn_like(mean)
        noise = eps * std * self.variance# Gaussian noise
        return mean +noise

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

class MyDataset(Dataset):
    def __init__(self, data, data_target, device):
        self.data = data
        self.data_target = data_target
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        sample = torch.tensor(self.data[idx], dtype=torch.float32).to(self.device)
        return sample

db_path = "/home/smart/Documents/IISC/sqlite_1.db"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features, coordinates = load_full_dataset(db_path)

# train_val_split = 0.8  
# train_split = 0.75  

# train_val_data, test_data = train_test_split(full_dataset, test_size=(1 - train_val_split), random_state=42)
# train_data, val_data = train_test_split(train_val_data, test_size=(1 - train_split), random_state=42)

# train_mean = np.mean(train_data, axis=0)
# train_std = np.std(train_data, axis=0)

# train_dataset = PolymerDataset(train_data, train_mean, train_std, device = device)
# val_dataset = PolymerDataset(val_data, train_mean, train_std, device = device)
# test_dataset = PolymerDataset(test_data, train_mean, train_std, device = device)

# batch_size = 32
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# sample = train_dataset[0]

features, coordinates = load_full_dataset(db_path)
train_data = features[:80000]
val_data = features[80000:100000]
test_data = features[100000:] 
train_mean = np.mean(train_data, axis=(0,1))
train_std = np.std(train_data, axis=(0,1))
# train_data = (train_data-train_mean)/train_std
# val_data = (val_data-train_mean)/train_std
# test_data = (test_data-train_mean)/train_std

test_dataset = MyDataset(test_data, coordinates[100000:], device=device)
batch_size = 32
train_dataset = MyDataset(train_data, coordinates[:10000], device=device)
val_dataset = MyDataset(val_data, coordinates[10001:12001], device=device)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
input_dim = 440

# input_dim = 384 # Dimensionality of the input features
bottleneck_dim = 4 # Bottleneck dimension (can be tuned)
output_dim = input_dim  # Output dimension should match input dimension for reconstruction

model = EncoderDecoder(input_dim, bottleneck_dim, output_dim)

criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.003)

model.to(device)

train_losses = []
val_losses = []

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for data in train_dataloader:
        input_data = data
        input_data = input_data.to(device)
        bottleneck, output = model(input_data)
        loss = criterion(output, input_data)

    # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_dataloader:
            input_data = data.to(device)
            bottleneck, output = model(input_data)
            loss = criterion(output, input_data)
            val_loss += loss.item()
    
    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

def test_model(model, test_dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    bottlenecks = []

    with torch.no_grad():
        for data in test_dataloader:
            input_data = data.to(device)
            bottleneck, output = model(input_data)
            print(output[0][315:].cpu())
            loss = criterion(output, input_data)
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

# Load the bottleneck values obtained from the test set
test_loss, test_bottlenecks = test_model(model, test_dataloader, criterion, device)
print(f'Test Loss: {test_loss:.4f}')

# Ensure the bottleneck values have exactly 2 dimensions
assert test_bottlenecks.shape[1] == 2, "Bottleneck dimension must be 2."

# Compute the 2D KDE of the bottleneck values
x_grid, y_grid, kde_values = compute_2d_kde(test_bottlenecks)

# Compute the free energy distribution
free_energy = compute_free_energy(kde_values)

# Plot the 3D surface plot for free energy
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, free_energy, cmap='viridis')

ax.set_xlabel('Bottleneck Dimension 1')
ax.set_ylabel('Bottleneck Dimension 2')
ax.set_zlabel('Free Energy')
ax.set_title('Free Energy Distribution')

min_bottleneck_index = np.argmin(np.linalg.norm(test_bottlenecks, axis=1))
min_bottleneck_value = test_bottlenecks[min_bottleneck_index]

print(f'Global Minimum Bottleneck Value: {min_bottleneck_value}')
print(f'Corresponding Frame Index: {min_bottleneck_index}')

# x_values, pdf = compute_kde_pdf(test_bottlenecks)

# # Compute the free energy distribution
# free_energy = compute_free_energy(pdf, temperature=1)

# # Plot the free energy distribution as a line graph
# plt.figure(figsize=(10, 5))
# plt.plot(x_values, free_energy, label='Free Energy Distribution')
# plt.xlabel('Bottleneck Value')
# plt.ylabel('Free Energy')
# plt.legend()
# plt.show()

plt.show()