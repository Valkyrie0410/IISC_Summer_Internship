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
    coordinates = np.array(coordinates).reshape(100001, 384)[1:]  # Adjust shape if needed
    conn.close()
    
    return coordinates



# Define your autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Bottleneck layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Output layer
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return bottleneck, reconstruction

# Define the custom dataset

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

# Create the dataset and dataloader
db_path = "/home/smart/Documents/IISC/sqlite_1.db"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_dataset = load_full_dataset(db_path)

train_val_split = 0.8  
train_split = 0.75  

train_val_data, test_data = train_test_split(full_dataset, test_size=(1 - train_val_split), random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=(1 - train_split), random_state=42)

train_mean = np.mean(train_data, axis=0)
train_std = np.std(train_data, axis=0)

train_dataset = PolymerDataset(train_data, train_mean, train_std, device = device)
val_dataset = PolymerDataset(val_data, train_mean, train_std, device = device)
test_dataset = PolymerDataset(test_data, train_mean, train_std, device = device)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

sample = train_dataset[0]
input_dim = 384  # Dimensionality of the input features
bottleneck_dim = 2 # Bottleneck dimension (can be tuned)
output_dim = input_dim  # Output dimension should match input dimension for reconstruction

model = Autoencoder(input_dim)

model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 11
train_losses = []
val_losses = []

num_epochs = 11
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
# After training, collect the bottleneck representations (order parameters)
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

plt.show()