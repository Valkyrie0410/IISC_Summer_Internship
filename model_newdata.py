import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sqlite3
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_full_dataset(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Load coordinates
    cur.execute("SELECT * FROM data")
    coordinates = cur.fetchall()
    coordinates = np.array(coordinates).reshape(200001, 384)[1:]  
    selected_columns = []
    for i in range(0, coordinates.shape[1], 6):
        selected_columns.extend([i, i+1, i+2])

    coordinates = coordinates[:, selected_columns]
    coordinates = coordinates.reshape(200000, 64, 3)
    conn.close()
    
    return coordinates

def calculate_bond_lengths(coordinates, bonds):
    bond_lengths = []
    for bond in bonds:
        atom1, atom2 = bond
        length = np.linalg.norm(coordinates[atom1] - coordinates[atom2])
        bond_lengths.append(length)
    return np.array(bond_lengths)

def calculate_bond_angles(coordinates, angles):
    bond_angles = []
    for angle in angles:
        atom1, atom2, atom3 = angle
        vec1 = coordinates[atom1] - coordinates[atom2]
        vec2 = coordinates[atom3] - coordinates[atom2]
        cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        bond_angles.append(np.degrees(angle_rad))
    return np.array(bond_angles)

def calculate_dihedral_angles(coordinates, dihedrals):
    dihedral_angles = []
    for dihedral in dihedrals:
        atom1, atom2, atom3, atom4 = dihedral
        b1 = coordinates[atom2] - coordinates[atom1]
        b2 = coordinates[atom3] - coordinates[atom2]
        b3 = coordinates[atom4] - coordinates[atom3]
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        
        m1 = np.cross(n1, b2/np.linalg.norm(b2))
        
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        dihedral_angle = np.arctan2(y, x)
        dihedral_angles.append(np.degrees(dihedral_angle))
    return np.array(dihedral_angles)

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

    def forward(self, x):
        x = self.linear1(x)
        return x

class StochasticDecoder(nn.Module):
    def __init__(self, bottleneck_dim, output_dim, variance=0.005):
        super(StochasticDecoder, self).__init__()
        self.fc1 = nn.Linear(bottleneck_dim, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.log_var = nn.Parameter(torch.zeros(output_dim))
        self.elu = nn.ELU()
        self.variance = variance

        init.uniform_(self.fc1.weight, -0.005, 0.005)
        init.uniform_(self.fc3.weight, -0.005, 0.005)

    def forward(self, x):
        x = self.elu(self.fc1(x))
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


db_path = "/home/smart/Documents/IISC/sqlite_1.db"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

coordinates = load_full_dataset(db_path)

num_atoms = 64
num_frames = 200000
bonds = [(i, i + 1) for i in range(num_atoms - 1)]
angles = [(i, i + 1, i + 2) for i in range(num_atoms - 2)]
dihedrals = [(i, i + 1, i + 2, i + 3) for i in range(num_atoms - 3)]

bond_angles = np.zeros((num_frames, len(angles)))
bond_lengths = np.zeros((num_frames, len(bonds)))
dihedral_angles = np.zeros((num_frames, len(dihedrals)))

for i in range(num_frames):
    frame_coordinates = coordinates[i]
    bond_length = calculate_bond_lengths(frame_coordinates, bonds)
    bond_angle = calculate_bond_angles(frame_coordinates, angles)
    dihedral_angle = calculate_dihedral_angles(frame_coordinates, dihedrals)
    bond_lengths[i] = bond_length
    bond_angles[i] = bond_angle 
    dihedral_angles[i] = dihedral_angle 

features = np.hstack([bond_lengths, bond_angles, dihedral_angles])
train_val_split = 0.8  
train_split = 0.75  

train_val_data, test_data = train_test_split(features, test_size=(1 - train_val_split), random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=(1 - train_split), random_state=42)

train_mean = np.mean(train_data, axis=0)
train_std = np.std(train_data, axis=0)
delta_t_values = np.arange(1, 11)
results = []
batch_size = 32

# input_dim = 2  # Dimensionality of the input features
# bottleneck_dim = 1  # Bottleneck dimension (can be tuned)
# output_dim = input_dim  # Output dimension should match input dimension for reconstruction

for delta_t in delta_t_values:
    print(f"Training with time delay: {delta_t}")
    
    train_dataset = TimeDelayedDataset(train_data, train_mean, train_std, delta_t, device=device)
    val_dataset = TimeDelayedDataset(val_data, train_mean, train_std, delta_t, device=device)
    test_dataset = TimeDelayedDataset(test_data, train_mean, train_std, delta_t, device=device)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = features.shape[0] # Dimensionality of the input features
    bottleneck_dim = 1  # Bottleneck dimension (can be tuned)
    output_dim = input_dim  # Output dimension should match input dimension for reconstruction

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
        
        if epoch == 10:  # Capture the absolute weights after the first epoch
            weights = model.encoder.linear1.weight.detach().cpu().numpy()
            absolute_weights_nn.append(np.abs(weights[0, 0]))
            absolute_weights_rg.append(np.abs(weights[0, 1]))

        model.eval()
        val_loss = 0.0
        
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
    
    test_loss, test_bottlenecks = test_model(model, test_dataloader, criterion, device=device)
    print(f'Test Loss for Δt={delta_t}: {test_loss:.4f}')
    
    results.append((delta_t, train_losses, val_losses, test_loss, test_bottlenecks))

# Plotting the absolute weights for different time delays
print(absolute_weights_nn)
print(absolute_weights_rg)

plt.figure(figsize=(12, 6))
plt.plot(delta_t_values, absolute_weights_nn, label='NN Distance Weights')
plt.plot(delta_t_values, absolute_weights_rg, label='Radius of Gyration Weights')
plt.xlabel('Predictive Time Delay (Δt)')
plt.ylabel('Absolute Weights (First Epoch)')
plt.title('Absolute Weights for Different Order Parameters as a Function of Δt')
plt.legend()
plt.grid(True)
plt.show()


