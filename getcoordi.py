import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sqlite3
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.fc5 = nn.Linear(400, 400)
        self.fc6 = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        # x = torch.tanh(self.fc2(x))
        # x = torch.tanh(self.fc3(x))
        # x = torch.tanh(self.fc4(x))
        # x = torch.tanh(self.fc5(x))
        x = (self.fc6(x))
        return x.float()

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(400, 400)
        self.fc6 = nn.Linear(64, input_dim)
        self.fc4 = nn.Linear(400, 400)
        self.fc5 = nn.Linear(400, 400)
        self.tanh = nn.Tanh()
        

    def forward(self, x, mdn_bool = True):
        x = self.tanh(self.fc1(x))
        # x = self.tanh(self.fc2(x))
        # x = self.tanh(self.fc3(x))
        # x = self.tanh(self.fc4(x))
        # x = self.tanh(self.fc5(x))
        
        x = self.fc6(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        
    def forward(self, x, mdn_bool = True):
        x = x.to(self.encoder.fc1.weight.dtype)
        latent_states = self.encoder(x)
        z = latent_states
        
        traj = self.decoder(z, mdn_bool)
        return traj

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

# Load and preprocess data
features, coordinates = load_full_dataset(db_path)
train_data = features[:80000]
val_data = features[80000:100000]
test_data = features[100000:1000001] 
test_dataset = MyDataset(test_data, coordinates[100000:], device=device)
batch_size = 32
train_dataset = MyDataset(train_data, coordinates[:10000], device=device)
val_dataset = MyDataset(val_data, coordinates[10001:12001], device=device)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
input_dim = 440
output_dim = 384
model = Autoencoder(input_dim, 2).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for inputs in val_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            valid_loss += loss.item()
        valid_loss /= len(val_dataloader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs in test_dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        test_loss += loss.item()
    test_loss /= len(test_dataloader)

print(f'Test Loss: {test_loss:.4f}')






