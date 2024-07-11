import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mdtraj as md
import sqlite3
from torch.utils.data import Dataset, DataLoader

# def datagen(batch_size):
#     conn = sqlite3.connect("sqlite.db")
#     cur = conn.cursor()
#     sql = f"""
#     SELECT frame_id FROM pairwise_distances ORDER BY random() LIMIT {batch_size}"""
#     sql_1 = f"""SELECT distances FROM pairwise_distances WHERE frame_id = ?"""
#     sql_2 = f"""SELECT * FROM data"""
#     cur.execute(sql_2)
#     coordinates = cur.fetchall()
#     coordinates = np.array(flattened_coordinates)
#     coordinates = coordinates.reshape(100001, 384)
#     while True:
#         pairwise_distances = []
#         flattened_coordinates =[]
#         cur.execute(sql)
#         frame_ids = cur.fetchall()
#         for frame_id in frame_ids:
#             cur.execute(sql_1, frame_id)
#             flattened_coordinates.append(coordinates[frame_id])
#             alldis = cur.fetchall()
#             pairwise_distances.append(alldis)
#         features = np.hstack([flattened_coordinates, pairwise_distances])
#         mean = np.mean(features, axis=0)
#         std = np.std(features, axis=0)
#         normalized_features = (features - mean) / std  
#         input_data = torch.tensor(normalized_features, dtype=torch.float32)
#         return input_data


# class PolymerDataset(Dataset):
#     def __init__(self, input_tensor):
#         self.data = input_tensor
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         return sample



class LinearEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(LinearEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, x):
        return self.linear(x)

class StochasticDecoder(nn.Module):
    def __init__(self, bottleneck_dim, output_dim):
        super(StochasticDecoder, self).__init__()
        self.fc1 = nn.Linear(bottleneck_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.log_var = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc3(x)
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = LinearEncoder(input_dim, bottleneck_dim)
        self.decoder = StochasticDecoder(bottleneck_dim, output_dim)

    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstructed = self.decoder(bottleneck)
        return bottleneck, reconstructed

def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    frames = []
    i = 0
    while i < len(lines):
        num_atoms = 128
        i += 2  # Skip the atom count line and the comment line
        frame = []
        for _ in range(num_atoms):
            if i<len(lines):
                atom_line = lines[i].strip().split()
            # Fix the unpacking error by checking the length of atom_line
            if len(atom_line) >= 4:
                    x, y, z = map(float, atom_line[1:])
            frame.append([x, y, z])
            i += 1
        frames.append(frame)

    return np.array(frames)
file_path = '/home/smart/Downloads/animation_1.xyz'
frames = read_xyz(file_path)

frames_test = frames[20001:40000]	
frames = frames[:20000]

flattened_coordinates = frames.reshape(frames.shape[0], -1)
flattened_coordinates_test = frames_test.reshape(frames_test.shape[0], -1)

def compute_pairwise_distances(frames):
    num_frames, num_atoms, _ = frames.shape
    pairwise_distances = []
    for frame in frames[:20000]:
        distances = []
        for i in range(128):
            for j in range(i, 128):
                if j!=i:
                   dist = np.linalg.norm(frame[i] - frame[j])
                   distances.append(dist)
                   j += 3
        pairwise_distances.append(distances)
    return np.array(pairwise_distances)
pairwise_distances = compute_pairwise_distances(frames)
pairwise_distances_test = compute_pairwise_distances(frames_test)

features = np.hstack([flattened_coordinates, pairwise_distances])
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
normalized_features = (features - mean) / std

features_test = np.hstack([flattened_coordinates_test, pairwise_distances_test])
mean = np.mean(features_test, axis=0)
std = np.std(features_test, axis=0)
normalized_features_test = (features_test - mean) / std

input_data = torch.tensor(normalized_features, dtype=torch.float32)
test_data = torch.tensor(normalized_features, dtype = torch.float32)

input_dim = input_data.shape[1]  # Dimensionality of the input features
print(input_dim)
bottleneck_dim = 1  # Bottleneck dimension (can be tuned)
output_dim = input_dim  # Output dimension should match input dimension for reconstruction

model = EncoderDecoder(input_dim, bottleneck_dim, output_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5000
for epoch in range(num_epochs):
    # Forward pass
    bottleneck, output = model(input_data)
    loss = criterion(output, input_data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    test_bottleneck, test_output = model(test_data)
    print(test_output)
    print(test_bottleneck)
    test_loss = criterion(test_output, test_data)
    print(f'Test Loss: {test_loss.item():.4f}')